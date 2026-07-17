import {Tensor} from 'onnxruntime-web/webgpu';
import {describe, expect, it, vi} from 'vitest';

import {ARTIWebModule} from '../src/runtime.js';
import type {ARTIWebManifest, TensorContract} from '../src/generated/contract.js';
import type {TensorMap} from '../src/types.js';

function contract(name: string, shape: Array<number | string>, role: 'primary' | 'workspace' | 'diagnostic' = 'diagnostic'): TensorContract {
  return {name, dtype: 'float32', logical_type: 'tensor', shape, dynamic_axes: {}, max_bytes: 1024, tolerance: {atol: 1e-4, rtol: 1e-3}, role};
}

function manifest(): ARTIWebManifest {
  return {
    format: 'arti.web', format_version: 2, package_version: 'test',
    producer: {backend: 'torch', graph_format: 'onnx'},
    module: {type: 'tests.Inspectable', config: {}, forward_kwargs: {return_info: true}},
    runtime: {dtype: 'float32', opset_version: 18, execution_providers: ['webgpu', 'wasm']},
    inputs: [contract('x', ['batch', 2])],
    outputs: [contract('fused', ['batch', 2], 'primary'), contract('workspace', ['batch', 4], 'workspace'), contract('survival', ['batch', 2])],
    files: {'model.onnx': {sha256: '0'.repeat(64), size: 1}},
  };
}

function input(): Tensor { return new Tensor('float32', new Float32Array([1, 2]), [1, 2]); }
function output(data: number[], dims: number[]): Tensor { return new Tensor('float32', Float32Array.from(data), dims); }

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((yes) => { resolve = yes; });
  return {promise, resolve};
}

describe('ARTIWebModule.inspect', () => {
  it('uses ORT selective fetches and returns explicitly owned outputs', async () => {
    const fused = output([1, 2], [1, 2]);
    const workspace = output([1, 2, 3, 4], [1, 4]);
    const fusedDispose = vi.spyOn(fused, 'dispose');
    const workspaceDispose = vi.spyOn(workspace, 'dispose');
    const session = {run: vi.fn(async () => ({fused, workspace})), release: vi.fn(async () => undefined)};
    const module = new ARTIWebModule(session as never, manifest(), 'wasm');
    const x = input();

    const result = await module.inspect({x}, {outputs: ['fused', 'workspace']});

    expect(session.run).toHaveBeenCalledWith({x}, ['fused', 'workspace']);
    expect(result.device).toBe('wasm');
    expect(result.timings.inferenceMs).toBeGreaterThanOrEqual(0);
    expect(Object.keys(result.outputs)).toEqual(['fused', 'workspace']);
    const downloaded = await result.download(['workspace']);
    expect(Array.from(downloaded.workspace!.data)).toEqual([1, 2, 3, 4]);
    await result.dispose();
    expect(fusedDispose).toHaveBeenCalledOnce();
    expect(workspaceDispose).toHaveBeenCalledOnce();
    await expect(result.download()).rejects.toThrow(/expired/);
    x.dispose();
    await module.dispose();
  });

  it('disposes outputs when an accepted run is cancelled before ORT returns', async () => {
    const execution = deferred<TensorMap>();
    const session = {run: vi.fn(() => execution.promise), release: vi.fn(async () => undefined)};
    const module = new ARTIWebModule(session as never, manifest(), 'wasm');
    const controller = new AbortController();
    const x = input();
    const running = module.inspect({x}, {outputs: ['fused'], signal: controller.signal});
    controller.abort('cancelled');
    const fused = output([1, 2], [1, 2]);
    const dispose = vi.spyOn(fused, 'dispose');
    execution.resolve({fused});

    await expect(running).rejects.toMatchObject({code: 'ABORTED'});
    expect(dispose).toHaveBeenCalledOnce();
    x.dispose();
    await module.dispose();
  });

  it('expires in-flight and retained runs before unloading or changing device', async () => {
    const execution = deferred<TensorMap>();
    const oldSession = {run: vi.fn(() => execution.promise), release: vi.fn(async () => undefined)};
    const oldModule = new ARTIWebModule(oldSession as never, manifest(), 'wasm');
    const x = input();
    const running = oldModule.inspect({x}, {outputs: ['fused']});
    const unloading = oldModule.dispose();
    const stale = output([1, 2], [1, 2]);
    const staleDispose = vi.spyOn(stale, 'dispose');
    execution.resolve({fused: stale});

    await expect(running).rejects.toThrow(/expired/);
    await unloading;
    expect(staleDispose).toHaveBeenCalledOnce();
    expect(oldSession.release).toHaveBeenCalledOnce();

    const current = output([3, 4], [1, 2]);
    const currentDispose = vi.spyOn(current, 'dispose');
    const newSession = {run: vi.fn(async () => ({fused: current})), release: vi.fn(async () => undefined)};
    const newModule = new ARTIWebModule(newSession as never, manifest(), 'webgpu');
    const currentRun = await newModule.inspect({x}, {outputs: ['fused']});
    expect(currentRun.device).toBe('webgpu');
    await newModule.dispose();
    expect(currentDispose).toHaveBeenCalledOnce();
    expect(newSession.release).toHaveBeenCalledOnce();
    await expect(currentRun.download()).rejects.toThrow(/expired/);
    x.dispose();
  });

  it('rejects unknown, duplicate, empty, and over-budget output selections', async () => {
    const session = {run: vi.fn(async () => ({fused: output([1, 2], [1, 2])})), release: vi.fn(async () => undefined)};
    const value = manifest();
    value.outputs[0]!.max_bytes = 1;
    const module = new ARTIWebModule(session as never, value, 'wasm');
    const x = input();
    await expect(module.inspect({x}, {outputs: []})).rejects.toThrow(/empty/);
    await expect(module.inspect({x}, {outputs: ['missing']})).rejects.toThrow(/not a declared/);
    await expect(module.inspect({x}, {outputs: ['fused', 'fused']})).rejects.toThrow(/duplicates/);
    await expect(module.inspect({x}, {outputs: ['fused']})).rejects.toThrow(/budget/);
    x.dispose();
    await module.dispose();
  });
});
