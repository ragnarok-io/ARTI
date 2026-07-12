import {Tensor} from 'onnxruntime-web/webgpu';
import {describe, expect, it, vi} from 'vitest';

import {ARTIWebModule} from '../src/runtime.js';
import type {ARTIWebManifest} from '../src/generated/contract.js';

const contract = (name: string) => ({name, dtype: 'float32' as const, shape: ['batch', 2]});
const manifest = {inputs: [contract('x')], outputs: [contract('y')]} as unknown as ARTIWebManifest;

describe('ARTIWebModule.predict', () => {
  it('converts CPU values, downloads outputs, and releases owned tensors', async () => {
    const output = new Tensor('float32', new Float32Array([2, 4]), [1, 2]);
    const originalOutputData = output.data;
    const outputDispose = vi.spyOn(output, 'dispose');
    let received: Tensor | undefined;
    const session = {run: vi.fn(async (feeds: Record<string, Tensor>) => { received = feeds.x; return {y: output}; }), release: vi.fn()} as never;
    const module = new ARTIWebModule(session, manifest, 'wasm');
    const result = await module.predict({x: {data: new Float32Array([1, 2]), dims: [1, 2]}});
    expect(Array.from(result.y!.data)).toEqual([2, 4]);
    expect(result.y!.data).not.toBe(originalOutputData);
    expect(outputDispose).toHaveBeenCalledOnce();
    expect(received).toBeDefined();
    expect(() => received!.data).toThrow();
  });

  it('does not dispose caller-owned ORT inputs and still cleans outputs on abort', async () => {
    const input = new Tensor('float32', new Float32Array([1, 2]), [1, 2]);
    const inputDispose = vi.spyOn(input, 'dispose');
    const output = new Tensor('float32', new Float32Array([2, 4]), [1, 2]);
    const outputDispose = vi.spyOn(output, 'dispose');
    const controller = new AbortController();
    let received: Tensor | undefined;
    const session = {run: vi.fn(async (feeds: Record<string, Tensor>) => { received = feeds.x; controller.abort(); return {y: output}; }), release: vi.fn()} as never;
    const module = new ARTIWebModule(session, manifest, 'wasm');
    await expect(module.predict({x: input}, {signal: controller.signal})).rejects.toMatchObject({code: 'ABORTED'});
    expect(inputDispose).not.toHaveBeenCalled();
    expect(received).toBe(input);
    expect(outputDispose).toHaveBeenCalledOnce();
  });

  it('keeps predict in flight until GPU output download completes', async () => {
    let finishDownload!: (value: Float32Array) => void;
    const output = new Tensor('float32', new Float32Array([2, 4]), [1, 2]);
    vi.spyOn(output, 'getData').mockImplementation(() => new Promise((resolve) => { finishDownload = resolve as (value: Float32Array) => void; }));
    const session = {run: vi.fn(async () => ({y: output})), release: vi.fn(async () => undefined)} as never;
    const module = new ARTIWebModule(session, manifest, 'webgpu');
    const prediction = module.predict({x: {data: new Float32Array([1, 2]), dims: [1, 2]}});
    await vi.waitFor(() => expect(finishDownload).toBeTypeOf('function'));
    const disposing = module.dispose();
    expect(session.release).not.toHaveBeenCalled();
    finishDownload(new Float32Array([2, 4]));
    await prediction;
    await disposing;
    expect(session.release).toHaveBeenCalledOnce();
  });
});
