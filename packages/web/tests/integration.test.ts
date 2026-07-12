import {createHash, webcrypto} from 'node:crypto';
import {beforeEach, describe, expect, it, vi} from 'vitest';

const ortMock = vi.hoisted(() => ({create: vi.fn()}));

vi.mock('onnxruntime-web/webgpu', async (load) => {
  const actual = await load<typeof import('onnxruntime-web/webgpu')>();
  return {...actual, InferenceSession: {...actual.InferenceSession, create: ortMock.create}};
});

import {loadArti, requireMatchingFileSets} from '../src/runtime.js';

Object.defineProperty(globalThis, 'crypto', {value: webcrypto, configurable: true});

describe('loadArti integration surface', () => {
  beforeEach(() => ortMock.create.mockReset().mockImplementation(async () => ({run: vi.fn(), release: vi.fn()})));
  it('loads a verified artifact with progress and device diagnostics', async () => {
    const model = new Uint8Array([1, 2, 3]);
    const modelHash = createHash('sha256').update(model).digest('hex');
    const manifest = {
      format: 'arti.web', format_version: 2, package_version: 'test',
      producer: {backend: 'torch', graph_format: 'onnx'}, module: {type: 'Test', config: {}},
      runtime: {dtype: 'float32', opset_version: 18, execution_providers: ['wasm']},
      inputs: [{name: 'x', dtype: 'float32', shape: [1]}], outputs: [{name: 'y', dtype: 'float32', shape: [1]}],
      files: {'model.onnx': {sha256: modelHash, size: model.byteLength}},
    };
    const manifestBytes = new TextEncoder().encode(JSON.stringify(manifest));
    const manifestHash = createHash('sha256').update(manifestBytes).digest('hex');
    const lock = {format: 'arti.web', format_version: 2, manifest: {file: 'arti-web.json', sha256: manifestHash}, files: {'model.onnx': {sha256: modelHash, size: model.byteLength}}};
    const files: Record<string, Uint8Array> = {'arti-web.json': manifestBytes, 'arti-web.lock.json': new TextEncoder().encode(JSON.stringify(lock)), 'model.onnx': model};
    const signals: (AbortSignal | null | undefined)[] = [];
    const fetcher: typeof fetch = async (input, init) => { signals.push(init?.signal); const name = new URL(input instanceof Request ? input.url : input.toString()).pathname.split('/').pop()!; return new Response(files[name], {status: files[name] ? 200 : 404}); };
    const stages: string[] = [];
    const controller = new AbortController();
    const module = await loadArti('https://arti.test/demo/', {device: 'wasm', fetch: fetcher, signal: controller.signal, onProgress: ({stage}) => stages.push(stage)});
    expect(signals).toEqual([controller.signal, controller.signal, controller.signal]);
    expect(stages).toEqual(expect.arrayContaining(['manifest', 'lock', 'model', 'verify', 'initialize', 'ready']));
    expect(module.diagnostics).toMatchObject({artifactUrl: 'https://arti.test/demo/', selectedDevice: 'wasm'});
    expect(module.diagnostics.attempts).toHaveLength(1);
    expect(module.diagnostics.attempts[0]).toMatchObject({device: 'wasm', success: true});
    await module.dispose();
  });

  it('releases a newly created session when cancellation wins initialization', async () => {
    const controller = new AbortController();
    const release = vi.fn(async () => undefined);
    ortMock.create.mockImplementationOnce(async () => {
      controller.abort('cancel after session creation');
      return {run: vi.fn(), release};
    });
    const model = new Uint8Array([1, 2, 3]);
    const modelHash = createHash('sha256').update(model).digest('hex');
    const manifest = {format: 'arti.web', format_version: 2, package_version: 'test', producer: {backend: 'torch', graph_format: 'onnx'}, module: {type: 'Test', config: {}}, runtime: {dtype: 'float32', opset_version: 18, execution_providers: ['wasm']}, inputs: [{name: 'x', dtype: 'float32', shape: [1]}], outputs: [{name: 'y', dtype: 'float32', shape: [1]}], files: {'model.onnx': {sha256: modelHash, size: model.byteLength}}};
    const manifestBytes = new TextEncoder().encode(JSON.stringify(manifest));
    const manifestHash = createHash('sha256').update(manifestBytes).digest('hex');
    const lock = {format: 'arti.web', format_version: 2, manifest: {file: 'arti-web.json', sha256: manifestHash}, files: {'model.onnx': {sha256: modelHash, size: model.byteLength}}};
    const files: Record<string, Uint8Array> = {'arti-web.json': manifestBytes, 'arti-web.lock.json': new TextEncoder().encode(JSON.stringify(lock)), 'model.onnx': model};
    const fetcher: typeof fetch = async (input) => {
      const name = new URL(input instanceof Request ? input.url : input.toString()).pathname.split('/').pop()!;
      return new Response(files[name], {status: files[name] ? 200 : 404});
    };
    await expect(loadArti('https://arti.test/demo/', {device: 'wasm', fetch: fetcher, signal: controller.signal})).rejects.toMatchObject({code: 'ABORTED'});
    expect(release).toHaveBeenCalledOnce();
  });

  it('rejects extra lock files for v2 and v3 file sets', () => {
    for (const files of [{'model.onnx': {}}, {'read.onnx': {}, 'update.onnx': {}}]) {
      expect(() => requireMatchingFileSets(files, {...files, 'extra.onnx': {}}, new URL('https://arti.test/demo/'))).toThrow(/file sets disagree/);
    }
  });
});
