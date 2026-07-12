import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { Tensor, loadArti } from '../src/index.js';

interface TensorPayload { dims: number[]; data: number[] }
interface FixtureCase { inputs: Record<string, TensorPayload>; expected: TensorPayload }
interface Fixture { name: string; tolerance: {atol: number; rtol: number}; cases: FixtureCase[] }

const defaultRoot = path.resolve(import.meta.dirname, '../../../.tmp/web-fixtures');
const root = process.env.ARTI_WEB_FIXTURES ?? defaultRoot;
const run = existsSync(root) ? describe : describe.skip;
const require = createRequire(import.meta.url);
const ortEntry = require.resolve('onnxruntime-web/webgpu');
const wasmPath = path.join(path.dirname(ortEntry), 'ort-wasm-simd-threaded.asyncify.wasm');
let wasmBinary: Uint8Array;

run('PyTorch to ONNX Runtime Web parity', () => {
  for (const name of ['half', 'fold-salience', 'fold-q', 'learned-pulse']) {
    it(`${name} matches PyTorch for dynamic token shapes`, async () => {
      const directory = path.resolve(root, name);
      const fixture = JSON.parse(await readFile(path.join(directory, 'case.json'), 'utf8')) as Fixture;
      wasmBinary ??= new Uint8Array(await readFile(wasmPath));
      const module = await loadArti(`http://arti.local/${name}/`, {device: 'wasm', fetch: fixtureFetch(root), wasmBinary, wasmNumThreads: 1});
      expect(module.device).toBe('wasm');
      for (const item of fixture.cases) {
        const tensors = Object.fromEntries(Object.entries(item.inputs).map(([key, value]) => [key, tensor(value)]));
        const first = await module.forward(tensors.x!, {q: tensors.q, mask: tensors.mask});
        const second = await module.forward(tensors.x!, {q: tensors.q, mask: tensors.mask});
        const firstData = Array.from(await first.getData()) as number[];
        const secondData = Array.from(await second.getData()) as number[];
        expect(first.dims).toEqual(item.expected.dims);
        expectError(firstData, item.expected.data, fixture.tolerance);
        expect(secondData).toEqual(firstData);
        first.dispose();
        second.dispose();
      }
      const sample = fixture.cases[0]!;
      const sampleInputs = Object.fromEntries(Object.entries(sample.inputs).map(([key, value]) => [key, tensor(value)]));
      const output = new Tensor('float32', new Float32Array(sample.expected.data.length), sample.expected.dims);
      await expect(module.forwardInto(output, sampleInputs.x!, {q: sampleInputs.q, mask: sampleInputs.mask})).rejects.toThrow(/gpu-buffer/);
      output.dispose();
      await module.dispose();
      const x = tensor(fixture.cases[0]!.inputs.x!);
      await expect(module.forward(x)).rejects.toThrow(/disposed/);
    });
  }

  it('enforces declared optional inputs and model hashes', async () => {
    const fetcher = fixtureFetch(root);
    wasmBinary ??= new Uint8Array(await readFile(wasmPath));
    const module = await loadArti('http://arti.local/fold-q/', {device: 'wasm', fetch: fetcher, wasmBinary, wasmNumThreads: 1});
    const fixture = JSON.parse(await readFile(path.join(root, 'fold-q', 'case.json'), 'utf8')) as Fixture;
    const x = tensor(fixture.cases[0]!.inputs.x!);
    await expect(module.forward(x)).rejects.toThrow(/q is required/);
    const q = tensor(fixture.cases[0]!.inputs.q!);
    const mask = tensor(fixture.cases[0]!.inputs.mask!);
    const wrong = new Tensor('float32', new Float32Array(2 * 5 * 3), [2, 5, 3]);
    await expect(module.forward(wrong, {q, mask})).rejects.toThrow(/dimension 2/);
    await module.dispose();

    const automatic = await loadArti('http://arti.local/half/', {device: 'auto', fetch: fetcher, wasmBinary, wasmNumThreads: 1});
    expect(automatic.device).toBe('wasm');
    await automatic.dispose();

    const corrupt: typeof fetch = async (input, init) => {
      const response = await fetcher(input, init);
      if (new URL(input instanceof Request ? input.url : input.toString()).pathname.endsWith('/model.onnx')) {
        return new Response(new Uint8Array([1, 2, 3]), {status: 200});
      }
      return response;
    };
    await expect(loadArti('http://arti.local/half/', {device: 'wasm', fetch: corrupt, wasmBinary, wasmNumThreads: 1})).rejects.toThrow(/size|SHA-256/);
  });
});

function tensor(value: TensorPayload): Tensor {
  return new Tensor('float32', Float32Array.from(value.data), value.dims);
}

function fixtureFetch(fixtures: string): typeof fetch {
  return async (input) => {
    const url = new URL(input instanceof Request ? input.url : input.toString());
    const relative = url.pathname.replace(/^\/+/, '');
    const target = path.resolve(fixtures, relative);
    if (!target.startsWith(path.resolve(fixtures) + path.sep)) return new Response(null, {status: 403});
    try {
      return new Response(await readFile(target), {status: 200});
    } catch {
      return new Response(null, {status: 404});
    }
  };
}

function expectError(actual: number[], expected: number[], tolerance: {atol: number; rtol: number}): void {
  expect(actual.length).toBe(expected.length);
  let maxAbsolute = 0;
  let maxRelative = 0;
  for (let index = 0; index < actual.length; index += 1) {
    const difference = Math.abs(actual[index]! - expected[index]!);
    maxAbsolute = Math.max(maxAbsolute, difference);
    maxRelative = Math.max(maxRelative, difference / Math.max(Math.abs(expected[index]!), 1e-8));
  }
  expect(maxAbsolute).toBeLessThanOrEqual(tolerance.atol);
  expect(maxRelative).toBeLessThanOrEqual(tolerance.rtol);
}
