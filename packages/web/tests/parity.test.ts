import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { Tensor, loadArti, loadArtiStateful } from '../src/index.js';

interface TensorPayload { dims: number[]; data: number[] }
interface FixtureCase { inputs: Record<string, TensorPayload>; outputs: Record<string, TensorPayload> }
interface Fixture { name: string; tolerance: {atol: number; rtol: number}; cases: FixtureCase[] }

const defaultRoot = path.resolve(import.meta.dirname, '../../../.tmp/web-fixtures');
const root = process.env.ARTI_WEB_FIXTURES ?? defaultRoot;
const run = existsSync(root) ? describe : describe.skip;
const require = createRequire(import.meta.url);
const ortEntry = require.resolve('onnxruntime-web/webgpu');
const wasmPath = path.join(path.dirname(ortEntry), 'ort-wasm-simd-threaded.asyncify.wasm');
let wasmBinary: Uint8Array;

run('Python graph to ONNX Runtime Web parity', () => {
  for (const name of ['half', 'fold-salience', 'fold-q', 'learned-pulse', 'generic-affine']) {
    it(`${name} executes without JavaScript mechanism logic`, async () => {
      const directory = path.resolve(root, name);
      const fixture = JSON.parse(await readFile(path.join(directory, 'case.json'), 'utf8')) as Fixture;
      wasmBinary ??= new Uint8Array(await readFile(wasmPath));
      const module = await loadArti(`http://arti.local/${name}/`, {device: 'wasm', fetch: fixtureFetch(root), wasmBinary, wasmNumThreads: 1});
      expect(module.device).toBe('wasm');
      for (const item of fixture.cases) {
        const inputs = tensors(item.inputs);
        const first = await module.run(inputs);
        const second = await module.run(inputs);
        for (const [outputName, expected] of Object.entries(item.outputs)) {
          const firstTensor = first[outputName]!;
          const secondTensor = second[outputName]!;
          const firstData = Array.from(await firstTensor.getData()) as number[];
          const secondData = Array.from(await secondTensor.getData()) as number[];
          expect(firstTensor.dims).toEqual(expected.dims);
          expectError(firstData, expected.data, fixture.tolerance);
          expect(secondData).toEqual(firstData);
          firstTensor.dispose();
          secondTensor.dispose();
        }
      }
      const sample = fixture.cases[0]!;
      const outputs = Object.fromEntries(Object.entries(sample.outputs).map(([outputName, expected]) => [outputName, new Tensor('float32', new Float32Array(expected.data.length), expected.dims)]));
      await expect(module.run(tensors(sample.inputs), outputs)).rejects.toThrow(/gpu-buffer/);
      for (const output of Object.values(outputs)) output.dispose();
      await module.dispose();
      await expect(module.run(tensors(sample.inputs))).rejects.toThrow(/disposed/);
    });
  }

  it('keeps forward as a single-input/single-output convenience only', async () => {
    const fetcher = fixtureFetch(root);
    wasmBinary ??= new Uint8Array(await readFile(wasmPath));
    const half = await loadArti('http://arti.local/half/', {device: 'wasm', fetch: fetcher, wasmBinary, wasmNumThreads: 1});
    const halfFixture = JSON.parse(await readFile(path.join(root, 'half', 'case.json'), 'utf8')) as Fixture;
    const y = await half.forward(tensor(halfFixture.cases[0]!.inputs.x!));
    y.dispose();
    await half.dispose();

    const generic = await loadArti('http://arti.local/generic-affine/', {device: 'wasm', fetch: fetcher, wasmBinary, wasmNumThreads: 1});
    await expect(generic.forward(tensor(halfFixture.cases[0]!.inputs.x!))).rejects.toThrow(/use run/);
    await generic.dispose();
  });

  it('enforces Python-declared names, shapes, and model hashes', async () => {
    const fetcher = fixtureFetch(root);
    wasmBinary ??= new Uint8Array(await readFile(wasmPath));
    const module = await loadArti('http://arti.local/generic-affine/', {device: 'wasm', fetch: fetcher, wasmBinary, wasmNumThreads: 1});
    const fixture = JSON.parse(await readFile(path.join(root, 'generic-affine', 'case.json'), 'utf8')) as Fixture;
    const inputs = tensors(fixture.cases[0]!.inputs);
    await expect(module.run({signal: inputs.signal!})).rejects.toThrow(/gate is required/);
    await expect(module.run({...inputs, invented: inputs.signal!})).rejects.toThrow(/invented is not declared/);
    const wrong = new Tensor('float32', new Float32Array(2 * 5 * 3), [2, 5, 3]);
    await expect(module.run({...inputs, signal: wrong})).rejects.toThrow(/dimension 2/);
    wrong.dispose();
    await module.dispose();

    const corrupt: typeof fetch = async (input, init) => {
      const response = await fetcher(input, init);
      if (new URL(input instanceof Request ? input.url : input.toString()).pathname.endsWith('/model.onnx')) return new Response(new Uint8Array([1, 2, 3]), {status: 200});
      return response;
    };
    await expect(loadArti('http://arti.local/half/', {device: 'wasm', fetch: corrupt, wasmBinary, wasmNumThreads: 1})).rejects.toThrow(/size|SHA-256/);
  });

  it('runs stateful Recall read/commit/snapshot/restore/fork/reset without parameter training', async () => {
    const directory = path.resolve(root, 'stateful-recall');
    const fixture = JSON.parse(await readFile(path.join(directory, 'case.json'), 'utf8')) as {
      inputs: Record<string, TensorPayload>; expected: Record<string, TensorPayload>; tolerance: {atol: number; rtol: number};
    };
    wasmBinary ??= new Uint8Array(await readFile(wasmPath));
    const url = 'http://arti.local/stateful-recall/';
    const options = {device: 'wasm' as const, fetch: fixtureFetch(root), wasmBinary, wasmNumThreads: 1};
    const module = await loadArtiStateful(url, options);
    const full = tensor(fixture.inputs.full!); const mask = tensor(fixture.inputs.mask!);
    const first = await module.run('read', {x: full, mask});
    await expectTensor(first.recognition!, fixture.expected.initial_recognition!, fixture.tolerance);
    for (let index = 0; index < 4; index += 1) {
      const diagnostics = await module.commit('update', {trace_key: first.trace_key!, observed: full, mask});
      dispose(diagnostics);
    }
    const corrupt = tensor(fixture.inputs.corrupt!); const novel = tensor(fixture.inputs.unseen!);
    const seen = await module.run('read', {x: corrupt, mask});
    const unseen = await module.run('read', {x: novel, mask});
    await expectTensor(seen.recognition!, fixture.expected.seen_recognition!, fixture.tolerance);
    await expectTensor(seen.delta!, fixture.expected.seen_delta!, fixture.tolerance);
    await expectTensorAbsolute(unseen.recognition!, fixture.expected.unseen_recognition!, fixture.tolerance.atol);
    const snapshot = await module.snapshot();
    expect(snapshot.tensors.strengths?.dims).toEqual(fixture.expected.strengths!.dims);
    expectError(Array.from(snapshot.tensors.strengths!.data), fixture.expected.strengths!.data, fixture.tolerance);
    await expect(module.commit('update', {trace_key: first.trace_key!, mask})).rejects.toThrow(/observed is required/);
    const afterFailedCommit = await module.snapshot();
    expect(Array.from(afterFailedCommit.tensors.strengths!.data)).toEqual(Array.from(snapshot.tensors.strengths!.data));
    const fork = await module.fork();
    const forked = await fork.run('read', {x: tensor(fixture.inputs.corrupt!), mask: tensor(fixture.inputs.mask!)});
    await expectTensor(forked.recognition!, fixture.expected.seen_recognition!, fixture.tolerance);
    fork.reset();
    const reset = await fork.run('read', {x: tensor(fixture.inputs.corrupt!), mask: tensor(fixture.inputs.mask!)});
    expect(Math.max(...Array.from(await reset.recognition!.getData()) as number[])).toBeLessThan(1e-4);
    await fork.restore(snapshot);
    await expect(module.run('read', {x: full, mask, keys: full})).rejects.toThrow(/managed/);
    await expect(module.run('read', {x: full, mask, invented: full})).rejects.toThrow(/not declared/);
    dispose(first); dispose(seen); dispose(unseen); dispose(forked); dispose(reset);
    full.dispose(); mask.dispose(); corrupt.dispose(); novel.dispose(); await fork.dispose(); await module.dispose();
    expect(() => module.stateInfo()).toThrow(/disposed/);

    const limited = await loadArtiStateful(url, {...options, maxStateBytes: 1});
    const limitedX = tensor(fixture.inputs.full!); const limitedMask = tensor(fixture.inputs.mask!);
    await expect(limited.run('read', {x: limitedX, mask: limitedMask})).rejects.toThrow(/budget/);
    limitedX.dispose(); limitedMask.dispose(); await limited.dispose();
    await expect(loadArtiStateful(url, {...options, maxArtifactBytes: 1})).rejects.toThrow(/artifact requires.*budget/);
  });
});

function tensor(value: TensorPayload): Tensor { return new Tensor('float32', Float32Array.from(value.data), value.dims); }
function tensors(values: Record<string, TensorPayload>): Record<string, Tensor> { return Object.fromEntries(Object.entries(values).map(([name, value]) => [name, tensor(value)])); }
function dispose(values: Record<string, Tensor>): void { for (const value of Object.values(values)) value.dispose(); }
async function expectTensor(actual: Tensor, expected: TensorPayload, tolerance: {atol: number; rtol: number}): Promise<void> {
  expect(actual.dims).toEqual(expected.dims); expectError(Array.from(await actual.getData()) as number[], expected.data, tolerance);
}
async function expectTensorAbsolute(actual: Tensor, expected: TensorPayload, atol: number): Promise<void> {
  expect(actual.dims).toEqual(expected.dims); const values = Array.from(await actual.getData()) as number[];
  expect(Math.max(...values.map((value, index) => Math.abs(value - expected.data[index]!)))).toBeLessThanOrEqual(atol);
}
function fixtureFetch(fixtures: string): typeof fetch {
  return async (input) => {
    const url = new URL(input instanceof Request ? input.url : input.toString());
    const target = path.resolve(fixtures, url.pathname.replace(/^\/+/, ''));
    if (!target.startsWith(path.resolve(fixtures) + path.sep)) return new Response(null, {status: 403});
    try { return new Response(await readFile(target), {status: 200}); } catch { return new Response(null, {status: 404}); }
  };
}
function expectError(actual: number[], expected: number[], tolerance: {atol: number; rtol: number}): void {
  expect(actual.length).toBe(expected.length);
  let maxAbsolute = 0; let maxRelative = 0;
  for (let index = 0; index < actual.length; index += 1) {
    const difference = Math.abs(actual[index]! - expected[index]!);
    maxAbsolute = Math.max(maxAbsolute, difference);
    maxRelative = Math.max(maxRelative, difference / Math.max(Math.abs(expected[index]!), 1e-8));
  }
  expect(maxAbsolute).toBeLessThanOrEqual(tolerance.atol);
  expect(maxRelative).toBeLessThanOrEqual(tolerance.rtol);
}
