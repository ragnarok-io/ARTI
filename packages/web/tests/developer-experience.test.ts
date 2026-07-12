import {describe, expect, it} from 'vitest';

import {ArtiWebError, fromCPU, isCPUTensor, tensor, toCPU, zeros} from '../src/index.js';
import type {TensorContract} from '../src/index.js';

const contract: TensorContract = {name: 'signal', dtype: 'float32', shape: ['batch', 2]};

describe('ArtiWebError', () => {
  it('preserves stable diagnostic fields and the cause', () => {
    const cause = new Error('offline');
    const error = new ArtiWebError('load failed', {
      code: 'FETCH_FAILED', stage: 'fetch', artifactUrl: new URL('https://example.test/model/'),
      tensorName: 'x', expected: 2, actual: 3, device: 'wasm', cause,
    });
    expect(error).toBeInstanceOf(Error);
    expect(error).toMatchObject({name: 'ArtiWebError', code: 'FETCH_FAILED', stage: 'fetch', artifactUrl: 'https://example.test/model/', tensorName: 'x', expected: 2, actual: 3, device: 'wasm', cause});
  });

  it('redacts public URLs and keeps causes out of serialization', () => {
    const cause = new Error('https://example.test/model?token=secret');
    const error = new ArtiWebError('load failed', {code: 'FETCH_FAILED', artifactUrl: 'https://user:pass@example.test/model?token=secret#part', cause});
    expect(error.artifactUrl).toBe('https://example.test/model');
    expect(error.cause).toBe(cause);
    expect(JSON.stringify(error)).not.toContain('secret');
    expect(structuredClone(error)).not.toHaveProperty('cause');
  });
});

describe('tensor helpers', () => {
  it('creates ordinary ORT tensors from explicit and contract shapes', () => {
    const explicit = tensor([1, 2, 3, 4], [2, 2]);
    const aware = tensor(contract, [1, 2, 3, 4], {batch: 2});
    expect(explicit.dims).toEqual([2, 2]);
    expect(aware.dims).toEqual([2, 2]);
    expect(Array.from(aware.data)).toEqual([1, 2, 3, 4]);
  });

  it('creates contract-aware zeros and reports contract mismatches', () => {
    const value = zeros(contract, {batch: 3});
    expect(value.dims).toEqual([3, 2]);
    expect(Array.from(value.data)).toEqual([0, 0, 0, 0, 0, 0]);
    expect(() => tensor(contract, [1, 2], {batch: 2})).toThrowError(ArtiWebError);
    expect(() => zeros(contract)).toThrow(/missing dynamic dimension batch/);
    expect(() => zeros(contract, {batch: 1, extra: 2})).toThrow(/unknown dynamic dimension extra/);
  });

  it('round-trips detached CPU-friendly values', async () => {
    const cpu = await toCPU(tensor([1, 2], [1, 2]));
    expect(isCPUTensor(cpu)).toBe(true);
    const ort = fromCPU(cpu);
    cpu.data[0] = 9;
    expect(Array.from(ort.data)).toEqual([1, 2]);
    const copy = await toCPU(cpu);
    expect(copy).not.toBe(cpu);
    expect(copy.data).not.toBe(cpu.data);
  });
});
