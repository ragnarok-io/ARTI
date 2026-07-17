import {describe, expect, it} from 'vitest';

import {decodeTensor, requestTransfers, responseTransfers, tensorMessage} from '../src/worker.js';
import type {WorkerRequest, WorkerResponse} from '../src/worker.js';

describe('worker protocol', () => {
  it('transfers run input ownership with dimensions intact', () => {
    const input = tensorMessage(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const request: WorkerRequest = {id: 7, type: 'run', inputs: {x: input}};
    const transfers = requestTransfers(request);
    const cloned = structuredClone(request, {transfer: transfers});

    expect(transfers).toEqual([input.data]);
    expect(input.data.byteLength).toBe(0);
    expect(cloned.inputs.x?.dims).toEqual([2, 2]);
    expect(Array.from(decodeTensor(cloned.inputs.x!))).toEqual([1, 2, 3, 4]);
  });

  it('returns each CPU output buffer as a transferable', () => {
    const response: WorkerResponse = {
      id: 8,
      type: 'result',
      outputs: {
        y: tensorMessage(new Float32Array([5, 6]), [1, 2]),
        score: tensorMessage(new Float32Array([0.5]), [1]),
      },
    };
    expect(responseTransfers(response)).toEqual([response.outputs.y?.data, response.outputs.score?.data]);
  });

  it('supports inspect selections and every inspectable tensor dtype', () => {
    const inputs = {
      x: tensorMessage(new Float32Array([1, 2]), [1, 2]),
      mask: tensorMessage(new Uint8Array([1, 0]), [1, 2]),
      index: tensorMessage(new BigInt64Array([0n, 1n]), [2]),
    };
    const request: WorkerRequest = {id: 9, type: 'inspect', inputs, outputs: ['workspace', 'index']};
    expect(requestTransfers(request)).toEqual([inputs.x.data, inputs.mask.data, inputs.index.data]);
    expect(decodeTensor(inputs.x)).toBeInstanceOf(Float32Array);
    expect(decodeTensor(inputs.mask)).toBeInstanceOf(Uint8Array);
    expect(decodeTensor(inputs.index)).toBeInstanceOf(BigInt64Array);

    const response: WorkerResponse = {
      id: 9,
      type: 'inspected',
      outputs: {index: inputs.index},
      timings: {startedAt: 1, finishedAt: 2, inferenceMs: 1},
      device: 'webgpu',
    };
    expect(responseTransfers(response)).toEqual([inputs.index.data]);
  });

  it('keeps control messages cloneable and transfer-free', () => {
    const messages: WorkerRequest[] = [
      {id: 1, type: 'load', baseUrl: '/model/', device: 'wasm'},
      {id: 2, type: 'cancel', targetId: 1},
      {id: 3, type: 'dispose'},
    ];
    for (const message of messages) {
      expect(requestTransfers(message)).toEqual([]);
      expect(structuredClone(message)).toEqual(message);
    }
  });

  it('keeps structured errors cloneable', () => {
    const response: WorkerResponse = {
      id: 4,
      type: 'error',
      error: {name: 'AbortError', message: 'worker operation was cancelled', code: 'CANCELLED', stage: 'worker'},
    };
    expect(responseTransfers(response)).toEqual([]);
    expect(structuredClone(response)).toEqual(response);
  });

  it('rejects malformed tensor shapes before posting', () => {
    expect(() => tensorMessage(new Float32Array(3), [2, 2])).toThrow(/shape requires 4 values/);
    expect(() => decodeTensor({dtype: 'float32', dims: [-1], data: new ArrayBuffer(0)})).toThrow(/dimensions/);
  });
});
