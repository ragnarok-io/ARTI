import {describe, expect, it} from 'vitest';

import {decodeTensor, requestTransfers, responseTransfers, tensorMessage} from '../examples/worker/protocol.js';
import type {WorkerRequest, WorkerResponse} from '../examples/worker/protocol.js';

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
