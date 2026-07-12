import {requestTransfers, tensorMessage} from './protocol.js';
import type {WorkerRequest, WorkerRequestWithoutId, WorkerResponse} from './protocol.js';

const worker = new Worker(new URL('./arti.worker.ts', import.meta.url), {type: 'module'});
let nextId = 1;

export function send(message: WorkerRequestWithoutId): number {
  const request = {...message, id: nextId++} as WorkerRequest;
  worker.postMessage(request, requestTransfers(request));
  return request.id;
}

worker.addEventListener('message', (event: MessageEvent<WorkerResponse>) => {
  if (event.data.type === 'loaded') {
    const values = new Float32Array([1, 2, 3, 4]);
    const runId = send({type: 'run', inputs: {x: tensorMessage(values, [1, 4])}});
    // Cancellation is cooperative: the worker suppresses a late result and disposes it.
    // send({type: 'cancel', targetId: runId});
    void runId;
  } else if (event.data.type === 'result') {
    // These are CPU ArrayBuffers owned by the main thread. GPU tensors never cross the boundary.
    console.log(event.data.outputs);
  } else if (event.data.type === 'error') {
    console.error(`ARTI worker ${event.data.error.code ?? event.data.error.name}: ${event.data.error.message}`);
  }
});

send({type: 'load', baseUrl: '/layer-web/', device: 'auto', wasmPath: '/ort-runtime.wasm'});
