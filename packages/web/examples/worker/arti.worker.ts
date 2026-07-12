import {Tensor, loadArti} from '../../src/index.js';
import type {ARTIWebModule} from '../../src/index.js';

import {decodeTensor, responseTransfers} from './protocol.js';
import type {SerializedError, SerializedTensor, WorkerRequest, WorkerResponse} from './protocol.js';

interface WorkerScope {
  addEventListener(type: 'message', listener: (event: MessageEvent<WorkerRequest>) => void): void;
  postMessage(message: WorkerResponse, transfer: Transferable[]): void;
}

const scope = globalThis as unknown as WorkerScope;
let runtime: ARTIWebModule | undefined;
let generation = 0;
let operationTail: Promise<void> = Promise.resolve();
let active: {id: number; generation: number; controller?: AbortController; cancelled: boolean} | undefined;

scope.addEventListener('message', (event) => {
  const request = event.data;
  if (request.type === 'cancel') {
    if (!active || active.id !== request.targetId) {
      post({id: request.id, type: 'error', error: serializeError(new Error(`cannot cancel inactive worker request ${request.targetId}`))});
      return;
    }
    active.cancelled = true;
    active.controller?.abort();
    post({id: request.id, type: 'cancelled', targetId: request.targetId});
    return;
  }
  if (request.type === 'dispose') {
    generation += 1;
    active?.controller?.abort();
    if (active) active.cancelled = true;
  }
  const requestGeneration = generation;
  const operation = operationTail.then(() => handle(request, requestGeneration));
  operationTail = operation.catch(() => undefined);
  void operation.catch((error: unknown) => post({id: request.id, type: 'error', error: serializeError(error)}));
});

async function handle(request: Exclude<WorkerRequest, {type: 'cancel'}>, requestGeneration: number): Promise<void> {
  const operation = {id: request.id, generation: requestGeneration, cancelled: false} as {id: number; generation: number; controller?: AbortController; cancelled: boolean};
  active = operation;
  try {
    await handleActive(request, operation);
  } finally {
    if (active === operation) active = undefined;
  }
}

async function handleActive(request: Exclude<WorkerRequest, {type: 'cancel'}>, operation: {id: number; generation: number; controller?: AbortController; cancelled: boolean}): Promise<void> {
  if (request.type !== 'dispose' && operation.generation !== generation) throw cancelledError();
  if (request.type === 'load') {
    if (runtime) await runtime.dispose();
    runtime = undefined;
    operation.controller = new AbortController();
    const loaded = await loadArti(request.baseUrl, {
      device: request.device ?? 'auto',
      signal: operation.controller.signal,
      ...(request.wasmPath ? {wasmPaths: {wasm: request.wasmPath}} : {}),
    });
    if (isCancelled(operation)) {
      await loaded.dispose();
      throw cancelledError();
    }
    runtime = loaded;
    post({id: request.id, type: 'loaded', device: runtime.device});
    return;
  }

  if (request.type === 'dispose') {
    await runtime?.dispose();
    runtime = undefined;
    post({id: request.id, type: 'disposed'});
    return;
  }

  if (!runtime) throw new Error('load an ARTI artifact before running inference');
  const inputs = Object.fromEntries(Object.entries(request.inputs).map(([name, value]) => [
    name,
    new Tensor('float32', decodeTensor(value), value.dims),
  ]));
  try {
    if (isCancelled(operation)) throw cancelledError();
    const outputs = await runtime.run(inputs);
    try {
      const serialized: Record<string, SerializedTensor> = {};
      for (const [name, tensor] of Object.entries(outputs)) {
        const values = await tensor.getData();
        serialized[name] = {dtype: 'float32', dims: [...tensor.dims], data: Float32Array.from(values as Float32Array).buffer};
      }
      if (isCancelled(operation)) throw cancelledError();
      post({id: request.id, type: 'result', outputs: serialized});
    } finally {
      for (const tensor of Object.values(outputs)) tensor.dispose();
    }
  } finally {
    for (const tensor of Object.values(inputs)) tensor.dispose();
  }
}

function post(message: WorkerResponse): void {
  scope.postMessage(message, responseTransfers(message));
}

function isCancelled(operation: {generation: number; cancelled: boolean}): boolean {
  return operation.cancelled || operation.generation !== generation;
}

function cancelledError(): Error & {code: string; stage: string} {
  return Object.assign(new Error('worker operation was cancelled'), {name: 'AbortError', code: 'CANCELLED', stage: 'worker'});
}

function serializeError(error: unknown): SerializedError {
  if (!(error instanceof Error)) return {name: 'Error', message: String(error)};
  const detail = error as Error & {code?: unknown; stage?: unknown};
  return {
    name: error.name,
    message: error.message,
    ...(typeof detail.code === 'string' ? {code: detail.code} : {}),
    ...(typeof detail.stage === 'string' ? {stage: detail.stage} : {}),
    ...(error.stack ? {stack: error.stack} : {}),
  };
}
