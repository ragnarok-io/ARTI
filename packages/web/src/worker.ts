import type {ARTIWebManifest} from './generated/contract.js';

export type SerializedTensorDType = 'float32' | 'bool' | 'int64';
export type SerializedTensorData = Float32Array | Uint8Array | BigInt64Array;

/** CPU tensor payload whose ArrayBuffer can cross a Worker boundary. */
export interface SerializedTensor {
  dtype: SerializedTensorDType;
  dims: number[];
  data: ArrayBuffer;
}

export interface WorkerRunTimings {
  startedAt: number;
  finishedAt: number;
  inferenceMs: number;
}

export type WorkerRequest =
  | {id: number; type: 'load'; baseUrl: string; device?: 'auto' | 'webgpu' | 'wasm'; wasmPath?: string}
  | {id: number; type: 'run'; inputs: Record<string, SerializedTensor>}
  | {id: number; type: 'inspect'; inputs: Record<string, SerializedTensor>; outputs?: string[]}
  | {id: number; type: 'dispose'}
  | {id: number; type: 'cancel'; targetId: number};

export type WorkerRequestWithoutId = WorkerRequest extends infer Request
  ? Request extends WorkerRequest ? Omit<Request, 'id'> : never
  : never;

export interface SerializedError {
  name: string;
  message: string;
  code?: string;
  stage?: string;
  stack?: string;
}

export type WorkerResponse =
  | {id: number; type: 'loaded'; device: 'webgpu' | 'wasm'; manifest: ARTIWebManifest}
  | {id: number; type: 'result'; outputs: Record<string, SerializedTensor>}
  | {id: number; type: 'inspected'; outputs: Record<string, SerializedTensor>; timings: WorkerRunTimings; device: 'webgpu' | 'wasm'}
  | {id: number; type: 'disposed'}
  | {id: number; type: 'cancelled'; targetId: number}
  | {id: number; type: 'error'; error: SerializedError};

/** Create a validated, transferable CPU tensor payload. */
export function tensorMessage(data: SerializedTensorData, dims: readonly number[]): SerializedTensor {
  validateDims(dims, data.length);
  const owned = data.buffer instanceof ArrayBuffer && data.byteOffset === 0 && data.byteLength === data.buffer.byteLength
    ? data.buffer
    : data.slice().buffer;
  return {dtype: tensorDType(data), dims: [...dims], data: owned};
}

export function requestTransfers(message: WorkerRequest): Transferable[] {
  return message.type === 'run' || message.type === 'inspect' ? tensorTransfers(message.inputs) : [];
}

export function responseTransfers(message: WorkerResponse): Transferable[] {
  return message.type === 'result' || message.type === 'inspected' ? tensorTransfers(message.outputs) : [];
}

/** Decode and validate a transferred tensor without assigning mechanism semantics. */
export function decodeTensor(value: SerializedTensor): SerializedTensorData {
  if (!(value.data instanceof ArrayBuffer)) throw new TypeError('worker tensors must contain a transferable ArrayBuffer');
  const data = value.dtype === 'float32'
    ? new Float32Array(value.data)
    : value.dtype === 'bool'
      ? new Uint8Array(value.data)
      : value.dtype === 'int64'
        ? new BigInt64Array(value.data)
        : unsupportedDType(value.dtype);
  validateDims(value.dims, data.length);
  return data;
}

function tensorDType(data: SerializedTensorData): SerializedTensorDType {
  if (data instanceof Float32Array) return 'float32';
  if (data instanceof BigInt64Array) return 'int64';
  if (data instanceof Uint8Array) return 'bool';
  throw new TypeError('worker tensors support Float32Array, Uint8Array, and BigInt64Array');
}

function unsupportedDType(value: never): never {
  throw new TypeError(`unsupported worker tensor dtype ${String(value)}`);
}

function tensorTransfers(values: Record<string, SerializedTensor>): Transferable[] {
  const buffers = new Set<ArrayBuffer>();
  for (const value of Object.values(values)) {
    decodeTensor(value);
    buffers.add(value.data);
  }
  return [...buffers];
}

function validateDims(dims: readonly number[], length: number): void {
  if (!Array.isArray(dims) || !dims.every((dim) => Number.isSafeInteger(dim) && dim >= 0)) throw new TypeError('worker tensor dimensions must be non-negative safe integers');
  const elements = dims.reduce((size, dim) => size * dim, 1);
  if (!Number.isSafeInteger(elements) || elements !== length) throw new TypeError(`worker tensor shape requires ${elements} values, got ${length}`);
}
