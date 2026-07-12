export interface SerializedTensor {
  dtype: 'float32';
  dims: number[];
  data: ArrayBuffer;
}

export type WorkerRequest =
  | {id: number; type: 'load'; baseUrl: string; device?: 'auto' | 'webgpu' | 'wasm'; wasmPath?: string}
  | {id: number; type: 'run'; inputs: Record<string, SerializedTensor>}
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
  | {id: number; type: 'loaded'; device: 'webgpu' | 'wasm'}
  | {id: number; type: 'result'; outputs: Record<string, SerializedTensor>}
  | {id: number; type: 'disposed'}
  | {id: number; type: 'cancelled'; targetId: number}
  | {id: number; type: 'error'; error: SerializedError};

export function tensorMessage(data: Float32Array, dims: readonly number[]): SerializedTensor {
  validateDims(dims, data.length);
  const owned = data.buffer instanceof ArrayBuffer && data.byteOffset === 0 && data.byteLength === data.buffer.byteLength
    ? data.buffer
    : data.slice().buffer;
  return {dtype: 'float32', dims: [...dims], data: owned};
}

export function requestTransfers(message: WorkerRequest): Transferable[] {
  return message.type === 'run' ? tensorTransfers(message.inputs) : [];
}

export function responseTransfers(message: WorkerResponse): Transferable[] {
  return message.type === 'result' ? tensorTransfers(message.outputs) : [];
}

export function decodeTensor(value: SerializedTensor): Float32Array {
  if (value.dtype !== 'float32' || !(value.data instanceof ArrayBuffer)) throw new TypeError('worker tensors must contain a transferable float32 ArrayBuffer');
  const data = new Float32Array(value.data);
  validateDims(value.dims, data.length);
  return data;
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
