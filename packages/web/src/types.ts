import type { Tensor } from 'onnxruntime-web';
import type { ActiveARTIDevice } from './generated/contract.js';
import type {LoadProgressCallback} from './diagnostics.js';

export type ARTIDevice = 'auto' | ActiveARTIDevice;
export type TensorMap = Record<string, Tensor>;

/** A serializable, CPU-resident float32 tensor value. */
export interface CPUTensor {
  data: Float32Array;
  dims: number[];
}

/** Values accepted by CPU-oriented convenience APIs. */
export type TensorInput = Tensor | CPUTensor;
/** Values returned by CPU-oriented convenience APIs. */
export type TensorOutput = CPUTensor;

export interface OperationOptions {
  signal?: AbortSignal;
}

export interface LoadArtiOptions extends OperationOptions {
  device?: ARTIDevice;
  fetch?: typeof globalThis.fetch;
  wasmBinary?: ArrayBuffer | Uint8Array;
  wasmPaths?: string | {wasm?: string | URL; mjs?: string | URL};
  wasmNumThreads?: number;
  /** Maximum mutable state memory for stateful artifacts. Defaults to 256 MiB. */
  maxStateBytes?: number;
  /** Maximum aggregate model bytes for stateful artifacts. Defaults to 512 MiB. */
  maxArtifactBytes?: number;
  /** Receives best-effort artifact loading and runtime initialization progress. */
  onProgress?: LoadProgressCallback;
}
