import type { Tensor } from 'onnxruntime-web';
import type { ActiveARTIDevice } from './generated/contract.js';

export type ARTIDevice = 'auto' | ActiveARTIDevice;
export type TensorMap = Record<string, Tensor>;

export interface LoadArtiOptions {
  device?: ARTIDevice;
  fetch?: typeof globalThis.fetch;
  wasmBinary?: ArrayBuffer | Uint8Array;
  wasmPaths?: string | {wasm?: string | URL; mjs?: string | URL};
  wasmNumThreads?: number;
  /** Maximum mutable state memory for stateful artifacts. Defaults to 256 MiB. */
  maxStateBytes?: number;
  /** Maximum aggregate model bytes for stateful artifacts. Defaults to 512 MiB. */
  maxArtifactBytes?: number;
}
