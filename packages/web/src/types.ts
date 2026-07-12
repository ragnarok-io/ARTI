import type { Tensor } from 'onnxruntime-web';

export type ARTIDevice = 'auto' | 'webgpu' | 'wasm';
export type ActiveARTIDevice = Exclude<ARTIDevice, 'auto'>;

export interface TensorContract {
  name: string;
  dtype: 'float32';
  shape: Array<number | string>;
}

export interface ARTIWebManifest {
  format: 'arti.web';
  format_version: 1;
  package_version: string;
  module: {type: 'Half' | 'Fold' | 'LearnedPulse'; config: Record<string, unknown>};
  runtime: {dtype: 'float32'; opset_version: number; execution_providers: ActiveARTIDevice[]};
  inputs: TensorContract[];
  output: TensorContract;
  files: Record<string, {sha256: string; size: number}>;
}

export interface ARTIWebLock {
  format: 'arti.web';
  format_version: 1;
  manifest: {file: string; sha256: string};
  files: Record<string, {sha256: string; size: number}>;
}

export interface LoadArtiOptions {
  device?: ARTIDevice;
  fetch?: typeof globalThis.fetch;
  wasmBinary?: ArrayBuffer | Uint8Array;
  wasmPaths?: string | {wasm?: string | URL; mjs?: string | URL};
  wasmNumThreads?: number;
}

export interface ForwardOptions {
  q?: Tensor;
  mask?: Tensor;
}
