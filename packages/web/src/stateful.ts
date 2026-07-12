import * as ort from 'onnxruntime-web/webgpu';

import {sha256, verifyFile} from './artifact.js';
import {parseStatefulLock, parseStatefulManifest} from './generated/contract.js';
import type {ActiveARTIDevice, ARTIStatefulWebManifest, StatefulEntrypoint, TensorContract} from './generated/contract.js';
import type {LoadArtiOptions, TensorMap} from './types.js';

const MANIFEST = 'arti-web.json';
const LOCK = 'arti-web.lock.json';

export interface ARTIStateSnapshot {
  readonly batchSize: number;
  readonly tensors: Record<string, {dims: number[]; data: Float32Array}>;
}

/** Generic explicit-state executor for Python-exported read/update graphs. */
export class ARTIStatefulWebModule {
  readonly manifest: ARTIStatefulWebManifest;
  readonly device: ActiveARTIDevice;
  private sessions: Map<string, ort.InferenceSession>;
  private state: TensorMap | null = null;
  private stateBuffers: Record<string, GPUBuffer> | null = null;
  private batchSize = 0;
  private disposed = false;
  private readonly forkFactory: () => Promise<ARTIStatefulWebModule>;
  private readonly maxStateBytes: number;

  /** @internal */
  constructor(sessions: Map<string, ort.InferenceSession>, manifest: ARTIStatefulWebManifest, device: ActiveARTIDevice, maxStateBytes: number, forkFactory: () => Promise<ARTIStatefulWebModule>) {
    this.sessions = sessions;
    this.manifest = manifest;
    this.device = device;
    this.forkFactory = forkFactory;
    this.maxStateBytes = maxStateBytes;
  }

  /** Execute an entrypoint without changing state. */
  async run(entrypoint: string, inputs: TensorMap): Promise<TensorMap> {
    this.requireActive();
    const contract = this.requireEntrypoint(entrypoint);
    await this.ensureState(inferBatch(inputs));
    const feeds = this.feeds(contract, inputs);
    const outputs = await this.sessions.get(entrypoint)!.run(feeds);
    try { validateOutputs(outputs, contract.outputs); return outputs; }
    catch (error) { disposeMap(outputs); throw error; }
  }

  /** Execute an entrypoint and atomically adopt its state-named outputs. */
  async commit(entrypoint: string, inputs: TensorMap): Promise<TensorMap> {
    const outputs = await this.run(entrypoint, inputs);
    const next: TensorMap = {};
    try {
      const bindings = this.requireEntrypoint(entrypoint).state_outputs ?? {};
      for (const contract of this.manifest.state) {
        const outputName = bindings[contract.name];
        const tensor = outputName === undefined ? undefined : outputs[outputName];
        if (!tensor) throw new Error(`ARTI stateful commit did not produce state ${contract.name}`);
        next[contract.name] = tensor;
      }
    } catch (error) {
      disposeMap(outputs);
      throw error;
    }
    const previous = this.state!; const previousBuffers = this.stateBuffers;
    this.state = next;
    this.stateBuffers = null;
    disposeState(previous, previousBuffers);
    const stateOutputNames = new Set(Object.values(this.requireEntrypoint(entrypoint).state_outputs ?? {}));
    return Object.fromEntries(Object.entries(outputs).filter(([name]) => !stateOutputNames.has(name)));
  }

  /** Explicitly download a portable state snapshot. */
  async snapshot(): Promise<ARTIStateSnapshot> {
    this.requireActive();
    if (!this.state) throw new Error('ARTI state has not been initialized; run an entrypoint first');
    const tensors: ARTIStateSnapshot['tensors'] = {};
    for (const contract of this.manifest.state) {
      const tensor = this.state[contract.name]!;
      tensors[contract.name] = {dims: [...tensor.dims], data: Float32Array.from(await tensor.getData() as Float32Array)};
    }
    return {batchSize: this.batchSize, tensors};
  }

  /** Atomically replace state from an explicit snapshot. */
  async restore(snapshot: ARTIStateSnapshot): Promise<void> {
    this.requireActive();
    const next = createStateFromSnapshot(snapshot, this.manifest);
    const previous = this.state;
    const previousBuffers = this.stateBuffers;
    this.state = next;
    this.stateBuffers = null;
    this.batchSize = snapshot.batchSize;
    if (previous) disposeState(previous, previousBuffers);
  }

  /** Create an independent session with a copied state. */
  async fork(): Promise<ARTIStatefulWebModule> {
    const snapshot = await this.snapshot();
    const clone = await this.forkFactory();
    await clone.restore(snapshot);
    return clone;
  }

  /** Return to empty, non-persistent state. */
  reset(): void {
    this.requireActive();
    if (this.state) disposeState(this.state, this.stateBuffers);
    this.state = null;
    this.stateBuffers = null;
    this.batchSize = 0;
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    if (this.state) disposeState(this.state, this.stateBuffers);
    this.state = null;
    this.stateBuffers = null;
    await Promise.all([...this.sessions.values()].map((session) => session.release()));
    this.sessions.clear();
  }

  /** Non-sensitive state diagnostics; never exposes stored values. */
  stateInfo(): {batchSize: number; bytes: number; locations: Record<string, string>} {
    this.requireActive();
    return {
      batchSize: this.batchSize,
      bytes: this.batchSize * this.manifest.limits.max_state_bytes_per_batch,
      locations: Object.fromEntries(this.manifest.state.map((contract) => [contract.name, this.state?.[contract.name]?.location ?? 'uninitialized'])),
    };
  }

  private requireActive(): void { if (this.disposed) throw new Error('ARTI stateful Web module has been disposed'); }
  private requireEntrypoint(name: string): StatefulEntrypoint {
    const entry = this.manifest.entrypoints[name];
    if (!entry || !this.sessions.has(name)) throw new Error(`unknown ARTI stateful entrypoint ${name}`);
    return entry;
  }
  private async ensureState(batch: number): Promise<void> {
    if (this.state) {
      if (batch !== this.batchSize) throw new Error(`ARTI state batch is ${this.batchSize}; reset before using batch ${batch}`);
      return;
    }
    const bytes = stateBytesForBatch(this.manifest, batch);
    if (bytes !== this.manifest.limits.max_state_bytes_per_batch * batch) throw new Error('ARTI state budget does not match declared state shapes');
    if (!Number.isSafeInteger(bytes) || bytes <= 0 || bytes > this.maxStateBytes) throw new Error(`ARTI state requires ${bytes} bytes, exceeding the ${this.maxStateBytes}-byte budget`);
    if (this.device === 'webgpu') {
      const gpuDevice = await ort.env.webgpu.device;
      const buffers: Record<string, GPUBuffer> = {};
      this.state = Object.fromEntries(this.manifest.state.map((contract) => {
        const dims = contract.shape.map((dim) => dim === 'batch' ? batch : requireStaticDimension(dim, contract.name));
        const byteLength = dims.reduce((product, value) => product * value, 1) * Float32Array.BYTES_PER_ELEMENT;
        const buffer = gpuDevice.createBuffer({size: Math.ceil(byteLength / 16) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, mappedAtCreation: true});
        new Uint8Array(buffer.getMappedRange()).fill(0); buffer.unmap(); buffers[contract.name] = buffer;
        return [contract.name, ort.Tensor.fromGpuBuffer(buffer, {dataType: 'float32', dims})];
      }));
      this.stateBuffers = buffers;
    } else {
      this.state = Object.fromEntries(this.manifest.state.map((contract) => {
        const dims = contract.shape.map((dim) => dim === 'batch' ? batch : requireStaticDimension(dim, contract.name));
        const size = dims.reduce((product, value) => product * value, 1);
        return [contract.name, new ort.Tensor('float32', new Float32Array(size), dims)];
      }));
      this.stateBuffers = null;
    }
    this.batchSize = batch;
  }
  private feeds(entry: StatefulEntrypoint, inputs: TensorMap): TensorMap {
    const stateNames = new Set(this.manifest.state.map((item) => item.name));
    const allowed = new Set(entry.inputs.filter((item) => !stateNames.has(item.name)).map((item) => item.name));
    for (const name of Object.keys(inputs)) if (stateNames.has(name)) throw new Error(`${name} is managed by the ARTI stateful session`);
    for (const name of Object.keys(inputs)) if (!allowed.has(name)) throw new Error(`${name} is not declared by this ARTI stateful entrypoint`);
    const merged = {...inputs, ...this.state};
    validateNamed(merged, entry.inputs);
    return Object.fromEntries(entry.inputs.map((contract) => [contract.name, merged[contract.name]!]));
  }
}

/** Load every graph in a Python-exported stateful artifact. */
export async function loadArtiStateful(baseUrl: string | URL, options: LoadArtiOptions = {}): Promise<ARTIStatefulWebModule> {
  const fetcher = options.fetch ?? globalThis.fetch;
  if (!fetcher) throw new Error('loadArtiStateful requires a fetch implementation');
  if (options.wasmBinary !== undefined) ort.env.wasm.wasmBinary = options.wasmBinary;
  if (options.wasmPaths !== undefined) ort.env.wasm.wasmPaths = options.wasmPaths;
  if (options.wasmNumThreads !== undefined) ort.env.wasm.numThreads = options.wasmNumThreads;
  const base = normalizeBase(baseUrl);
  const maxStateBytes = options.maxStateBytes ?? 256 * 1024 * 1024;
  if (!Number.isSafeInteger(maxStateBytes) || maxStateBytes <= 0) throw new Error('maxStateBytes must be a positive safe integer');
  const maxArtifactBytes = options.maxArtifactBytes ?? 512 * 1024 * 1024;
  if (!Number.isSafeInteger(maxArtifactBytes) || maxArtifactBytes <= 0) throw new Error('maxArtifactBytes must be a positive safe integer');
  const [manifestResponse, lockResponse] = await Promise.all([fetcher(new URL(MANIFEST, base)), fetcher(new URL(LOCK, base))]);
  requireOk(manifestResponse, MANIFEST); requireOk(lockResponse, LOCK);
  const [manifestBuffer, lockValue] = await Promise.all([manifestResponse.arrayBuffer(), lockResponse.json()]);
  const lock = parseStatefulLock(lockValue);
  if (await sha256(manifestBuffer) !== lock.manifest.sha256) throw new Error('arti-web.json SHA-256 does not match its lock');
  const manifest = parseStatefulManifest(JSON.parse(new TextDecoder().decode(manifestBuffer)));
  const declaredArtifactBytes = Object.values(manifest.files).reduce((total, file) => total + file.size, 0);
  if (!Number.isSafeInteger(declaredArtifactBytes) || declaredArtifactBytes > maxArtifactBytes) throw new Error(`ARTI artifact requires ${declaredArtifactBytes} bytes, exceeding the ${maxArtifactBytes}-byte budget`);
  const modelBuffers = new Map<string, ArrayBuffer>();
  for (const [name, record] of Object.entries(manifest.files)) {
    const locked = lock.files[name];
    if (!locked || locked.sha256 !== record.sha256 || locked.size !== record.size) throw new Error(`${name} manifest and lock records disagree`);
    const response = await fetcher(artifactUrl(name, base)); requireOk(response, name);
    const buffer = await response.arrayBuffer(); await verifyFile(name, buffer, locked); modelBuffers.set(name, buffer);
  }
  const requested = options.device ?? 'auto';
  const candidates: ActiveARTIDevice[] = requested === 'auto' ? (hasWebGPU() ? ['webgpu', 'wasm'] : ['wasm']) : [requested];
  let lastError: unknown;
  for (const device of candidates) {
    const sessions = new Map<string, ort.InferenceSession>();
    try {
      if (!manifest.runtime.execution_providers.includes(device)) throw new Error(`${device} is not supported by this artifact`);
      if (device === 'webgpu' && !hasWebGPU()) throw new Error('WebGPU is not available in this environment');
      for (const [name, entry] of Object.entries(manifest.entrypoints)) {
        const buffer = modelBuffers.get(entry.file)!;
        sessions.set(name, await ort.InferenceSession.create(new Uint8Array(buffer), {executionProviders: [device], preferredOutputLocation: device === 'webgpu' ? 'gpu-buffer' : 'cpu'}));
      }
      return new ARTIStatefulWebModule(sessions, manifest, device, maxStateBytes, () => loadArtiStateful(base, options));
    } catch (error) {
      lastError = error; await Promise.all([...sessions.values()].map((session) => session.release()));
      if (requested !== 'auto') break;
    }
  }
  throw new Error(`unable to initialize ARTI stateful Web runtime for ${requested}`, {cause: lastError});
}

function validateNamed(values: TensorMap, contracts: TensorContract[]): void {
  const expected = new Set(contracts.map((item) => item.name));
  for (const name of Object.keys(values)) if (!expected.has(name)) continue;
  const symbols = new Map<string, number>();
  for (const contract of contracts) {
    const tensor = values[contract.name]; if (!tensor) throw new Error(`${contract.name} is required by this entrypoint`);
    if (tensor.type !== contract.dtype || tensor.dims.length !== contract.shape.length) throw new Error(`${contract.name} does not match its tensor contract`);
    contract.shape.forEach((expectedDim, index) => {
      const actual = tensor.dims[index]!;
      if (typeof expectedDim === 'number' && expectedDim !== actual) throw new Error(`${contract.name} dimension ${index} must be ${expectedDim}, got ${actual}`);
      if (typeof expectedDim === 'string') { const prior = symbols.get(expectedDim); if (prior !== undefined && prior !== actual) throw new Error(`${contract.name} conflicts with dynamic axis ${expectedDim}`); symbols.set(expectedDim, actual); }
    });
  }
}
function validateOutputs(outputs: TensorMap, contracts: TensorContract[]): void { validateNamed(outputs, contracts); }
function inferBatch(inputs: TensorMap): number { const first = Object.values(inputs)[0]; if (!first || first.dims.length === 0 || !first.dims[0]) throw new Error('cannot infer ARTI state batch size'); return first.dims[0]; }
function requireStaticDimension(value: number | string, name: string): number { if (typeof value !== 'number') throw new Error(`${name} state has unsupported dynamic dimension ${value}`); return value; }
function stateBytesForBatch(manifest: ARTIStatefulWebManifest, batch: number): number {
  let bytes = 0;
  for (const contract of manifest.state) {
    let elements = 1;
    for (const dim of contract.shape) {
      const value = dim === 'batch' ? batch : requireStaticDimension(dim, contract.name);
      elements *= value;
      if (!Number.isSafeInteger(elements)) throw new Error('ARTI state shape exceeds safe integer bounds');
    }
    bytes += elements * Float32Array.BYTES_PER_ELEMENT;
    if (!Number.isSafeInteger(bytes)) throw new Error('ARTI state size exceeds safe integer bounds');
  }
  return bytes;
}
function createStateFromSnapshot(snapshot: ARTIStateSnapshot, manifest: ARTIStatefulWebManifest): TensorMap {
  if (!Number.isSafeInteger(snapshot.batchSize) || snapshot.batchSize <= 0) throw new Error('invalid ARTI state snapshot batch');
  const result: TensorMap = {};
  for (const contract of manifest.state) {
    const value = snapshot.tensors[contract.name]; if (!value) { disposeMap(result); throw new Error(`snapshot is missing ${contract.name}`); }
    const dims = contract.shape.map((dim) => dim === 'batch' ? snapshot.batchSize : requireStaticDimension(dim, contract.name));
    if (dims.length !== value.dims.length || dims.some((dim, index) => dim !== value.dims[index])) { disposeMap(result); throw new Error(`${contract.name} snapshot shape does not match`); }
    result[contract.name] = new ort.Tensor('float32', Float32Array.from(value.data), dims);
  }
  return result;
}
function disposeMap(values: TensorMap): void { for (const tensor of Object.values(values)) tensor.dispose(); }
function disposeState(values: TensorMap, buffers: Record<string, GPUBuffer> | null): void {
  if (buffers) { for (const buffer of Object.values(buffers)) buffer.destroy(); return; }
  disposeMap(values);
}
function normalizeBase(value: string | URL): URL { const url = value instanceof URL ? new URL(value.href) : new URL(value, globalThis.location?.href ?? 'http://localhost/'); if (!url.pathname.endsWith('/')) url.pathname += '/'; return url; }
function artifactUrl(name: string, base: URL): URL {
  const url = new URL(name, base);
  if (url.origin !== base.origin || !url.pathname.startsWith(base.pathname) || url.search || url.hash) throw new Error(`ARTI artifact file ${name} escapes its base URL`);
  return url;
}
function requireOk(response: Response, name: string): void { if (!response.ok) throw new Error(`failed to load ${name}: HTTP ${response.status}`); }
function hasWebGPU(): boolean { return typeof navigator !== 'undefined' && 'gpu' in navigator && navigator.gpu !== undefined; }
