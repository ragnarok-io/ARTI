import * as ort from 'onnxruntime-web/webgpu';

import {sha256, verifyFile} from './artifact.js';
import {ArtiWebError, sanitizeArtifactUrl} from './errors.js';
import {parseStatefulLock, parseStatefulManifest} from './generated/contract.js';
import type {ActiveARTIDevice, ARTIStatefulWebManifest, StatefulEntrypoint, TensorContract} from './generated/contract.js';
import {summarizeDiagnosticError, type LoadAttemptDiagnostic, type LoadDiagnostics} from './diagnostics.js';
import {download, progress, requireBudget, requireMatchingFileSets, structuredLoadError, throwIfAborted} from './runtime.js';
import type {LoadArtiOptions, TensorMap} from './types.js';

const MANIFEST = 'arti-web.json';
const LOCK = 'arti-web.lock.json';
const MAX_METADATA_BYTES = 1024 * 1024;

export interface ARTIStateSnapshot {
  readonly batchSize: number;
  readonly tensors: Record<string, {dims: number[]; data: Float32Array}>;
}

/** Generic explicit-state executor for Python-exported read/update graphs. */
export class ARTIStatefulWebModule {
  readonly manifest: ARTIStatefulWebManifest;
  readonly device: ActiveARTIDevice;
  readonly diagnostics: LoadDiagnostics;
  private sessions: Map<string, ort.InferenceSession>;
  private state: TensorMap | null = null;
  private stateBuffers: Record<string, GPUBuffer> | null = null;
  private batchSize = 0;
  private disposed = false;
  private acceptingOperations = true;
  private operationTail: Promise<void> = Promise.resolve();
  private disposePromise: Promise<void> | null = null;
  private readonly forkFactory: () => Promise<ARTIStatefulWebModule>;
  private readonly maxStateBytes: number;

  /** @internal */
  constructor(sessions: Map<string, ort.InferenceSession>, manifest: ARTIStatefulWebManifest, device: ActiveARTIDevice, maxStateBytes: number, forkFactory: () => Promise<ARTIStatefulWebModule>, diagnostics?: LoadDiagnostics) {
    this.sessions = sessions;
    this.manifest = manifest;
    this.device = device;
    this.diagnostics = diagnostics ?? {artifactUrl: '', startedAt: 0, finishedAt: 0, attempts: [], selectedDevice: device};
    this.forkFactory = forkFactory;
    this.maxStateBytes = maxStateBytes;
  }

  /** Execute an entrypoint without changing state. */
  async run(entrypoint: string, inputs: TensorMap): Promise<TensorMap> {
    return this.enqueue(() => this.runInternal(entrypoint, inputs));
  }

  private async runInternal(entrypoint: string, inputs: TensorMap): Promise<TensorMap> {
    const contract = this.requireEntrypoint(entrypoint);
    await this.ensureState(inferBatch(inputs));
    const feeds = this.feeds(contract, inputs);
    const outputs = await this.sessions.get(entrypoint)!.run(feeds);
    try { validateOutputs(outputs, contract.outputs); return outputs; }
    catch (error) { disposeMap(outputs); throw error; }
  }

  /** Execute an entrypoint and atomically adopt its state-named outputs. */
  async commit(entrypoint: string, inputs: TensorMap): Promise<TensorMap> {
    return this.enqueue(() => this.commitInternal(entrypoint, inputs));
  }

  private async commitInternal(entrypoint: string, inputs: TensorMap): Promise<TensorMap> {
    const outputs = await this.runInternal(entrypoint, inputs);
    const next: TensorMap = {};
    try {
      const bindings = this.requireEntrypoint(entrypoint).state_outputs ?? {};
      for (const contract of this.manifest.state) {
        const outputName = bindings[contract.name];
        const tensor = outputName === undefined ? undefined : outputs[outputName];
        if (!tensor) throw new Error(`ARTI stateful commit did not produce state ${contract.name}`);
        next[contract.name] = tensor;
      }
      assertUnaliasedCommit(outputs, next, this.state!, new Set(Object.values(bindings)));
    } catch (error) {
      disposeMapExcept(outputs, new Set(Object.values(this.state!)));
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
    return this.enqueue(() => this.snapshotInternal());
  }

  private async snapshotInternal(): Promise<ARTIStateSnapshot> {
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
    return this.enqueue(() => this.restoreInternal(snapshot));
  }

  private restoreInternal(snapshot: ARTIStateSnapshot): void {
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
    return this.enqueue(async () => {
      const snapshot = await this.snapshotInternal();
      const clone = await this.forkFactory();
      await clone.restore(snapshot);
      return clone;
    });
  }

  /** Return to empty, non-persistent state. */
  reset(): Promise<void> {
    return this.enqueue(() => {
      if (this.state) disposeState(this.state, this.stateBuffers);
      this.state = null;
      this.stateBuffers = null;
      this.batchSize = 0;
    });
  }

  async dispose(): Promise<void> {
    if (this.disposePromise) return this.disposePromise;
    this.acceptingOperations = false;
    this.disposePromise = this.appendOperation(async () => {
      this.disposed = true;
      if (this.state) disposeState(this.state, this.stateBuffers);
      this.state = null;
      this.stateBuffers = null;
      await Promise.all([...this.sessions.values()].map((session) => session.release()));
      this.sessions.clear();
    });
    return this.disposePromise;
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

  private requireActive(): void { if (!this.acceptingOperations || this.disposed) throw new ArtiWebError('ARTI stateful Web module has been disposed', {code: 'DISPOSED', stage: 'disposed', device: this.device}); }
  private enqueue<T>(operation: () => T | Promise<T>): Promise<T> {
    this.requireActive();
    return this.appendOperation(operation).catch((error: unknown) => {
      if (error instanceof ArtiWebError) throw error;
      const detail = error instanceof Error ? `: ${error.message}` : '';
      throw new ArtiWebError(`ARTI stateful operation failed${detail}`, {code: 'STATE_INVALID', stage: 'state', device: this.device, cause: error});
    });
  }
  private appendOperation<T>(operation: () => T | Promise<T>): Promise<T> {
    const result = this.operationTail.then(operation);
    this.operationTail = result.then(() => undefined, () => undefined);
    return result;
  }
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
      const tensors: TensorMap = {};
      try {
        for (const contract of this.manifest.state) {
          const dims = contract.shape.map((dim) => dim === 'batch' ? batch : requireStaticDimension(dim, contract.name));
          const byteLength = dims.reduce((product, value) => product * value, 1) * Float32Array.BYTES_PER_ELEMENT;
          const buffer = gpuDevice.createBuffer({size: Math.ceil(byteLength / 16) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, mappedAtCreation: true});
          buffers[contract.name] = buffer;
          new Uint8Array(buffer.getMappedRange()).fill(0);
          buffer.unmap();
          tensors[contract.name] = ort.Tensor.fromGpuBuffer(buffer, {dataType: 'float32', dims});
        }
      } catch (error) {
        disposeState(tensors, buffers);
        throw error;
      }
      this.state = tensors;
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
  if (!fetcher) throw new ArtiWebError('loadArtiStateful requires a fetch implementation', {code: 'FETCH_FAILED', stage: 'fetch'});
  if (options.wasmBinary !== undefined) ort.env.wasm.wasmBinary = options.wasmBinary;
  if (options.wasmPaths !== undefined) ort.env.wasm.wasmPaths = options.wasmPaths;
  if (options.wasmNumThreads !== undefined) ort.env.wasm.numThreads = options.wasmNumThreads;
  const base = normalizeBase(baseUrl);
  const startedAt = Date.now();
  const attempts: LoadAttemptDiagnostic[] = [];
  const maxStateBytes = requireBudget(options.maxStateBytes ?? 256 * 1024 * 1024, 'maxStateBytes', base);
  const maxArtifactBytes = requireBudget(options.maxArtifactBytes ?? 512 * 1024 * 1024, 'maxArtifactBytes', base);
  let manifestBuffer: ArrayBuffer;
  let lockBuffer: ArrayBuffer;
  try {
    [manifestBuffer, lockBuffer] = await Promise.all([
      download(new URL(MANIFEST, base), MANIFEST, MAX_METADATA_BYTES, fetcher, options.signal, options.onProgress, 'manifest'),
      download(new URL(LOCK, base), LOCK, MAX_METADATA_BYTES, fetcher, options.signal, options.onProgress, 'lock'),
    ]);
  } catch (error) { throw structuredLoadError(error, base); }
  let lock;
  let manifest;
  try {
    throwIfAborted(options.signal, base);
    lock = parseStatefulLock(JSON.parse(new TextDecoder().decode(lockBuffer)));
    if (await sha256(manifestBuffer) !== lock.manifest.sha256) throw new Error('arti-web.json SHA-256 does not match its lock');
    throwIfAborted(options.signal, base);
    manifest = parseStatefulManifest(JSON.parse(new TextDecoder().decode(manifestBuffer)));
  } catch (error) { throw structuredLoadError(error, base, 'ARTIFACT_INVALID', 'parse'); }
  requireMatchingFileSets(manifest.files, lock.files, base);
  const declaredArtifactBytes = Object.values(manifest.files).reduce((total, file) => total + file.size, 0);
  if (!Number.isSafeInteger(declaredArtifactBytes) || declaredArtifactBytes > maxArtifactBytes) throw new ArtiWebError(`ARTI artifact requires ${declaredArtifactBytes} bytes, exceeding the ${maxArtifactBytes}-byte budget`, {code: 'ARTIFACT_INVALID', stage: 'verify', artifactUrl: base, expected: maxArtifactBytes, actual: declaredArtifactBytes});
  const modelBuffers = new Map<string, ArrayBuffer>();
  let downloadedBytes = 0;
  for (const [name, record] of Object.entries(manifest.files)) {
    const locked = lock.files[name];
    if (!locked || locked.sha256 !== record.sha256 || locked.size !== record.size) throw new ArtiWebError(`${name} manifest and lock records disagree`, {code: 'ARTIFACT_INVALID', stage: 'verify', artifactUrl: base});
    try {
      const remaining = maxArtifactBytes - downloadedBytes;
      const buffer = await download(artifactUrl(name, base), name, remaining, fetcher, options.signal, (event) => progress(options.onProgress, {...event, loadedBytes: downloadedBytes + (event.loadedBytes ?? 0), totalBytes: declaredArtifactBytes}), 'model', locked.size);
      downloadedBytes += buffer.byteLength;
      throwIfAborted(options.signal, base);
      progress(options.onProgress, {stage: 'verify', loadedBytes: downloadedBytes, totalBytes: declaredArtifactBytes});
      await verifyFile(name, buffer, locked);
      throwIfAborted(options.signal, base);
      modelBuffers.set(name, buffer);
    } catch (error) { throw structuredLoadError(error, base, 'ARTIFACT_INVALID', 'verify'); }
  }
  const requested = options.device ?? 'auto';
  const candidates: ActiveARTIDevice[] = requested === 'auto' ? (hasWebGPU() ? ['webgpu', 'wasm'] : ['wasm']) : [requested];
  let lastError: unknown;
  for (const device of candidates) {
    const attemptStarted = Date.now();
    progress(options.onProgress, {stage: 'initialize', device});
    const sessions = new Map<string, ort.InferenceSession>();
    try {
      throwIfAborted(options.signal, base);
      if (!manifest.runtime.execution_providers.includes(device)) throw new Error(`${device} is not supported by this artifact`);
      if (device === 'webgpu' && !hasWebGPU()) throw new Error('WebGPU is not available in this environment');
      for (const [name, entry] of Object.entries(manifest.entrypoints)) {
        const buffer = modelBuffers.get(entry.file)!;
        sessions.set(name, await ort.InferenceSession.create(new Uint8Array(buffer), {executionProviders: [device], preferredOutputLocation: device === 'webgpu' ? 'gpu-buffer' : 'cpu'}));
        throwIfAborted(options.signal, base);
      }
      attempts.push({device, startedAt: attemptStarted, finishedAt: Date.now(), success: true});
      const diagnostics: LoadDiagnostics = {artifactUrl: sanitizeArtifactUrl(base), startedAt, finishedAt: Date.now(), attempts, selectedDevice: device};
      progress(options.onProgress, {stage: 'ready', device});
      return new ARTIStatefulWebModule(sessions, manifest, device, maxStateBytes, () => loadArtiStateful(base, {...options, signal: undefined}), diagnostics);
    } catch (error) {
      lastError = error; await Promise.all([...sessions.values()].map((session) => session.release()));
      attempts.push({device, startedAt: attemptStarted, finishedAt: Date.now(), success: false, error: summarizeDiagnosticError(error)});
      if (error instanceof ArtiWebError && error.code === 'ABORTED') break;
      if (requested !== 'auto') break;
    }
  }
  if (lastError instanceof ArtiWebError && lastError.code === 'ABORTED') throw lastError;
  throw new ArtiWebError(`unable to initialize ARTI stateful Web runtime for ${requested}`, {code: requested === 'auto' ? 'INITIALIZATION_FAILED' : 'DEVICE_UNAVAILABLE', stage: 'initialize', artifactUrl: base, device: requested, cause: lastError});
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
function disposeMapExcept(values: TensorMap, preserved: Set<ort.Tensor>): void {
  const disposed = new Set<ort.Tensor>();
  for (const tensor of Object.values(values)) if (!preserved.has(tensor) && !disposed.has(tensor)) { disposed.add(tensor); tensor.dispose(); }
}
function assertUnaliasedCommit(outputs: TensorMap, next: TensorMap, previous: TensorMap, stateOutputNames: Set<string>): void {
  const nextOwners = new Map<ort.Tensor, string>();
  for (const [name, tensor] of Object.entries(next)) {
    const owner = nextOwners.get(tensor);
    if (owner) throw new Error(`ARTI stateful commit aliases new state ${owner} and ${name}`);
    nextOwners.set(tensor, name);
  }
  const previousOwners = new Map(Object.entries(previous).map(([name, tensor]) => [tensor, name]));
  for (const [tensor, name] of nextOwners) {
    const previousName = previousOwners.get(tensor);
    if (previousName) throw new Error(`ARTI stateful commit aliases new state ${name} with previous state ${previousName}`);
  }
  for (const [name, tensor] of Object.entries(outputs)) {
    if (stateOutputNames.has(name)) continue;
    const stateName = nextOwners.get(tensor);
    if (stateName) throw new Error(`ARTI stateful commit aliases output ${name} with state ${stateName}`);
  }
}
function disposeState(values: TensorMap, buffers: Record<string, GPUBuffer> | null): void {
  disposeMap(values);
  if (buffers) for (const buffer of Object.values(buffers)) buffer.destroy();
}
function normalizeBase(value: string | URL): URL { const url = value instanceof URL ? new URL(value.href) : new URL(value, globalThis.location?.href ?? 'http://localhost/'); if (!url.pathname.endsWith('/')) url.pathname += '/'; return url; }
function artifactUrl(name: string, base: URL): URL {
  const url = new URL(name, base);
  if (url.origin !== base.origin || !url.pathname.startsWith(base.pathname) || url.search || url.hash) throw new Error(`ARTI artifact file ${name} escapes its base URL`);
  return url;
}
function hasWebGPU(): boolean { return typeof navigator !== 'undefined' && 'gpu' in navigator && navigator.gpu !== undefined; }
