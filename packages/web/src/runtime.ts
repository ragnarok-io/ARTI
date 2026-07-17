import * as ort from 'onnxruntime-web/webgpu';

import { parseLock, parseManifest, sha256, verifyFile } from './artifact.js';
import {ArtiWebError, sanitizeArtifactUrl} from './errors.js';
import type { ActiveARTIDevice, ARTIWebManifest, TensorContract, TensorDType } from './generated/contract.js';
import {fromCPU, isCPUTensor, toCPU} from './tensor.js';
import {summarizeDiagnosticError, type LoadAttemptDiagnostic, type LoadDiagnostics, type LoadProgressCallback, type LoadStage} from './diagnostics.js';
import type {CPUTensor, InspectedCPUTensor, InspectOptions, LoadArtiOptions, OperationOptions, RunTimings, TensorInput, TensorMap} from './types.js';

const MANIFEST = 'arti-web.json';
const MODEL = 'model.onnx';
const TYPESCRIPT = 'artifact.ts';
const LOCK = 'arti-web.lock.json';
const MAX_METADATA_BYTES = 1024 * 1024;
const DEFAULT_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024;

/** Owns the tensors produced by one inspectable ORT run. */
export class OwnedRunResult {
  readonly outputs: Readonly<TensorMap>;
  readonly timings: Readonly<RunTimings>;
  readonly device: ActiveARTIDevice;
  private readonly contracts: ReadonlyMap<string, TensorContract>;
  private readonly downloads = new Set<Promise<unknown>>();
  private readonly onDisposed: () => void;
  private disposing = false;
  private disposePromise: Promise<void> | null = null;

  /** @internal */
  constructor(outputs: TensorMap, contracts: TensorContract[], device: ActiveARTIDevice, timings: RunTimings, onDisposed: () => void) {
    this.outputs = Object.freeze({...outputs});
    this.contracts = new Map(contracts.map((contract) => [contract.name, contract]));
    this.device = device;
    this.timings = Object.freeze({...timings});
    this.onDisposed = onDisposed;
  }

  /** Download only the named retained outputs. Omit names to download all retained outputs. */
  async download(outputs?: readonly string[]): Promise<Record<string, InspectedCPUTensor>> {
    if (this.disposing) throw new ArtiWebError('ARTI inspect result has expired', {code: 'DISPOSED', stage: 'disposed', device: this.device});
    const names = selectNames(outputs, [...this.contracts.keys()], 'retained inspect output');
    const operation = (async () => {
      const result: Record<string, InspectedCPUTensor> = {};
      for (const name of names) {
        const tensor = this.outputs[name];
        const contract = this.contracts.get(name);
        if (!tensor || !contract) throw new ArtiWebError(`${name} is not retained by this inspect result`, {code: 'CONTRACT_MISMATCH', stage: 'output', tensorName: name, device: this.device});
        result[name] = await downloadTensor(tensor, contract.dtype);
      }
      if (this.disposing) throw new ArtiWebError('ARTI inspect result expired while downloading', {code: 'DISPOSED', stage: 'disposed', device: this.device});
      return result;
    })();
    this.downloads.add(operation);
    try { return await operation; }
    finally { this.downloads.delete(operation); }
  }

  /** Release every retained ORT tensor. Calling it more than once is safe. */
  async dispose(): Promise<void> {
    if (this.disposePromise) return this.disposePromise;
    this.disposing = true;
    this.disposePromise = (async () => {
      await Promise.allSettled([...this.downloads]);
      disposeUnique(this.outputs as TensorMap);
      this.onDisposed();
    })();
    return this.disposePromise;
  }
}

/** A generic executor for a Python-exported ARTI graph. */
export class ARTIWebModule {
  readonly manifest: ARTIWebManifest;
  readonly device: ActiveARTIDevice;
  readonly diagnostics: LoadDiagnostics;
  private session: ort.InferenceSession | null;
  private readonly inFlight = new Set<Promise<unknown>>();
  private readonly ownedResults = new Set<OwnedRunResult>();
  private disposing = false;
  private disposePromise: Promise<void> | null = null;
  private lifecycleEpoch = 0;

  /** @internal */
  constructor(session: ort.InferenceSession, manifest: ARTIWebManifest, device: ActiveARTIDevice, diagnostics?: LoadDiagnostics) {
    this.session = session;
    this.manifest = manifest;
    this.device = device;
    this.diagnostics = diagnostics ?? {artifactUrl: '', startedAt: 0, finishedAt: 0, attempts: [], selectedDevice: device};
  }

  /** Execute exactly the named inputs and outputs declared by Python. */
  async run(inputs: TensorMap, outputs?: TensorMap): Promise<TensorMap> {
    if (this.disposing || !this.session) throw new ArtiWebError('ARTI Web module has been disposed', {code: 'DISPOSED', stage: 'disposed', device: this.device});
    const session = this.session;
    const symbols = new Map<string, number>();
    const feeds = validateTensorMap(inputs, this.manifest.inputs, symbols, 'input');
    const fetches = outputs === undefined ? undefined : validateTensorMap(outputs, this.manifest.outputs, symbols, 'output');
    if (fetches && Object.values(fetches).some((tensor) => tensor.location !== 'gpu-buffer')) throw new ArtiWebError('preallocated ARTI Web outputs must use gpu-buffer tensors', {code: 'CONTRACT_MISMATCH', stage: 'output', device: this.device});
    const operation = (async () => {
      let results: TensorMap;
      try { results = fetches ? await session.run(feeds, fetches) : await session.run(feeds); }
      catch (error) { throw new ArtiWebError('ARTI Web inference failed', {code: 'INFERENCE_FAILED', stage: 'execution', device: this.device, cause: error}); }
      for (const contract of this.manifest.outputs) {
        if (!results[contract.name]) throw new ArtiWebError(`ARTI Web runtime did not produce ${contract.name}`, {code: 'CONTRACT_MISMATCH', stage: 'output', tensorName: contract.name, device: this.device});
      }
      return results;
    })();
    this.inFlight.add(operation);
    try { return await operation; }
    finally { this.inFlight.delete(operation); }
  }

  /** Execute Python-exported outputs and retain them until the result is disposed. */
  async inspect(inputs: TensorMap, options: InspectOptions = {}): Promise<OwnedRunResult> {
    throwIfOperationAborted(options.signal, this.device);
    if (this.disposing || !this.session) throw new ArtiWebError('ARTI Web module has been disposed', {code: 'DISPOSED', stage: 'disposed', device: this.device});
    const session = this.session;
    const epoch = this.lifecycleEpoch;
    const symbols = new Map<string, number>();
    const feeds = validateTensorMap(inputs, this.manifest.inputs, symbols, 'input');
    const outputNames = selectNames(options.outputs, this.manifest.outputs.map((contract) => contract.name), 'artifact output');
    const contracts = outputNames.map((name) => this.manifest.outputs.find((contract) => contract.name === name)!);
    const operation = this.inspectOperation(session, feeds, contracts, symbols, options.signal, epoch);
    this.inFlight.add(operation);
    try { return await operation; }
    finally { this.inFlight.delete(operation); }
  }

  private async inspectOperation(
    session: ort.InferenceSession,
    feeds: TensorMap,
    contracts: TensorContract[],
    symbols: Map<string, number>,
    signal: AbortSignal | undefined,
    epoch: number,
  ): Promise<OwnedRunResult> {
    const startedAt = now();
    let results: TensorMap | undefined;
    try {
      results = await session.run(feeds, contracts.map((contract) => contract.name));
      const finishedAt = now();
      if (signal?.aborted) throw operationAbortError(signal, this.device);
      if (this.disposing || this.lifecycleEpoch !== epoch || !this.session) throw new ArtiWebError('ARTI inspect run expired before ownership transfer', {code: 'DISPOSED', stage: 'disposed', device: this.device});
      validateSelectedOutputs(results, contracts, symbols, this.device);
      let owned!: OwnedRunResult;
      owned = new OwnedRunResult(
        results,
        contracts,
        this.device,
        {startedAt, finishedAt, inferenceMs: finishedAt - startedAt},
        () => this.ownedResults.delete(owned),
      );
      this.ownedResults.add(owned);
      results = undefined;
      return owned;
    } catch (error) {
      if (signal?.aborted && !(error instanceof ArtiWebError && error.code === 'ABORTED')) throw operationAbortError(signal, this.device, error);
      if (error instanceof ArtiWebError) throw error;
      throw new ArtiWebError('ARTI inspect inference failed', {code: 'INFERENCE_FAILED', stage: 'execution', device: this.device, cause: error});
    } finally {
      if (results) disposeUnique(results);
    }
  }

  /** Convenience path for artifacts with exactly one input and one output. */
  async forward(x: ort.Tensor): Promise<ort.Tensor> {
    const [input] = requireSingle(this.manifest.inputs, 'input');
    const [output] = requireSingle(this.manifest.outputs, 'output');
    const results = await this.run({[input.name]: x});
    return results[output.name]!;
  }

  /** Single-input/single-output execution into a caller-owned GPU buffer. */
  async forwardInto(outputTensor: ort.Tensor, x: ort.Tensor): Promise<ort.Tensor> {
    const [input] = requireSingle(this.manifest.inputs, 'input');
    const [output] = requireSingle(this.manifest.outputs, 'output');
    const results = await this.run({[input.name]: x}, {[output.name]: outputTensor});
    return results[output.name]!;
  }

  /** CPU-friendly inference which owns and releases all temporary ORT tensors. */
  async predict(inputs: Readonly<Record<string, TensorInput>>, options: OperationOptions = {}): Promise<Record<string, CPUTensor>> {
    throwIfAborted(options.signal);
    if (this.disposing || !this.session) throw new ArtiWebError('ARTI Web module has been disposed', {code: 'DISPOSED', stage: 'disposed', device: this.device});
    const operation = this.predictOperation(inputs, options);
    this.inFlight.add(operation);
    try { return await operation; }
    finally { this.inFlight.delete(operation); }
  }

  private async predictOperation(inputs: Readonly<Record<string, TensorInput>>, options: OperationOptions): Promise<Record<string, CPUTensor>> {
    const feeds: TensorMap = {};
    const temporaryInputs: ort.Tensor[] = [];
    let results: TensorMap | undefined;
    try {
      for (const [name, value] of Object.entries(inputs)) {
        if (isCPUTensor(value)) {
          const converted = fromCPU(value);
          temporaryInputs.push(converted);
          feeds[name] = converted;
        } else feeds[name] = value;
      }
      results = await this.run(feeds);
      throwIfAborted(options.signal);
      const output: Record<string, CPUTensor> = {};
      for (const contract of this.manifest.outputs) {
        output[contract.name] = await toCPU(results[contract.name]!);
        throwIfAborted(options.signal);
      }
      return output;
    } catch (error) {
      if (error instanceof ArtiWebError) throw error;
      throw new ArtiWebError('ARTI Web prediction failed', {code: 'INFERENCE_FAILED', stage: 'predict', device: this.device, cause: error});
    } finally {
      if (results) disposeUnique(results);
      for (const tensor of temporaryInputs) tensor.dispose();
    }
  }

  /** Release the ONNX Runtime session. Calling it more than once is safe. */
  async dispose(): Promise<void> {
    if (this.disposePromise) return this.disposePromise;
    this.disposing = true;
    this.lifecycleEpoch += 1;
    const session = this.session;
    this.session = null;
    this.disposePromise = (async () => {
      await Promise.allSettled([...this.inFlight]);
      await Promise.allSettled([...this.ownedResults].map((result) => result.dispose()));
      if (session) await session.release();
    })();
    return this.disposePromise;
  }
}

/** Load and verify an artifact compiled by the Python package. */
export async function loadArti(baseUrl: string | URL, options: LoadArtiOptions = {}): Promise<ARTIWebModule> {
  const fetcher = options.fetch ?? globalThis.fetch;
  if (!fetcher) throw new ArtiWebError('loadArti requires a fetch implementation', {code: 'FETCH_FAILED', stage: 'fetch'});
  if (options.wasmBinary !== undefined) ort.env.wasm.wasmBinary = options.wasmBinary;
  if (options.wasmPaths !== undefined) ort.env.wasm.wasmPaths = options.wasmPaths;
  if (options.wasmNumThreads !== undefined) ort.env.wasm.numThreads = options.wasmNumThreads;
  const base = normalizeBase(baseUrl);
  const startedAt = Date.now();
  const attempts: LoadAttemptDiagnostic[] = [];
  const maxArtifactBytes = requireBudget(options.maxArtifactBytes ?? DEFAULT_MAX_ARTIFACT_BYTES, 'maxArtifactBytes', base);
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
    lock = parseLock(JSON.parse(new TextDecoder().decode(lockBuffer)));
    if ((await sha256(manifestBuffer)) !== lock.manifest.sha256) throw new Error('arti-web.json SHA-256 does not match its lock');
    throwIfAborted(options.signal, base);
    manifest = parseManifest(JSON.parse(new TextDecoder().decode(manifestBuffer)));
  } catch (error) { throw structuredLoadError(error, base, 'ARTIFACT_INVALID', 'parse'); }
  const declaredFiles = Object.entries(manifest.files);
  requireMatchingFileSets(manifest.files, lock.files, base);
  for (const [name] of declaredFiles) if (name !== MODEL && name !== TYPESCRIPT) throw new ArtiWebError(`unsupported ARTI Web artifact file ${name}`, {code: 'ARTIFACT_INVALID', stage: 'verify', artifactUrl: base});
  const declaredArtifactBytes = declaredFiles.reduce((total, [, record]) => total + record.size, 0);
  if (!Number.isSafeInteger(declaredArtifactBytes) || declaredArtifactBytes > maxArtifactBytes) throw artifactBudgetError(declaredArtifactBytes, maxArtifactBytes, base);
  const manifestModel = manifest.files[MODEL];
  const lockModel = lock.files[MODEL];
  if (!manifestModel || !lockModel || manifestModel.sha256 !== lockModel.sha256 || manifestModel.size !== lockModel.size) throw new ArtiWebError('model.onnx manifest and lock records disagree', {code: 'ARTIFACT_INVALID', stage: 'verify', artifactUrl: base});
  if (lockModel.size > maxArtifactBytes) throw artifactBudgetError(lockModel.size, maxArtifactBytes, base);
  let modelBuffer: ArrayBuffer;
  try {
    modelBuffer = await download(new URL(MODEL, base), MODEL, maxArtifactBytes, fetcher, options.signal, options.onProgress, 'model', lockModel.size);
    throwIfAborted(options.signal, base);
    progress(options.onProgress, {stage: 'verify', loadedBytes: modelBuffer.byteLength, totalBytes: lockModel.size});
    await verifyFile(MODEL, modelBuffer, lockModel);
    const typescriptRecord = manifest.files[TYPESCRIPT];
    if (typescriptRecord) {
      const locked = lock.files[TYPESCRIPT];
      if (!locked || locked.sha256 !== typescriptRecord.sha256 || locked.size !== typescriptRecord.size) throw new ArtiWebError('artifact.ts manifest and lock records disagree', {code: 'ARTIFACT_INVALID', stage: 'verify', artifactUrl: base});
      const sidecar = await download(new URL(TYPESCRIPT, base), TYPESCRIPT, maxArtifactBytes - modelBuffer.byteLength, fetcher, options.signal, options.onProgress, 'verify', locked.size);
      await verifyFile(TYPESCRIPT, sidecar, locked);
    }
    throwIfAborted(options.signal, base);
  } catch (error) { throw structuredLoadError(error, base, 'ARTIFACT_INVALID', 'verify'); }

  const requested = options.device ?? 'auto';
  const candidates: ActiveARTIDevice[] = requested === 'auto' ? (hasWebGPU() ? ['webgpu', 'wasm'] : ['wasm']) : [requested];
  let lastError: unknown;
  for (const device of candidates) {
    const attemptStarted = Date.now();
    let session: ort.InferenceSession | undefined;
    progress(options.onProgress, {stage: 'initialize', device});
    try {
      throwIfAborted(options.signal, base);
      if (!manifest.runtime.execution_providers.includes(device)) throw new Error(`${device} is not supported by this ARTI Web artifact`);
      if (device === 'webgpu' && !hasWebGPU()) throw new Error('WebGPU is not available in this environment');
      session = await ort.InferenceSession.create(new Uint8Array(modelBuffer), {
        executionProviders: [device],
        preferredOutputLocation: device === 'webgpu' ? 'gpu-buffer' : 'cpu',
      });
      throwIfAborted(options.signal, base);
      attempts.push({device, startedAt: attemptStarted, finishedAt: Date.now(), success: true});
      const diagnostics: LoadDiagnostics = {artifactUrl: sanitizeArtifactUrl(base), startedAt, finishedAt: Date.now(), attempts, selectedDevice: device};
      progress(options.onProgress, {stage: 'ready', device});
      return new ARTIWebModule(session, manifest, device, diagnostics);
    } catch (error) {
      if (session) await session.release();
      lastError = error;
      attempts.push({device, startedAt: attemptStarted, finishedAt: Date.now(), success: false, error: summarizeDiagnosticError(error)});
      if (error instanceof ArtiWebError && error.code === 'ABORTED') break;
      if (requested !== 'auto') break;
    }
  }
  const detail = lastError instanceof Error ? `: ${lastError.message}` : '';
  if (lastError instanceof ArtiWebError && lastError.code === 'ABORTED') throw lastError;
  throw new ArtiWebError(`unable to initialize ARTI Web runtime for ${requested}${detail}`, {code: requested === 'auto' ? 'INITIALIZATION_FAILED' : 'DEVICE_UNAVAILABLE', stage: 'initialize', artifactUrl: base, device: requested, cause: lastError});
}

export async function download(url: URL, name: string, limit: number, fetcher: typeof globalThis.fetch, signal: AbortSignal | undefined, onProgress: LoadProgressCallback | undefined, stage: LoadStage, expectedSize?: number): Promise<ArrayBuffer> {
  throwIfAborted(signal, url);
  let response: Response;
  try { response = await fetcher(url, {signal}); }
  catch (error) { throw structuredLoadError(error, url); }
  if (!response.ok) throw new ArtiWebError(`failed to load ${name}: HTTP ${response.status}`, {code: 'FETCH_FAILED', stage: 'fetch', artifactUrl: url});
  const contentLength = parseContentLength(response.headers.get('content-length'));
  if (expectedSize !== undefined && contentLength !== undefined && contentLength !== expectedSize) throw new ArtiWebError(`${name} Content-Length does not match its declared size`, {code: 'ARTIFACT_INVALID', stage: 'fetch', artifactUrl: url, expected: expectedSize, actual: contentLength});
  if (contentLength !== undefined && contentLength > limit) throw artifactBudgetError(contentLength, limit, url);
  if (expectedSize !== undefined && expectedSize > limit) throw artifactBudgetError(expectedSize, limit, url);
  const readLimit = expectedSize === undefined ? limit : Math.min(limit, expectedSize);
  const totalBytes = contentLength ?? expectedSize;
  const reader = response.body?.getReader();
  if (!reader) {
    const value = await response.arrayBuffer();
    if (value.byteLength > readLimit) throw artifactBudgetError(value.byteLength, readLimit, url);
    progress(onProgress, {stage, loadedBytes: value.byteLength, totalBytes});
    throwIfAborted(signal, url);
    return value;
  }
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  try {
    while (true) {
      throwIfAborted(signal, url);
      const {done, value} = await reader.read();
      if (done) break;
      loaded += value.byteLength;
      if (!Number.isSafeInteger(loaded) || loaded > readLimit) throw artifactBudgetError(loaded, readLimit, url);
      chunks.push(value);
      progress(onProgress, {stage, loadedBytes: loaded, totalBytes});
    }
  } catch (error) { await reader.cancel().catch(() => undefined); throw error; }
  throwIfAborted(signal, url);
  const result = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) { result.set(chunk, offset); offset += chunk.byteLength; }
  return result.buffer;
}

export function throwIfAborted(signal?: AbortSignal, artifactUrl?: string | URL): void {
  if (signal?.aborted) throw new ArtiWebError('ARTI Web operation was aborted', {code: 'ABORTED', stage: 'fetch', artifactUrl, cause: signal.reason});
}
export function requireBudget(value: number, name: string, artifactUrl: URL): number {
  if (!Number.isSafeInteger(value) || value <= 0) throw new ArtiWebError(`${name} must be a positive safe integer`, {code: 'CONTRACT_MISMATCH', stage: 'input', artifactUrl, expected: 'positive safe integer', actual: value});
  return value;
}
export function progress(callback: LoadProgressCallback | undefined, value: Parameters<LoadProgressCallback>[0]): void { try { callback?.(value); } catch { /* Progress observers cannot break loading. */ } }
export function structuredLoadError(error: unknown, url: string | URL, code: 'ARTIFACT_INVALID' | 'FETCH_FAILED' = 'FETCH_FAILED', stage: 'fetch' | 'parse' | 'verify' = 'fetch'): ArtiWebError {
  if (error instanceof ArtiWebError) return error;
  if (error instanceof DOMException && error.name === 'AbortError') return new ArtiWebError('ARTI Web operation was aborted', {code: 'ABORTED', stage: 'fetch', artifactUrl: url, cause: error});
  const detail = error instanceof Error && error.message ? `: ${error.message}` : '';
  return new ArtiWebError(`${stage === 'fetch' ? 'failed to fetch ARTI Web artifact' : 'invalid ARTI Web artifact'}${detail}`, {code, stage, artifactUrl: url, cause: error});
}
function artifactBudgetError(actual: number, limit: number, url: string | URL): ArtiWebError { return new ArtiWebError(`ARTI artifact requires ${actual} bytes, exceeding the ${limit}-byte budget`, {code: 'ARTIFACT_INVALID', stage: 'fetch', artifactUrl: url, expected: limit, actual}); }
function parseContentLength(value: string | null): number | undefined { if (value === null || !/^\d+$/.test(value)) return undefined; const result = Number(value); return Number.isSafeInteger(result) ? result : undefined; }
export function requireMatchingFileSets(manifestFiles: Record<string, unknown>, lockFiles: Record<string, unknown>, url: URL): void {
  const manifestNames = Object.keys(manifestFiles).sort(); const lockNames = Object.keys(lockFiles).sort();
  if (manifestNames.length !== lockNames.length || manifestNames.some((name, index) => name !== lockNames[index])) throw new ArtiWebError('manifest and lock file sets disagree', {code: 'ARTIFACT_INVALID', stage: 'verify', artifactUrl: url, expected: manifestNames, actual: lockNames});
}
function disposeUnique(values: TensorMap): void { for (const tensor of new Set(Object.values(values))) tensor.dispose(); }

function selectNames(requested: readonly string[] | undefined, available: readonly string[], kind: string): string[] {
  if (requested === undefined) return [...available];
  if (requested.length === 0) throw new ArtiWebError(`inspect ${kind} selection cannot be empty`, {code: 'CONTRACT_MISMATCH', stage: 'output', expected: available, actual: requested});
  const names = [...requested];
  if (new Set(names).size !== names.length) throw new ArtiWebError(`inspect ${kind} selection contains duplicates`, {code: 'CONTRACT_MISMATCH', stage: 'output', expected: available, actual: names});
  const known = new Set(available);
  for (const name of names) if (!known.has(name)) throw new ArtiWebError(`${name} is not a declared ${kind}`, {code: 'CONTRACT_MISMATCH', stage: 'output', tensorName: name, expected: available, actual: name});
  return names;
}

function validateSelectedOutputs(outputs: TensorMap, contracts: TensorContract[], symbols: Map<string, number>, device: ActiveARTIDevice): void {
  for (const contract of contracts) {
    const tensor = outputs[contract.name];
    if (!tensor) throw new ArtiWebError(`ARTI Web runtime did not produce ${contract.name}`, {code: 'CONTRACT_MISMATCH', stage: 'output', tensorName: contract.name, device});
    validateTensor(tensor, contract, symbols, 'output');
  }
}

function validateTensorMap(values: TensorMap, contracts: TensorContract[], symbols: Map<string, number>, kind: string): TensorMap {
  const expected = new Set(contracts.map((contract) => contract.name));
  for (const name of Object.keys(values)) if (!expected.has(name)) throw tensorContractError(`${name} is not declared as an ARTI Web ${kind}`, name, [...expected], name, kind);
  const resolved: TensorMap = {};
  for (const contract of contracts) {
    const tensor = values[contract.name];
    if (!tensor) throw tensorContractError(`${contract.name} is required by this ARTI Web artifact`, contract.name, contract, undefined, kind);
    validateTensor(tensor, contract, symbols, kind);
    resolved[contract.name] = tensor;
  }
  return resolved;
}

function validateTensor(tensor: ort.Tensor, contract: TensorContract, symbols: Map<string, number>, kind: string): void {
  if (tensor.type !== contract.dtype) throw tensorContractError(`${contract.name} must use ${contract.dtype}, got ${tensor.type}`, contract.name, contract.dtype, tensor.type, kind);
  if (tensor.dims.length !== contract.shape.length) throw tensorContractError(`${contract.name} rank does not match its artifact contract`, contract.name, contract.shape.length, tensor.dims.length, kind);
  contract.shape.forEach((expected, index) => {
    const actual = tensor.dims[index];
    if (actual === undefined) throw tensorContractError(`${contract.name} is missing dimension ${index}`, contract.name, expected, actual, kind);
    if (typeof expected === 'number' && expected !== actual) throw tensorContractError(`${contract.name} dimension ${index} must be ${expected}, got ${actual}`, contract.name, expected, actual, kind);
    if (typeof expected === 'string') {
      const previous = symbols.get(expected);
      if (previous !== undefined && previous !== actual) throw tensorContractError(`${contract.name} dimension ${index} conflicts with dynamic axis ${expected}`, contract.name, previous, actual, kind);
      symbols.set(expected, actual);
    }
  });
  const bytes = tensorBytes(tensor, contract.dtype, kind);
  if (contract.max_bytes !== undefined && bytes > contract.max_bytes) throw tensorContractError(`${contract.name} requires ${bytes} bytes, exceeding its declared budget`, contract.name, contract.max_bytes, bytes, kind);
}

function tensorBytes(tensor: ort.Tensor, dtype: TensorDType, kind: string): number {
  let elements = 1;
  for (const dim of tensor.dims) {
    elements *= dim;
    if (!Number.isSafeInteger(elements)) throw tensorContractError('tensor element count exceeds safe integer bounds', '', 'safe integer', elements, kind);
  }
  const bytes = elements * (dtype === 'int64' ? 8 : dtype === 'bool' ? 1 : 4);
  if (!Number.isSafeInteger(bytes)) throw tensorContractError('tensor byte size exceeds safe integer bounds', '', 'safe integer', bytes, kind);
  return bytes;
}

async function downloadTensor(tensor: ort.Tensor, dtype: TensorDType): Promise<InspectedCPUTensor> {
  const raw = await tensor.getData();
  let data: InspectedCPUTensor['data'];
  if (dtype === 'float32' && raw instanceof Float32Array) data = Float32Array.from(raw);
  else if (dtype === 'bool' && raw instanceof Uint8Array) data = Uint8Array.from(raw);
  else if (dtype === 'int64' && raw instanceof BigInt64Array) data = BigInt64Array.from(raw);
  else throw new ArtiWebError(`ORT returned an unexpected ${dtype} data representation`, {code: 'CONTRACT_MISMATCH', stage: 'output', expected: dtype, actual: raw.constructor.name});
  return {type: dtype, data, dims: [...tensor.dims]};
}

function now(): number { return globalThis.performance?.now() ?? Date.now(); }
function operationAbortError(signal: AbortSignal, device: ActiveARTIDevice, cause: unknown = signal.reason): ArtiWebError {
  return new ArtiWebError('ARTI inspect operation was aborted', {code: 'ABORTED', stage: 'execution', device, cause});
}
function throwIfOperationAborted(signal: AbortSignal | undefined, device: ActiveARTIDevice): void {
  if (signal?.aborted) throw operationAbortError(signal, device);
}

function requireSingle(contracts: TensorContract[], kind: string): [TensorContract] {
  if (contracts.length !== 1) throw new ArtiWebError(`forward requires exactly one artifact ${kind}; use run() for named tensors`, {code: 'CONTRACT_MISMATCH', stage: kind === 'input' ? 'input' : 'output', expected: 1, actual: contracts.length});
  return [contracts[0]!];
}
function tensorContractError(message: string, tensorName: string, expected: unknown, actual: unknown, kind: string): ArtiWebError {
  return new ArtiWebError(message, {code: 'CONTRACT_MISMATCH', stage: kind === 'output' ? 'output' : 'input', tensorName, expected, actual});
}

function normalizeBase(value: string | URL): URL {
  const url = value instanceof URL ? new URL(value.href) : new URL(value, globalThis.location?.href ?? 'http://localhost/');
  if (!url.pathname.endsWith('/')) url.pathname += '/';
  return url;
}
function hasWebGPU(): boolean { return typeof navigator !== 'undefined' && 'gpu' in navigator && navigator.gpu !== undefined; }
