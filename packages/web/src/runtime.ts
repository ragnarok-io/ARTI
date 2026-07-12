import * as ort from 'onnxruntime-web/webgpu';

import { parseLock, parseManifest, sha256, verifyFile } from './artifact.js';
import type { ActiveARTIDevice, ARTIWebManifest, TensorContract } from './generated/contract.js';
import type { LoadArtiOptions, TensorMap } from './types.js';

const MANIFEST = 'arti-web.json';
const MODEL = 'model.onnx';
const LOCK = 'arti-web.lock.json';

/** A generic executor for a Python-exported ARTI graph. */
export class ARTIWebModule {
  readonly manifest: ARTIWebManifest;
  readonly device: ActiveARTIDevice;
  private session: ort.InferenceSession | null;

  /** @internal */
  constructor(session: ort.InferenceSession, manifest: ARTIWebManifest, device: ActiveARTIDevice) {
    this.session = session;
    this.manifest = manifest;
    this.device = device;
  }

  /** Execute exactly the named inputs and outputs declared by Python. */
  async run(inputs: TensorMap, outputs?: TensorMap): Promise<TensorMap> {
    if (!this.session) throw new Error('ARTI Web module has been disposed');
    const symbols = new Map<string, number>();
    const feeds = validateTensorMap(inputs, this.manifest.inputs, symbols, 'input');
    const fetches = outputs === undefined ? undefined : validateTensorMap(outputs, this.manifest.outputs, symbols, 'output');
    if (fetches && Object.values(fetches).some((tensor) => tensor.location !== 'gpu-buffer')) {
      throw new Error('preallocated ARTI Web outputs must use gpu-buffer tensors');
    }
    const results = fetches ? await this.session.run(feeds, fetches) : await this.session.run(feeds);
    for (const contract of this.manifest.outputs) {
      if (!results[contract.name]) throw new Error(`ARTI Web runtime did not produce ${contract.name}`);
    }
    return results;
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

  /** Release the ONNX Runtime session. Calling it more than once is safe. */
  async dispose(): Promise<void> {
    const session = this.session;
    this.session = null;
    if (session) await session.release();
  }
}

/** Load and verify an artifact compiled by the Python package. */
export async function loadArti(baseUrl: string | URL, options: LoadArtiOptions = {}): Promise<ARTIWebModule> {
  const fetcher = options.fetch ?? globalThis.fetch;
  if (!fetcher) throw new Error('loadArti requires a fetch implementation');
  if (options.wasmBinary !== undefined) ort.env.wasm.wasmBinary = options.wasmBinary;
  if (options.wasmPaths !== undefined) ort.env.wasm.wasmPaths = options.wasmPaths;
  if (options.wasmNumThreads !== undefined) ort.env.wasm.numThreads = options.wasmNumThreads;
  const base = normalizeBase(baseUrl);
  const [manifestResponse, lockResponse] = await Promise.all([fetcher(new URL(MANIFEST, base)), fetcher(new URL(LOCK, base))]);
  requireOk(manifestResponse, MANIFEST);
  requireOk(lockResponse, LOCK);
  const [manifestBuffer, lockValue] = await Promise.all([manifestResponse.arrayBuffer(), lockResponse.json()]);
  const lock = parseLock(lockValue);
  if ((await sha256(manifestBuffer)) !== lock.manifest.sha256) throw new Error('arti-web.json SHA-256 does not match its lock');
  const manifest = parseManifest(JSON.parse(new TextDecoder().decode(manifestBuffer)));
  const manifestModel = manifest.files[MODEL];
  const lockModel = lock.files[MODEL];
  if (!manifestModel || !lockModel || manifestModel.sha256 !== lockModel.sha256 || manifestModel.size !== lockModel.size) throw new Error('model.onnx manifest and lock records disagree');
  const modelResponse = await fetcher(new URL(MODEL, base));
  requireOk(modelResponse, MODEL);
  const modelBuffer = await modelResponse.arrayBuffer();
  await verifyFile(MODEL, modelBuffer, lockModel);

  const requested = options.device ?? 'auto';
  const candidates: ActiveARTIDevice[] = requested === 'auto' ? (hasWebGPU() ? ['webgpu', 'wasm'] : ['wasm']) : [requested];
  let lastError: unknown;
  for (const device of candidates) {
    if (!manifest.runtime.execution_providers.includes(device)) { lastError = new Error(`${device} is not supported by this ARTI Web artifact`); continue; }
    if (device === 'webgpu' && !hasWebGPU()) { lastError = new Error('WebGPU is not available in this environment'); continue; }
    try {
      const session = await ort.InferenceSession.create(new Uint8Array(modelBuffer), {
        executionProviders: [device],
        preferredOutputLocation: device === 'webgpu' ? 'gpu-buffer' : 'cpu',
      });
      return new ARTIWebModule(session, manifest, device);
    } catch (error) {
      lastError = error;
      if (requested !== 'auto') break;
    }
  }
  const detail = lastError instanceof Error ? `: ${lastError.message}` : '';
  throw new Error(`unable to initialize ARTI Web runtime for ${requested}${detail}`, {cause: lastError});
}

function validateTensorMap(values: TensorMap, contracts: TensorContract[], symbols: Map<string, number>, kind: string): TensorMap {
  const expected = new Set(contracts.map((contract) => contract.name));
  for (const name of Object.keys(values)) if (!expected.has(name)) throw new Error(`${name} is not declared as an ARTI Web ${kind}`);
  const resolved: TensorMap = {};
  for (const contract of contracts) {
    const tensor = values[contract.name];
    if (!tensor) throw new Error(`${contract.name} is required by this ARTI Web artifact`);
    validateTensor(tensor, contract, symbols);
    resolved[contract.name] = tensor;
  }
  return resolved;
}

function validateTensor(tensor: ort.Tensor, contract: TensorContract, symbols: Map<string, number>): void {
  if (tensor.type !== contract.dtype) throw new Error(`${contract.name} must use ${contract.dtype}, got ${tensor.type}`);
  if (tensor.dims.length !== contract.shape.length) throw new Error(`${contract.name} rank does not match its artifact contract`);
  contract.shape.forEach((expected, index) => {
    const actual = tensor.dims[index];
    if (actual === undefined) throw new Error(`${contract.name} is missing dimension ${index}`);
    if (typeof expected === 'number' && expected !== actual) throw new Error(`${contract.name} dimension ${index} must be ${expected}, got ${actual}`);
    if (typeof expected === 'string') {
      const previous = symbols.get(expected);
      if (previous !== undefined && previous !== actual) throw new Error(`${contract.name} dimension ${index} conflicts with dynamic axis ${expected}`);
      symbols.set(expected, actual);
    }
  });
}

function requireSingle(contracts: TensorContract[], kind: string): [TensorContract] {
  if (contracts.length !== 1) throw new Error(`forward requires exactly one artifact ${kind}; use run() for named tensors`);
  return [contracts[0]!];
}

function normalizeBase(value: string | URL): URL {
  const url = value instanceof URL ? new URL(value.href) : new URL(value, globalThis.location?.href ?? 'http://localhost/');
  if (!url.pathname.endsWith('/')) url.pathname += '/';
  return url;
}
function requireOk(response: Response, name: string): void { if (!response.ok) throw new Error(`failed to load ${name}: HTTP ${response.status}`); }
function hasWebGPU(): boolean { return typeof navigator !== 'undefined' && 'gpu' in navigator && navigator.gpu !== undefined; }
