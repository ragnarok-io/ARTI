import { Tensor, loadArti, loadArtiStateful } from '../src/index.js';
import { env } from 'onnxruntime-web/webgpu';
const wasmUrl = '/ort-runtime.wasm';

interface TensorPayload { dtype?: 'float32' | 'bool' | 'int64'; dims: number[]; data: number[] }
interface FixtureCase { inputs: Record<string, TensorPayload>; outputs: Record<string, TensorPayload> }
interface Fixture { tolerance: {atol: number; rtol: number}; cases: FixtureCase[] }

declare global {
  interface Window {
    artiWebGPUInfo: () => Promise<Record<string, unknown>>;
    runArtiParity: (name: string, warmup?: number, runs?: number) => Promise<Record<string, unknown>>;
    runFusionPulseInspect: () => Promise<Record<string, unknown>>;
    runWorkerInspect: () => Promise<Record<string, unknown>>;
    runStatefulRecall: () => Promise<Record<string, unknown>>;
    runWorkerSmoke: () => Promise<Record<string, unknown>>;
  }
}

window.runWorkerInspect = async () => {
  const fixture = (await (await fetch('/fusion-pulse-inspect/case.json')).json()) as Fixture;
  const sample = fixture.cases[0]!;
  const worker = new Worker(new URL('../examples/worker/arti.worker.ts', import.meta.url), {type: 'module'});
  let nextId = 1;
  const call = <T extends {id: number; type: string}>(request: Record<string, unknown>, transfers: Transferable[] = []) => new Promise<T>((resolve, reject) => {
    const id = nextId++;
    const listener = (event: MessageEvent<T & {error?: {message: string}}>) => {
      if (event.data.id !== id) return;
      worker.removeEventListener('message', listener);
      if (event.data.type === 'error') reject(new Error(event.data.error?.message ?? 'worker failed'));
      else resolve(event.data);
    };
    worker.addEventListener('message', listener);
    worker.postMessage({...request, id}, transfers);
  });
  try {
    const loaded = await call<{
      id: number; type: 'loaded'; device: 'webgpu' | 'wasm';
      manifest: {outputs: Array<{name: string; role?: string; tolerance?: {atol: number; rtol: number}}>};
    }>({type: 'load', baseUrl: '/fusion-pulse-inspect/', device: 'webgpu', wasmPath: wasmUrl});
    const selected = ['fused', 'survival', 'pulse_mask', 'workspace', 'unfold_source_index'];
    const inputs: Record<string, {dtype: 'float32' | 'bool' | 'int64'; dims: number[]; data: ArrayBuffer}> = {};
    const transfers: ArrayBuffer[] = [];
    for (const [name, value] of Object.entries(sample.inputs)) {
      const data = payloadBuffer(value);
      inputs[name] = {dtype: value.dtype ?? 'float32', dims: value.dims, data};
      transfers.push(data);
    }
    const result = await call<{
      id: number; type: 'inspected'; device: 'webgpu' | 'wasm';
      timings: {inferenceMs: number};
      outputs: Record<string, {dtype: 'float32' | 'bool' | 'int64'; dims: number[]; data: ArrayBuffer}>;
    }>({type: 'inspect', inputs, outputs: selected}, transfers);
    let maxAbsolute = 0;
    for (const name of selected) {
      const contract = loaded.manifest.outputs.find((output) => output.name === name)!;
      const expected = sample.outputs[name]!;
      const actual = workerOutputValues(result.outputs[name]!);
      if ((expected.dtype ?? 'float32') === 'float32') {
        if (!contract.tolerance) throw new Error(`${name} does not declare tolerance`);
        for (let index = 0; index < actual.length; index += 1) {
          const expectedValue = Number(expected.data[index]!);
          const difference = Math.abs(Number(actual[index]!) - expectedValue);
          maxAbsolute = Math.max(maxAbsolute, difference);
          if (difference > contract.tolerance.atol + contract.tolerance.rtol * Math.abs(expectedValue)) {
            throw new Error(`${name} worker parity exceeded its declared tolerance`);
          }
        }
      } else if (actual.map(String).join(',') !== expected.data.map((value) => String(Number(value))).join(',')) {
        throw new Error(`${name} worker exact parity failed`);
      }
    }
    await call({type: 'dispose'});
    return {
      device: result.device,
      inferenceMs: result.timings.inferenceMs,
      maxAbsolute,
      outputNames: Object.keys(result.outputs),
      roles: Object.fromEntries(loaded.manifest.outputs.map((output) => [output.name, output.role])),
      inputBuffersDetached: transfers.every((buffer) => buffer.byteLength === 0),
    };
  } finally {
    worker.terminate();
  }
};

window.runWorkerSmoke = async () => {
  const fixture = (await (await fetch('/generic-affine/case.json')).json()) as Fixture;
  const sample = fixture.cases[0]!;
  const worker = new Worker(new URL('../examples/worker/arti.worker.ts', import.meta.url), {type: 'module'});
  let nextId = 1;
  const call = <T extends {id: number; type: string}>(request: Record<string, unknown>, transfers: Transferable[] = []) => new Promise<T>((resolve, reject) => {
    const id = nextId++;
    const listener = (event: MessageEvent<T & {error?: {message: string}}>) => {
      if (event.data.id !== id) return;
      worker.removeEventListener('message', listener);
      if (event.data.type === 'error') reject(new Error(event.data.error?.message ?? 'worker failed'));
      else resolve(event.data);
    };
    worker.addEventListener('message', listener);
    worker.postMessage({...request, id}, transfers);
  });
  try {
    const loaded = await call<{id: number; type: 'loaded'; device: 'webgpu' | 'wasm'}>({type: 'load', baseUrl: '/generic-affine/', device: 'webgpu', wasmPath: wasmUrl});
    const inputs: Record<string, {dtype: 'float32'; dims: number[]; data: ArrayBuffer}> = {};
    const transfers: ArrayBuffer[] = [];
    for (const [name, value] of Object.entries(sample.inputs)) {
      const data = Float32Array.from(value.data).buffer;
      inputs[name] = {dtype: 'float32', dims: value.dims, data};
      transfers.push(data);
    }
    const result = await call<{id: number; type: 'result'; outputs: Record<string, {dims: number[]; data: ArrayBuffer}>}>({type: 'run', inputs}, transfers);
    let maxAbsolute = 0;
    for (const [name, expected] of Object.entries(sample.outputs)) {
      const actual = new Float32Array(result.outputs[name]!.data);
      for (let index = 0; index < actual.length; index += 1) maxAbsolute = Math.max(maxAbsolute, Math.abs(actual[index]! - expected.data[index]!));
    }
    await call({type: 'dispose'});
    return {device: loaded.device, maxAbsolute, inputBuffersDetached: transfers.every((buffer) => buffer.byteLength === 0)};
  } finally {
    worker.terminate();
  }
};

window.artiWebGPUInfo = async () => {
  if (!navigator.gpu) throw new Error('WebGPU is unavailable');
  const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
  if (!adapter) throw new Error('WebGPU adapter is unavailable');
  const info = adapter.info;
  return {
    vendor: info.vendor,
    architecture: info.architecture,
    device: info.device,
    description: info.description,
    isFallbackAdapter: 'isFallbackAdapter' in adapter ? Boolean(adapter.isFallbackAdapter) : false,
  };
};

window.runStatefulRecall = async () => {
  const fixture = await (await fetch('/stateful-recall/case.json')).json() as {
    inputs: Record<string, TensorPayload>; expected: Record<string, TensorPayload>;
  };
  const module = await loadArtiStateful('/stateful-recall/', {device: 'webgpu', wasmPaths: {wasm: wasmUrl}});
  const full = tensor(fixture.inputs.full!); const mask = tensor(fixture.inputs.mask!);
  const first = await module.run('read', {x: full, mask});
  let firstCommittedBytes = 0;
  for (let index = 0; index < 64; index += 1) {
    const diagnostics = await module.commit('update', {trace_key: first.trace_key!, observed: full, mask});
    for (const value of Object.values(diagnostics)) value.dispose();
    if (index === 0) firstCommittedBytes = module.stateInfo().bytes;
  }
  const state = module.stateInfo();
  const corrupt = tensor(fixture.inputs.corrupt!); const novel = tensor(fixture.inputs.unseen!);
  const seen = await module.run('read', {x: corrupt, mask}); const unseen = await module.run('read', {x: novel, mask});
  const seenRecognition = Array.from(await seen.recognition!.getData() as Float32Array);
  const unseenRecognition = Array.from(await unseen.recognition!.getData() as Float32Array);
  for (const output of [...Object.values(first), ...Object.values(seen), ...Object.values(unseen)]) output.dispose();
  full.dispose(); mask.dispose(); corrupt.dispose(); novel.dispose(); await module.dispose();
  return {
    device: module.device, state, firstCommittedBytes,
    seenRecognition: seenRecognition.reduce((a, b) => a + b, 0) / seenRecognition.length,
    unseenRecognition: unseenRecognition.reduce((a, b) => a + b, 0) / unseenRecognition.length,
  };
};

window.runArtiParity = async (name, warmup = 0, runs = 0) => {
  const fixture = (await (await fetch(`/${name}/case.json`)).json()) as Fixture;
  const module = await loadArti(`/${name}/`, {device: 'webgpu', wasmPaths: {wasm: wasmUrl}});
  let maxAbsolute = 0;
  let maxRelative = 0;
  for (const item of fixture.cases) {
    const inputs = Object.fromEntries(Object.entries(item.inputs).map(([key, value]) => [key, tensor(value)]));
    const outputs = await module.run(inputs);
    for (const [outputName, expected] of Object.entries(item.outputs)) {
      const y = outputs[outputName]!;
      const raw = await y.getData();
      if (!(raw instanceof Float32Array)) throw new Error('ARTI Web output is not float32');
      const actual = Array.from(raw);
      for (let index = 0; index < actual.length; index += 1) {
        const difference = Math.abs(actual[index]! - expected.data[index]!);
        maxAbsolute = Math.max(maxAbsolute, difference);
        maxRelative = Math.max(maxRelative, difference / Math.max(Math.abs(expected.data[index]!), 1e-8));
      }
      y.dispose();
    }
  }

  const sample = fixture.cases[0]!;
  const inputs = Object.fromEntries(Object.entries(sample.inputs).map(([key, value]) => [key, tensor(value)]));
  let forwardIntoMaxAbsolute = 0;
  if (Object.keys(sample.outputs).length === 1) {
    const outputName = Object.keys(sample.outputs)[0]!;
    const expected = sample.outputs[outputName]!;
    const gpuDevice = await env.webgpu.device;
    const outputBytes = expected.data.length * Float32Array.BYTES_PER_ELEMENT;
    const outputBuffer = gpuDevice.createBuffer({size: Math.ceil(outputBytes / 16) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST});
    const outputTensor = Tensor.fromGpuBuffer(outputBuffer, {dataType: 'float32', dims: expected.dims});
    const into = (await module.run(inputs, {[outputName]: outputTensor}))[outputName]!;
    if (into !== outputTensor) throw new Error('forwardInto did not preserve the preallocated tensor');
    const staging = gpuDevice.createBuffer({size: Math.ceil(outputBytes / 4) * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
    const encoder = gpuDevice.createCommandEncoder();
    encoder.copyBufferToBuffer(outputBuffer, 0, staging, 0, outputBytes);
    gpuDevice.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const intoData = Array.from(new Float32Array(staging.getMappedRange().slice(0, outputBytes)));
    staging.unmap(); staging.destroy(); outputBuffer.destroy();
    for (let index = 0; index < intoData.length; index += 1) forwardIntoMaxAbsolute = Math.max(forwardIntoMaxAbsolute, Math.abs(intoData[index]! - expected.data[index]!));
  }
  for (let index = 0; index < warmup; index += 1) {
    const outputs = await module.run(inputs);
    for (const tensor of Object.values(outputs)) tensor.dispose();
  }
  const start = performance.now();
  for (let index = 0; index < runs; index += 1) {
    const outputs = await module.run(inputs);
    for (const tensor of Object.values(outputs)) tensor.dispose();
  }
  const meanMilliseconds = runs === 0 ? 0 : (performance.now() - start) / runs;
  await module.dispose();
  return {device: module.device, maxAbsolute, maxRelative, forwardIntoMaxAbsolute, tolerance: fixture.tolerance, meanMilliseconds};
};

window.runFusionPulseInspect = async () => {
  const fixture = (await (await fetch('/fusion-pulse-inspect/case.json')).json()) as Fixture;
  const module = await loadArti('/fusion-pulse-inspect/', {device: 'webgpu', wasmPaths: {wasm: wasmUrl}});
  const selected = ['fused', 'survival', 'source_index', 'pulse_mask', 'unfold_source_index', 'workspace'];
  let maxAbsolute = 0;
  let maxRelative = 0;
  const observations: Record<string, Array<number | bigint>>[] = [];
  let inferenceMs = 0;
  for (const item of fixture.cases) {
    const inputs = Object.fromEntries(Object.entries(item.inputs).map(([key, value]) => [key, tensor(value)]));
    const result = await module.inspect(inputs, {outputs: selected});
    inferenceMs += result.timings.inferenceMs;
    const downloaded = await result.download();
    const observation: Record<string, Array<number | bigint>> = {};
    for (const name of selected) {
      const contract = module.manifest.outputs.find((value) => value.name === name)!;
      const actual = Array.from(
        downloaded[name]!.data as Iterable<number | bigint>,
      );
      const expected = item.outputs[name]!;
      observation[name] = actual;
      if (contract.tolerance === undefined) throw new Error(`${name} does not declare tolerance`);
      if ((expected.dtype ?? 'float32') === 'float32') {
        for (let index = 0; index < actual.length; index += 1) {
          const expectedValue = Number(expected.data[index]!);
          const difference = Math.abs(Number(actual[index]!) - expectedValue);
          maxAbsolute = Math.max(maxAbsolute, difference);
          maxRelative = Math.max(maxRelative, difference / Math.max(Math.abs(expectedValue), 1e-8));
          const allowed = contract.tolerance.atol + contract.tolerance.rtol * Math.abs(expectedValue);
          if (difference > allowed) throw new Error(`${name} parity exceeded its declared tolerance`);
        }
      } else if (actual.map(String).join(',') !== expected.data.map((value) => String(Number(value))).join(',')) {
        throw new Error(`${name} exact parity failed`);
      }
    }
    observations.push(observation);
    await result.dispose();
    for (const value of Object.values(inputs)) value.dispose();
  }
  const roles = Object.fromEntries(module.manifest.outputs.map((value) => [value.name, value.role]));
  await module.dispose();
  return {
    device: module.device,
    maxAbsolute,
    maxRelative,
    inferenceMs,
    roles,
    survivalChanged: observations[0]!.survival!.map(String).join(',') !== observations[1]!.survival!.map(String).join(','),
    sourceMappingChanged: observations[0]!.unfold_source_index!.map(String).join(',') !== observations[1]!.unfold_source_index!.map(String).join(','),
  };
};

function tensor(value: TensorPayload): Tensor {
  if (value.dtype === 'bool') return new Tensor('bool', Uint8Array.from(value.data.map((item) => item ? 1 : 0)), value.dims);
  if (value.dtype === 'int64') return new Tensor('int64', BigInt64Array.from(value.data.map((item) => BigInt(Number(item)))), value.dims);
  return new Tensor('float32', Float32Array.from(value.data.map(Number)), value.dims);
}

function payloadBuffer(value: TensorPayload): ArrayBuffer {
  if (value.dtype === 'bool') return Uint8Array.from(value.data.map((item) => item ? 1 : 0)).buffer;
  if (value.dtype === 'int64') return BigInt64Array.from(value.data.map((item) => BigInt(Number(item)))).buffer;
  return Float32Array.from(value.data.map(Number)).buffer;
}

function workerOutputValues(value: {dtype: 'float32' | 'bool' | 'int64'; data: ArrayBuffer}): Array<number | bigint> {
  if (value.dtype === 'bool') return Array.from(new Uint8Array(value.data));
  if (value.dtype === 'int64') return Array.from(new BigInt64Array(value.data));
  return Array.from(new Float32Array(value.data));
}
