import { Tensor, loadArti } from '../src/index.js';
import { env } from 'onnxruntime-web/webgpu';
const wasmUrl = '/ort-runtime.wasm';

interface TensorPayload { dims: number[]; data: number[] }
interface FixtureCase { inputs: Record<string, TensorPayload>; expected: TensorPayload }
interface Fixture { tolerance: {atol: number; rtol: number}; cases: FixtureCase[] }

declare global {
  interface Window {
    artiWebGPUInfo: () => Promise<Record<string, unknown>>;
    runArtiParity: (name: string, warmup?: number, runs?: number) => Promise<Record<string, unknown>>;
  }
}

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

window.runArtiParity = async (name, warmup = 0, runs = 0) => {
  const fixture = (await (await fetch(`/${name}/case.json`)).json()) as Fixture;
  const module = await loadArti(`/${name}/`, {device: 'webgpu', wasmPaths: {wasm: wasmUrl}});
  let maxAbsolute = 0;
  let maxRelative = 0;
  for (const item of fixture.cases) {
    const inputs = Object.fromEntries(Object.entries(item.inputs).map(([key, value]) => [key, tensor(value)]));
    const y = await module.forward(inputs.x!, {q: inputs.q, mask: inputs.mask});
    const raw = await y.getData();
    if (!(raw instanceof Float32Array)) throw new Error('ARTI Web output is not float32');
    const actual = Array.from(raw);
    for (let index = 0; index < actual.length; index += 1) {
      const difference = Math.abs(actual[index]! - item.expected.data[index]!);
      maxAbsolute = Math.max(maxAbsolute, difference);
      maxRelative = Math.max(maxRelative, difference / Math.max(Math.abs(item.expected.data[index]!), 1e-8));
    }
    y.dispose();
  }

  const sample = fixture.cases[0]!;
  const inputs = Object.fromEntries(Object.entries(sample.inputs).map(([key, value]) => [key, tensor(value)]));
  const gpuDevice = await env.webgpu.device;
  const outputBytes = sample.expected.data.length * Float32Array.BYTES_PER_ELEMENT;
  const outputBuffer = gpuDevice.createBuffer({
    size: Math.ceil(outputBytes / 16) * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outputTensor = Tensor.fromGpuBuffer(outputBuffer, {dataType: 'float32', dims: sample.expected.dims});
  const into = await module.forwardInto(outputTensor, inputs.x!, {q: inputs.q, mask: inputs.mask});
  if (into !== outputTensor) throw new Error('forwardInto did not preserve the preallocated tensor');
  const staging = gpuDevice.createBuffer({size: Math.ceil(outputBytes / 4) * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
  const encoder = gpuDevice.createCommandEncoder();
  encoder.copyBufferToBuffer(outputBuffer, 0, staging, 0, outputBytes);
  gpuDevice.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const intoData = Array.from(new Float32Array(staging.getMappedRange().slice(0, outputBytes)));
  staging.unmap();
  staging.destroy();
  outputBuffer.destroy();
  let forwardIntoMaxAbsolute = 0;
  for (let index = 0; index < intoData.length; index += 1) {
    forwardIntoMaxAbsolute = Math.max(forwardIntoMaxAbsolute, Math.abs(intoData[index]! - sample.expected.data[index]!));
  }
  for (let index = 0; index < warmup; index += 1) {
    const y = await module.forward(inputs.x!, {q: inputs.q, mask: inputs.mask});
    y.dispose();
  }
  const start = performance.now();
  for (let index = 0; index < runs; index += 1) {
    const y = await module.forward(inputs.x!, {q: inputs.q, mask: inputs.mask});
    y.dispose();
  }
  const meanMilliseconds = runs === 0 ? 0 : (performance.now() - start) / runs;
  await module.dispose();
  return {device: module.device, maxAbsolute, maxRelative, forwardIntoMaxAbsolute, tolerance: fixture.tolerance, meanMilliseconds};
};

function tensor(value: TensorPayload): Tensor {
  return new Tensor('float32', Float32Array.from(value.data), value.dims);
}
