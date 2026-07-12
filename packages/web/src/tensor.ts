import {Tensor} from 'onnxruntime-web/webgpu';

import {ArtiWebError} from './errors.js';
import type {TensorContract} from './generated/contract.js';
import type {CPUTensor, TensorInput, TensorOutput} from './types.js';

export type TensorDimensions = readonly number[];
export type DynamicDimensions = Readonly<Record<string, number>>;
export type Float32TensorData = Float32Array | readonly number[];

export function tensor(data: Float32TensorData, dims: TensorDimensions): Tensor;
export function tensor(contract: TensorContract, data: Float32TensorData, dimensions?: DynamicDimensions): Tensor;
export function tensor(
  contractOrData: TensorContract | Float32TensorData,
  dataOrDims: Float32TensorData | TensorDimensions,
  dimensions: DynamicDimensions = {},
): Tensor {
  if (isContract(contractOrData)) {
    const data = asFloat32(dataOrDims as Float32TensorData);
    const dims = resolveShape(contractOrData, dimensions);
    requireElementCount(contractOrData.name, dims, data.length);
    return new Tensor(contractOrData.dtype, data, dims);
  }
  const data = asFloat32(contractOrData);
  const dims = validateDims(dataOrDims as TensorDimensions);
  requireElementCount(undefined, dims, data.length);
  return new Tensor('float32', data, dims);
}

export function zeros(dims: TensorDimensions): Tensor;
export function zeros(contract: TensorContract, dimensions?: DynamicDimensions): Tensor;
export function zeros(contractOrDims: TensorContract | TensorDimensions, dimensions: DynamicDimensions = {}): Tensor {
  const contract = isContract(contractOrDims) ? contractOrDims : undefined;
  const dims = contract ? resolveShape(contract, dimensions) : validateDims(contractOrDims as TensorDimensions);
  const size = elementCount(dims, contract?.name);
  return new Tensor('float32', new Float32Array(size), dims);
}

export function isCPUTensor(value: unknown): value is CPUTensor {
  if (value instanceof Tensor) return false;
  if (typeof value !== 'object' || value === null) return false;
  const item = value as Partial<CPUTensor>;
  return item.data instanceof Float32Array && Array.isArray(item.dims) && item.dims.every(isDimension);
}

export function fromCPU(value: CPUTensor): Tensor {
  if (!isCPUTensor(value)) throw contractError('invalid CPU tensor value', undefined, 'a Float32Array and valid dimensions', value);
  return tensor(Float32Array.from(value.data), value.dims);
}

export async function toCPU(value: TensorInput): Promise<TensorOutput> {
  if (value instanceof Tensor) {
    if (value.type !== 'float32') throw contractError('only float32 tensors are supported', undefined, 'float32', value.type);
    const data = await value.getData();
    return {data: Float32Array.from(data as Float32Array), dims: [...value.dims]};
  }
  if (isCPUTensor(value)) return {data: Float32Array.from(value.data), dims: [...value.dims]};
  throw contractError('invalid tensor input', undefined, 'an ORT Tensor or CPU tensor', value);
}

function resolveShape(contract: TensorContract, dimensions: DynamicDimensions): number[] {
  const known = new Set(contract.shape.filter((dim): dim is string => typeof dim === 'string'));
  for (const name of Object.keys(dimensions)) if (!known.has(name)) throw contractError(`unknown dynamic dimension ${name}`, contract.name, [...known], name);
  return contract.shape.map((dim) => {
    if (typeof dim === 'number') return dim;
    const value = dimensions[dim];
    if (value === undefined) throw contractError(`missing dynamic dimension ${dim}`, contract.name, dim, undefined);
    if (!isDimension(value)) throw contractError(`invalid dynamic dimension ${dim}`, contract.name, 'a non-negative safe integer', value);
    return value;
  });
}

function validateDims(dims: TensorDimensions): number[] {
  if (!Array.isArray(dims) || !dims.every(isDimension)) throw contractError('invalid tensor dimensions', undefined, 'non-negative safe integers', dims);
  return [...dims];
}

function elementCount(dims: readonly number[], name?: string): number {
  let size = 1;
  for (const dim of dims) {
    size *= dim;
    if (!Number.isSafeInteger(size)) throw contractError('tensor element count exceeds safe integer bounds', name, 'a safe integer', size);
  }
  return size;
}

function requireElementCount(name: string | undefined, dims: readonly number[], actual: number): void {
  const expected = elementCount(dims, name);
  if (expected !== actual) throw contractError('tensor data length does not match its shape', name, expected, actual);
}

function asFloat32(data: Float32TensorData): Float32Array {
  if (data instanceof Float32Array) return data;
  if (!Array.isArray(data) || !data.every((value) => typeof value === 'number')) throw contractError('invalid tensor data', undefined, 'Float32Array or numbers', data);
  return Float32Array.from(data);
}

function isContract(value: unknown): value is TensorContract {
  return typeof value === 'object' && value !== null && 'name' in value && 'dtype' in value && 'shape' in value;
}
function isDimension(value: unknown): value is number { return Number.isSafeInteger(value) && Number(value) >= 0; }
function contractError(message: string, tensorName: string | undefined, expected: unknown, actual: unknown): ArtiWebError {
  return new ArtiWebError(message, {code: 'CONTRACT_MISMATCH', stage: 'input', tensorName, expected, actual});
}
