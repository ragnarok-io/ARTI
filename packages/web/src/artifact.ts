import type { ARTIWebLock, ARTIWebManifest } from './types.js';

const SHA256 = /^[0-9a-f]{64}$/;

export function parseManifest(value: unknown): ARTIWebManifest {
  if (!isRecord(value) || value.format !== 'arti.web' || value.format_version !== 1) throw new Error('invalid ARTI Web manifest format');
  if (!isRecord(value.module) || !['Half', 'Fold', 'LearnedPulse'].includes(String(value.module.type))) throw new Error('unsupported ARTI Web module type');
  if (!isRecord(value.runtime) || value.runtime.dtype !== 'float32' || !Array.isArray(value.runtime.execution_providers)) throw new Error('invalid ARTI Web runtime contract');
  if (!Array.isArray(value.inputs) || value.inputs.length === 0 || !value.inputs.every(isTensorContract)) throw new Error('invalid ARTI Web input contract');
  if (!isTensorContract(value.output) || !isRecord(value.files)) throw new Error('invalid ARTI Web output or file contract');
  if (!isFileRecord(value.files['model.onnx'])) throw new Error('ARTI Web manifest is missing model.onnx');
  return value as unknown as ARTIWebManifest;
}

export function parseLock(value: unknown): ARTIWebLock {
  if (!isRecord(value) || value.format !== 'arti.web' || value.format_version !== 1 || !isRecord(value.manifest)) throw new Error('invalid ARTI Web lock format');
  if (value.manifest.file !== 'arti-web.json' || !isSha(value.manifest.sha256) || !isRecord(value.files)) throw new Error('invalid ARTI Web lock manifest record');
  if (!isFileRecord(value.files['model.onnx'])) throw new Error('ARTI Web lock is missing model.onnx');
  return value as unknown as ARTIWebLock;
}

export async function sha256(value: ArrayBuffer): Promise<string> {
  const digest = await globalThis.crypto.subtle.digest('SHA-256', value);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

export async function verifyFile(name: string, value: ArrayBuffer, expected: {sha256: string; size: number}): Promise<void> {
  if (value.byteLength !== expected.size) throw new Error(`${name} size does not match its ARTI Web lock`);
  if ((await sha256(value)) !== expected.sha256) throw new Error(`${name} SHA-256 does not match its ARTI Web lock`);
}

function isRecord(value: unknown): value is Record<string, unknown> { return typeof value === 'object' && value !== null && !Array.isArray(value); }
function isSha(value: unknown): value is string { return typeof value === 'string' && SHA256.test(value); }
function isFileRecord(value: unknown): value is {sha256: string; size: number} { return isRecord(value) && isSha(value.sha256) && Number.isSafeInteger(value.size) && Number(value.size) >= 0; }
function isTensorContract(value: unknown): boolean {
  return isRecord(value) && typeof value.name === 'string' && value.dtype === 'float32' && Array.isArray(value.shape) && value.shape.every((dim) => (Number.isSafeInteger(dim) && Number(dim) >= 0) || typeof dim === 'string');
}
