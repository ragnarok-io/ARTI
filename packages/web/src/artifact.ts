import type { ARTIWebLock, ARTIWebManifest } from './generated/contract.js';
export { parseLock, parseManifest } from './generated/contract.js';

export async function sha256(value: ArrayBuffer): Promise<string> {
  const digest = await globalThis.crypto.subtle.digest('SHA-256', value);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

export async function verifyFile(name: string, value: ArrayBuffer, expected: {sha256: string; size: number}): Promise<void> {
  if (value.byteLength !== expected.size) throw new Error(`${name} size does not match its ARTI Web lock`);
  if ((await sha256(value)) !== expected.sha256) throw new Error(`${name} SHA-256 does not match its ARTI Web lock`);
}

export type { ARTIWebLock, ARTIWebManifest };
