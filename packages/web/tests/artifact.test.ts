import { createHash, webcrypto } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import { parseLock, parseManifest, sha256, verifyFile } from '../src/artifact.js';

Object.defineProperty(globalThis, 'crypto', {value: webcrypto});
const hash = '0'.repeat(64);
const manifest = {
  format: 'arti.web', format_version: 1, package_version: '0.2.0', module: {type: 'Half', config: {}},
  runtime: {dtype: 'float32', opset_version: 18, execution_providers: ['webgpu', 'wasm']},
  inputs: [{name: 'x', dtype: 'float32', shape: ['batch', 'tokens', 4]}],
  output: {name: 'y', dtype: 'float32', shape: ['batch', 'tokens', 4]},
  files: {'model.onnx': {sha256: hash, size: 3}},
};

describe('ARTI Web artifact validation', () => {
  it('accepts the versioned manifest and lock schema', () => {
    expect(parseManifest(manifest).module.type).toBe('Half');
    expect(parseLock({format: 'arti.web', format_version: 1, manifest: {file: 'arti-web.json', sha256: hash}, files: {'model.onnx': {sha256: hash, size: 3}}}).format_version).toBe(1);
  });
  it('rejects malformed and unsupported manifests', () => {
    expect(() => parseManifest({...manifest, format_version: 2})).toThrow(/format/);
    expect(() => parseManifest({...manifest, module: {type: 'RecallRefiner'}})).toThrow(/module/);
  });
  it('verifies file size and SHA-256', async () => {
    const bytes = new Uint8Array([1, 2, 3]);
    const expected = createHash('sha256').update(bytes).digest('hex');
    expect(await sha256(bytes.buffer)).toBe(expected);
    await expect(verifyFile('model.onnx', bytes.buffer, {sha256: expected, size: 3})).resolves.toBeUndefined();
    await expect(verifyFile('model.onnx', bytes.buffer, {sha256: hash, size: 3})).rejects.toThrow(/SHA-256/);
  });
});
