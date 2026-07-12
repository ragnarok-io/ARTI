import {describe, expect, it} from 'vitest';

import {parseStatefulLock, parseStatefulManifest} from '../src/generated/contract.js';

const sha = '0'.repeat(64);

function manifest(): Record<string, unknown> {
  const state = [
    {name: 'keys', dtype: 'float32', shape: ['batch', 3, 4], initializer: 'zeros'},
    {name: 'values', dtype: 'float32', shape: ['batch', 3, 4], initializer: 'zeros'},
    {name: 'strengths', dtype: 'float32', shape: ['batch', 3], initializer: 'zeros'},
  ];
  const inputs = [{name: 'x', dtype: 'float32', shape: ['batch', 'tokens', 4]}, ...state];
  return {
    format: 'arti.web', format_version: 3, artifact_kind: 'stateful', package_version: 'test', persistence: 'explicit',
    producer: {backend: 'torch', graph_format: 'onnx'}, module: {type: 'StatefulRecall', config: {}},
    runtime: {dtype: 'float32', opset_version: 18, execution_providers: ['wasm']}, state,
    entrypoints: {read: {file: 'read.onnx', inputs, outputs: [{name: 'y', dtype: 'float32', shape: ['batch', 'tokens', 4]}]}},
    files: {'read.onnx': {sha256: sha, size: 32}}, limits: {max_state_bytes_per_batch: 108},
  };
}

describe('stateful artifact security boundaries', () => {
  it('accepts a bounded artifact with an exact shape-derived state budget', () => {
    expect(parseStatefulManifest(manifest()).limits.max_state_bytes_per_batch).toBe(108);
  });

  it('rejects a state budget smaller than the declared state shapes', () => {
    const value = manifest(); (value.limits as Record<string, unknown>).max_state_bytes_per_batch = 1;
    expect(() => parseStatefulManifest(value)).toThrow(/does not match declared state shapes/);
  });

  it.each(['../read.onnx', '/read.onnx', 'https:read.onnx', 'read.onnx?x=1', 'dir/read.onnx'])(
    'rejects artifact file name %s',
    (name) => {
      const value = manifest();
      value.files = {[name]: {sha256: sha, size: 32}};
      (value.entrypoints as Record<string, Record<string, unknown>>).read!.file = name;
      expect(() => parseStatefulManifest(value)).toThrow(/file name/);
    },
  );

  it('rejects excessive file and entrypoint fan-out', () => {
    const files = Object.fromEntries(Array.from({length: 17}, (_, index) => [`m${index}.onnx`, {sha256: sha, size: 1}]));
    const value = manifest(); value.files = files;
    expect(() => parseStatefulManifest(value)).toThrow(/file count/);
  });

  it('applies the same path rules to lock records', () => {
    const lock = {format: 'arti.web', format_version: 3, manifest: {file: 'arti-web.json', sha256: sha}, files: {'../read.onnx': {sha256: sha, size: 1}}};
    expect(() => parseStatefulLock(lock)).toThrow(/lock file name/);
  });
});
