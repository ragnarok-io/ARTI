"""Canonical Python-owned contract for ARTI Web artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path


ARTI_WEB_FORMAT = "arti.web"
ARTI_WEB_FORMAT_VERSION = 2
ARTI_WEB_MANIFEST = "arti-web.json"
ARTI_WEB_MODEL = "model.onnx"
ARTI_WEB_LOCK = "arti-web.lock.json"
ARTI_WEB_TYPESCRIPT = "artifact.ts"
MAX_STATEFUL_FILES = 16
MAX_STATEFUL_ENTRYPOINTS = 16
MAX_STATEFUL_ARTIFACT_BYTES = 512 * 1024 * 1024


def artifact_schema() -> dict[str, object]:
    """Return the canonical JSON Schema for v2 manifests and locks."""

    file_record = {
        "type": "object",
        "required": ["sha256", "size"],
        "properties": {
            "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "size": {"type": "integer", "minimum": 0},
        },
        "additionalProperties": False,
    }
    tensor = {
        "type": "object",
        "required": ["name", "dtype", "shape"],
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "dtype": {"const": "float32"},
            "shape": {
                "type": "array",
                "minItems": 1,
                "items": {"oneOf": [{"type": "integer", "minimum": 0}, {"type": "string", "minLength": 1}]},
            },
        },
        "additionalProperties": False,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://arti.fit/schema/web-artifact-v2.json",
        "title": "ARTI Web Artifact v2",
        "$defs": {"file": file_record, "tensor": tensor},
        "type": "object",
        "required": ["manifest", "lock"],
        "properties": {
            "manifest": {
                "type": "object",
                "required": ["format", "format_version", "package_version", "producer", "module", "runtime", "inputs", "outputs", "files"],
                "properties": {
                    "format": {"const": ARTI_WEB_FORMAT},
                    "format_version": {"const": ARTI_WEB_FORMAT_VERSION},
                    "package_version": {"type": "string"},
                    "producer": {
                        "type": "object",
                        "required": ["backend", "graph_format"],
                        "properties": {"backend": {"type": "string"}, "graph_format": {"const": "onnx"}},
                        "additionalProperties": False,
                    },
                    "module": {
                        "type": "object",
                        "required": ["type", "config"],
                        "properties": {"type": {"type": "string", "minLength": 1}, "config": {"type": "object"}},
                        "additionalProperties": False,
                    },
                    "runtime": {
                        "type": "object",
                        "required": ["dtype", "opset_version", "execution_providers"],
                        "properties": {
                            "dtype": {"const": "float32"},
                            "opset_version": {"type": "integer", "minimum": 18},
                            "execution_providers": {"type": "array", "items": {"enum": ["webgpu", "wasm"]}, "minItems": 1},
                        },
                        "additionalProperties": False,
                    },
                    "inputs": {"type": "array", "items": {"$ref": "#/$defs/tensor"}, "minItems": 1},
                    "outputs": {"type": "array", "items": {"$ref": "#/$defs/tensor"}, "minItems": 1},
                    "files": {"type": "object", "required": [ARTI_WEB_MODEL], "additionalProperties": {"$ref": "#/$defs/file"}},
                },
                "additionalProperties": False,
            },
            "lock": {
                "type": "object",
                "required": ["format", "format_version", "manifest", "files"],
                "properties": {
                    "format": {"const": ARTI_WEB_FORMAT},
                    "format_version": {"const": ARTI_WEB_FORMAT_VERSION},
                    "manifest": {
                        "type": "object",
                        "required": ["file", "sha256"],
                        "properties": {"file": {"const": ARTI_WEB_MANIFEST}, "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}},
                        "additionalProperties": False,
                    },
                    "files": {"type": "object", "required": [ARTI_WEB_MODEL], "additionalProperties": {"$ref": "#/$defs/file"}},
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    }


def stateful_artifact_schema() -> dict[str, object]:
    """Return the canonical JSON Schema for explicit-state v3 artifacts."""

    base = artifact_schema()
    file_record = deepcopy(base["$defs"]["file"])
    tensor = deepcopy(base["$defs"]["tensor"])
    state_tensor = deepcopy(tensor)
    state_tensor["required"] = [*state_tensor["required"], "initializer"]
    state_tensor["properties"]["initializer"] = {"const": "zeros"}
    entrypoint = {
        "type": "object",
        "required": ["file", "inputs", "outputs"],
        "properties": {
            "file": {"type": "string", "minLength": 1},
            "inputs": {"type": "array", "items": {"$ref": "#/$defs/tensor"}, "minItems": 1},
            "outputs": {"type": "array", "items": {"$ref": "#/$defs/tensor"}, "minItems": 1},
            "state_outputs": {"type": "object", "additionalProperties": {"type": "string", "minLength": 1}},
        },
        "additionalProperties": False,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://arti.fit/schema/web-stateful-artifact-v3.json",
        "title": "ARTI Stateful Web Artifact v3",
        "$defs": {"file": file_record, "tensor": tensor, "state_tensor": state_tensor, "entrypoint": entrypoint},
        "type": "object",
        "required": ["manifest", "lock"],
        "properties": {
            "manifest": {
                "type": "object",
                "required": ["format", "format_version", "artifact_kind", "package_version", "producer", "module", "runtime", "state", "entrypoints", "files", "limits", "persistence"],
                "properties": {
                    "format": {"const": ARTI_WEB_FORMAT}, "format_version": {"const": 3}, "artifact_kind": {"const": "stateful"},
                    "package_version": {"type": "string"}, "producer": {"type": "object"}, "module": {"type": "object"}, "runtime": {"type": "object"},
                    "state": {"type": "array", "items": {"$ref": "#/$defs/state_tensor"}, "minItems": 1},
                    "entrypoints": {"type": "object", "additionalProperties": {"$ref": "#/$defs/entrypoint"}, "minProperties": 1, "maxProperties": MAX_STATEFUL_ENTRYPOINTS},
                    "files": {"type": "object", "additionalProperties": {"$ref": "#/$defs/file"}, "minProperties": 1, "maxProperties": MAX_STATEFUL_FILES},
                    "limits": {"type": "object", "required": ["max_state_bytes_per_batch"], "properties": {"max_state_bytes_per_batch": {"type": "integer", "minimum": 1}}, "additionalProperties": False},
                    "persistence": {"const": "explicit"},
                },
                "additionalProperties": False,
            },
            "lock": {"type": "object"},
        },
        "additionalProperties": False,
    }


def render_typescript_contract() -> str:
    """Render the TypeScript contract and validator from the Python schema."""

    return '''// Generated by arti.web.contract. Do not edit by hand.
export type ActiveARTIDevice = 'webgpu' | 'wasm';
export interface TensorContract { name: string; dtype: 'float32'; shape: Array<number | string>; }
export interface ARTIWebManifest {
  format: 'arti.web'; format_version: 2; package_version: string;
  producer: {backend: string; graph_format: 'onnx'};
  module: {type: string; config: Record<string, unknown>};
  runtime: {dtype: 'float32'; opset_version: number; execution_providers: ActiveARTIDevice[]};
  inputs: TensorContract[]; outputs: TensorContract[];
  files: Record<string, {sha256: string; size: number}>;
}
export interface ARTIWebLock {
  format: 'arti.web'; format_version: 2;
  manifest: {file: 'arti-web.json'; sha256: string};
  files: Record<string, {sha256: string; size: number}>;
}
const SHA256 = /^[0-9a-f]{64}$/;
const MAX_STATEFUL_FILES = 16;
const MAX_STATEFUL_ENTRYPOINTS = 16;
const MAX_STATEFUL_ARTIFACT_BYTES = 512 * 1024 * 1024;
export function parseManifest(value: unknown): ARTIWebManifest {
  const item = record(value, 'manifest');
  if (item.format !== 'arti.web') throw new Error('invalid ARTI Web manifest format');
  if (item.format_version !== 2) throw new Error('unsupported ARTI Web artifact version; re-export it with the current Python package');
  if (typeof item.package_version !== 'string') throw new Error('invalid ARTI Web package version');
  const producer = record(item.producer, 'producer');
  if (typeof producer.backend !== 'string' || producer.backend.length === 0 || producer.graph_format !== 'onnx') throw new Error('invalid ARTI Web producer');
  const module = record(item.module, 'module');
  if (typeof module.type !== 'string' || module.type.length === 0) throw new Error('invalid ARTI Web module type');
  record(module.config, 'module config');
  const runtime = record(item.runtime, 'runtime');
  if (runtime.dtype !== 'float32' || !Number.isInteger(runtime.opset_version) || Number(runtime.opset_version) < 18) throw new Error('invalid ARTI Web runtime contract');
  if (!Array.isArray(runtime.execution_providers) || !runtime.execution_providers.every((entry) => entry === 'webgpu' || entry === 'wasm')) throw new Error('invalid ARTI Web execution providers');
  tensorList(item.inputs, 'inputs'); tensorList(item.outputs, 'outputs');
  const files = record(item.files, 'files'); fileRecord(files['model.onnx'], 'model.onnx');
  return item as unknown as ARTIWebManifest;
}
export function parseLock(value: unknown): ARTIWebLock {
  const item = record(value, 'lock');
  if (item.format !== 'arti.web') throw new Error('invalid ARTI Web lock format');
  if (item.format_version !== 2) throw new Error('unsupported ARTI Web artifact version; re-export it with the current Python package');
  const manifest = record(item.manifest, 'manifest');
  if (manifest.file !== 'arti-web.json' || !isSha(manifest.sha256)) throw new Error('invalid ARTI Web lock manifest record');
  const files = record(item.files, 'files'); fileRecord(files['model.onnx'], 'model.onnx');
  return item as unknown as ARTIWebLock;
}
function tensorList(value: unknown, name: string): void {
  if (!Array.isArray(value) || value.length === 0) throw new Error(`invalid ARTI Web ${name} contract`);
  const names = new Set<string>();
  for (const entry of value) {
    const item = record(entry, name);
    if (typeof item.name !== 'string' || item.name.length === 0 || names.has(item.name) || item.dtype !== 'float32' || !Array.isArray(item.shape) || item.shape.length === 0) throw new Error(`invalid ARTI Web ${name} contract`);
    if (!item.shape.every((dim) => (Number.isSafeInteger(dim) && Number(dim) >= 0) || (typeof dim === 'string' && dim.length > 0))) throw new Error(`invalid ARTI Web ${name} shape`);
    names.add(item.name);
  }
}
function record(value: unknown, name: string): Record<string, unknown> { if (typeof value !== 'object' || value === null || Array.isArray(value)) throw new Error(`invalid ARTI Web ${name}`); return value as Record<string, unknown>; }
function isSha(value: unknown): value is string { return typeof value === 'string' && SHA256.test(value); }
function fileRecord(value: unknown, name: string): void { const item = record(value, name); if (!isSha(item.sha256) || !Number.isSafeInteger(item.size) || Number(item.size) < 0) throw new Error(`invalid ARTI Web ${name} file record`); }
function safeArtifactFileName(value: string): boolean { return value.length > 0 && value.length <= 255 && !value.includes('\\\\') && !value.includes('/') && value !== '.' && value !== '..' && !value.includes('?') && !value.includes('#') && !value.includes(':'); }
export interface StatefulTensorContract extends TensorContract { initializer?: 'zeros'; }
export interface StatefulEntrypoint { file: string; inputs: TensorContract[]; outputs: TensorContract[]; state_outputs?: Record<string, string>; }
export interface ARTIStatefulWebManifest {
  format: 'arti.web'; format_version: 3; artifact_kind: 'stateful'; package_version: string;
  producer: {backend: string; graph_format: 'onnx'};
  module: {type: string; config: Record<string, unknown>};
  runtime: {dtype: 'float32'; opset_version: number; execution_providers: ActiveARTIDevice[]};
  state: StatefulTensorContract[];
  entrypoints: Record<string, StatefulEntrypoint>;
  files: Record<string, {sha256: string; size: number}>;
  limits: {max_state_bytes_per_batch: number}; persistence: 'explicit';
}
export interface ARTIStatefulWebLock {
  format: 'arti.web'; format_version: 3;
  manifest: {file: 'arti-web.json'; sha256: string};
  files: Record<string, {sha256: string; size: number}>;
}
export function parseStatefulManifest(value: unknown): ARTIStatefulWebManifest {
  const item = record(value, 'stateful manifest');
  if (item.format !== 'arti.web' || item.format_version !== 3 || item.artifact_kind !== 'stateful') throw new Error('invalid ARTI stateful Web manifest');
  if (typeof item.package_version !== 'string' || item.persistence !== 'explicit') throw new Error('invalid ARTI stateful Web metadata');
  const producer = record(item.producer, 'producer'); if (typeof producer.backend !== 'string' || producer.graph_format !== 'onnx') throw new Error('invalid ARTI stateful producer');
  const module = record(item.module, 'module'); if (typeof module.type !== 'string' || module.type.length === 0) throw new Error('invalid ARTI stateful module'); record(module.config, 'module config');
  const runtime = record(item.runtime, 'runtime');
  if (runtime.dtype !== 'float32' || !Number.isSafeInteger(runtime.opset_version) || Number(runtime.opset_version) < 18 || !Array.isArray(runtime.execution_providers) || !runtime.execution_providers.every((entry) => entry === 'webgpu' || entry === 'wasm')) throw new Error('invalid ARTI stateful runtime');
  tensorList(item.state, 'state');
  for (const state of item.state as Array<Record<string, unknown>>) if (state.initializer !== 'zeros') throw new Error('unsupported ARTI state initializer');
  const entrypoints = record(item.entrypoints, 'entrypoints');
  if (Object.keys(entrypoints).length === 0 || Object.keys(entrypoints).length > MAX_STATEFUL_ENTRYPOINTS) throw new Error('invalid ARTI stateful entrypoint count');
  const files = record(item.files, 'files');
  const fileEntries = Object.entries(files);
  if (fileEntries.length === 0 || fileEntries.length > MAX_STATEFUL_FILES) throw new Error('invalid ARTI stateful file count');
  let artifactBytes = 0;
  for (const [name, file] of fileEntries) {
    if (!safeArtifactFileName(name)) throw new Error(`invalid ARTI stateful file name ${name}`);
    fileRecord(file, name);
    artifactBytes += Number((file as Record<string, unknown>).size);
    if (!Number.isSafeInteger(artifactBytes) || artifactBytes > MAX_STATEFUL_ARTIFACT_BYTES) throw new Error('ARTI stateful artifact exceeds the model byte limit');
  }
  for (const [name, raw] of Object.entries(entrypoints)) {
    const entry = record(raw, `entrypoint ${name}`);
    if (typeof entry.file !== 'string' || !(entry.file in files)) throw new Error(`invalid ARTI stateful entrypoint ${name}`);
    tensorList(entry.inputs, `${name} inputs`); tensorList(entry.outputs, `${name} outputs`); fileRecord(files[entry.file], entry.file);
    if (entry.state_outputs !== undefined) {
      const bindings = record(entry.state_outputs, `${name} state outputs`);
      const outputNames = new Set((entry.outputs as Array<Record<string, unknown>>).map((output) => output.name));
      for (const [stateName, outputName] of Object.entries(bindings)) if (typeof outputName !== 'string' || !outputNames.has(outputName)) throw new Error(`invalid state output binding ${stateName}`);
    }
  }
  const limits = record(item.limits, 'limits');
  if (!Number.isSafeInteger(limits.max_state_bytes_per_batch) || Number(limits.max_state_bytes_per_batch) <= 0) throw new Error('invalid ARTI state budget');
  let stateBytes = 0;
  for (const state of item.state as Array<Record<string, unknown>>) {
    let elements = 1;
    for (const dim of state.shape as Array<number | string>) {
      if (dim === 'batch') continue;
      if (!Number.isSafeInteger(dim) || Number(dim) <= 0) throw new Error(`invalid static state dimension for ${state.name}`);
      elements *= Number(dim);
      if (!Number.isSafeInteger(elements)) throw new Error('ARTI state shape exceeds safe integer bounds');
    }
    stateBytes += elements * Float32Array.BYTES_PER_ELEMENT;
    if (!Number.isSafeInteger(stateBytes)) throw new Error('ARTI state size exceeds safe integer bounds');
  }
  if (stateBytes !== Number(limits.max_state_bytes_per_batch)) throw new Error('ARTI state budget does not match declared state shapes');
  return item as unknown as ARTIStatefulWebManifest;
}
export function parseStatefulLock(value: unknown): ARTIStatefulWebLock {
  const item = record(value, 'stateful lock');
  if (item.format !== 'arti.web' || item.format_version !== 3) throw new Error('invalid ARTI stateful Web lock');
  const manifest = record(item.manifest, 'manifest');
  if (manifest.file !== 'arti-web.json' || !isSha(manifest.sha256)) throw new Error('invalid ARTI stateful manifest lock');
  const files = record(item.files, 'files');
  const fileEntries = Object.entries(files);
  if (fileEntries.length === 0 || fileEntries.length > MAX_STATEFUL_FILES) throw new Error('invalid ARTI stateful lock file count');
  let artifactBytes = 0;
  for (const [name, file] of fileEntries) {
    if (!safeArtifactFileName(name)) throw new Error(`invalid ARTI stateful lock file name ${name}`);
    fileRecord(file, name);
    artifactBytes += Number((file as Record<string, unknown>).size);
    if (!Number.isSafeInteger(artifactBytes) || artifactBytes > MAX_STATEFUL_ARTIFACT_BYTES) throw new Error('ARTI stateful lock exceeds the model byte limit');
  }
  return item as unknown as ARTIStatefulWebLock;
}
'''


def write_typescript_contract(path: str | Path) -> Path:
    """Write the generated TypeScript contract to ``path``."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_typescript_contract(), encoding="utf-8")
    return target


def render_artifact_typescript(manifest: Mapping[str, object]) -> str:
    """Render a typed client whose names come from one exported v2 manifest."""

    inputs = _manifest_tensor_names(manifest, "inputs")
    outputs = _manifest_tensor_names(manifest, "outputs")
    if manifest.get("format") != ARTI_WEB_FORMAT or manifest.get("format_version") != ARTI_WEB_FORMAT_VERSION:
        raise ValueError("artifact TypeScript generation requires an ARTI Web v2 manifest")

    input_type = _typescript_tensor_map(inputs)
    output_type = _typescript_tensor_map(outputs)
    descriptor = json.dumps(
        {"format": ARTI_WEB_FORMAT, "format_version": ARTI_WEB_FORMAT_VERSION, "inputs": inputs, "outputs": outputs},
        separators=(",", ":"),
    )
    if len(inputs) == 1 and len(outputs) == 1:
        method = """  async forward(value: Tensor): Promise<Tensor> {
    return this.module.forward(value);
  }"""
    else:
        method = """  async run(inputs: ArtifactInputs): Promise<ArtifactOutputs> {
    return await this.module.run(inputs) as ArtifactOutputs;
  }"""
    return f'''// Generated from arti-web.json by arti.web.contract. Do not edit by hand.
import type {{ ARTIWebModule, Tensor }} from '@arti-fit/web';

export const descriptor = {descriptor} as const;
export type ArtifactInputs = {input_type};
export type ArtifactOutputs = {output_type};

export class ArtifactClient {{
  constructor(readonly module: ARTIWebModule) {{}}

{method}
}}
'''


def write_artifact_typescript(manifest: Mapping[str, object], path: str | Path) -> Path:
    """Write artifact-specific TypeScript generated from ``manifest``."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_artifact_typescript(manifest), encoding="utf-8")
    return target


def _manifest_tensor_names(manifest: Mapping[str, object], field: str) -> list[str]:
    values = manifest.get(field)
    if not isinstance(values, list) or not values:
        raise ValueError(f"manifest {field} must contain at least one tensor")
    names: list[str] = []
    for value in values:
        if not isinstance(value, Mapping) or not isinstance(value.get("name"), str) or not value["name"]:
            raise ValueError(f"manifest {field} contains an invalid tensor name")
        names.append(value["name"])
    if len(names) != len(set(names)):
        raise ValueError(f"manifest {field} tensor names must be unique")
    return names


def _typescript_tensor_map(names: list[str]) -> str:
    properties = "; ".join(f"{json.dumps(name)}: Tensor" for name in names)
    return "{ " + properties + " }"
