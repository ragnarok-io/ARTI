"""Build ARTI distributions and smoke-test wheel installation."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from pathlib import PurePosixPath

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
RELEASE_MANIFEST = ROOT / "release-artifact-manifest.json"
LOCAL_USER_PATH = re.compile(
    rb"(?:[A-Za-z]:[\\/]+(?:Users|Documents and Settings)[\\/]+[^\\/\r\n\t \"']+|/(?:home|Users|root)/[^/\r\n\t \"']+)",
    re.IGNORECASE,
)
FORBIDDEN_ARCHIVE_PARTS = {
    ".artifacts",
    ".cache",
    ".git",
    ".ruff_cache",
    ".tmp",
    ".venv",
    "artifacts",
    "benchmarks",
    "checkpoints",
    "__pycache__",
}
SECRET_PATTERNS = {
    "GitHub token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    "PyPI token": re.compile(rb"pypi-[A-Za-z0-9_-]{50,}"),
    "Hugging Face token": re.compile(rb"hf_[A-Za-z0-9]{30,}"),
    "GitLab token": re.compile(rb"glpat-[A-Za-z0-9_-]{20,}"),
    "Slack token": re.compile(rb"xox[baprs]-[A-Za-z0-9-]{20,}"),
    "Google API key": re.compile(rb"AIza[0-9A-Za-z_-]{35}"),
    "AWS access key": re.compile(rb"(?:AKIA|ASIA)[0-9A-Z]{16}"),
    "API token": re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{32,}"),
    "Bearer token": re.compile(rb"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{24,}"),
    "private key": re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    "JWT": re.compile(rb"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    "credential assignment": re.compile(
        rb"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password|passwd|secret)"
        rb"\b\s*[:=]\s*[\"']?[A-Za-z0-9/+_.=-]{24,}"
    ),
}
RECORD_HASH_LENGTHS = {"sha256": 43, "sha384": 64, "sha512": 86}


def content_findings(label: str, content: bytes) -> list[str]:
    findings = []
    if LOCAL_USER_PATH.search(content):
        findings.append(f"{label}: contains an absolute user-directory path")
    findings.extend(
        f"{label}: contains a possible {name}"
        for name, pattern in SECRET_PATTERNS.items()
        if pattern.search(content)
    )
    return findings


def wheel_record_findings(label: str, content: bytes, members: dict[str, bytes]) -> list[str]:
    """Scan RECORD paths and verify digest fields against their wheel members."""
    findings = []
    if LOCAL_USER_PATH.search(content):
        findings.append(f"{label}: contains an absolute user-directory path")

    try:
        rows = csv.reader(io.StringIO(content.decode("utf-8"), newline=""))
        recorded_paths = set()
        for row in rows:
            if len(row) != 3:
                return [f"{label}: malformed wheel RECORD"]
            path, digest, size = row
            if path in recorded_paths:
                return [f"{label}: malformed wheel RECORD"]
            recorded_paths.add(path)
            findings.extend(content_findings(f"{label} path", path.encode("utf-8")))
            if path == label:
                if digest or size:
                    return [f"{label}: malformed wheel RECORD"]
                continue

            match = re.fullmatch(r"(sha256|sha384|sha512)=([A-Za-z0-9_-]+)", digest)
            member = members.get(path)
            if (
                match is None
                or len(match.group(2)) != RECORD_HASH_LENGTHS[match.group(1)]
                or member is None
                or not size.isdecimal()
                or int(size) != len(member)
            ):
                return [f"{label}: malformed wheel RECORD"]
            expected_digest = (
                base64.urlsafe_b64encode(hashlib.new(match.group(1), member).digest())
                .decode("ascii")
                .rstrip("=")
            )
            if match.group(2) != expected_digest:
                return [f"{label}: malformed wheel RECORD"]
    except (csv.Error, UnicodeDecodeError):
        return [f"{label}: malformed wheel RECORD"]
    if recorded_paths != members.keys():
        return [f"{label}: malformed wheel RECORD"]
    return findings


def archive_inventory(names: list[str]) -> tuple[int, str]:
    normalized = sorted(name.replace("\\", "/") for name in names)
    inventory = "".join(f"{name}\n" for name in normalized).encode("utf-8")
    return len(normalized), hashlib.sha256(inventory).hexdigest()


def archive_member_findings(label: str, names: list[str], expected: dict) -> list[str]:
    findings = []
    normalized = sorted(name.replace("\\", "/") for name in names)
    for name in normalized:
        path = PurePosixPath(name)
        if (
            LOCAL_USER_PATH.search(name.encode("utf-8"))
            or path.is_absolute()
            or ".." in path.parts
            or any(part.lower() in FORBIDDEN_ARCHIVE_PARTS for part in path.parts)
        ):
            findings.append(f"{label}: contains a disallowed archive path")
            break

    file_count, digest = archive_inventory(normalized)
    if file_count != expected["file_count"] or digest != expected["paths_sha256"]:
        findings.append(f"{label}: file inventory differs from the reviewed release manifest")
    return findings


def audit_archive_contents(sdist: Path, wheel: Path, manifest: dict) -> list[str]:
    findings = []
    with tarfile.open(sdist, mode="r:gz") as archive:
        members = archive.getmembers()
        files = [member for member in members if member.isfile()]
        expected_root = f"{manifest['distribution'].replace('-', '_')}-{manifest['version']}"
        names = []
        for member in members:
            normalized_name = member.name.replace("\\", "/")
            parts = PurePosixPath(normalized_name).parts
            if not parts or parts[0] != expected_root:
                findings.append("sdist: contains an unexpected archive root")
                break
            if not member.isdir() and not member.isfile():
                findings.append("sdist: contains an unexpected non-file member")
        for member in files:
            relative_name = "/".join(PurePosixPath(member.name.replace("\\", "/")).parts[1:])
            names.append(relative_name)
            stream = archive.extractfile(member)
            if stream is not None:
                findings.extend(content_findings(relative_name, stream.read()))
        findings.extend(archive_member_findings("sdist", names, manifest["sdist"]))

    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        members = {name: archive.read(name) for name in names}
        for name in names:
            content = members[name]
            if name.endswith(".dist-info/RECORD"):
                findings.extend(wheel_record_findings(name, content, members))
            else:
                findings.extend(content_findings(name, content))
        findings.extend(archive_member_findings("wheel", names, manifest["wheel"]))
    return findings


def run(command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True, env=env)


def main() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    expected_version = project["version"]
    distribution_stem = project["name"].replace("-", "_")
    manifest = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("distribution") != project["name"]
        or manifest.get("version") != expected_version
    ):
        raise SystemExit("release archive manifest does not match the package identity")
    if DIST.exists():
        shutil.rmtree(DIST)

    run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--sdist",
            "--wheel",
            "--outdir",
            str(DIST),
        ]
    )

    wheels = sorted(DIST.glob(f"{distribution_stem}-*.whl"))
    sdists = sorted(DIST.glob(f"{distribution_stem}-*.tar.gz"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel, found {len(wheels)}")
    if len(sdists) != 1:
        raise SystemExit(f"expected exactly one sdist, found {len(sdists)}")

    with tarfile.open(sdists[0], mode="r:gz") as archive:
        members = [member.name.replace("\\", "/") for member in archive.getmembers()]
    relative_members = [name.split("/", 1)[1] if "/" in name else name for name in members]
    leaked = sorted(
        name
        for name in relative_members
        if name.startswith(("benchmarks/", "data/", "artifacts/", ".artifacts/"))
    )
    if leaked:
        raise SystemExit(f"sdist contains local-only paths: {leaked[:5]}")
    if "LICENSE" not in relative_members:
        raise SystemExit("sdist is missing the root LICENSE file")

    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())
        required = {
            "arti/__init__.py",
            "arti/_version.py",
            "arti/_toml.py",
            "arti/py.typed",
            "arti/torch/__init__.py",
            "arti/torch/cuda.py",
            "arti/jax/__init__.py",
            "arti/backend.py",
            "arti/experimental/__init__.py",
            "arti/experimental/web/__init__.py",
            "arti/experimental/web/contract.py",
            "arti/experimental/web/exporter.py",
            "arti/experimental/web/stateful.py",
            "arti/serialization.py",
            "arti/alpha/__init__.py",
            "arti/mechanisms/__init__.py",
            "arti/execution.py",
            "arti/legacy/__init__.py",
            "arti/legacy/layered_recall.py",
            "arti/legacy/refine.py",
            "arti/legacy/refine_contracts.py",
            "arti/legacy/stateful_recall.py",
            "arti/arti_layer.py",
            "arti/attachment_layer.py",
            "arti/formula_v2.py",
            "arti/formula_learning.py",
            "arti/branch_search.py",
            "arti/branch_formula.py",
            "arti/branch_search_runtime.py",
            "arti/gpu_resident.py",
            "arti/runtime_checkpoint.py",
            "arti/tensor_binding.py",
            "arti/tensor_transaction.py",
            "arti/providers.py",
            "arti/pretrained.py",
            "arti/pretrained_cli.py",
            "arti/fit/__init__.py",
            "arti/fit/project.py",
            "arti/fit/batch_schema.py",
            "arti/fit/runtime.py",
            "arti/fit/scanner.py",
            "arti/fit/insertion.py",
            "arti/fit/artifacts.py",
            "arti/fit/doctor.py",
            "arti/fit/plugins.py",
            "arti/fit/strategies.py",
            "arti/integrations/__init__.py",
            "arti/integrations/qwen.py",
            "arti/schemas/fit-config.schema.json",
            "arti/schemas/task-graph.schema.json",
            "arti/torch/fit.py",
        }
        missing = sorted(required - names)
        if missing:
            raise SystemExit(f"wheel is missing expected files: {missing}")

    content_leaks = audit_archive_contents(sdists[0], wheels[0], manifest)
    if content_leaks:
        raise SystemExit(f"distribution contains sensitive content markers: {content_leaks[:10]}")

    with tempfile.TemporaryDirectory(prefix="arti-wheel-smoke-") as tmp:
        target = Path(tmp) / "target"
        target.mkdir()
        with zipfile.ZipFile(wheels[0]) as wheel:
            wheel.extractall(target)
        if not (target / "arti" / "__init__.py").exists():
            raise SystemExit("extracted wheel does not contain importable arti package")
        if not (target / "arti" / "py.typed").exists():
            raise SystemExit("extracted wheel does not contain py.typed")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(target) + os.pathsep + env.get("PYTHONPATH", "")
        env["ARTI_WHEEL_ROOT"] = str(target)
        run(
            [
                sys.executable,
                "-c",
                (
                    "import arti, arti.functional, arti.torch, arti.jax, arti.experimental.web, arti.mechanisms, arti.legacy, importlib.util, json, torch; "
                    "import os, pathlib; "
                    "assert pathlib.Path(arti.__file__).resolve().parent == pathlib.Path(os.environ['ARTI_WHEEL_ROOT']) / 'arti'; "
                    "import arti.cli; "
                    "assert callable(arti.experimental.web.export); "
                    "assert importlib.util.find_spec('arti.web') is None; "
                    "assert importlib.util.find_spec('arti.layered_recall') is None; "
                    "assert importlib.util.find_spec('arti.stateful_recall') is None; "
                    f"assert arti.__version__ == {expected_version!r}; "
                    "assert 'torch' in arti.available_backends(); "
                    "assert arti.jax.backend_status() in {'available', 'broken', 'unavailable'}; "
                    "assert callable(arti.jax.init_layer); "
                    "assert callable(arti.jax.apply_layer); "
                    "assert callable(arti.jax.apply_layer_single); "
                    "assert callable(arti.jax.apply_coord_frame_inverse); "
                    "assert callable(arti.jax.smoke_report); "
                    "assert callable(arti.jax.masked_mean); "
                    "assert callable(arti.jax.masked_softmax); "
                    "assert callable(arti.jax.mask_coverage); "
                    "assert callable(arti.jax.ensure_visibility); "
                    "assert callable(arti.jax.attention_mask_to_visibility); "
                    "assert callable(arti.cuda_runtime_available); "
                    "assert callable(arti.cuda_device_report); "
                    "assert callable(arti.cuda_smoke_report); "
                    "assert callable(arti.require_cuda); "
                    "assert callable(arti.virtual_recall_alignment_loss); "
                    "assert callable(arti.experiential_recall_alignment_loss); "
                    "assert callable(arti.recall_route_exterior_penalty); "
                    "assert callable(arti.experiential_recall_selectivity_loss); "
                    "assert callable(arti.ARTIClassifier); "
                    "assert arti.torch.ARTIClassifier is arti.ARTIClassifier; "
                    "assert callable(arti.ARTIHostBridge); "
                    "assert arti.torch.ARTIHostBridge is arti.ARTIHostBridge; "
                    "assert callable(arti.Half); "
                    "assert arti.component_ref(arti.Half()).startswith('arti/half@sha256:'); "
                    "assert callable(arti.component_provenance); "
                    "assert callable(arti.component_catalog); "
                    "assert callable(arti.component_state_contract); "
                    "assert callable(arti.validate_component_state_contract); "
                    "assert callable(arti.state_dict_schema); "
                    "assert callable(arti.resolve_component); "
                    "assert callable(arti.validate_component_provenance); "
                    "assert arti.torch.Half is arti.Half; "
                    "assert callable(arti.TensorContext); "
                    "assert callable(arti.FrameContext); "
                    "assert callable(arti.EmissionRouter); "
                    "assert arti.nn.EmissionRouter is arti.EmissionRouter; "
                    "assert arti.torch.EmissionRouter is arti.EmissionRouter; "
                    "assert callable(arti.Retrieve); "
                    "assert arti.nn.Retrieve is arti.Retrieve; "
                    "assert arti.torch.Retrieve is arti.Retrieve; "
                    "assert callable(arti.module_behavior_fingerprint); "
                    "assert callable(arti.validate_formula); "
                    "assert callable(arti.RecallFormulaId); "
                    "assert callable(arti.register_formula); "
                    "assert callable(arti.list_formulas); "
                    "assert callable(arti.UnFold); "
                    "assert arti.torch.UnFold is arti.UnFold; "
                    "assert callable(arti.Fold); "
                    "assert arti.torch.Fold is arti.Fold; "
                    "assert callable(arti.Pulse); "
                    "assert arti.Pulse is arti.LearnedPulse; "
                    "assert arti.torch.Pulse is arti.Pulse; "
                    "assert callable(arti.LearnedPulse); "
                    "assert arti.torch.LearnedPulse is arti.LearnedPulse; "
                    "assert callable(arti.RecallExecutor); "
                    "assert arti.torch.RecallExecutor is arti.RecallExecutor; "
                    "assert callable(arti.RetrieveExecutor); "
                    "assert arti.torch.RetrieveExecutor is arti.RetrieveExecutor; "
                    "assert callable(arti.RecallCapacityPlan); "
                    "assert callable(arti.RecallCapacityDecision); "
                    "assert callable(arti.RecallBankAssembly); "
                    "assert callable(arti.RecallBankContract); "
                    "assert callable(arti.RecallBankError); "
                    "assert callable(arti.RecallBankProvenance); "
                    "assert callable(arti.migrate_recall_bank); "
                    "assert arti.RECALL_BANK_PROVENANCE_VERSION == 1; "
                    "assert arti.alpha is arti.mechanisms; "
                    "default_layer = arti.ARTILayer(); "
                    "assert default_layer.program is None and default_layer.graph is None; "
                    "assert arti.component_ref(default_layer).startswith('arti/layer@sha256:'); "
                    "assert arti.component_spec(default_layer).lifecycle == 'alpha'; "
                    "assert arti.torch.ARTILayer is arti.ARTILayer; "
                    "assert callable(arti.legacy.LayerRecall); "
                    "assert callable(arti.legacy.StatefulRecall); "
                    "assert callable(arti.legacy.ARTILayer); "
                    "assert callable(arti.alpha.RecallValueUpdater); "
                    "assert callable(arti.alpha.query_recall_branches); "
                    "assert callable(arti.alpha.run_branch_search); "
                    "assert callable(arti.alpha.RefineStepTraining); "
                    "assert callable(arti.alpha.RefineRollout); "
                    "assert callable(arti.alpha.FormulaRefineExit); "
                    "assert callable(arti.alpha.RefineExitControl); "
                    "assert callable(arti.alpha.RefineExitCurve); "
                    "assert callable(arti.alpha.RefineExitTraining); "
                    "assert callable(arti.alpha.PortSpec); "
                    "assert callable(arti.alpha.OperableTensorPort); "
                    "assert callable(arti.alpha.SharedCanvasFold); "
                    "assert callable(arti.alpha.TensorEditFormula); "
                    "assert callable(arti.alpha.TensorOperationQuery); "
                    "assert callable(arti.alpha.TensorOperationBank); "
                    "assert callable(arti.alpha.TensorOperationSelector); "
                    "assert callable(arti.alpha.TensorOperationLoop); "
                    "assert callable(arti.alpha.TensorOperationFieldSpec); "
                    "assert callable(arti.alpha.TensorOperationRouteSelection); "
                    "assert callable(arti.alpha.TensorInvocation); "
                    "port_spec = arti.alpha.PortSpec(canvas_tokens=2, tensor_shape=(1,), dim=2, tensor_to_canvas=(0,)); "
                    "operable_port = arti.alpha.OperableTensorPort(port_spec, batch_size=1); "
                    "assert arti.component_ref(port_spec).startswith('arti/operable-tensor-port-spec@sha256:'); "
                    "assert arti.component_ref(operable_port).startswith('arti/operable-tensor-port@sha256:'); "
                    "assert callable(arti.ExecutionTraceV3); "
                    "assert arti.torch.ExecutionTraceV3 is arti.ExecutionTraceV3; "
                    "assert arti.EXECUTION_TRACE_V3_SCHEMA_REF == 'arti/execution-trace@3'; "
                    "assert not hasattr(arti, 'RefinePolicy'); "
                    "assert callable(arti.legacy.RefinePolicy); "
                    "exit_atom = arti.alpha.FormulaRefineExit(input_kind='logit'); "
                    "assert arti.component_ref(exit_atom).startswith('arti/formula-atom-refine-exit@sha256:'); "
                    "exit_request = exit_atom(torch.ones(1, 1), mask=torch.ones(1, 1, dtype=torch.bool)); "
                    "assert arti.component_ref(exit_request).startswith('arti/refine-exit-request@sha256:'); "
                    "assert callable(arti.alpha.BranchSearchPlan); "
                    "assert callable(arti.alpha.BranchBatchHarness); "
                    "assert callable(arti.alpha.HotPagePool); "
                    "assert callable(arti.alpha.bind_hot_page_pool); "
                    "assert callable(arti.alpha.save_runtime_checkpoint); "
                    "assert callable(arti.alpha.load_runtime_checkpoint); "
                    "assert callable(arti.alpha.VolatileTensorRuntime); "
                    "assert callable(arti.alpha.TensorSchema); "
                    "assert callable(arti.alpha.TerminalOutputABI); "
                    "assert callable(arti.alpha.BankExecutionSignature); "
                    "assert callable(arti.alpha.AutonomousBankProgram); "
                    "assert callable(arti.alpha.FederalCandidate); "
                    "assert callable(arti.alpha.FederalRecall); "
                    "assert callable(arti.alpha.FederalTrace); "
                    "assert callable(arti.alpha.FormulaFabricV2); "
                    "assert callable(arti.alpha.FormulaExecutionPlanV2); "
                    "assert callable(arti.alpha.FormulaOperandBank); "
                    "assert callable(arti.alpha.FormulaProgramCandidate); "
                    "assert callable(arti.alpha.FormulaProgramQuery); "
                    "assert callable(arti.alpha.ExactFormulaProgramQueryTraining); "
                    "assert callable(arti.alpha.build_lora_program); "
                    "assert callable(arti.alpha.build_routed_lora_program); "
                    "assert callable(arti.alpha.hard_formula_route); "
                    "formula_program = arti.alpha.build_lora_program(input_dim=2, output_dim=2, rank=1, source_ref='arti/package-formula-bank@1', dtype='float32'); "
                    "formula_payload = json.loads(json.dumps(formula_program.to_dict())); "
                    "assert arti.alpha.FormulaProgram.from_dict(formula_payload).fingerprint == formula_program.fingerprint; "
                    "formula_bank = arti.alpha.FormulaOperandBank(keys=torch.tensor([[1.0, 0.0]]), operands={'A': torch.ones(1, 1, 2), 'B': torch.ones(1, 2, 1), 'gain': torch.ones(1)}, source_ref='arti/formula-operand-bank@1', bundle_id='lora'); "
                    "routed_program = arti.alpha.build_routed_lora_program(input_dim=2, output_dim=2, rank=1, candidate_count=1, source_ref=formula_bank.source_ref, bundle_id=formula_bank.bundle_id, member_ids=formula_bank.member_ids, dtype='float32'); "
                    "route = formula_bank.route(torch.tensor([[1.0, 0.0]]), estimator='hard').route; "
                    "formula_result = arti.alpha.FormulaFabricV2(routed_program)(inputs={'x': torch.ones(1, 1, 2), 'base': torch.zeros(1, 1, 2), 'formula.route': route}, banks=formula_bank.bind(routed_program), return_trace=True); "
                    "assert formula_result.values[0].shape == (1, 1, 2); "
                    "assert formula_result.trace.to_dict()['schema_ref'].startswith('arti/formula-trace@sha256:'); "
                    "formula_state_contract = arti.component_state_contract(formula_bank, formula_bank.state_dict(), scope='trainable'); "
                    "assert arti.validate_component_state_contract(formula_state_contract, state_dict=formula_bank.state_dict(), model=formula_bank) == formula_state_contract; "
                    "assert not hasattr(arti, 'StatefulRecall'); "
                    "assert not hasattr(arti, 'LayerRecall'); "
                    "assert callable(arti.VisualField); "
                    "assert callable(arti.concat_visual_fields); "
                    "assert callable(arti.VisualScan); "
                    "assert callable(arti.VisualScanConfig); "
                    "assert callable(arti.nn.Layer); "
                    "assert callable(arti.features); "
                    "assert callable(arti.profile); "
                    "assert callable(arti.inspect); "
                    "assert callable(arti.ARTI.attach); "
                    "assert arti.torch.ARTI is arti.ARTI; "
                    "assert callable(arti.load_attach_config); "
                    "assert callable(arti.write_attach_config); "
                    "assert callable(arti.ARTITrainingSession); "
                    "assert arti.torch.ARTITrainingSession is arti.ARTITrainingSession; "
                    "assert callable(arti.ARTI.from_pretrained); "
                    "assert callable(arti.ARTICheckpointCallback); "
                    "assert callable(arti.ARTIDoctorReport); "
                    "assert callable(arti.pixel_shift_observe); "
                    "assert callable(arti.shift_and_add); "
                    "assert arti.torch.RecallCapacityPlan is arti.RecallCapacityPlan; "
                    "assert arti.torch.RecallCapacityDecision is arti.RecallCapacityDecision; "
                    "assert callable(arti.functional.half); "
                    "assert arti.torch.half is arti.functional.half; "
                    "assert callable(arti.MembraneVisibilityRouter); "
                    "assert callable(arti.build_membrane_visibility); "
                    "assert arti.torch.MembraneVisibilityRouter is arti.MembraneVisibilityRouter; "
                    "assert callable(arti.build_participant_context); "
                    "assert callable(arti.last_non_assistant_participant); "
                    "assert arti.torch.build_participant_context is arti.build_participant_context; "
                    "assert callable(arti.RuntimeVocabInput); "
                    "assert callable(arti.RuntimeVocabHead); "
                    "assert callable(arti.LiteralInput); "
                    "assert callable(arti.OutputLexiconContext); "
                    "assert callable(arti.LiteralOutputHead); "
                    "assert callable(arti.LiteralVocabModel); "
                    "assert callable(arti.LiteralSequenceDecoder); "
                    "assert callable(arti.LiteralSequenceOutput); "
                    "assert callable(arti.fit_literal_sequence); "
                    "assert callable(arti.save); "
                    "assert callable(arti.load); "
                    "assert arti.ARTI_ST_FORMAT == 'arti.st'; "
                    "assert callable(arti.ARTIPlan); "
                    "assert callable(arti.pretrained); "
                    "assert callable(arti.from_pretrained); "
                    "assert callable(arti.validate_pretrained_lock); "
                    "assert callable(arti.register_provider); "
                    "assert {row['name'] for row in arti.provider_report()} >= {'torch', 'transformers', 'peft', 'diffusers'}; "
                    "assert callable(arti.RuntimeVocabModel); "
                    "assert callable(arti.RuntimeVocabPulseAdapter); "
                    "assert callable(arti.attach_runtime_vocab_semantics); "
                    "assert callable(arti.permute_runtime_vocab); "
                    "assert callable(arti.remap_token_ids); "
                    "assert arti.torch.RuntimeVocabHead is arti.RuntimeVocabHead; "
                    "assert arti.torch.OutputLexiconContext is arti.OutputLexiconContext; "
                    "assert arti.torch.LiteralOutputHead is arti.LiteralOutputHead; "
                    "assert arti.torch.LiteralSequenceDecoder is arti.LiteralSequenceDecoder; "
                    "assert arti.torch.fit_literal_sequence is arti.fit_literal_sequence; "
                    "assert arti.torch.save is arti.save; "
                    "assert arti.torch.load is arti.load; "
                    "assert arti.torch.RuntimeVocabPulseAdapter is arti.RuntimeVocabPulseAdapter; "
                    "assert arti.torch.attach_runtime_vocab_semantics is arti.attach_runtime_vocab_semantics; "
                    "assert callable(arti.PulseCompressor); "
                    "assert callable(arti.pulse_compress); "
                    "assert callable(arti.fixed_width_pulse_ids); "
                    "assert callable(arti.pulse_distinctness_report); "
                    "assert callable(arti.assert_pulse_distinct); "
                    "assert callable(arti.latent_distinctness_report); "
                    "assert callable(arti.assert_latent_distinct); "
                    "assert arti.torch.PulseCompressor is arti.PulseCompressor; "
                    "assert callable(arti.BitmapTextRenderer); "
                    "assert callable(arti.render_text_bitmap); "
                    "assert callable(arti.render_text_vocab); "
                    "assert callable(arti.bitmap_vocab_report); "
                    "assert callable(arti.assert_bitmap_vocab_distinct); "
                    "assert arti.torch.BitmapTextRenderer is arti.BitmapTextRenderer; "
                    "assert callable(arti.TextTensorRenderer); "
                    "assert callable(arti.render_text_layout); "
                    "assert callable(arti.render_text_tensor); "
                    "assert 'glyph_only' in arti.TEXT_IDENTITY_MODES; "
                    "assert arti.torch.TextTensorRenderer is arti.TextTensorRenderer; "
                    "from arti.integrations.qwen import QwenGlyphRuntimeAdapter; "
                    "assert callable(QwenGlyphRuntimeAdapter); "
                    "assert callable(arti.make_source_integrity_basis); "
                    "assert callable(arti.SourceIntegrityCarrier); "
                    "assert callable(arti.superpose_sources); "
                    "assert callable(arti.read_sources); "
                    "assert callable(arti.encode_source_tokens); "
                    "assert callable(arti.decode_source_tokens); "
                    "assert callable(arti.source_integrity_report); "
                    "assert callable(arti.assert_source_integrity); "
                    "assert arti.torch.SourceIntegrityCarrier is arti.SourceIntegrityCarrier; "
                    "assert callable(arti.fit); "
                    "assert callable(arti.project); "
                    "assert callable(arti.apply_adapter); "
                    "assert callable(arti.validate_plan); "
                    "assert callable(arti.create_build_lock); "
                    "assert callable(arti.create_deployment_manifest); "
                    "assert callable(arti.create_task_graph_payload); "
                    "assert callable(arti.validate_build_lock); "
                    "assert callable(arti.validate_deployment_manifest); "
                    "assert callable(arti.validate_task_graph); "
                    "assert callable(arti.validate_task_graph_payload); "
                    "assert callable(arti.write_task_graph_artifact); "
                    "assert callable(arti.plan_provenance_fingerprint); "
                    "assert callable(arti.cli.main); "
                    "assert callable(arti.capabilities); "
                    "assert callable(arti.backend_capabilities); "
                    "assert callable(arti.doctor_report); "
                    "assert callable(arti.doctor_report_markdown); "
                    "assert callable(arti.validate_backend_capabilities); "
                    "assert callable(arti.write_doctor_report); "
                    "assert callable(arti.generate_capabilities_markdown); "
                    "assert callable(arti.write_generated_docs); "
                    "assert callable(arti.check_generated_docs); "
                    "assert callable(arti.generate_fit_config_schema); "
                    "assert callable(arti.generate_fit_config_schema_json); "
                    "assert callable(arti.generate_task_graph_schema); "
                    "assert callable(arti.generate_task_graph_schema_json); "
                    "assert callable(arti.packaged_fit_config_schema_json); "
                    "assert callable(arti.packaged_task_graph_schema_json); "
                    "assert arti.packaged_fit_config_schema_json() == arti.generate_fit_config_schema_json(); "
                    "assert arti.packaged_task_graph_schema_json() == arti.generate_task_graph_schema_json(); "
                    "assert callable(arti.write_fit_config_schema); "
                    "assert callable(arti.write_task_graph_schema); "
                    "assert callable(arti.check_fit_config_schema); "
                    "assert callable(arti.check_task_graph_schema); "
                    "assert callable(arti.list_profiles); "
                    "assert callable(arti.load_fit_config); "
                    "assert callable(arti.write_fit_config_template); "
                    "assert callable(arti.validate_fit_config); "
                    "assert callable(arti.resolve_fit_config_mechanism); "
                    "assert callable(arti.apply_mechanism_overrides); "
                    "assert callable(arti.MechanismOverrides); "
                    "assert callable(arti.RuntimeFieldConfig); "
                    "assert callable(arti.infer_batch_schema); "
                    "assert callable(arti.attention_mask_to_visibility); "
                    "assert arti.get_plugin('torch').available; "
                    "assert arti.torch.fit is arti.fit; "
                    "assert arti.torch.apply_adapter is arti.apply_adapter; "
                    "assert arti.torch.validate_plan is arti.validate_plan; "
                    "assert arti.torch.create_build_lock is arti.create_build_lock; "
                    "assert arti.torch.create_deployment_manifest is arti.create_deployment_manifest; "
                    "assert arti.torch.create_task_graph_payload is arti.create_task_graph_payload; "
                    "assert arti.torch.validate_build_lock is arti.validate_build_lock; "
                    "assert arti.torch.validate_deployment_manifest is arti.validate_deployment_manifest; "
                    "assert arti.torch.validate_task_graph is arti.validate_task_graph; "
                    "assert arti.torch.validate_task_graph_payload is arti.validate_task_graph_payload; "
                    "assert arti.torch.write_task_graph_artifact is arti.write_task_graph_artifact; "
                    "assert arti.torch.plan_provenance_fingerprint is arti.plan_provenance_fingerprint; "
                    "assert arti.torch.capabilities is arti.capabilities; "
                    "assert arti.torch.doctor_report is arti.doctor_report; "
                    "assert arti.torch.write_doctor_report is arti.write_doctor_report; "
                    "assert arti.torch.generate_capabilities_markdown is arti.generate_capabilities_markdown; "
                    "assert arti.torch.write_generated_docs is arti.write_generated_docs; "
                    "assert arti.torch.check_generated_docs is arti.check_generated_docs; "
                    "assert arti.torch.generate_fit_config_schema is arti.generate_fit_config_schema; "
                    "assert arti.torch.generate_fit_config_schema_json is arti.generate_fit_config_schema_json; "
                    "assert arti.torch.generate_task_graph_schema is arti.generate_task_graph_schema; "
                    "assert arti.torch.generate_task_graph_schema_json is arti.generate_task_graph_schema_json; "
                    "assert arti.torch.packaged_fit_config_schema_json is arti.packaged_fit_config_schema_json; "
                    "assert arti.torch.packaged_task_graph_schema_json is arti.packaged_task_graph_schema_json; "
                    "assert arti.torch.write_fit_config_schema is arti.write_fit_config_schema; "
                    "assert arti.torch.write_task_graph_schema is arti.write_task_graph_schema; "
                    "assert arti.torch.check_fit_config_schema is arti.check_fit_config_schema; "
                    "assert arti.torch.check_task_graph_schema is arti.check_task_graph_schema; "
                    "assert arti.torch.load_fit_config is arti.load_fit_config; "
                    "assert arti.torch.write_fit_config_template is arti.write_fit_config_template; "
                    "assert arti.torch.validate_fit_config is arti.validate_fit_config; "
                    "assert arti.torch.resolve_fit_config_mechanism is arti.resolve_fit_config_mechanism; "
                    "assert arti.torch.apply_mechanism_overrides is arti.apply_mechanism_overrides; "
                    "assert arti.torch.MechanismOverrides is arti.MechanismOverrides; "
                    "assert arti.torch.RuntimeFieldConfig is arti.RuntimeFieldConfig; "
                    "assert arti.torch.cuda_runtime_available is arti.cuda_runtime_available; "
                    "assert arti.torch.cuda_device_report is arti.cuda_device_report; "
                    "assert arti.torch.cuda_smoke_report is arti.cuda_smoke_report; "
                    "assert arti.torch.experiential_recall_alignment_loss is arti.experiential_recall_alignment_loss; "
                    "assert arti.torch.recall_route_exterior_penalty is arti.recall_route_exterior_penalty; "
                    "print(arti.ARTILayer.__name__)"
                ),
            ],
            cwd=Path(tmp),
            env=env,
        )

    print(f"Built and smoke-tested {wheels[0].name} and {sdists[0].name}")


if __name__ == "__main__":
    main()
