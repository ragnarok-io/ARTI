"""Build ARTI distributions and smoke-test wheel installation."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import os
import tarfile
import zipfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 release runner
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def run(command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True, env=env)


def main() -> None:
    if DIST.exists():
        shutil.rmtree(DIST)

    run([sys.executable, "-m", "build", "--no-isolation", "--sdist", "--wheel", "--outdir", str(DIST)])

    wheels = sorted(DIST.glob("*.whl"))
    sdists = sorted(DIST.glob("*.tar.gz"))
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
            "arti/web/__init__.py",
            "arti/web/contract.py",
            "arti/web/exporter.py",
            "arti/serialization.py",
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
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        env["ARTI_EXPECTED_VERSION"] = str(project["project"]["version"])
        run(
            [
                sys.executable,
                "-c",
                    (
                    "import arti, arti.functional, arti.torch, arti.jax, arti.web; "
                    "from arti.alpha import TargetBankUpdater, WriteRefinePolicy; "
                    "import os, pathlib; "
                    "assert pathlib.Path(arti.__file__).resolve().parent == pathlib.Path(os.environ['ARTI_WHEEL_ROOT']) / 'arti'; "
                    "import arti.cli; "
                    "assert callable(arti.web.export); "
                    "assert arti.__version__ == os.environ['ARTI_EXPECTED_VERSION']; "
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
                    "assert arti.torch.Half is arti.Half; "
                    "assert arti.component_ref(TargetBankUpdater(4, 3)) == 'arti/target-bank-updater@1'; "
                    "assert arti.component_ref(arti.resolve_component('arti/target-bank-updater@2', hidden_dim=4, slots=3)) == 'arti/target-bank-updater@2'; "
                    "assert WriteRefinePolicy.adaptive(max_steps=4, min_steps=2).budget.max_steps == 4; "
                    "assert callable(arti.Recall); "
                    "assert arti.nn.Recall is arti.Recall; "
                    "assert arti.torch.Recall is arti.Recall; "
                    "assert callable(arti.validate_formula); "
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
                    "assert callable(arti.RecallRefiner); "
                    "assert arti.torch.RecallRefiner is arti.RecallRefiner; "
                    "assert callable(arti.RecallCapacityPlan); "
                    "assert callable(arti.RecallCapacityDecision); "
                    "assert callable(arti.RecallExpertPool); "
                    "assert callable(arti.RecallExpertContract); "
                    "assert callable(arti.RecallExpertAssembly); "
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
                    "assert callable(arti.LayerRecall); "
                    "assert callable(arti.LayeredRecallModel); "
                    "assert callable(arti.layered_recall_trajectory_loss); "
                    "assert callable(arti.pixel_shift_observe); "
                    "assert callable(arti.shift_and_add); "
                    "assert arti.torch.RecallCapacityPlan is arti.RecallCapacityPlan; "
                    "assert arti.torch.RecallCapacityDecision is arti.RecallCapacityDecision; "
                    "assert arti.torch.RecallExpertPool is arti.RecallExpertPool; "
                    "assert arti.torch.RecallExpertContract is arti.RecallExpertContract; "
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
                    "assert callable(arti.migrate_pt); "
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
