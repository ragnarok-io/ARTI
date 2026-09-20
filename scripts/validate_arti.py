"""Quick mechanism validation for ARTI.

This script is intentionally lightweight. It does not claim downstream task
quality; it checks that the package mechanisms promised by the architecture
contract are active and stable.
"""

from __future__ import annotations

import tempfile
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import torch
import torch.nn as nn

import arti
from arti import ARTIResidualBlock, apply_adapter, create_build_lock, create_deployment_manifest, fit, project, validate_deployment_manifest
from arti.legacy import ARTILayer as ClassicARTILayer
from arti.cli import main as arti_cli_main
from arti.fit.artifacts import hash_tensor_state_dict


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_default_layer() -> None:
    layer = arti.ARTILayer()
    x = torch.randn(2, 5, 8)
    y, result = layer(x, return_info=True)

    _assert(arti.component_ref(layer) == "arti/layer@2", "default layer identity mismatch")
    _assert(isinstance(layer.pulse, arti.mechanisms.AdaptivePulse), "default layer is not AdaptivePulse-backed")
    _assert(torch.equal(y, x), "empty AdaptivePulse graph is not identity")
    _assert(result.value_identity, "identity Pulse did not retain value identity")


def validate_shapes() -> None:
    layer = ClassicARTILayer(input_dim=8, coord_dim=3, hidden_dim=16)
    x = torch.randn(2, 5, 8)
    coord = torch.randn(2, 5, 3)
    out = layer(x, coord=coord)

    _assert(out.y.shape == (2, 5, 16), "sequence output shape mismatch")
    _assert(out.pooled.shape == (2, 16), "pooled shape mismatch")
    _assert("operator_weights" in out.diagnostics, "missing operator diagnostics")

    vector_out = ClassicARTILayer(input_dim=8, hidden_dim=8)(torch.randn(2, 8))
    _assert(vector_out.y.shape == (2, 8), "vector output rank was not restored")


def validate_masking() -> None:
    torch.manual_seed(1)
    layer = ClassicARTILayer(input_dim=4, hidden_dim=4, recall_steps=0)
    x = torch.randn(1, 4, 4)
    mask = torch.tensor([[True, True, False, False]])

    out_a = layer(x, mask=mask).pooled
    x_changed = x.clone()
    x_changed[:, 2:] = torch.randn_like(x_changed[:, 2:]) * 1000.0
    out_b = layer(x_changed, mask=mask).pooled

    _assert(torch.allclose(out_a, out_b, atol=1e-5), "masked tokens leaked into pooled output")


def validate_visibility() -> None:
    torch.manual_seed(2)
    layer = ClassicARTILayer(input_dim=4, hidden_dim=4, recall_steps=0)
    x = torch.randn(1, 3, 4)
    visibility = torch.tensor([[[True, True, False], [True, True, False], [False, False, True]]])
    out = layer(x, visibility=visibility)

    blocked = out.diagnostics["visibility_weights"][0, :2, 2]
    _assert(torch.all(blocked == 0), "visibility did not block hidden token")


def validate_coordinate_sensitivity() -> None:
    torch.manual_seed(3)
    layer = ClassicARTILayer(input_dim=6, coord_dim=2, hidden_dim=6, recall_steps=0)
    x = torch.randn(2, 4, 6)
    coord_a = torch.zeros(2, 4, 2)
    coord_b = torch.ones(2, 4, 2)

    out_a = layer(x, coord=coord_a).y
    out_b = layer(x, coord=coord_b).y

    _assert(not torch.allclose(out_a, out_b), "coordinates did not affect latent output")


def validate_backward_and_serialization() -> None:
    torch.manual_seed(4)
    layer = ClassicARTILayer(input_dim=8, coord_dim=2, hidden_dim=8)
    x = torch.randn(2, 5, 8, requires_grad=True)
    coord = torch.randn(2, 5, 2)
    loss = layer(x, coord=coord).pooled.square().mean()
    loss.backward()

    _assert(x.grad is not None, "missing input gradient")
    _assert(torch.isfinite(x.grad).all().item(), "non-finite input gradient")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "arti.pt"
        torch.save(layer.state_dict(), path)
        loaded = ClassicARTILayer(input_dim=8, coord_dim=2, hidden_dim=8)
        loaded.load_state_dict(torch.load(path, weights_only=True))
        _assert(loaded(x.detach(), coord=coord).y.shape == (2, 5, 8), "loaded model shape mismatch")


def validate_sequential_block() -> None:
    block = ARTIResidualBlock(dim=8)
    x = torch.randn(2, 8)
    y = block(x)
    _assert(y.shape == x.shape, "residual block is not shape-stable")


def validate_build_pipeline() -> None:
    torch.manual_seed(5)
    sample = torch.randn(2, 4)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        plan = project(nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))).scan(sample).write_plan(root / "plan.json", where="0")
        artifact = fit(nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)), sample_batch=sample, target_modules="0").export(root / "adapter.pt")
        lock = create_build_lock(root / "arti.lock.json", artifact=artifact, plan=plan)
        applied = apply_adapter(nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)), artifact, sample_batch=sample)
        applied_report = applied.write_report(root / "applied.json")
        state_dict = applied.model.state_dict()
        state_path = root / "patched-state.pt"
        torch.save(state_dict, state_path)
        deployment = create_deployment_manifest(
            root / "deployment.json",
            lock=lock,
            artifact=artifact,
            applied_report=applied_report,
            state_dict=state_path,
        )
        payload = validate_deployment_manifest(deployment)

    _assert(payload["kind"] == "deployment-manifest", "deployment manifest kind mismatch")
    _assert(payload["state_dict"]["state_dict_sha256"] == hash_tensor_state_dict(state_dict), "deployment state_dict hash mismatch")
    _assert(payload["artifact"]["adapter_state_sha256"], "deployment manifest missing adapter hash")


def run_cli(args: list[str]) -> int:
    with redirect_stdout(StringIO()):
        return arti_cli_main(args)


def validate_cli_build_pipeline() -> None:
    torch.manual_seed(6)
    sample = torch.randn(2, 4)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        module_path = root / "cli_fixture_model.py"
        module_path.write_text(
            "\n".join(
                [
                    "import torch.nn as nn",
                    "",
                    "def make_model():",
                    "    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        sys.path.insert(0, str(root))
        try:
            artifact = fit(nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)), sample_batch=sample, target_modules="0").export(root / "adapter.pt")
            plan = root / "plan.json"
            lock = root / "arti.lock.json"
            applied_report = root / "applied.json"
            state_path = root / "patched-state.pt"
            deployment = root / "deployment.json"
            _assert(run_cli(["plan", "cli_fixture_model:make_model", str(plan), "--sample-shape", "2,4", "--target-modules", "0"]) == 0, "CLI plan failed")
            _assert(run_cli(["lock", str(lock), "--artifact", str(artifact), "--plan", str(plan)]) == 0, "CLI lock failed")
            _assert(
                run_cli(
                    [
                        "apply",
                        "cli_fixture_model:make_model",
                        str(artifact),
                        str(applied_report),
                        "--sample-shape",
                        "2,4",
                        "--lock",
                        str(lock),
                        "--save-state-dict",
                        str(state_path),
                    ]
                )
                == 0,
                "CLI apply failed",
            )
            _assert(
                run_cli(
                    [
                        "deployment-manifest",
                        str(deployment),
                        "--lock",
                        str(lock),
                        "--artifact",
                        str(artifact),
                        "--applied-report",
                        str(applied_report),
                        "--state-dict",
                        str(state_path),
                    ]
                )
                == 0,
                "CLI deployment manifest failed",
            )
            payload = validate_deployment_manifest(deployment)
            _assert(
                run_cli(
                    [
                        "validate",
                        "deployment",
                        str(deployment),
                        "--expect-adapter-state-sha256",
                        payload["artifact"]["adapter_state_sha256"],
                        "--expect-state-dict-sha256",
                        payload["state_dict"]["state_dict_sha256"],
                    ]
                )
                == 0,
                "CLI deployment validation failed",
            )
        finally:
            try:
                sys.path.remove(str(root))
            except ValueError:
                pass


def main() -> None:
    validations = [
        validate_default_layer,
        validate_shapes,
        validate_masking,
        validate_visibility,
        validate_coordinate_sensitivity,
        validate_backward_and_serialization,
        validate_sequential_block,
        validate_build_pipeline,
        validate_cli_build_pipeline,
    ]
    for validation in validations:
        validation()
        print(f"PASS {validation.__name__}")
    print("ARTI quick validation passed.")


if __name__ == "__main__":
    main()
