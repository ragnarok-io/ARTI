"""Installed environment diagnostics for ARTI adaptation builds."""

from __future__ import annotations

import platform
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .._version import __version__
from ..backend import available_backends, planned_backends
from ..jax import backend_status as jax_backend_status
from ..jax import smoke_report as jax_smoke_report


@dataclass(frozen=True)
class BackendCapabilities:
    python: str
    platform: str
    arti_version: str
    available_backends: tuple[str, ...]
    planned_backends: tuple[str, ...]
    torch_version: str
    torch_cuda_version: str | None
    nvidia_smi_available: bool
    nvidia_driver_version: str | None
    nvidia_cuda_version: str | None
    nvidia_devices: list[dict[str, Any]]
    torch_cuda_available: bool
    torch_cuda_device_count: int
    torch_cuda_devices: list[dict[str, Any]]
    torch_cuda_smoke: dict[str, Any]
    torch_cuda_smoke_status: str
    torch_cuda_amp_available: bool
    torch_compile_available: bool
    jax_backend_status: str
    jax_smoke: dict[str, Any]
    jax_smoke_status: str
    gpu_readiness_level: str
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cuda_devices() -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    if not torch.cuda.is_available():
        return devices
    for index in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(index)
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(capability),
                "total_memory_bytes": int(props.total_memory),
            }
        )
    return devices


def cuda_smoke_probe(size: int = 16) -> dict[str, Any]:
    """Run the doctor CUDA allocation/compute probe."""

    if not torch.cuda.is_available():
        return {
            "smoke_status": "skipped",
            "allocation_ok": False,
            "compute_ok": False,
            "reason": "PyTorch CUDA runtime is not available.",
        }
    device = torch.device("cuda", torch.cuda.current_device())
    try:
        x = torch.eye(size, device=device)
        y = x @ x
        total = float(y.sum().detach().cpu().item())
        torch.cuda.synchronize(device)
    except RuntimeError as exc:
        return {
            "smoke_status": "failed",
            "allocation_ok": False,
            "compute_ok": False,
            "device": str(device),
            "reason": str(exc),
        }
    return {
        "smoke_status": "passed",
        "allocation_ok": True,
        "compute_ok": total == float(size),
        "device": str(device),
        "smoke_tensor_shape": [size, size],
        "smoke_sum": total,
    }


def nvidia_smi_report() -> tuple[bool, str | None, str | None, list[dict[str, Any]]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False, None, None, []

    devices: list[dict[str, Any]] = []
    driver_version = None
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        index, name, driver, memory_mib = parts[:4]
        driver_version = driver_version or driver
        devices.append(
            {
                "index": int(index),
                "name": name,
                "driver_version": driver,
                "total_memory_mib": int(memory_mib),
            }
        )

    cuda_version = None
    try:
        plain = subprocess.run(["nvidia-smi"], check=True, capture_output=True, text=True).stdout
        match = re.search(r"CUDA Version:\s+([0-9.]+)", plain)
        cuda_version = match.group(1) if match else None
    except (FileNotFoundError, subprocess.CalledProcessError):
        cuda_version = None

    return True, driver_version, cuda_version, devices


def backend_capabilities() -> BackendCapabilities:
    torch_cuda_available = torch.cuda.is_available()
    torch_devices = cuda_devices()
    torch_cuda_smoke = cuda_smoke_probe()
    jax_smoke = jax_smoke_report()
    smi_available, driver_version, smi_cuda_version, smi_devices = nvidia_smi_report()
    if torch_cuda_available:
        gpu_readiness_level = "torch_cuda_runtime_available"
        interpretation = "PyTorch CUDA runtime is available and the local CUDA smoke probe can be used."
    elif smi_devices:
        gpu_readiness_level = "nvidia_hardware_detected_torch_cpu"
        interpretation = (
            "NVIDIA hardware is visible through nvidia-smi, but the current PyTorch build is CPU-only or cannot access CUDA. "
            "Install a CUDA-enabled PyTorch build to use ARTI on this device."
        )
    else:
        gpu_readiness_level = "cpu_only"
        interpretation = "No NVIDIA CUDA hardware is visible locally; ARTI will use the CPU runtime."
    return BackendCapabilities(
        python=sys.version.split()[0],
        platform=platform.platform(),
        arti_version=__version__,
        available_backends=available_backends(),
        planned_backends=planned_backends(),
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
        nvidia_smi_available=smi_available,
        nvidia_driver_version=driver_version,
        nvidia_cuda_version=smi_cuda_version,
        nvidia_devices=smi_devices,
        torch_cuda_available=torch_cuda_available,
        torch_cuda_device_count=torch.cuda.device_count() if torch_cuda_available else 0,
        torch_cuda_devices=torch_devices,
        torch_cuda_smoke=torch_cuda_smoke,
        torch_cuda_smoke_status=str(torch_cuda_smoke.get("smoke_status")),
        torch_cuda_amp_available=torch_cuda_available,
        torch_compile_available=hasattr(torch, "compile"),
        jax_backend_status=jax_backend_status(),
        jax_smoke=jax_smoke,
        jax_smoke_status=str(jax_smoke.get("smoke_status")),
        gpu_readiness_level=gpu_readiness_level,
        interpretation=interpretation,
    )


def validate_backend_capabilities(payload: dict[str, Any], *, allow_cpu_torch: bool = False) -> list[str]:
    failures: list[str] = []
    required = {
        "python",
        "platform",
        "arti_version",
        "available_backends",
        "planned_backends",
        "torch_version",
        "nvidia_smi_available",
        "nvidia_driver_version",
        "nvidia_cuda_version",
        "nvidia_devices",
        "torch_cuda_available",
        "torch_cuda_device_count",
        "torch_cuda_devices",
        "torch_cuda_smoke",
        "torch_cuda_smoke_status",
        "torch_cuda_amp_available",
        "torch_compile_available",
        "jax_backend_status",
        "jax_smoke",
        "jax_smoke_status",
        "gpu_readiness_level",
        "interpretation",
    }
    missing = required - set(payload)
    if missing:
        return [f"missing fields: {sorted(missing)}"]
    if "torch" not in payload["available_backends"]:
        failures.append("available_backends must include torch")
    if payload["jax_backend_status"] not in {"available", "unavailable"}:
        failures.append("jax_backend_status must be available or unavailable")
    if payload["jax_backend_status"] == "available" and "jax" not in payload["available_backends"]:
        failures.append("available JAX backend must be listed in available_backends")
    if payload["jax_backend_status"] == "unavailable" and "jax" not in payload["planned_backends"]:
        failures.append("unavailable JAX backend must be listed in planned_backends")
    if payload["jax_smoke_status"] not in {"passed", "skipped", "failed"}:
        failures.append("jax_smoke_status must be passed, skipped, or failed")
    if not isinstance(payload["jax_smoke"], dict):
        failures.append("jax_smoke must be dictionary")
    elif payload["jax_smoke"].get("smoke_status") != payload["jax_smoke_status"]:
        failures.append("jax_smoke_status must match jax_smoke.smoke_status")
    if payload["jax_backend_status"] == "available" and payload["jax_smoke_status"] != "passed":
        failures.append("available JAX backend requires passing jax_smoke_status")
    if payload["jax_backend_status"] == "unavailable" and payload["jax_smoke_status"] != "skipped":
        failures.append("unavailable JAX backend requires skipped jax_smoke_status")
    if payload["gpu_readiness_level"] not in {"cpu_only", "nvidia_hardware_detected_torch_cpu", "torch_cuda_runtime_available"}:
        failures.append("gpu_readiness_level must be cpu_only, nvidia_hardware_detected_torch_cpu, or torch_cuda_runtime_available")
    if not isinstance(payload["torch_cuda_available"], bool):
        failures.append("torch_cuda_available must be boolean")
    if not isinstance(payload["torch_cuda_device_count"], int):
        failures.append("torch_cuda_device_count must be integer")
    if payload["torch_cuda_available"] and payload["torch_cuda_device_count"] <= 0:
        failures.append("torch_cuda_available=true requires torch_cuda_device_count > 0")
    if not payload["torch_cuda_available"] and payload["torch_cuda_device_count"] != 0:
        failures.append("torch_cuda_available=false requires torch_cuda_device_count == 0")
    if not isinstance(payload["nvidia_devices"], list):
        failures.append("nvidia_devices must be list")
    if not isinstance(payload["torch_cuda_devices"], list):
        failures.append("torch_cuda_devices must be list")
    if payload["torch_cuda_smoke_status"] not in {"passed", "skipped", "failed"}:
        failures.append("torch_cuda_smoke_status must be passed, skipped, or failed")
    if not isinstance(payload["torch_cuda_smoke"], dict):
        failures.append("torch_cuda_smoke must be dictionary")
    elif payload["torch_cuda_smoke"].get("smoke_status") != payload["torch_cuda_smoke_status"]:
        failures.append("torch_cuda_smoke_status must match torch_cuda_smoke.smoke_status")
    if payload["torch_cuda_available"] and payload["torch_cuda_smoke_status"] != "passed":
        failures.append("torch_cuda_available=true requires passing torch_cuda_smoke_status")
    if not payload["torch_cuda_available"] and payload["torch_cuda_smoke_status"] != "skipped":
        failures.append("torch_cuda_available=false requires skipped torch_cuda_smoke_status")
    for index, device in enumerate(payload.get("torch_cuda_devices", [])):
        for field in ("index", "name", "capability", "total_memory_bytes"):
            if field not in device:
                failures.append(f"torch_cuda_devices[{index}] missing {field}")
    if payload.get("nvidia_devices") and not payload["torch_cuda_available"] and not allow_cpu_torch:
        failures.append(
            "NVIDIA hardware is visible but current PyTorch cannot use CUDA. "
            f"torch={payload['torch_version']}, torch.version.cuda={payload.get('torch_cuda_version')!r}."
        )
    return failures


def doctor_report(*, allow_cpu_torch: bool = False) -> dict[str, Any]:
    payload = backend_capabilities().to_dict()
    failures = validate_backend_capabilities(payload, allow_cpu_torch=allow_cpu_torch)
    return {
        "ok": not failures,
        "kind": "doctor",
        "failures": failures,
        "capabilities": payload,
    }


def doctor_report_markdown(report: dict[str, Any]) -> str:
    capabilities = report["capabilities"]
    lines = [
        "# ARTI Doctor Report",
        "",
        f"Status: `{'PASS' if report['ok'] else 'FAIL'}`",
        f"ARTI version: `{capabilities['arti_version']}`",
        f"Python: `{capabilities['python']}`",
        f"Platform: `{capabilities['platform']}`",
        f"PyTorch version: `{capabilities['torch_version']}`",
        f"PyTorch CUDA version: `{capabilities['torch_cuda_version']}`",
        f"GPU readiness level: `{capabilities['gpu_readiness_level']}`",
        f"CUDA smoke status: `{capabilities['torch_cuda_smoke_status']}`",
        "",
        capabilities["interpretation"],
        "",
        "## Backends",
        "",
        f"- Available: `{capabilities['available_backends']}`",
        f"- Planned: `{capabilities['planned_backends']}`",
        f"- JAX status: `{capabilities['jax_backend_status']}`",
        f"- JAX smoke: `{capabilities['jax_smoke']}`",
        f"- torch.compile available: `{capabilities['torch_compile_available']}`",
        f"- CUDA smoke: `{capabilities['torch_cuda_smoke']}`",
        "",
        "## NVIDIA Devices",
        "",
    ]
    if capabilities["nvidia_devices"]:
        lines.extend(["| Index | Name | Driver | Memory MiB |", "| ---: | --- | --- | ---: |"])
        for device in capabilities["nvidia_devices"]:
            lines.append(f"| {device['index']} | {device['name']} | {device['driver_version']} | {device['total_memory_mib']} |")
    else:
        lines.append("No NVIDIA devices detected through nvidia-smi.")
    lines.extend(["", "## PyTorch CUDA Devices", ""])
    if capabilities["torch_cuda_devices"]:
        lines.extend(["| Index | Name | Capability | Memory GB |", "| ---: | --- | --- | ---: |"])
        for device in capabilities["torch_cuda_devices"]:
            memory_gb = int(device["total_memory_bytes"]) / 1_000_000_000
            lines.append(f"| {device['index']} | {device['name']} | {device['capability']} | {memory_gb:.2f} |")
    else:
        lines.append("No CUDA devices detected by PyTorch.")
    if report["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in report["failures"]:
            lines.append(f"- {failure}")
    lines.append("")
    return "\n".join(lines)


def write_doctor_report(report: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".md":
        target.write_text(doctor_report_markdown(report), encoding="utf-8")
    else:
        target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return target
