"""PyTorch CUDA runtime helpers for ARTI."""

from __future__ import annotations

from typing import Any

import torch


def cuda_runtime_available() -> bool:
    """Return whether the current PyTorch runtime can allocate CUDA tensors."""

    return bool(torch.cuda.is_available())


def cuda_device_report() -> dict[str, Any]:
    """Return a compact report of PyTorch-visible CUDA devices."""

    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": int(props.total_memory),
                }
            )
    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": devices,
    }


def cuda_smoke_report(size: int = 16) -> dict[str, Any]:
    """Run a tiny CUDA allocation/compute smoke check and return its report.

    The function is intentionally lightweight so it can be used by release
    checks and application startup diagnostics. CPU-only environments return a
    structured skipped report instead of raising.
    """

    if size <= 0:
        raise ValueError("ARTI CUDA smoke size must be positive")
    report = cuda_device_report()
    if not torch.cuda.is_available():
        return {
            **report,
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
            **report,
            "smoke_status": "failed",
            "allocation_ok": False,
            "compute_ok": False,
            "device": str(device),
            "reason": str(exc),
        }
    return {
        **report,
        "smoke_status": "passed",
        "allocation_ok": True,
        "compute_ok": total == float(size),
        "device": str(device),
        "smoke_tensor_shape": [size, size],
        "smoke_sum": total,
    }


def require_cuda() -> torch.device:
    """Return the current CUDA device or raise a clear runtime error."""

    if not torch.cuda.is_available():
        raise RuntimeError("ARTI CUDA requires a CUDA-enabled PyTorch runtime. Install torch with CUDA support first.")
    return torch.device("cuda", torch.cuda.current_device())


__all__ = ["cuda_device_report", "cuda_runtime_available", "cuda_smoke_report", "require_cuda"]
