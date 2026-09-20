# GPU Validation

ARTI is PyTorch-first and should run on CPU and CUDA devices when a CUDA-enabled PyTorch build is installed.

GPU-readiness has three distinct levels:

| Level | Meaning | Evidence |
| --- | --- | --- |
| `cpu_only` | No NVIDIA CUDA hardware is visible to the local process. | CPU tests only. |
| `nvidia_hardware_detected_torch_cpu` | NVIDIA hardware is visible through `nvidia-smi`, but PyTorch is CPU-only or cannot access CUDA. | Hardware report plus `torch.cuda.is_available() == False`. |
| `torch_cuda_runtime_available` | PyTorch can allocate CUDA tensors. | CUDA runtime smoke tests and CUDA scaling profile. |

Run the backend capability report first:

```bash
uv run --extra dev arti doctor --allow-cpu-torch --output benchmarks/results/doctor.md
uv run --extra dev python benchmarks/report_backend_capabilities.py
uv run --extra dev python benchmarks/verify_backend_capabilities.py
```

`arti doctor` is the installed-package entrypoint for the same backend readiness model used by the benchmark reports. It writes JSON when `--output` ends in `.json` and Markdown when it ends in `.md`. Omit `--allow-cpu-torch` on machines where visible NVIDIA hardware must be backed by a CUDA-enabled PyTorch runtime.
The doctor payload includes `torch_cuda_smoke_status` and `torch_cuda_smoke`, so CI can distinguish a CUDA-enabled PyTorch runtime that actually passed a tiny allocation/compute smoke check from a CPU-only or failed runtime.

On a machine with NVIDIA hardware, require PyTorch CUDA runtime before claiming GPU-ready execution:

```bash
uv run --extra dev python benchmarks/verify_torch_cuda_runtime.py
uv run --extra dev arti doctor --require-cuda-smoke
```

For an installed-package runtime smoke check, call:

```python
import arti

report = arti.cuda_smoke_report()
print(report["smoke_status"])
```

`cuda_smoke_report()` allocates a tiny tensor and runs a tiny CUDA compute step when PyTorch CUDA is available. A passing smoke report supports local runtime readiness only; CUDA scaling claims still require the scaling profile and evidence packet below.

CI or CPU-only environments may run the same verifier in reporting mode:

```bash
uv run --extra dev python benchmarks/verify_torch_cuda_runtime.py --allow-cpu-torch
```

If `nvidia-smi` shows a GPU but the report says `nvidia_hardware_detected_torch_cpu`, install a CUDA-enabled PyTorch build for your platform. Use the official PyTorch install selector and verify with:

```python
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
```

PyTorch's official local installation page says to choose the OS, package manager, Python language, and compute platform in the selector, then verify GPU access with `torch.cuda.is_available()`.

For example, after switching from a CPU-only wheel to a CUDA wheel, the report should change from:

```text
gpu_readiness_level = nvidia_hardware_detected_torch_cpu
```

to:

```text
gpu_readiness_level = torch_cuda_runtime_available
```

Run the device smoke test:

```bash
uv run --extra dev pytest tests/test_device.py tests/test_serialization.py tests/test_torch_backend_runtime.py
```

`tests/test_torch_backend_runtime.py` includes:

- CPU forward/backward coverage through `arti.torch`.
- CUDA autocast smoke coverage when CUDA is available.
- `torch.compile(..., backend="eager")` smoke coverage when `torch.compile` exists.

Run the scaling profile:

```bash
uv run --extra dev python benchmarks/profile_scaling.py --device cuda
```

Run CUDA evidence packet generation and verification when a CUDA device is available:

```bash
uv run --extra dev python benchmarks/verify_torch_cuda_runtime.py
uv run --extra dev python benchmarks/profile_scaling.py --device cuda
uv run --extra dev python benchmarks/build_cuda_evidence_packet.py
uv run --extra dev python benchmarks/verify_cuda_evidence_packet.py
```

CPU-only runs are useful for correctness but do not prove GPU memory scaling.
