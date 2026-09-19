"""Use RCC with capture and consumption points in an existing PyTorch model."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import weakref

from torch import Tensor, nn


@dataclass(frozen=True)
class RCCHostResult:
    output: Any
    context: Tensor


@dataclass
class _ActiveRun:
    captures: dict[str, list[Any]]
    context: Tensor | None


class RCCHostAdapter(nn.Module):
    """Compile C from host layers and consume a previous C at an independent layer.

    ``bind_context`` owns the host's position and attention-mask conventions. It
    receives the input to ``consume_path`` and returns replacement args/kwargs.
    The adapter never assigns an absolute position to a saved C.
    """

    def __init__(
        self,
        host: nn.Module,
        compiler: nn.Module,
        *,
        capture_paths: Sequence[str],
        consume_path: str,
        bind_context: Callable[
            [Tensor, tuple[Any, ...], dict[str, Any]],
            tuple[tuple[Any, ...], dict[str, Any]],
        ],
        capture_inputs: Mapping[str, Callable[[Any], Any]] | None = None,
        prepare_host: Callable[
            [Tensor, tuple[Any, ...], dict[str, Any]],
            tuple[tuple[Any, ...], dict[str, Any]],
        ]
        | None = None,
    ) -> None:
        super().__init__()
        if not capture_paths or len(set(capture_paths)) != len(capture_paths):
            raise ValueError("capture_paths must contain distinct host module paths")
        if not callable(bind_context):
            raise TypeError("bind_context must be callable")
        selectors = dict(capture_inputs or {})
        if set(selectors) - set(capture_paths):
            raise ValueError("capture_inputs contains an unknown capture path")
        # Resolve before installing hooks so an invalid declaration never runs the host.
        for path in (*capture_paths, consume_path):
            host.get_submodule(path)
        self.host = host
        self.compiler = compiler
        self.capture_paths = tuple(capture_paths)
        self.consume_path = consume_path
        self.bind_context = bind_context
        self.capture_inputs = selectors
        self.capture_modules = nn.ModuleDict(
            {
                str(index): selectors[path]
                for index, path in enumerate(self.capture_paths)
                if isinstance(selectors.get(path), nn.Module)
            }
        )
        self.prepare_host = prepare_host
        self._active_run: ContextVar[_ActiveRun | None] = ContextVar(
            f"rcc_host_run_{id(self)}", default=None
        )
        self._hook_handles = []
        owner = weakref.ref(self)

        for path in self.capture_paths:

            def capture(_module: nn.Module, _inputs: tuple, output: Any, *, path: str = path):
                adapter = owner()
                if adapter is None:
                    return
                run = adapter._active_run.get()
                if run is not None:
                    selector = adapter.capture_inputs.get(path)
                    run.captures[path].append(selector(output) if selector else output)

            self._hook_handles.append(self.host.get_submodule(path).register_forward_hook(capture))

        def consume(_module: nn.Module, inputs: tuple, keyword: dict) -> tuple | None:
            adapter = owner()
            if adapter is None:
                return None
            run = adapter._active_run.get()
            if run is None or run.context is None:
                return None
            return adapter.bind_context(run.context, inputs, keyword)

        self._hook_handles.append(
            self.host.get_submodule(self.consume_path).register_forward_pre_hook(
                consume, with_kwargs=True
            )
        )

    def close(self) -> None:
        """Detach the persistent hooks when the adapter is no longer used."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def save(self, directory: str | Path, *, integration_ref: str) -> None:
        """Save module weights and the host-specific integration contract."""
        from safetensors.torch import save_model

        if not integration_ref:
            raise ValueError("integration_ref must identify the host binding")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": "arti/rcc-host-adapter@1",
            "capture_paths": list(self.capture_paths),
            "consume_path": self.consume_path,
            "capture_inputs": sorted(self.capture_inputs),
            "prepare_host": self.prepare_host is not None,
            "integration_ref": integration_ref,
        }
        save_model(self, str(path / "model.safetensors"))
        (path / "config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        directory: str | Path,
        host: nn.Module,
        compiler: nn.Module,
        *,
        bind_context: Callable,
        integration_ref: str,
        capture_inputs: Mapping[str, Callable[[Any], Any]] | None = None,
        prepare_host: Callable | None = None,
    ) -> RCCHostAdapter:
        """Reload into caller-constructed modules with the same binding contract."""
        from safetensors.torch import load_model

        path = Path(directory)
        manifest = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if manifest["schema"] != "arti/rcc-host-adapter@1":
            raise ValueError("unsupported RCC host adapter schema")
        if (
            manifest["integration_ref"] != integration_ref
            or manifest["capture_inputs"] != sorted(capture_inputs or {})
            or manifest["prepare_host"] != (prepare_host is not None)
        ):
            raise ValueError("RCC host integration does not match the saved configuration")
        adapter = cls(
            host,
            compiler,
            capture_paths=manifest["capture_paths"],
            consume_path=manifest["consume_path"],
            bind_context=bind_context,
            capture_inputs=capture_inputs,
            prepare_host=prepare_host,
        )
        try:
            load_model(adapter, str(path / "model.safetensors"), strict=True)
        except Exception:
            adapter.close()
            raise
        return adapter

    def forward(
        self,
        *args: Any,
        context: Tensor | None = None,
        **kwargs: Any,
    ) -> RCCHostResult:
        if not self._hook_handles:
            raise RuntimeError("RCC host adapter is closed")
        run = _ActiveRun({path: [] for path in self.capture_paths}, context)
        token = self._active_run.set(run)
        try:
            if context is not None and self.prepare_host is not None:
                args, kwargs = self.prepare_host(context, args, kwargs)
            output = self.host(*args, **kwargs)
        finally:
            self._active_run.reset(token)

        inputs = []
        for path in self.capture_paths:
            if not run.captures[path]:
                raise RuntimeError(f"capture path {path!r} was not executed")
            inputs.extend(run.captures[path])
        compiled = self.compiler(tuple(inputs))
        if not isinstance(compiled, Tensor) or compiled.ndim != 3:
            raise ValueError("compiler must return C as a [B,M,D] Tensor")
        return RCCHostResult(output=output, context=compiled)
