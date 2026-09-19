"""Host-side immutable input plans for recursive context compilation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .recursive_context_compiler import ContextInput, LexicalEmbedding, RecursiveContextCompiler


@dataclass(frozen=True)
class ContextNode:
    """Older compiled children followed by a nonempty local lexical frontier.

    Placements describe children and frontier, in that order, at this consumer.
    They never mutate a child's intrinsic representation.
    """

    name: str
    source: str
    start: int
    stop: int
    children: tuple[str, ...] = ()
    placements: tuple[tuple[float, ...], ...] = ()
    context_width: int | None = None
    compile_depth: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "children", tuple(self.children))
        object.__setattr__(self, "placements", tuple(tuple(p) for p in self.placements))
        if not self.name or not self.source:
            raise ValueError("node name and source must be nonempty")
        if any(type(v) is not int for v in (self.start, self.stop)):
            raise ValueError("source interval must use integer offsets")
        if not 0 <= self.start < self.stop:
            raise ValueError("each node needs a nonempty lexical frontier")
        if len(self.placements) != len(self.children) + 1:
            raise ValueError("one placement is required per child and lexical frontier")
        sizes = {len(p) for p in self.placements}
        if len(sizes) != 1 or 0 in sizes:
            raise ValueError("placements must have one consistent nonzero dimension")
        if any(not math.isfinite(v) for p in self.placements for v in p):
            raise ValueError("placements must be finite")
        if self.context_width is not None and (
            type(self.context_width) is not int or self.context_width < 1
        ):
            raise ValueError("context_width must be a positive integer")
        if type(self.compile_depth) is not int or self.compile_depth < 1:
            raise ValueError("compile_depth must be a positive integer")


@dataclass(frozen=True, init=False)
class ContextPlan:
    """Validated DAG over explicitly supplied, already-known source prefixes.

    Each source is a prefix snapshot, not a complete future-bearing document.
    Overlap is forbidden within a node's represented context; independent roots
    may share children. Only reachable nodes participate in execution.
    """

    sources: Mapping[str, tuple[int, ...]]
    nodes: Mapping[str, ContextNode]
    roots: tuple[str, ...]
    coverage: Mapping[str, tuple[tuple[str, int, int], ...]]
    keys: Mapping[str, str]
    levels: tuple[tuple[str, ...], ...]

    def __init__(
        self,
        sources: Mapping[str, Sequence[int]],
        nodes: Sequence[ContextNode],
        roots: Sequence[str],
        *,
        prefix_limits: Mapping[str, int] | None = None,
    ) -> None:
        source_map = {name: tuple(tokens) for name, tokens in sources.items()}
        if prefix_limits is not None:
            if set(prefix_limits) != set(source_map):
                raise ValueError("prefix limits must cover exactly the supplied sources")
            for name, limit in prefix_limits.items():
                if type(limit) is not int or not 0 <= limit <= len(source_map[name]):
                    raise ValueError("invalid source prefix limit")
                source_map[name] = source_map[name][:limit]
        for tokens in source_map.values():
            if any(type(token) is not int or token < 0 for token in tokens):
                raise ValueError("source tokens must be nonnegative integers")
        node_map = {node.name: node for node in nodes}
        if len(node_map) != len(nodes):
            raise ValueError("duplicate node names")
        roots = tuple(roots)
        if not roots:
            raise ValueError("at least one root is required")
        coverage: dict[str, tuple[tuple[str, int, int], ...]] = {}
        keys: dict[str, str] = {}
        depth: dict[str, int] = {}
        visiting: set[str] = set()

        def visit(name: str) -> None:
            if name in coverage:
                return
            if name in visiting:
                raise ValueError("context graph contains a cycle")
            if name not in node_map:
                raise ValueError(f"unknown context node: {name}")
            visiting.add(name)
            node = node_map[name]
            if node.source not in source_map or node.stop > len(source_map[node.source]):
                raise ValueError("lexical frontier exceeds supplied source prefix")
            intervals = []
            for child in node.children:
                visit(child)
                for source, start, stop in coverage[child]:
                    if source == node.source and stop > node.start:
                        raise ValueError("child context must precede the local frontier")
                    intervals.append((source, start, stop))
            intervals.append((node.source, node.start, node.stop))
            intervals.sort()
            for left, right in zip(intervals, intervals[1:]):
                if left[0] == right[0] and left[2] > right[1]:
                    raise ValueError("source coverage overlaps within a context node")
            coverage[name] = tuple(intervals)
            depth[name] = 1 + max((depth[c] for c in node.children), default=-1)
            payload = {
                "schema": "arti/context-plan@1",
                "source": node.source,
                "interval": (node.start, node.stop),
                "tokens": source_map[node.source][node.start : node.stop],
                "children": [keys[c] for c in node.children],
                "placements": node.placements,
                "context_width": node.context_width,
                "compile_depth": node.compile_depth,
            }
            keys[name] = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            visiting.remove(name)

        for root in roots:
            visit(root)
        object.__setattr__(self, "sources", MappingProxyType(source_map))
        object.__setattr__(
            self, "nodes", MappingProxyType({name: node_map[name] for name in coverage})
        )
        object.__setattr__(self, "roots", roots)
        object.__setattr__(self, "coverage", MappingProxyType(coverage))
        object.__setattr__(self, "keys", MappingProxyType(keys))
        object.__setattr__(
            self,
            "levels",
            tuple(
                tuple(name for name in coverage if depth[name] == level)
                for level in range(max(depth.values()) + 1)
            ),
        )


@dataclass(frozen=True)
class ContextBatchWork:
    """Logical forward work, not measured launches, latency or backward work."""

    nodes: tuple[str, ...]
    level: int
    valid_input_positions: int
    padded_input_positions: int
    output_positions: int
    activation_recomputation: bool


@dataclass(frozen=True)
class ContextResult:
    roots: tuple[Tensor, ...]
    values: Mapping[str, Tensor]
    front_calls: int
    computed_nodes: int
    cache_hits: int = 0
    batches: tuple[ContextBatchWork, ...] = ()


class ContextMemo:
    """Local inference-only memo; ordinary parameter updates invalidate entries.

    Tensor version counters cover optimizer/copy_/load_state_dict changes. Direct
    .data writes bypass PyTorch's tracking and require an explicit clear().
    Model content fingerprints for portable persistence are a separate boundary.
    """

    def __init__(self) -> None:
        self._stamp: tuple | None = None
        self._values: dict[str, Tensor] = {}

    def clear(self) -> None:
        self._stamp = None
        self._values.clear()

    def bind(self, model: nn.Module, device: torch.device) -> None:
        stamp = (
            id(model),
            tuple(
                (name, id(p), p._version, p.device, p.dtype, tuple(p.shape))
                for name, p in (*model.named_parameters(), *model.named_buffers())
            ),
            repr(model.front.config()),
            model.front._component_reference,
            repr(model.lexical.config()),
            torch.is_autocast_enabled(device.type),
            torch.get_autocast_dtype(device.type),
            torch.get_float32_matmul_precision(),
            torch.backends.cuda.matmul.allow_tf32,
        )
        if stamp != self._stamp:
            self._values.clear()
            self._stamp = stamp

    def get(self, key: str) -> Tensor | None:
        value = self._values.get(key)
        return None if value is None else value.clone()

    def put(self, key: str, value: Tensor) -> None:
        self._values[key] = value.detach().clone()

    def __len__(self) -> int:
        return len(self._values)

    @staticmethod
    def _portable_identity(model: ContextEvaluator) -> dict:
        from .recall_experts import canonical_tensor_state_sha256

        device = model.lexical.embedding.weight.device
        return {
            "schema": "arti/context-memo@1",
            "front": model.front.config(),
            "front_reference": model.front._component_reference,
            "lexical": model.lexical.config(),
            "state": canonical_tensor_state_sha256(model.state_dict()),
            "torch": torch.__version__,
            "device": str(device),
            "autocast": torch.is_autocast_enabled(device.type),
            "autocast_dtype": str(torch.get_autocast_dtype(device.type)),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "tf32": torch.backends.cuda.matmul.allow_tf32,
        }

    def save(self, path: str | Path, model: ContextEvaluator) -> None:
        """Persist only entries valid for this exact generating model/mode."""
        from safetensors.torch import save_file

        if torch.is_grad_enabled() or any(m.training for m in model.modules()):
            raise ValueError("memo persistence requires eval() and disabled gradients")
        self.bind(model, model.lexical.embedding.weight.device)
        identity = json.dumps(self._portable_identity(model), sort_keys=True)
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in self._values.items()},
            str(path),
            metadata={"identity": identity},
        )

    @classmethod
    def load(cls, path: str | Path, model: ContextEvaluator) -> ContextMemo:
        """Restore a memo only when model content and numerical policy match."""
        from safetensors import safe_open

        if torch.is_grad_enabled() or any(m.training for m in model.modules()):
            raise ValueError("memo persistence requires eval() and disabled gradients")
        identity = cls._portable_identity(model)
        memo = cls()
        device = model.lexical.embedding.weight.device
        memo.bind(model, device)
        with safe_open(str(path), framework="pt", device="cpu") as file:
            metadata = file.metadata() or {}
            if json.loads(metadata.get("identity", "null")) != identity:
                raise ValueError("memo generating model or numerical policy differs")
            for key in file.keys():
                value = file.get_tensor(key)
                if (
                    value.ndim != 3
                    or value.shape[0] != 1
                    or not 1 <= value.shape[1] <= model.front.width
                    or value.shape[2] != model.front.dim
                    or not value.is_floating_point()
                ):
                    raise ValueError("memo value violates context compiler output schema")
                memo._values[key] = value.to(device)
        return memo


class ContextEvaluator(nn.Module):
    """Evaluate ready siblings together while sharing live child activations.

    One plan node represents one sample. Independent roots and their ready
    siblings form the batch dimension. No tensor values drive host scheduling.
    A fresh activation map is created on every call, never reused after backward.
    """

    def __init__(self, front: RecursiveContextCompiler, lexical: LexicalEmbedding) -> None:
        super().__init__()
        if front.dim != lexical.dim:
            raise ValueError("front and lexical embedding dimensions must agree")
        self.front = front
        self.lexical = lexical

    def forward(
        self,
        plan: ContextPlan,
        *,
        serial: bool = False,
        recompute: bool = False,
        memo: ContextMemo | None = None,
        root_output: str = "context",
    ) -> ContextResult:
        if root_output not in {"context", "downstream"}:
            raise ValueError("root_output must be context or downstream")
        if root_output == "downstream":
            children = {child for node in plan.nodes.values() for child in node.children}
            if children.intersection(plan.roots):
                raise ValueError("a downstream root cannot also be consumed as a recursive child")
        values: dict[str, Tensor] = {}
        calls = 0
        parameter = self.lexical.embedding.weight
        device = parameter.device
        hits = 0
        computed = 0
        work = []
        if memo is not None:
            if torch.is_grad_enabled() or any(module.training for module in self.modules()):
                raise ValueError("inference memo requires eval() and disabled gradients")
            memo.bind(self, device)
        for level_index, level in enumerate(plan.levels):
            buckets: dict[tuple[tuple[int, ...], int, int, int, str], list[str]] = {}
            for name in level:
                output_mode = (
                    "downstream"
                    if root_output == "downstream" and name in plan.roots
                    else "context"
                )
                if memo is not None and output_mode == "context":
                    cached = memo.get(plan.keys[name])
                    if cached is not None:
                        values[name] = cached
                        hits += 1
                        continue
                node = plan.nodes[name]
                if len(node.placements[0]) != self.front.relation_dim:
                    raise ValueError("node placement dimension differs from front")
                width = node.context_width or self.front.width
                if width > self.front.width:
                    raise ValueError("node context_width exceeds configured capacity")
                length = node.stop - node.start
                # Power-of-two padding bounds bucket waste without specializing
                # an invocation for every individual lexical length.
                bucket = (
                    tuple(values[child].shape[1] for child in node.children),
                    1 << (length - 1).bit_length(),
                    width,
                    node.compile_depth if output_mode == "context" else 1,
                    output_mode,
                )
                buckets.setdefault(bucket, []).append(name)
            for (child_widths, length, width, compile_depth, output_mode), names in buckets.items():
                groups = [[name] for name in names] if serial else [names]
                for group in groups:
                    nodes = [plan.nodes[name] for name in group]
                    ids = []
                    masks = []
                    for node in nodes:
                        tokens = plan.sources[node.source][node.start : node.stop]
                        ids.append(tokens + (0,) * (length - len(tokens)))
                        masks.append((True,) * len(tokens) + (False,) * (length - len(tokens)))
                    embedded = self.lexical(torch.tensor(ids, device=device, dtype=torch.long))
                    valid = torch.tensor(masks, device=device, dtype=torch.bool)
                    coordinate_dtype = (
                        torch.float64 if embedded.dtype == torch.float64 else torch.float32
                    )
                    channels = []
                    for index in range(len(nodes[0].children)):
                        child = torch.cat([values[node.children[index]] for node in nodes])
                        # AMP may produce reduced-precision child outputs. The
                        # consuming view follows the lexical input dtype without
                        # severing the child's autograd connection.
                        child = child.to(embedded.dtype)
                        channels.append(
                            ContextInput(
                                child,
                                torch.ones(child.shape[:2], device=device, dtype=torch.bool),
                                torch.tensor(
                                    [node.placements[index] for node in nodes],
                                    device=device,
                                    dtype=coordinate_dtype,
                                ),
                            )
                        )
                    channels.append(
                        ContextInput(
                            embedded,
                            valid,
                            torch.tensor(
                                [node.placements[-1] for node in nodes],
                                device=device,
                                dtype=coordinate_dtype,
                            ),
                        )
                    )
                    if recompute and torch.is_grad_enabled():
                        flat = tuple(t for c in channels for t in (c.value, c.valid, c.placement))

                        def fuse(*args: Tensor) -> Tensor:
                            inputs = tuple(
                                ContextInput(*args[i : i + 3]) for i in range(0, len(args), 3)
                            )
                            if output_mode == "context":
                                return self.front.compile(
                                    inputs, depth=compile_depth, context_width=width
                                )
                            return self.front(inputs, output="downstream", context_width=width)

                        output = checkpoint(fuse, *flat, use_reentrant=False)
                    else:
                        if output_mode == "context":
                            output = self.front.compile(
                                channels, depth=compile_depth, context_width=width
                            )
                        else:
                            output = self.front(channels, output="downstream", context_width=width)
                    for name, value in zip(group, output.split(1)):
                        values[name] = value
                        if memo is not None and output_mode == "context":
                            memo.put(plan.keys[name], value)
                    computed += len(group)
                    child_positions = sum(child_widths)
                    output_positions = (
                        len(group) if output_mode == "downstream" else len(group) * width
                    )
                    work.append(
                        ContextBatchWork(
                            tuple(group),
                            level_index,
                            sum(node.stop - node.start + child_positions for node in nodes),
                            len(group) * (length + child_positions + 1),
                            output_positions,
                            recompute and torch.is_grad_enabled(),
                        )
                    )
                    calls += 1
        return ContextResult(
            tuple(values[name] for name in plan.roots),
            MappingProxyType(values),
            calls,
            computed,
            hits,
            tuple(work),
        )
