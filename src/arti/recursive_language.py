"""Native recursive-context language model and legal prefix construction."""

from __future__ import annotations

from typing import Sequence
from pathlib import Path
import json

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .recursive_context import ContextEvaluator, ContextMemo, ContextNode, ContextPlan


def prefix_plan(
    tokens: Sequence[int],
    stops: Sequence[int],
    *,
    chunk_size: int = 16,
    depth: int = 1,
    fanout: int = 4,
    relation_dim: int = 2,
    context_width: int | None = None,
    compile_depth: int = 1,
) -> ContextPlan:
    """Build roots using only tokens strictly before each prediction offset.

    Depth zero is fully expanded. At greater depths, older chunks become child
    nodes and the last (possibly full) chunk stays lexical. Chunk boundaries are
    fixed from source origin, so extending the frontier preserves settled nodes.
    Relations encode source starts; physical channel order is not chronology.
    """
    if any(type(v) is not int or v < 1 for v in (chunk_size, fanout, relation_dim)):
        raise ValueError("chunk_size, fanout and relation_dim must be positive integers")
    if type(depth) is not int or depth < 0:
        raise ValueError("depth must be a nonnegative integer")
    if context_width is not None and (type(context_width) is not int or context_width < 1):
        raise ValueError("context_width must be a positive integer")
    if type(compile_depth) is not int or compile_depth < 1:
        raise ValueError("compile_depth must be a positive integer")
    tokens, stops = tuple(tokens), tuple(stops)
    if not stops or any(type(s) is not int or not 1 <= s <= len(tokens) for s in stops):
        raise ValueError("prediction offsets must identify nonempty known prefixes")
    nodes = {}

    def make(start: int, stop: int, remaining: int) -> str:
        name = f"{start}:{stop}:{remaining}"
        if name in nodes:
            return name
        children = []
        positions = []
        frontier = start
        if remaining:
            block = chunk_size * fanout ** (remaining - 1)
            while frontier + block < stop:
                children.append(make(frontier, frontier + block, remaining - 1))
                positions.append(frontier)
                frontier += block
        positions.append(frontier)
        placements = tuple((float(p),) + (0.0,) * (relation_dim - 1) for p in positions)
        nodes[name] = ContextNode(
            name,
            "tokens",
            frontier,
            stop,
            tuple(children),
            placements,
            context_width,
            compile_depth,
        )
        return name

    roots = [make(0, stop, depth) for stop in stops]
    return ContextPlan(
        {"tokens": tokens}, tuple(nodes.values()), roots, prefix_limits={"tokens": max(stops)}
    )


class RecursiveLanguageModel(nn.Module):
    """RCC C enters only the input head; the language head consumes normal Y."""

    _component_reference = "arti/recursive-language@1"

    def __init__(self, evaluator: ContextEvaluator, language_head: nn.Module) -> None:
        super().__init__()
        self.evaluator = evaluator
        self.language_head = language_head

    def save(self, directory: str | Path) -> None:
        """Save registered F/embedding/D configuration and exact tensor state."""
        from safetensors.torch import save_file
        from .component_registry import component_spec

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        parts = {
            "front": self.evaluator.front,
            "lexical": self.evaluator.lexical,
            "language_head": self.language_head,
        }
        config = {
            name: {
                "reference": component_spec(part).reference,
                "config": dict(component_spec(part).config),
            }
            for name, part in parts.items()
        }
        state = {
            name: value.detach().cpu().contiguous() for name, value in self.state_dict().items()
        }
        save_file(state, str(path / "model.safetensors"))
        (path / "config.json").write_text(
            json.dumps({"schema": "arti/recursive-language@1", "components": config}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls, directory: str | Path, *, device: str | torch.device = "cpu"
    ) -> RecursiveLanguageModel:
        """Reconstruct registered components without pickled executable objects."""
        from safetensors.torch import load_file
        from .component_registry import resolve_component

        path = Path(directory)
        payload = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if payload["schema"] != "arti/recursive-language@1":
            raise ValueError("unsupported recursive language model schema")
        parts = {
            name: resolve_component(spec["reference"], **spec["config"])
            for name, spec in payload["components"].items()
        }
        model = cls(ContextEvaluator(parts["front"], parts["lexical"]), parts["language_head"])
        # assign preserves saved per-tensor dtypes rather than silently casting
        # a bf16 or float64 asset into the constructor's default float32.
        model.load_state_dict(load_file(str(path / "model.safetensors")), assign=True)
        return model.to(device).eval()

    def forward(
        self, plan: ContextPlan, *, recompute: bool = False, memo: ContextMemo | None = None
    ) -> Tensor:
        result = self.evaluator(plan, recompute=recompute, memo=memo, root_output="downstream")
        return self.language_head(torch.cat(result.roots, dim=0))

    def token_loss(
        self,
        tokens: Sequence[int],
        *,
        stops: Sequence[int] | None = None,
        chunk_size: int = 16,
        depth: int = 1,
        fanout: int = 4,
        context_width: int | None = None,
        compile_depth: int = 1,
        recompute: bool = False,
        substitution_weight: float = 0.0,
    ) -> Tensor:
        """True next-token CE; optional detached expanded-input distribution KL."""
        return self.batch_token_loss(
            [tokens],
            stops=None if stops is None else [stops],
            chunk_size=chunk_size,
            depth=depth,
            fanout=fanout,
            context_width=context_width,
            compile_depth=compile_depth,
            recompute=recompute,
            substitution_weight=substitution_weight,
        )

    def batch_token_loss(
        self,
        sequences: Sequence[Sequence[int]],
        *,
        stops: Sequence[Sequence[int]] | None = None,
        chunk_size: int = 16,
        depth: int = 1,
        fanout: int = 4,
        context_width: int | None = None,
        compile_depth: int = 1,
        recompute: bool = False,
        substitution_weight: float = 0.0,
    ) -> Tensor:
        """Mean loss per supervised token across independent source sequences.

        All ready nodes share physical batches, but source identities and root
        contexts never mix. For long sequences supply bounded prediction offsets
        and accumulate sums weighted by their token counts in the trainer.
        """
        sequences = tuple(tuple(seq) for seq in sequences)
        if not sequences:
            raise ValueError("at least one training sequence is required")
        offsets = (
            tuple(tuple(range(1, len(seq))) for seq in sequences)
            if stops is None
            else tuple(tuple(s) for s in stops)
        )
        if len(offsets) != len(sequences):
            raise ValueError("one prediction-offset sequence is required per source")
        for seq, selected in zip(sequences, offsets):
            if not selected or any(type(s) is not int or not 1 <= s < len(seq) for s in selected):
                raise ValueError("each supervised prefix must have a next-token target")
        if substitution_weight < 0:
            raise ValueError("substitution_weight must be nonnegative")
        kwargs = dict(
            chunk_size=chunk_size,
            fanout=fanout,
            relation_dim=self.evaluator.front.relation_dim,
            context_width=context_width,
            compile_depth=compile_depth,
        )

        def combined(selected_depth: int) -> ContextPlan:
            sources, nodes, roots = {}, [], []
            for index, (seq, selected) in enumerate(zip(sequences, offsets)):
                part = prefix_plan(seq, selected, depth=selected_depth, **kwargs)
                source = str(index)
                sources[source] = part.sources["tokens"]
                for node in part.nodes.values():
                    nodes.append(
                        ContextNode(
                            f"{index}/{node.name}",
                            source,
                            node.start,
                            node.stop,
                            tuple(f"{index}/{child}" for child in node.children),
                            node.placements,
                            node.context_width,
                            node.compile_depth,
                        )
                    )
                roots.extend(f"{index}/{root}" for root in part.roots)
            return ContextPlan(sources, nodes, roots)

        logits = self(combined(depth), recompute=recompute)
        targets = torch.tensor(
            [seq[s] for seq, selected in zip(sequences, offsets) for s in selected],
            device=logits.device,
        )
        loss = F.cross_entropy(logits.float(), targets)
        if substitution_weight:
            with torch.no_grad():
                expanded = self(combined(0))
            loss = loss + substitution_weight * F.kl_div(
                logits.float().log_softmax(-1), expanded.float().softmax(-1), reduction="batchmean"
            )
        return loss

    @torch.no_grad()
    def generate(
        self,
        tokens: Sequence[int],
        *,
        max_new_tokens: int,
        chunk_size: int = 16,
        depth: int = 1,
        fanout: int = 4,
        context_width: int | None = None,
        compile_depth: int = 1,
        eos_token_id: int | None = None,
        memo: ContextMemo | None = None,
    ) -> tuple[int, ...]:
        """Greedy reference generation; no stale recursive-context reuse."""
        if self.training:
            raise ValueError("generation requires eval()")
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        output = list(tokens)
        if not output:
            raise ValueError("generation requires a nonempty lexical prefix")
        for _ in range(max_new_tokens):
            plan = prefix_plan(
                output,
                [len(output)],
                chunk_size=chunk_size,
                depth=depth,
                fanout=fanout,
                context_width=context_width,
                compile_depth=compile_depth,
                relation_dim=self.evaluator.front.relation_dim,
            )
            token = int(self(plan, memo=memo).argmax(-1).item())
            output.append(token)
            if token == eos_token_id:
                break
        return tuple(output)
