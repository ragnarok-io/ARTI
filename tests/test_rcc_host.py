from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import torch
from torch import nn

from arti.alpha import ContextInput, RCCHostAdapter, RecursiveContextCompiler


class Host(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_head = nn.Identity()
        self.first = nn.Linear(8, 8)
        self.middle = nn.Identity()
        self.late = nn.Linear(8, 8)
        self.position_scale = nn.Parameter(torch.tensor(0.25))

    def native_place(self, context, source):
        # A real host would use its own RoPE/relative-position and mask path here.
        position = torch.arange(context.shape[1], device=context.device, dtype=context.dtype)
        return context + self.position_scale * position[None, :, None]

    def forward(self, x):
        original_length = x.shape[1]
        x = self.input_head(x)
        x = self.first(x).tanh()
        x = self.middle(x)
        x = x + x.mean(dim=1, keepdim=True)
        x = self.late(x).tanh()
        return x[:, -original_length:]


def _bind(host):
    def bind(context, args, kwargs):
        source = args[0]
        return (torch.cat((host.native_place(context, source), source), dim=1),), kwargs

    return bind


@pytest.mark.parametrize("consume_path", ["input_head", "middle"])
def test_late_capture_compiles_context_for_independent_consumption(consume_path):
    torch.manual_seed(31)
    host = Host()
    adapter = RCCHostAdapter(
        host,
        RecursiveContextCompiler(8, context_width=2, heads=2, layers=2),
        capture_paths=("middle", "late"),
        consume_path=consume_path,
        bind_context=_bind(host),
    )
    x = torch.randn(2, 3, 8)
    first = adapter(x)
    assert first.output.shape == (2, 3, 8)
    assert first.context.shape == (2, 2, 8)
    second = adapter(x, context=first.context)
    assert second.output.shape == first.output.shape
    assert not torch.allclose(first.output, second.output)
    second.output.square().mean().backward()
    assert host.position_scale.grad is not None
    assert adapter.compiler.context_seed.grad is not None
    adapter.close()
    assert not host.input_head._forward_pre_hooks
    assert not host.middle._forward_hooks
    assert not host.late._forward_hooks


def test_missing_capture_resets_run_without_detaching_adapter():
    class TupleHost(Host):
        def forward(self, x):
            x = self.input_head(x)
            x = self.first(x)
            x = self.middle(x)
            return x

    host = TupleHost()
    adapter = RCCHostAdapter(
        host,
        RecursiveContextCompiler(8, context_width=2, heads=2, layers=2),
        capture_paths=("middle", "late"),
        consume_path="input_head",
        bind_context=_bind(host),
    )
    with pytest.raises(RuntimeError, match="late.*not executed"):
        adapter(torch.randn(1, 2, 8))
    host(torch.randn(1, 2, 8))
    adapter.close()
    assert not host.middle._forward_hooks
    assert not host.late._forward_hooks


def test_capture_callback_preserves_host_padding_mask():
    class MaskedHost(nn.Module):
        def __init__(self):
            super().__init__()
            self.source = nn.Identity()

        def forward(self, x):
            return self.source(x)

    mask = torch.tensor([[True, False]])

    def channel(output):
        return ContextInput(output, mask, output.new_zeros(1, 2))

    adapter = RCCHostAdapter(
        MaskedHost(),
        RecursiveContextCompiler(8, context_width=2, heads=2, layers=2),
        capture_paths=("source",),
        consume_path="source",
        bind_context=lambda context, args, kwargs: (args, kwargs),
        capture_inputs={"source": channel},
    ).eval()
    x = torch.randn(1, 2, 8)
    changed = x.clone()
    changed[:, 1] += 100
    torch.testing.assert_close(adapter(x).context, adapter(changed).context)


def test_custom_compiler_uses_standard_forward_with_heterogeneous_captures():
    class HeterogeneousHost(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_head = nn.Identity()
            self.wide = nn.Linear(8, 8)
            self.narrow = nn.Linear(8, 4)
            self.context_projection = nn.Linear(6, 8)

        def forward(self, x):
            length = x.shape[1]
            x = self.input_head(x)
            wide = self.wide(x)
            narrow = self.narrow(wide)
            return (wide + narrow.mean(dim=-1, keepdim=True))[:, -length:]

    class CustomCompiler(nn.Module):
        def __init__(self):
            super().__init__()
            self.from_wide = nn.Linear(8, 6)
            self.from_narrow = nn.Linear(4, 6)

        def forward(self, captures):
            wide, narrow = captures
            return torch.stack(
                (self.from_wide(wide.mean(dim=1)), self.from_narrow(narrow.mean(dim=1))),
                dim=1,
            )

    host = HeterogeneousHost()
    compiler = CustomCompiler()  # No dim, relation_dim, or compile method.

    def bind(context, args, kwargs):
        return (torch.cat((host.context_projection(context), args[0]), dim=1),), kwargs

    adapter = RCCHostAdapter(
        host,
        compiler,
        capture_paths=("wide", "narrow"),
        consume_path="input_head",
        bind_context=bind,
    )
    first = adapter(torch.randn(2, 3, 8))
    second = adapter(torch.randn(2, 3, 8), context=first.context)
    assert first.context.shape == (2, 2, 6)
    second.output.square().mean().backward()
    assert compiler.from_wide.weight.grad is not None
    assert compiler.from_narrow.weight.grad is not None
    assert "compiler.from_wide.weight" in adapter.state_dict()


def test_mid_layer_binding_prepares_host_mask_and_positions():
    class MaskedMiddle(nn.Module):
        def forward(self, x, *, attention_mask, positions):
            assert x.shape[:2] == attention_mask.shape == positions.shape
            return x + positions.unsqueeze(-1) * 0.01

    class MaskedHost(nn.Module):
        def __init__(self):
            super().__init__()
            self.middle = MaskedMiddle()

        def forward(self, x, *, attention_mask):
            length = x.shape[1]
            positions = attention_mask.long().cumsum(dim=1) - 1
            return self.middle(x, attention_mask=attention_mask, positions=positions)[:, -length:]

    host = MaskedHost()

    def prepare(context, args, kwargs):
        mask = kwargs["attention_mask"]
        return args, {
            **kwargs,
            "attention_mask": torch.cat(
                (torch.ones(context.shape[:2], dtype=mask.dtype, device=mask.device), mask), dim=1
            ),
        }

    adapter = RCCHostAdapter(
        host,
        RecursiveContextCompiler(8, context_width=2, heads=2, layers=2),
        capture_paths=("middle",),
        consume_path="middle",
        bind_context=lambda context, args, kwargs: (
            (torch.cat((context, args[0]), dim=1),),
            kwargs,
        ),
        prepare_host=prepare,
    )
    x = torch.randn(2, 3, 8)
    mask = torch.ones(2, 3, dtype=torch.bool)
    first = adapter(x, attention_mask=mask)
    second = adapter(x, attention_mask=mask, context=first.context)
    assert second.output.shape == first.output.shape
    adapter.close()


def test_overlapping_calls_keep_captures_and_contexts_separate():
    barrier = Barrier(2)

    class Gate(nn.Module):
        def forward(self, x):
            barrier.wait(timeout=5)
            return x

    class ConcurrentHost(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = Gate()

        def forward(self, x):
            return self.gate(x)

    class CaptureCompiler(nn.Module):
        def forward(self, captures):
            return captures[0]

    adapter = RCCHostAdapter(
        ConcurrentHost(),
        CaptureCompiler(),
        capture_paths=("gate",),
        consume_path="gate",
        bind_context=lambda context, args, kwargs: ((args[0] + context,), kwargs),
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                adapter, torch.full((1, 1, 8), float(i)), context=torch.full((1, 1, 8), 10.0 * i)
            )
            for i in (1, 2)
        ]
        results = [future.result() for future in futures]
    for i, result in zip((1, 2), results):
        torch.testing.assert_close(result.output, torch.full((1, 1, 8), 11.0 * i))
        torch.testing.assert_close(result.context, result.output)
    adapter.close()


def test_save_reload_preserves_tied_host_weights_and_selector(tmp_path):
    class TiedHost(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 8)
            self.middle = nn.Identity()
            self.readout = nn.Linear(8, 16, bias=False)
            self.readout.weight = self.embedding.weight

        def forward(self, token_ids):
            length = token_ids.shape[1]
            return self.readout(self.middle(self.embedding(token_ids)))[:, -length:]

    def make_adapter(host):
        return RCCHostAdapter(
            host,
            RecursiveContextCompiler(8, context_width=2, heads=2, layers=2),
            capture_paths=("middle",),
            consume_path="middle",
            bind_context=lambda context, args, kwargs: (
                (torch.cat((context, args[0]), dim=1),),
                kwargs,
            ),
            capture_inputs={"middle": nn.Linear(8, 8)},
        ).eval()

    torch.manual_seed(12)
    original = make_adapter(TiedHost())
    token_ids = torch.tensor([[1, 2, 3]])
    context = torch.randn(1, 2, 8)
    expected = original(token_ids, context=context)
    assert "capture_modules.0.weight" in original.state_dict()
    original.save(tmp_path, integration_ref="test/tied-host@1")
    original.close()

    with pytest.raises(ValueError, match="integration does not match"):
        RCCHostAdapter.load(
            tmp_path,
            TiedHost(),
            RecursiveContextCompiler(8, context_width=2, heads=2, layers=2),
            bind_context=lambda context, args, kwargs: (args, kwargs),
            capture_inputs={"middle": nn.Linear(8, 8)},
            integration_ref="test/other-host@1",
        )

    host = TiedHost()
    restored = RCCHostAdapter.load(
        tmp_path,
        host,
        RecursiveContextCompiler(8, context_width=2, heads=2, layers=2),
        bind_context=lambda context, args, kwargs: (
            (torch.cat((context, args[0]), dim=1),),
            kwargs,
        ),
        capture_inputs={"middle": nn.Linear(8, 8)},
        integration_ref="test/tied-host@1",
    ).eval()
    actual = restored(token_ids, context=context)
    torch.testing.assert_close(actual.output, expected.output)
    torch.testing.assert_close(actual.context, expected.context)
    assert host.readout.weight is host.embedding.weight
    restored.close()
