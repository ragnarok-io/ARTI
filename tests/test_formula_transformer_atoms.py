from __future__ import annotations

import json

import pytest
import torch

import arti
from arti import mechanisms


@pytest.mark.parametrize("mode", ("gelu", "relu", "rsqrt", "sigmoid", "silu", "tanh"))
def test_scalar_map_atom_matches_torch(mode: str) -> None:
    value_type = mechanisms.TensorType.axes(("B", "D"), sizes=(2, 3), dtype="float64")
    atom = mechanisms.ScalarMapAtom(value_type, mode=mode)
    value = torch.rand(2, 3, dtype=torch.float64) + 0.25

    actual = atom(value)
    expected = {
        "gelu": torch.nn.functional.gelu,
        "relu": torch.relu,
        "rsqrt": torch.rsqrt,
        "sigmoid": torch.sigmoid,
        "silu": torch.nn.functional.silu,
        "tanh": torch.tanh,
    }[mode](value)

    torch.testing.assert_close(actual, expected)
    assert arti.component_ref(atom) == "arti/formula-atom-scalar-map@1"
    assert arti.validate_component_provenance(arti.component_provenance(atom))


def _build_composite_program() -> tuple[
    mechanisms.FormulaProgram,
    mechanisms.BankBinding,
]:
    table_type = mechanisms.TensorType.axes(
        ("V", "D"), sizes=(7, 4), dtype="float32", domain="model"
    )
    index_type = mechanisms.TensorType.axes(
        ("B", "S"), sizes=("B", "S"), dtype="int64", domain="tokens"
    )
    token_type = mechanisms.TensorType.axes(
        ("B", "S", "D"), sizes=("B", "S", 4), dtype="float32", domain="model"
    )
    token_mask_type = mechanisms.TensorType.axes(
        ("B", "S"), sizes=("B", "S"), dtype="boolean", domain="validity"
    )
    logits_type = mechanisms.TensorType.axes(
        ("B", "Q", "K"),
        sizes=("B", "Q", "K"),
        dtype="float32",
        domain="attention",
    )
    attention_mask_type = mechanisms.TensorType.axes(
        ("B", "Q", "K"),
        sizes=("B", "Q", "K"),
        dtype="boolean",
        domain="validity",
    )
    scalar_type = mechanisms.TensorType.axes((), sizes=(), dtype="float32", domain="model")

    table = mechanisms.BankBinding(
        "embedding.table",
        "arti/test-transformer-atom-bank@1",
        "embedding",
        table_type,
    )
    token_ids = mechanisms.InputBinding("token_ids", index_type)
    token_mask = mechanisms.InputBinding("token_mask", token_mask_type)
    fill = mechanisms.InputBinding("fill", scalar_type)
    logits = mechanisms.InputBinding("logits", logits_type)
    attention_mask = mechanisms.InputBinding("attention_mask", attention_mask_type)

    embedded = mechanisms.lookup(table, token_ids, table_axis="V")
    left = mechanisms.slice_tensor(embedded, axis="D", start=0, stop=2)
    right = mechanisms.slice_tensor(embedded, axis="D", start=2, stop=4)
    reassembled = mechanisms.concat(left, right, axis="D")
    fill_tensor = mechanisms.broadcast(
        fill,
        output_axes=token_type.axis_names,
        output_sizes=token_type.sizes,
    )
    chosen = mechanisms.select(token_mask, reassembled, fill_tensor)
    activated = mechanisms.scalar_map(chosen, mode="silu")
    probabilities = mechanisms.masked_softmax(
        logits,
        attention_mask,
        axis="K",
    )
    return mechanisms.FormulaProgram.build(outputs=(activated, probabilities)), table


@pytest.mark.parametrize(("batch", "sequence"), ((1, 2), (3, 5)))
def test_transformer_atom_program_is_dynamic_differentiable_and_serializable(
    batch: int,
    sequence: int,
) -> None:
    program, table_binding = _build_composite_program()
    restored = mechanisms.FormulaProgram.from_dict(json.loads(json.dumps(program.to_dict())))
    assert restored.fingerprint == program.fingerprint
    fabric = mechanisms.FormulaFabricV2(restored)

    table = torch.randn(7, 4, requires_grad=True)
    token_ids = torch.randint(0, 7, (batch, sequence), dtype=torch.int64)
    token_mask = torch.rand(batch, sequence) > 0.35
    fill = torch.randn((), requires_grad=True)
    logits = torch.randn(batch, sequence, sequence, requires_grad=True)
    attention_mask = torch.tril(
        torch.ones(sequence, sequence, dtype=torch.bool)
    ).unsqueeze(0).expand(batch, -1, -1).clone()
    attention_mask[0, 0] = False

    result = fabric(
        inputs={
            "token_ids": token_ids,
            "token_mask": token_mask,
            "fill": fill,
            "logits": logits,
            "attention_mask": attention_mask,
        },
        banks={"embedding.table": table_binding.bind(table)},
        return_trace=True,
    )

    embedded = table[token_ids]
    chosen = torch.where(token_mask.unsqueeze(-1), embedded, fill)
    expected_activated = torch.nn.functional.silu(chosen)
    masked = logits.masked_fill(~attention_mask, -torch.inf)
    expected_probabilities = torch.softmax(masked, dim=-1)
    expected_probabilities = torch.nan_to_num(expected_probabilities)
    torch.testing.assert_close(result.values[0], expected_activated)
    torch.testing.assert_close(result.values[1], expected_probabilities)
    assert torch.equal(result.values[1][0, 0], torch.zeros(sequence))

    (result.values[0].square().mean() + result.values[1].square().mean()).backward()
    assert table.grad is not None and torch.isfinite(table.grad).all()
    assert fill.grad is not None and torch.isfinite(fill.grad)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    expected_dependencies = {
        "arti/formula-atom-broadcast@1",
        "arti/formula-atom-concat@1",
        "arti/formula-atom-lookup@1",
        "arti/formula-atom-masked-softmax@1",
        "arti/formula-atom-scalar-map@1",
        "arti/formula-atom-select@1",
        "arti/formula-atom-slice@1",
    }
    assert set(arti.component_spec(fabric).dependencies) == expected_dependencies
    assert result.trace is not None
    assert expected_dependencies == set(result.trace.atom_refs)
    assert arti.validate_component_provenance(arti.component_provenance(fabric))


def test_transformer_atom_component_contracts_round_trip() -> None:
    scalar = mechanisms.TensorType.axes((), sizes=(), dtype="float32", domain="model")
    values = mechanisms.TensorType.axes(("B", "D"), sizes=(2, 4), dtype="float32")
    mask = mechanisms.TensorType.axes(("B",), sizes=(2,), dtype="boolean")
    table = mechanisms.TensorType.axes(("V", "D"), sizes=(5, 4), dtype="float32")
    indices = mechanisms.TensorType.axes(("B",), sizes=(2,), dtype="int64")
    half = mechanisms.TensorType.axes(("B", "D"), sizes=(2, 2), dtype="float32")
    atoms = (
        mechanisms.BroadcastAtom(scalar, output_axes=("B", "D"), output_sizes=(2, 4)),
        mechanisms.SelectAtom(mask, values),
        mechanisms.LookupAtom(table, indices, table_axis="V"),
        mechanisms.SliceAtom(values, axis="D", start=0, stop=2),
        mechanisms.ConcatAtom(half, half, axis="D"),
        mechanisms.MaskedSoftmaxAtom(values, mask, axis="D"),
    )

    for atom in atoms:
        provenance = arti.component_provenance(atom)
        assert arti.validate_component_provenance(provenance) == provenance


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_transformer_atom_program_matches_cuda() -> None:
    program, table_binding = _build_composite_program()
    cpu = mechanisms.FormulaFabricV2(program)
    cuda = mechanisms.FormulaFabricV2(program).cuda()
    token_ids = torch.tensor([[1, 4, 2]], dtype=torch.int64)
    token_mask = torch.tensor([[True, False, True]])
    attention_mask = torch.tril(torch.ones(3, 3, dtype=torch.bool)).unsqueeze(0)
    values = {
        "fill": torch.tensor(0.25),
        "logits": torch.randn(1, 3, 3),
        "table": torch.randn(7, 4),
    }

    expected = cpu(
        inputs={
            "token_ids": token_ids,
            "token_mask": token_mask,
            "fill": values["fill"],
            "logits": values["logits"],
            "attention_mask": attention_mask,
        },
        banks={"embedding.table": table_binding.bind(values["table"])},
    ).values
    actual = cuda(
        inputs={
            "token_ids": token_ids.cuda(),
            "token_mask": token_mask.cuda(),
            "fill": values["fill"].cuda(),
            "logits": values["logits"].cuda(),
            "attention_mask": attention_mask.cuda(),
        },
        banks={"embedding.table": table_binding.bind(values["table"].cuda())},
    ).values

    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left, right.cpu(), atol=1e-6, rtol=1e-5)
