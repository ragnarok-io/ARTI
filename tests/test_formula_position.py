from __future__ import annotations

import json
import shutil

import pytest
import torch

import arti
from arti import mechanisms as m


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _fabric(output):
    program = m.FormulaProgram.build(outputs=(output,))
    restored = m.FormulaProgram.from_dict(json.loads(json.dumps(program.to_dict())))
    assert restored.fingerprint == program.fingerprint
    return m.FormulaFabricV2(restored)


@pytest.mark.parametrize("device", DEVICES)
def test_axis_coordinates_form_self_contained_sinusoidal_position(device):
    value_type = m.TensorType(("B", "N", "D"), ("B", "N", 6), dtype="float32", domain="tokens")
    value = m.InputBinding("value", value_type)
    coordinates = m.axis_index(value, axis="N")
    position = m.sinusoidal_position(coordinates, feature_axis="D", feature_size=6, dtype="float32")
    field = m.broadcast(position, output_axes=("B", "N", "D"), output_sizes=("B", "N", 6))
    fabric = _fabric(m.add(value, field))
    x = torch.zeros(2, 5, 6, device=device, requires_grad=True)
    result = fabric(inputs={"value": x}, banks={}).values[0]
    expected_position = torch.arange(5, device=device, dtype=torch.float32)
    expected_frequency = torch.exp(-torch.log(torch.tensor(10000.0, device=device)) * torch.arange(0, 6, 2, device=device) / 6)
    expected = torch.stack(
        ((expected_position[:, None] * expected_frequency).sin(), (expected_position[:, None] * expected_frequency).cos()),
        dim=-1,
    ).flatten(-2)
    torch.testing.assert_close(result[0], expected)
    torch.testing.assert_close(result[1], expected)
    result.square().mean().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("device", DEVICES)
def test_relative_position_preserves_shared_batch_axis(device):
    query = m.InputBinding("query", m.TensorType(("B", "Q"), ("B", 3), dtype="int64", domain="tokens"))
    key = m.InputBinding("key", m.TensorType(("B", "K"), ("B", 4), dtype="int64", domain="tokens"))
    result = _fabric(m.relative_position(query, key))(
        inputs={
            "query": torch.tensor([[4, 8, 12], [2, 7, 9]], dtype=torch.int64, device=device),
            "key": torch.tensor([[1, 3, 10, 13], [0, 2, 8, 10]], dtype=torch.int64, device=device),
        },
        banks={},
    ).values[0]
    assert result.shape == (2, 3, 4)
    assert result.dtype == torch.int64
    assert result[0, 1, 2].item() == -2
    assert result[1, 2, 3].item() == -1


@pytest.mark.parametrize("device", DEVICES)
def test_rotary_position_is_identity_at_zero_and_preserves_pair_norm(device):
    value = m.InputBinding("value", m.TensorType(("B", "N", "H", "D"), ("B", 4, 2, 8), domain="tokens"))
    coordinates = m.InputBinding("coordinates", m.TensorType(("B", "N"), ("B", 4), dtype="int64", domain="tokens"))
    fabric = _fabric(m.rotary_position(value, coordinates, feature_axis="D"))
    torch.manual_seed(932)
    source = torch.randn(2, 4, 2, 8, device=device, requires_grad=True)
    zero = torch.zeros(2, 4, dtype=torch.int64, device=device)
    unchanged = fabric(inputs={"value": source, "coordinates": zero}, banks={}).values[0]
    torch.testing.assert_close(unchanged, source)
    stepped = fabric(
        inputs={"value": source, "coordinates": torch.arange(4, device=device).expand(2, -1)},
        banks={},
    ).values[0]
    torch.testing.assert_close(stepped.reshape(*stepped.shape[:-1], 4, 2).square().sum(-1), source.reshape(*source.shape[:-1], 4, 2).square().sum(-1))
    stepped.square().mean().backward()
    assert torch.isfinite(source.grad).all()


def test_position_atoms_are_parameter_free_components_and_validate_contracts():
    coordinate_type = m.TensorType(("N",), (7,), dtype="int64", domain="tokens")
    sinusoidal = m.SinusoidalPositionAtom((coordinate_type,), feature_axis="D", feature_size=8)
    relative = m.RelativePositionAtom((coordinate_type, coordinate_type))
    rotary = m.RotaryPositionAtom(
        (m.TensorType(("N", "D"), (7, 8), domain="tokens"), coordinate_type), feature_axis="D"
    )
    for atom in (sinusoidal, relative, rotary):
        assert not list(atom.parameters()) and not atom.state_dict()
        assert arti.component_ref(atom) == arti.canonical_contract_reference(atom._component_reference)
        assert arti.validate_component_provenance(arti.component_provenance(atom))
    with pytest.raises(m.FormulaTypeError, match="even"):
        m.rotary_position(
            m.InputBinding("odd", m.TensorType(("N", "D"), (7, 7), domain="tokens")),
            m.InputBinding("positions", coordinate_type),
            feature_axis="D",
        )


def test_position_kernel_compiles_without_python_position_dispatch():
    from arti.formula_position import execute_position

    coordinate_type = m.TensorType(("N",), (5,), dtype="int64", domain="tokens")
    attributes = {"feature_axis": "D", "feature_size": 8, "dtype": "float32", "base": 10000.0}

    def kernel(coordinates):
        return execute_position(
            "arti/formula-atom-position-sinusoidal@1",
            (coordinates,),
            (coordinate_type,),
            attributes,
        )

    if torch.cuda.is_available():
        device = "cuda"
    elif shutil.which("cl") is not None:
        device = "cpu"
    else:
        pytest.skip("CPU Inductor requires an MSVC compiler on this host")
    coordinates = torch.arange(5, dtype=torch.int64, device=device)
    compiled = torch.compile(kernel, fullgraph=True)
    torch.testing.assert_close(compiled(coordinates), kernel(coordinates))
