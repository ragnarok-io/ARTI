from __future__ import annotations

import copy

import pytest
import torch

import arti
from arti.mechanisms import ObjectiveExposureBank, ObjectiveExposureOutput


def test_component_is_versioned_and_alpha_only() -> None:
    bank = ObjectiveExposureBank(4, 2, key_layout="hypercube")

    assert arti.component_ref(bank) == "arti/objective-exposure-bank@1"
    assert not hasattr(arti, "ObjectiveExposureBank")
    assert not hasattr(arti.nn, "ObjectiveExposureBank")


def test_exposure_is_bounded_and_reports_real_routes() -> None:
    bank = ObjectiveExposureBank(
        4,
        2,
        key_layout="hypercube",
        temperature=4.0,
        min_exposure=0.1,
        max_exposure=0.9,
    )
    query = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])

    output = bank(query, return_info=True)

    assert isinstance(output, ObjectiveExposureOutput)
    assert output.exposure.shape == (2,)
    assert output.route_weights.shape == (2, 4)
    torch.testing.assert_close(
        output.route_weights.sum(dim=-1), torch.ones(2), rtol=0, atol=1e-6
    )
    assert torch.all(output.exposure >= 0.1)
    assert torch.all(output.exposure <= 0.9)


def test_only_values_receive_gradients_from_a_fixed_query() -> None:
    bank = ObjectiveExposureBank(4, 2, key_layout="hypercube")
    query = torch.randn(3, 2, requires_grad=True)

    bank(query).sum().backward()

    assert bank.values.grad is not None
    assert torch.count_nonzero(bank.values.grad) > 0
    assert query.grad is None


def test_state_dict_round_trip_preserves_output() -> None:
    bank = ObjectiveExposureBank(4, 2, key_layout="hypercube", key_seed=7)
    with torch.no_grad():
        bank.values.copy_(torch.tensor([-2.0, 1.0, 3.0, -1.0]))
    query = torch.randn(5, 2)
    expected = bank(query)

    restored = ObjectiveExposureBank(4, 2, key_layout="hypercube", key_seed=7)
    restored.load_state_dict(copy.deepcopy(bank.state_dict()))

    torch.testing.assert_close(restored(query), expected, rtol=0, atol=0)


def test_invalid_configuration_and_query_fail_closed() -> None:
    with pytest.raises(ValueError, match=r"2\*\*query_dim"):
        ObjectiveExposureBank(5, 2, key_layout="hypercube")
    bank = ObjectiveExposureBank(4, 2, key_layout="hypercube")
    with pytest.raises(ValueError, match="trailing shape"):
        bank(torch.randn(3, 3))
    with pytest.raises(ValueError, match="finite"):
        bank(torch.tensor([[float("nan"), 0.0]]))


def test_circle_layout_is_a_fixed_addressable_ring() -> None:
    bank = ObjectiveExposureBank(8, 2, key_layout="circle", temperature=8.0)
    query = bank.keys.clone()

    output = bank(query, return_info=True)

    assert torch.equal(output.route_weights.argmax(dim=-1), torch.arange(8))
    with pytest.raises(ValueError, match="query_dim=2"):
        ObjectiveExposureBank(8, 3, key_layout="circle")
