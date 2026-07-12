import pytest
import torch

from arti.nn import StatefulRecall


def _identity_recall(dim=4, slots=4):
    module = StatefulRecall(
        dim,
        slots=slots,
        recognition_threshold=0.97,
        recognition_temperature=0.01,
        write_rate=1.0,
        decay=1.0,
    ).eval()
    with torch.no_grad():
        for projection in (module.query, module.key, module.value, module.emit):
            projection.weight.copy_(torch.eye(dim))
        module.slot_anchors.copy_(torch.eye(slots, dim))
        module.write_quality.weight.zero_()
        module.write_quality.bias.fill_(6.0)
    return module


def test_stateful_recall_observes_then_recognizes_without_parameter_updates():
    module = _identity_recall()
    parameters_before = {name: value.detach().clone() for name, value in module.named_parameters()}
    state = module.initial_state(1)
    full = torch.tensor([[[2.0, 0.0, 0.0, 0.0], [1.8, 0.0, 0.0, 0.0]]])
    corrupt = full + torch.tensor([[[0.0, 0.03, 0.0, 0.0], [0.0, -0.02, 0.0, 0.0]]])
    unseen = torch.tensor([[[2.0, 0.8, 0.0, 0.0], [1.8, 0.72, 0.0, 0.0]]])
    first = module.read(full, **state)
    assert first["recognition"].max() < 1e-4
    for _ in range(4):
        update = module.update(first["trace_key"], full, **state)
        state = {name: update[name] for name in module.state_names}
    seen = module.read(corrupt, **state)
    novel = module.read(unseen, **state)
    assert seen["recognition"].mean() > 0.8
    assert novel["recognition"].mean() < 0.05
    assert seen["delta"].norm() > novel["delta"].norm() * 10
    for name, value in module.named_parameters():
        assert torch.equal(value, parameters_before[name])


def test_stateful_recall_state_is_fixed_shape_and_differentiable():
    module = StatefulRecall(6, slots=3, key_dim=4)
    x = torch.randn(2, 5, 6, requires_grad=True)
    state = module.initial_state(2)
    read = module.read(x, **state)
    update = module.update(read["trace_key"], x, **state)
    loss = read["y"].square().mean() + sum(update[name].square().mean() for name in module.state_names)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert module.write_rate_logit.grad is not None
    assert module.decay_logit.grad is not None
    assert update["keys"].shape == (2, 3, 4)
    assert update["values"].shape == (2, 3, 6)
    assert update["strengths"].shape == (2, 3)


@pytest.mark.parametrize("method", ["read", "update"])
def test_stateful_recall_rejects_broadcastable_mask_shapes(method):
    module = StatefulRecall(6, slots=3, key_dim=4)
    x = torch.randn(2, 5, 6)
    state = module.initial_state(2)
    args = (x, *state.values()) if method == "read" else (torch.randn(2, 4), x, *state.values())
    with pytest.raises(ValueError, match=r"mask must have shape \(2, 5\)"):
        getattr(module, method)(*args, mask=torch.ones(2, 1, 1))


def test_stateful_recall_accepts_exact_boolean_mask():
    module = StatefulRecall(6, slots=3, key_dim=4)
    x = torch.randn(2, 5, 6)
    state = module.initial_state(2)
    result = module.read(x, **state, mask=torch.ones(2, 5, dtype=torch.bool))
    assert result["y"].shape == x.shape
