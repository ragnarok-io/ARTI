import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from arti import ARTIHostBridge, ARTIResidualBlock


@pytest.fixture(autouse=True)
def preserve_torch_rng_state():
    state = torch.random.get_rng_state()
    try:
        yield
    finally:
        torch.random.set_rng_state(state)


def test_residual_block_works_in_sequential_for_vectors():
    model = nn.Sequential(
        nn.Linear(32, 64),
        ARTIResidualBlock(dim=64),
        nn.Linear(64, 10),
    )

    y = model(torch.randn(8, 32))

    assert y.shape == (8, 10)


def test_zero_init_output_is_identity_and_unblocks_radial_training_path():
    torch.manual_seed(7)
    block = ARTIResidualBlock(dim=8, hidden_dim=4, zero_init_output=True)
    x = torch.randn(3, 5, 8)
    target = torch.randn_like(x)

    assert torch.equal(block(x), x)
    assert isinstance(block.out, ARTIHostBridge)
    assert block.out.mode == "radial"

    optimizer = torch.optim.AdamW(block.parameters(), lr=1e-3)
    (block(x) - target).square().mean().backward()
    assert block.out.radius.grad is not None
    assert torch.count_nonzero(block.out.radius.grad) > 0
    assert block.out.linear.weight.grad is not None
    assert torch.count_nonzero(block.out.linear.weight.grad) == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    (block(x) - target).square().mean().backward()
    internal_grads = [
        parameter.grad
        for name, parameter in block.named_parameters()
        if not name.startswith("out.") and parameter.grad is not None
    ]
    assert internal_grads
    assert any(torch.count_nonzero(gradient) > 0 for gradient in internal_grads)


def test_zero_init_output_does_not_bound_residual_amplitude():
    block = ARTIResidualBlock(dim=4, hidden_dim=3, zero_init_output=True)
    assert isinstance(block.out, ARTIHostBridge)
    with torch.no_grad():
        block.out.radius.fill_(1_000.0)

    x = torch.ones(2, 4)
    delta = block(x) - x

    assert torch.max(torch.abs(delta)) > 1.0


def test_radial_bridge_first_step_budget_is_width_and_depth_bounded():
    torch.manual_seed(11)
    lr = 1e-3
    rms_limit = 2.0
    for hidden_dim in (4, 31):
        for depth in (1, 7):
            bridges = nn.ModuleList(
                ARTIHostBridge(
                    hidden_dim,
                    5,
                    residual_budget=1.0 / depth,
                    rms_limit=rms_limit,
                )
                for _ in range(depth)
            )
            x = torch.randn(3, hidden_dim) * 10.0
            optimizer = torch.optim.AdamW(bridges.parameters(), lr=lr, weight_decay=0.0)
            loss = sum(bridge(x).sum() for bridge in bridges)
            loss.backward()
            optimizer.step()

            total = sum(bridge(x) for bridge in bridges)
            total_rms = total.square().mean(dim=-1).sqrt()
            assert torch.max(total_rms) <= lr * rms_limit * 1.01


def test_radial_bridge_folded_weight_matches_forward_in_normal_range():
    torch.manual_seed(13)
    bridge = ARTIHostBridge(6, 4, residual_budget=0.25, rms_limit=10.0)
    with torch.no_grad():
        bridge.radius.copy_(torch.tensor([-2.0, -0.5, 1.0, 3.0]))
    x = torch.randn(2, 3, 6)

    expected = F.linear(x, bridge.effective_weight())

    assert torch.allclose(bridge(x), expected, rtol=1e-6, atol=1e-6)


def test_radial_bridge_keeps_radius_fp32_when_module_dtype_changes():
    bridge = ARTIHostBridge(4, 3).to(dtype=torch.bfloat16)

    assert bridge.linear.weight.dtype == torch.bfloat16
    assert bridge.radius.dtype == torch.float32


def test_dense_bridge_retains_legacy_zero_linear_behavior():
    bridge = ARTIHostBridge(4, 3, mode="dense", bias=True)
    x = torch.randn(2, 4)

    assert torch.equal(bridge(x), torch.zeros(2, 3))
    assert bridge.radius is None
