import torch

from arti import ARTILayer
from arti.layers import ARTIVirtualInterfaceMixer


def test_masked_tokens_do_not_contribute_to_pooling_or_output():
    layer = ARTILayer(input_dim=4, hidden_dim=4, recall_steps=0)
    x = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, True, False], [True, False, False]])

    out = layer(x, mask=mask)

    assert torch.allclose(out.y[~mask], torch.zeros_like(out.y[~mask]), atol=1e-6)
    assert torch.allclose(out.diagnostics["mask_coverage"], torch.tensor([2 / 3, 1 / 3]))


def test_visibility_blocks_hidden_token_in_attention_weights():
    layer = ARTILayer(input_dim=4, hidden_dim=4, recall_steps=0)
    x = torch.randn(1, 3, 4)
    visibility = torch.tensor([[[True, True, False], [True, True, False], [False, False, True]]])

    out = layer(x, visibility=visibility)

    assert torch.all(out.diagnostics["visibility_weights"][0, :2, 2] == 0)


def test_virtual_interface_respects_visibility_for_query_context():
    torch.manual_seed(11)
    mixer = ARTIVirtualInterfaceMixer(hidden_dim=4, slots=3)
    z = torch.randn(1, 3, 4)
    mask = torch.ones(1, 3, dtype=torch.bool)
    visibility = torch.tensor([[[True, True, False], [True, True, True], [True, True, True]]])

    context_a, _, _ = mixer(z, mask, visibility)
    z_changed = z.clone()
    z_changed[:, 2] = z_changed[:, 2] * 100.0
    context_b, _, _ = mixer(z_changed, mask, visibility)

    assert torch.allclose(context_a[:, 0], context_b[:, 0], atol=1e-5)
    assert not torch.allclose(context_a[:, 1], context_b[:, 1])
