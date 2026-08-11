import torch

from arti import ARTILayer


def test_state_dict_serialization_round_trip(tmp_path):
    layer = ARTILayer(input_dim=8, coord_dim=2, hidden_dim=16)
    path = tmp_path / "arti.pt"
    torch.save(layer.state_dict(), path)

    loaded = ARTILayer(input_dim=8, coord_dim=2, hidden_dim=16)
    loaded.load_state_dict(torch.load(path, weights_only=True))

    x = torch.randn(2, 5, 8)
    coord = torch.randn(2, 5, 2)

    assert loaded(x, coord=coord).y.shape == (2, 5, 16)


def test_grouped_recall_state_dict_round_trip_is_exact(tmp_path):
    torch.manual_seed(11)
    kwargs = dict(
        input_dim=8,
        hidden_dim=8,
        recall_slots=32,
        recall_steps=1,
        recall_routing="grouped",
        recall_key_dim=4,
        recall_group_size=8,
        recall_group_topk=2,
        use_pairwise_context=False,
    )
    layer = ARTILayer(**kwargs).eval()
    x = torch.randn(2, 5, 8)
    expected = layer(x).y
    path = tmp_path / "grouped-recall.pt"
    torch.save(layer.state_dict(), path)

    loaded = ARTILayer(**kwargs).eval()
    loaded.load_state_dict(torch.load(path, weights_only=True))

    assert torch.equal(loaded(x).y, expected)


def test_backward_is_stable():
    layer = ARTILayer(input_dim=8, coord_dim=2, hidden_dim=16)
    x = torch.randn(2, 5, 8, requires_grad=True)
    coord = torch.randn(2, 5, 2)

    loss = layer(x, coord=coord).pooled.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
