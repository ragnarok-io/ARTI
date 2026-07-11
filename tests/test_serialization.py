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


def test_backward_is_stable():
    layer = ARTILayer(input_dim=8, coord_dim=2, hidden_dim=16)
    x = torch.randn(2, 5, 8, requires_grad=True)
    coord = torch.randn(2, 5, 2)

    loss = layer(x, coord=coord).pooled.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
