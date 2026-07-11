import torch

from arti import ARTILayer


def test_layer_runs_on_available_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = ARTILayer(input_dim=8, coord_dim=2, hidden_dim=16).to(device)
    x = torch.randn(2, 5, 8, device=device)
    coord = torch.randn(2, 5, 2, device=device)

    out = layer(x, coord=coord)

    assert out.y.device.type == device.type
    assert out.pooled.device.type == device.type
    if device.type == "cuda":
        assert out.y.device.index == torch.cuda.current_device()
