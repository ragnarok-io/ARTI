import pytest
import torch

from arti.torch import ARTILayer


def test_torch_backend_forward_backward_cpu():
    layer = ARTILayer(input_dim=8, coord_dim=2, hidden_dim=16, recall_steps=1)
    x = torch.randn(2, 5, 8, requires_grad=True)
    coord = torch.randn(2, 5, 2)
    mask = torch.ones(2, 5, dtype=torch.bool)

    out = layer(x, coord=coord, mask=mask)
    loss = out.y.square().mean() + out.virtual_y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is not available")
def test_torch_backend_cuda_amp_smoke():
    device = torch.device("cuda")
    layer = ARTILayer(input_dim=8, coord_dim=2, hidden_dim=16, recall_steps=1).to(device)
    x = torch.randn(2, 5, 8, device=device)
    coord = torch.randn(2, 5, 2, device=device)
    mask = torch.ones(2, 5, dtype=torch.bool, device=device)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = layer(x, coord=coord, mask=mask)

    assert out.y.device.type == "cuda"
    assert out.virtual_y is not None
    assert out.virtual_y.device.type == "cuda"


def test_torch_backend_compile_smoke_if_available():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is not available")

    class Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = ARTILayer(input_dim=4, coord_dim=0, hidden_dim=4, recall_steps=0, use_pairwise_context=False)

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return self.layer(x, mask=mask).pooled

    model = Wrapper()
    compiled = torch.compile(model, backend="eager")
    x = torch.randn(2, 3, 4)
    mask = torch.ones(2, 3, dtype=torch.bool)

    out = compiled(x, mask)

    assert out.shape == (2, 4)
