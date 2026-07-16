"""Fuse variable-size Pulse workspaces with one shared output workspace."""

import torch

from arti.nn import FusionPulse, Pulse


def main() -> None:
    torch.manual_seed(7)
    left = Pulse(k=6, dim=16)(torch.randn(2, 24, 16))
    right = Pulse(k=4, dim=16)(torch.randn(2, 40, 16))
    fusion = FusionPulse(k=5, dim=16)

    output, info = fusion.concat(left, right, return_info=True)

    print("output", tuple(output.shape))
    print("survival", tuple(info["survival"].shape))
    print("structural_loss", info["structural_loss"].detach().item())


if __name__ == "__main__":
    main()
