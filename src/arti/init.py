"""Initialization and parameter inspection utilities."""

from __future__ import annotations

from collections import OrderedDict

import torch.nn as nn


def init_arti_module(module: nn.Module) -> nn.Module:
    """Apply a conservative default initialization to linear layers."""

    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
    return module


def parameter_report(module: nn.Module) -> OrderedDict[str, int]:
    """Return trainable and total parameter counts."""

    total = sum(param.numel() for param in module.parameters())
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return OrderedDict(total=total, trainable=trainable)
