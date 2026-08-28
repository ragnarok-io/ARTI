"""Experimental and internal integration surfaces.

These modules are intentionally outside the stable root namespace.  They may
change or disappear without compatibility promises while the tensor-native
core API remains stable.
"""

from . import web
from .layered_recall import (
    LayerRecall,
    LayerRecallSpec,
    LayerRecallStack,
    LayerRecallWrapper,
    LayeredRecallCalibration,
    LayeredRecallConfig,
    LayeredRecallLoss,
    LayeredRecallModel,
    calibrate_layered_recall,
    layered_recall_trajectory_loss,
)
from .stateful_recall import StatefulRecall

__all__ = [
    "LayerRecall",
    "LayerRecallSpec",
    "LayerRecallStack",
    "LayerRecallWrapper",
    "LayeredRecallCalibration",
    "LayeredRecallConfig",
    "LayeredRecallLoss",
    "LayeredRecallModel",
    "StatefulRecall",
    "web",
    "calibrate_layered_recall",
    "layered_recall_trajectory_loss",
]
