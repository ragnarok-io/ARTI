"""Shared settings for attachment, pretrained and native Torch training."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ARTITrainingConfig:
    engine: str = "torch"
    objective: str = "model_loss"
    learning_rate: float = 1e-3
    steps: int = 1
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "no"
    corruption_probability: float = 0.15
    device: str = "auto"
    distributed: bool = False

    def __post_init__(self) -> None:
        if self.engine not in {"torch", "transformers", "accelerate"}:
            raise ValueError("training.engine must be 'torch', 'transformers', or 'accelerate'")
        if self.objective not in {"tensor_alignment", "model_loss"}:
            raise ValueError("training.objective must be 'tensor_alignment' or 'model_loss'")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("training.learning_rate must be finite and positive")
        if type(self.steps) is not int or self.steps < 0:
            raise ValueError("training.steps must be a nonnegative integer")
        if type(self.gradient_accumulation_steps) is not int or self.gradient_accumulation_steps <= 0:
            raise ValueError("training.gradient_accumulation_steps must be a positive integer")
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("training.mixed_precision must be 'no', 'fp16', or 'bf16'")
        if not 0 <= self.corruption_probability < 1:
            raise ValueError("training.corruption_probability must be in [0, 1)")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("training.device must be 'auto', 'cpu', or 'cuda'")
        if type(self.distributed) is not bool:
            raise TypeError("training.distributed must be boolean")


__all__ = ["ARTITrainingConfig"]
