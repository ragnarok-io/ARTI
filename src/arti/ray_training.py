"""Optional Ray Train handoff at completed ARTI checkpoint boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from .attachment_training import ARTITrainingSession


def _ray_train():
    from ray import train

    if not hasattr(train, "CheckpointUploadMode"):
        raise RuntimeError(
            "Ray Train V2 requires RAY_TRAIN_V2_ENABLED=1 before importing ray.train"
        )
    return train


class RayTrainCheckpointHandler:
    """Critical checkpoint handler, not a failure-isolated experiment reporter.

    Ray owns scheduling and remote storage. Local mode keeps a checkpoint path
    reference only, so checkpoint directories remain owned by the caller.
    """

    def __init__(
        self,
        *,
        checkpoint_dir: str | Path | None = None,
        metrics: Callable[[ARTITrainingSession], Mapping[str, float]] | None = None,
    ) -> None:
        self.checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self.metrics = metrics

    def __call__(self, session: ARTITrainingSession, path: Path) -> None:
        train = _ray_train()

        if train.get_context().get_world_size() != 1 or session.config.engine != "torch":
            raise ValueError(
                "RayTrainCheckpointHandler currently requires single-worker Torch training"
            )
        tracker = session._validation
        requires_closure = (
            tracker is not None
            and tracker.best_checkpoint is not None
            and tracker.best_checkpoint.parent.resolve() != path.parent.resolve()
        )
        artifact = path
        if self.checkpoint_dir is not None or requires_closure:
            root = self.checkpoint_dir or session._run.directory / "ray-checkpoints"
            target = root / f"step-{session.global_step:08d}-{uuid.uuid4().hex}" / "model.arti.st"
            artifact = session.save_checkpoint(target, portable=True)
        metrics = {
            "optimizer_step": session.global_step,
            "micro_batches_seen": session.micro_batches_seen,
        }
        if session.loss_history:
            metrics["train/loss"] = session.loss_history[-1]
        receipt = session.selection_receipt()
        for key in ("selected_step", "best_step", "best_value"):
            if receipt.get(key) is not None:
                metrics[f"selection/{key}"] = receipt[key]
        if self.metrics is not None:
            metrics.update(self.metrics(session))
        train.report(
            metrics,
            checkpoint=train.Checkpoint.from_directory(str(artifact.parent.resolve())),
            checkpoint_upload_mode=train.CheckpointUploadMode.SYNC,
        )


def restore_ray_checkpoint(
    session: ARTITrainingSession,
    *,
    directory: str | Path,
    checkpoint=None,
) -> Path | None:
    """Materialize a Ray checkpoint before loading, keeping best assets alive."""
    train = _ray_train()

    checkpoint = train.get_checkpoint() if checkpoint is None else checkpoint
    if checkpoint is None:
        return None
    materialized = Path(checkpoint.to_directory(str(Path(directory).resolve())))
    artifact = materialized / "model.arti.st"
    session.load_checkpoint(artifact)
    return artifact


__all__ = ["RayTrainCheckpointHandler", "restore_ray_checkpoint"]
