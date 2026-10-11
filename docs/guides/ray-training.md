# Ray Train Checkpoint Handoff

ARTI's shared Torch session can participate in a caller-owned Ray Train job.
Ray controls job execution and its storage; ARTI keeps the same model binding,
objective, optimizer and asset format. No Ray cluster, cloud account or new
training loop is created by the adapter.

## Install and Enable

The optional `arti-fit[ray]` extra selects Ray Train **2.59.x**, with available SDK
wheels on Python 3.10-3.13. It is deliberately not selected on Python 3.14+;
use a supported interpreter for this integration. The validated local profile
uses Ray 2.59.0 and CPU Torch.

Enable Train V2 **before** importing `ray.train`:

```bash
RAY_TRAIN_V2_ENABLED=1 python your_training_job.py
```

```powershell
$env:RAY_TRAIN_V2_ENABLED = '1'
python your_training_job.py
```

`arti` does not import Ray at package import time, change this flag, call
`ray.init()`, connect to an account, or launch workers on its own.

## Use the Existing Session

```python
from pathlib import Path
from ray import train
from ray.train.torch import TorchTrainer
import arti

def train_worker(config):
    model = build_your_model(config)
    session = arti.trainer(model, objective=your_task_loss)
    arti.restore_ray_checkpoint(session, directory=Path(config["work_dir"]) / "resume")
    remaining = max(0, config["total_steps"] - session.global_step)
    session.fit(
        your_resumable_data(config),
        steps=remaining,
        run=arti.ARTITrainingRunConfig(
            output_dir=Path(config["work_dir"]) / "new-attempt",
            checkpoint_steps=50,
        ),
        on_checkpoint=arti.RayTrainCheckpointHandler(),
    )

job = TorchTrainer(
    train_worker,
    train_loop_config=your_config,
    scaling_config=train.ScalingConfig(num_workers=0),
)
result = job.fit()
```

Choose a fresh `output_dir` for each attempt. `steps` remains additional updates
in ARTI, while a job configuration's `total_steps` is the complete horizon.
Rebuild the same model, optimizer/scheduler constructors and dataset before
resuming. Indexed sequences and stateful loaders retain their supported data
position; a generic iterator is not exact data replay.

`on_checkpoint(session, path)` fires only after a completed run checkpoint has
been written and its local latest pointer published. It is available through
attachment, standalone, pretrained and classification/Qwen recipe entrances.
A callback failure propagates, marking the run failed while retaining the saved
local asset. This differs from optional W&B/MLflow reporting, which can fail
without invalidating training. Do not put `RayTrainCheckpointHandler` in
`ReporterGroup` or pass it as `reporter`.

The callback runs at optimizer/checkpoint boundaries, not within neural CUDA
graphs. Other engines retain their native callback lifecycle; `on_checkpoint`
currently belongs to the local Torch engine.

## Complete Assets and Scalar Metrics

The handler reports optimizer/microbatch counts, latest training loss and
selection scalars. `metrics=lambda session: {...}` can add caller-owned metrics.
Default metrics do not include prompts, record contents or local paths.

When an earlier best checkpoint is referenced, the adapter produces a portable
closure containing that best bundle and all existing sidecars. An explicit
`checkpoint_dir` requests independent durable snapshots on every handoff:

```python
handler = arti.RayTrainCheckpointHandler(checkpoint_dir="outputs/handoff")
```

Without that directory and without external best dependencies, the already
completed checkpoint directory is reported directly, avoiding duplicate saves.
Dependent best bundles are placed in the run's `ray-checkpoints` directory.
Snapshots use distinct names; a previously handed-off snapshot is not overwritten.
Local directories remain caller-owned; the adapter does not delete training
assets after reporting. Ray's own checkpoint retention policy governs its
stored copies, not the caller's local run directory.

`session.save_checkpoint(path, portable=True)` explicitly includes the best
dependency even when the destination is inside the active run. This reuses the
existing ARTI format, rather than introducing another tensor/checkpoint protocol.
Place a portable destination outside the source best bundle to avoid recursive
directory copying.

`restore_ray_checkpoint` uses `Checkpoint.to_directory` before loading. The
materialized directory stays alive for later best selection, export or resume;
it does not leave a dangling best path after a temporary `as_directory` context.
Supply a checkpoint explicitly outside a worker or for a deliberate local handoff:

```python
arti.restore_ray_checkpoint(
    session,
    directory="outputs/restored",
    checkpoint=result.checkpoint,
)
```

The default inside a worker uses `ray.train.get_checkpoint()`. If none exists,
it returns `None`. Upload mode is synchronous: a completed checkpoint handoff is
acknowledged before training proceeds. Remote storage/upload happens only through
the caller's Ray job/storage configuration, never by ARTI selecting a cloud.

## Local Mode and Recovery Boundaries

Ray Train V2's `num_workers=0` executes in the current process and does not launch
a Ray cluster. In Ray 2.59 local mode, `report` retains a checkpoint reference;
it does **not** implement a storage context, copy/upload it, automatically retry
workers, or restore a previous completed run by its name. Keep the referenced
directory and explicitly pass it to a fresh local job/process.

The qualified adapter is single-worker, ordinary Torch state. Sharded/DDP
checkpoint synchronization, automatic remote worker retries, Ray+HF/TRL native
checkpoint conversion and cloud storage are separate profiles. Do not treat one
rank's local state as a complete distributed checkpoint. See the native
[Ray failure recovery guide](https://docs.ray.io/en/latest/train/user-guides/fault-tolerance.html)
for job/storage identity and retry configuration when qualifying those profiles.

## Fresh-Process Example

```bash
python examples/ray_training_workflow.py --output .tmp/ray-workflow
```

The example runs an actual ProgramGraph/ARTILayer task in three independent CPU
processes: continuous six updates, two updates plus a handoff, and four resumed
updates. It checks exact ARTI weights, optimizer/scheduler, loss/RNG/data parity,
six total updates, twelve physical microbatches and different save/resume PIDs.
No cluster or GPU is started. This establishes lifecycle correctness, not neural
task superiority or automatic distributed recovery.

References: [Ray checkpoints](https://docs.ray.io/en/latest/train/user-guides/checkpoints.html),
[local mode](https://docs.ray.io/en/latest/train/user-guides/local_mode.html).
