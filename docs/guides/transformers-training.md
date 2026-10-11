# Transformers and TRL Training

Use the existing `ARTITrainingSession` with a native Hugging Face trainer.
Transformers or TRL owns preprocessing, batching, the training loop, evaluation,
callbacks and native checkpoint state. ARTI owns its selected parameter role and
its asset codec. No second training loop or LoRA conversion is introduced.

## Native Trainer

Start from a model already attached through `arti.ARTI.attach`, or from a
`TrainingBinding`/native model accepted by `arti.trainer`:

```python
from transformers import TrainingArguments

session = model.arti.trainer(engine="transformers", learning_rate=2e-4)
hf = session.transformers_trainer(
    train_dataset,
    trainer_kwargs={
        "args": TrainingArguments(
            output_dir="outputs/training",
            max_steps=500,
            learning_rate=2e-4,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            save_steps=25,
            save_only_model=False,
            report_to=[],
        ),
        "eval_dataset": validation_dataset,
        "data_collator": collator,
    },
)
hf.train()
hf.evaluate()
hf.save_model("outputs/inference")
```

`trainer_class=YourTrainer` accepts a `transformers.Trainer` subclass. The bridge
delegates native model loss to that class, so `compute_loss_func`, label handling,
metrics and subclass-specific objectives are retained. An explicitly supplied
ARTI session objective instead remains the objective; do not use that mode when
you intend TRL's own task loss.

Native `TrainingArguments` controls the optimizer family, learning rate, decay,
betas and scheduler when no caller optimizer is supplied. Parameter ownership
still comes from `TrainingBinding`, including an explicit parameter subset.
An optimizer supplied to the ARTI session is used unchanged. Recreate a custom
optimizer/scheduler when reloading its portable training asset; constructor
behavior cannot be inferred from numerical moments alone.

The arguments object is copied. Gradient checkpointing without explicit kwargs
uses `use_reentrant=False`, retaining gradients to adapters inside a frozen host.
An explicit checkpointing configuration remains the caller's choice.

## Checkpoint Continuation

There are three distinct outputs:

| Output | Contents | Use |
| --- | --- | --- |
| `checkpoint-N/` | `model.arti.st` plus existing ARTI sidecars, tokenizer/processor, TrainerState, optimizer, scheduler and RNG | Native continuation of the same data trajectory |
| `session.save_checkpoint("training.arti.st")` | Portable ARTI weights, optimizer/scheduler and shared session state with sidecars | Continue learning with a new Trainer data trajectory |
| `hf.save_model("inference")` | ARTI model asset, processor and training arguments, without native optimizer/TrainerState | Inference/loading with the corresponding host and ARTI construction |

Defaults keep native optimizer/RNG state, save the first completed update and
the final update, and use Trainer's periodic cadence in between. Explicit
`save_strategy="no"` disables this checkpoint policy. `save_only_model=True`
intentionally omits native optimizer/RNG recovery.

```python
# Direct native API: max_steps is the absolute Trainer horizon.
hf.train(resume_from_checkpoint="outputs/training/checkpoint-250")

# Shared API: steps means additional optimizer updates.
resumed = model.arti.trainer(
    engine="transformers",
    resume_from_checkpoint="outputs/training/checkpoint-250",
)
result = resumed.fit(
    train_dataset,
    steps=250,
    trainer_kwargs={"args": training_arguments, "data_collator": collator},
)
```

`resume_from_checkpoint=True` selects the latest native checkpoint in the
configured output directory. Supplying its `model.arti.st` also resolves to the
native directory when `trainer_state.json` accompanies it. A portable file
without TrainerState emits a warm-resume warning rather than claiming exact
data continuation. `fit(..., steps=0)` can restore without consuming a batch.

Use the same dataset, seeds, batching, accumulation and scheduler horizon for
continuous-versus-interrupted parity. Extending a schedule is a new experiment,
not parity with the old horizon. The qualified exact profile is a map-style
dataset, single process, zero data-loader workers and the native checkpoint
directory. Streaming state, worker-prefetch/RNG, distributed and sharded
checkpointing need separate qualification. `ignore_data_skip=True` deliberately
starts another data trajectory.

With `load_best_model_at_end=True`, direct `hf.train()` retains the native best
weight view. ARTI saves a separate coherent last-state training asset before
loading best weights; `session.save_checkpoint()` exports that last state and
preserves the current best view. The shared `session.fit()` returns the best
asset reference in `selected_checkpoint_path` while keeping the live session
and its optimizer at the last update. It never combines best weights with last
optimizer moments in a training checkpoint.

An ARTI-only inference directory is not an ordinary full-model
`AutoModel.from_pretrained` directory. Reload the host reference, construct the
same ARTI program/layer, and use `arti.ARTI.load` or the binding's asset loader.

## TRL SFTTrainer

The optional tested profile is TRL **0.29.1**, Transformers **5.13.0**, Accelerate
**1.14.0**, and single-process CPU. Install `arti-fit[trl]` for the bounded TRL 0.29.x
profile. Other TRL releases and fused LM-head kernels are not qualified here;
ARTI does not patch that kernel or claim CPU support for it.

```python
from trl import SFTConfig, SFTTrainer

session = model.arti.trainer(engine="transformers", learning_rate=2e-4)
sft = session.transformers_trainer(
    dataset,
    trainer_class=SFTTrainer,
    trainer_kwargs={
        "args": SFTConfig(
            output_dir="outputs/sft",
            max_steps=500,
            learning_rate=2e-4,
            completion_only_loss=True,
            max_length=None,
            packing=False,
            save_only_model=False,
            report_to=[],
        ),
        "processing_class": tokenizer,
        "eval_dataset": validation_dataset,
    },
)
sft.train()
```

Use a prompt-completion dataset to supervise only the final answer. A
conversational prompt contains the full earlier messages and latest user
message; completion contains only the final assistant response. A plain
`messages` dataset with assistant-only loss may supervise earlier assistant
turns too, so it is not automatically equivalent to ARTI's final-answer recipe.
Pretokenized `input_ids`/`attention_mask`/`labels` are also supported; excluded
targets use `-100`. The bridge does not replace these labels with language-model
targets or distill chain-of-thought.

TRL owns its truncation/packing policy. Choose `max_length=None` or validate a
chosen limit against your records before training. ARTI's strict Qwen recipe
and TRL's native preprocessing are separate choices, not silently interchangeable.

The bridge does not initiate model/data upload or download. Pass an already
loaded processor as well as the model for offline operation; the native trainer
may otherwise resolve its own tokenizer. Transformers/TRL
callbacks can use their own explicit reporting integrations. The existing local
recipe reporter remains separate from the engine-owned lifecycle. Ray,
FSDP/DeepSpeed, distributed resume and optimized serving profiles are not implied.

References: [Trainer API](https://huggingface.co/docs/transformers/en/main_classes/trainer),
[TRL SFT guide](https://huggingface.co/docs/trl/sft_trainer).

## Offline Workflow Example

```bash
python examples/transformers_training_workflow.py --output .tmp/hf-workflow
python examples/transformers_training_workflow.py --trainer trl --output .tmp/trl-workflow
```

Both examples initialize a tiny native Qwen locally, interrupt a six-update
schedule after two updates, resume the remaining four, export and reload. They
check actual ARTI parameter changes, a frozen host, twelve accumulated
microbatches and exact inference reload parity. They establish integration
correctness, not pretrained Qwen task quality or GPU throughput.
