# Formula-Controlled Refine Exit

`arti.mechanisms.FormulaRefineExit` is a stateless control atom that converts a
boolean predicate or floating-point halt logit into a hard
`RefineExitRequest`. It belongs to a control lane; it is not a Formula@2 data
slot and it never replaces the current Recall transition.

```python
import torch
import arti

head = torch.nn.Linear(64, 1)
exit_control = arti.mechanisms.RefineExitControl(
    head,
    input_kind="logit",
    threshold=0.0,
)

policy = arti.RefinePolicy.adaptive(
    max_steps=32,
    min_steps=4,
    scope="token",
    executor="static_masked",
    trace_level="routes",
)

y, trace = recall(
    x,
    refine_policy=policy,
    refine_exit=exit_control,
    model_exit=True,
    return_trace=True,
)
```

The runtime commits transition `d` before evaluating its exit request. An
effective request prevents transition `d + 1`; it cannot roll back transition
`d`. Requests before `min_steps` are recorded in
`trace.blocked_by_min_steps` but do not stop execution. `max_steps` remains a
hard host bound, and non-finite, convergence, and cycle monitors retain
precedence over model exit.

The stable lane requires `executor="static_masked"` and finite-state checking.
This keeps token and K-branch execution on a fixed tensor schedule and prevents
a controller from committing a non-finite state.

The default scope is token-local. With K-wide Batched Refine, each flattened
branch evaluates its own request and no request is reduced across K. Use
`model_exit=True` to enable the lane explicitly. Omitting it keeps the existing
Recall path unchanged even when the caller keeps a control module mounted.

`input_kind="logit"` accepts floating-point scores. Use
`input_kind="predicate"` for an already discrete boolean request; the two
contracts are not inferred from dtype. The convenience `RefineExitControl` is
a runtime-only composition around a caller-owned neural source. The stateless
atom is portable, while portability of a trained source belongs to the model
that owns it.

For token scope, the source maps independent `[B*N, D]` rows to `[B*N]` logits.
For branch scope, `RefineExitControl` first masked-pools each branch and the
source maps `[B, D]` to `[B]`. This structural boundary keeps padding, tokens,
and K branches out of one another's source layout. Sources with shared mutable
batch state or unkeyed randomness are outside the deterministic contract.

`RecallTraceV3` records the hard request, allowance, effective request,
minimum-depth block, actual model stop, differentiable score, and terminal
step. The score can train a controller with a separately defined surrogate;
the hard deployment decision itself is not presented as differentiable.

Logical masking is not evidence of lower FLOPs, memory traffic, or latency.
Such claims require a separate physical runtime measurement.

## Training From The Model's Own Depth Curve

`RefineExitTraining` trains only the exit controller. It does not require a
teacher stop sequence and it does not change the sequential Recall executor.
First train Bank and Formula with neural exit disabled across the full Refine
depth. Then freeze that task path and build a detached curve from the existing
flattened training contracts:

```python
from arti.mechanisms import RefineExitTraining, RefineStepTraining

step_training = RefineStepTraining()
exit_training = RefineExitTraining(
    compute_weight=0.01,
    quality_tolerance=0.02,
)

rollout = step_training.capture(
    recall,
    hidden,
    mask=mask,
    policy=full_depth_policy,  # min_steps == max_steps
)
step_result = step_training.replay(recall, rollout)
per_token_task_loss = task_loss(step_result.value)
curve = exit_training.build_curve(
    rollout,
    step_result,
    per_token_task_loss,
    scope="token",
)

optimizer = torch.optim.AdamW(exit_control.parameters(), lr=1e-3)
exit_training.assert_optimizer_contract(recall, exit_control, optimizer)
loss = exit_training.loss(exit_control, curve, min_steps=4)
loss.total.backward()
optimizer.step()
```

`arti/refine-exit-curve@1` contains detached post-transition states, real task
loss, masks, lineage, and snapshot fingerprints. It contains no stop label,
teacher hidden state, route cache, or model-exit trace. Token scope requires an
explicit `[P, N]` task loss; branch scope requires `[P]`. The API does not
silently broadcast a sequence loss across tokens or aggregate across K
branches.

The training loss treats controller logits as a continuous stop hazard. The
last depth always stops, depths below `min_steps` cannot stop, and the loss
combines expected task quality with expected logical depth plus a per-item
quality constraint against the full-depth result. `compute_weight` defaults to
zero and must be enabled explicitly. Logical expected depth is a training
quantity, not a claim about physical FLOPs or latency.

Curve states and task losses are detached before controller training. The
optimizer contract requires exactly the trainable controller parameters, so
the fixed Query, Bank, Formula, task head, and Recall transition receive no
gradient during this phase. Deployment still uses the hard post-transition
request from `FormulaRefineExit`.

Two bounded benchmarks keep the evidence levels separate:

- `benchmarks/train_refine_exit_task_curve.py` is a synthetic objective smoke
  test. It checks that the hazard can assign different depths when states carry
  different convergence rates.
- `benchmarks/train_refine_exit_combined.py` trains a real `arti.Recall` at full
  depth, freezes it, trains the controller from replayed task loss, and finally
  evaluates the hard `model_exit=True` runtime against fixed-depth Recall.
- `benchmarks/train_qwen_refine_exit_next_token.py` freezes a locally cached
  Qwen model, trains Recall from full-vocabulary next-token cross-entropy on
  complete and disjoint train/control/validation/test text splits, then evaluates
  learned hard exit against fixed Refine depths on the held-out test split. It
  uses controlled latent corruption to expose a bounded repair task.

The combined benchmark uses a controlled classification task so its result is
mechanism evidence, not a claim about a particular pretrained model or
downstream dataset. It fails unless learned hard-exit loss remains within the
explicit `--quality-loss-tolerance` of the fixed full-depth loss for the same
seed. A production evaluation should keep complete samples
separate across train, validation, and test, select quality tolerances on
validation data, and execute the hard Recall runtime once on the held-out test
set. It should report task quality, actual logical step distribution, and all
non-model termination reasons separately.

The Qwen gate is pretrained-model downstream evidence, but its scope remains
narrow: next-token prediction from cached final hidden states under controlled
corruption. It does not establish open-generation quality, Qwen fine-tuning, or
physical runtime acceleration. Its default hard-exit gate preserves full-depth
loss within `0.05` while requiring at least 25% fewer logical Refine steps.
