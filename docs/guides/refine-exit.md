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

policy = arti.ExecutionPolicy.adaptive(
    max_steps=32,
    min_steps=4,
    scope="token",
    executor="static_masked",
    trace_level="routes",
)

y, trace = recall(
    x,
    execution_policy=policy,
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

The default scope is token-local. With K-wide K-wide branch search, each flattened
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

Evaluate a learned exit against fixed-depth Recall on disjoint validation and
test samples. Select quality tolerances on validation data, then execute the
hard runtime once on the held-out test set. Report task quality, the logical
step distribution, and every non-model termination reason separately. Logical
step reduction alone is not evidence of physical runtime acceleration.
