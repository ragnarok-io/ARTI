# Flattened Refine Training

`arti/refine-rollout@1` and `arti/refine-step-training@1` are stable training
contracts for deep Recall. They reduce backward graph depth without changing
the sequential deployment behavior.

The workflow has two phases:

1. Run the canonical Recall executor without gradients and record the detached
   hidden state before each committed Refine transition.
2. Replay every valid state through one differentiable Recall step. Each replay
   computes a fresh Query from its own hidden state and is supervised only by
   the real downstream task loss.

```python
import torch
import torch.nn.functional as F

import arti
from arti.mechanisms import RefineStepTraining

recall = arti.Recall(64, 128, activation="none")
trainer = RefineStepTraining(max_snapshot_staleness=0)
policy = arti.RefinePolicy.adaptive(
    max_steps=16,
    min_steps=16,
    scope="token",
    relative_tolerance=1e-5,
)

hidden = torch.randn(8, 32, 64)
mask = torch.ones(8, 32, dtype=torch.bool)
training_step = 0
rollout = trainer.capture(
    recall,
    hidden,
    mask=mask,
    policy=policy,
    snapshot_generation=training_step,
)

result = trainer.replay(
    recall,
    rollout,
    current_generation=training_step,
)
logits = frozen_or_trainable_task_head(result.value)
per_token_loss = F.cross_entropy(
    logits.transpose(1, 2),
    target_tokens_for_each_flattened_state,
    reduction="none",
)
loss = trainer.reduce_task_loss(
    per_token_loss,
    result,
)
loss.total.backward()
```

The caller owns the task head, targets, optimizer, gradient accumulation, and
snapshot refresh schedule. `RefineStepTraining` does not perform an optimizer
step. Before training, use `assert_optimizer_contract(recall, optimizer)` to
verify that the fixed Query is absent from optimizer parameter groups.

## Contract Boundaries

- The Query producer is fixed and is never optimizer-owned. Its runtime value
  still depends on the latest hidden state, so gradients may flow to that
  hidden input.
- The rollout stores pre-transition hidden states, masks, transition status,
  branch identity, step identity, and fingerprints. It does not store a
  next-hidden teacher.
- Replay accepts no route plan. The first item in each K-wide branch regenerates
  the current Top-K candidates; later items query directly from their own
  branch-local hidden state.
- Only live, finite, committed token transitions enter the loss. Stopped,
  padded, masked, and rejected transitions remain in provenance but have zero
  training weight. Finiteness is checked from the actual post-transition state;
  that temporary state is discarded and never exposed as a training target.
- Loss is averaged inside each source trajectory before trajectories are
  averaged inside each source sample. A longer rollout or wider K therefore
  does not receive more statistical weight merely because it produced more
  adjacent pairs or branches. A wholly inactive source sample has zero weight;
  a partially surviving depth trajectory or K-branch set fails closed rather
  than being silently renormalized.
- Version 1 accepts fixed-depth rollouts only. Learned/adaptive exit is trained
  later from full-depth quality curves; stopped sampling cannot silently claim
  an unbiased full-depth objective.
- Version 1 also requires deterministic Recall execution. Use `activation="none"`
  and disable dropout and route exploration. Identity-keyed stochastic replay is
  a separate future contract; it is not approximated with ambient RNG state.
- A data-dependent state Bank must be calibrated before capture. Rollout capture
  is read-only and fails rather than silently performing lazy Bank mutation.
- A rollout replayed at its recorded snapshot generation is on-policy. Reuse
  after model updates is explicit bounded off-policy replay and requires a
  nonzero `max_snapshot_staleness`; structure,
  behavior, execution mode, Formula program, and fixed Query identity remain
  fail-closed. K-wide rollouts require zero staleness so candidate branch
  identity cannot drift across parameter generations.
- The capture/replay coordinator is an eager Python training utility. It reuses
  the canonical Recall kernels, but version 1 does not claim that the whole
  orchestration function is a single `torch.compile` graph.

An optional local-improvement hinge may compare candidate task loss with the
same task loss evaluated at the detached source state. It is an auxiliary
pressure only. Current-model next hidden states are never imitation targets.

This mechanism does not parallelize the causal Refine trajectory. Rollout
generation is still sequential. It makes the backward graphs one step deep and
lets compatible states across depth, samples, and branches share a training
batch.
