# Recall Training

For a bank-first Recall adapter, train the ordinary model output against the
real task target. The task loss is the signal that teaches bank values what is
useful; Recall does not need a separate image, language, or task-specific target
API.

```python
from arti import recall_route_exterior_penalty

anchor = layer(clean_x, mask=mask)
similar = layer(augmented_x, mask=mask)

task_loss = criterion(head(similar.y), target)
route_penalty = recall_route_exterior_penalty(
    anchor,
    similar,
    tolerance=0.01,
    mask=mask,
)
loss = task_loss + 0.01 * route_penalty
loss.backward()
```

`recall_route_exterior_penalty` is optional. It only penalizes route drift
beyond the tolerance and detaches the anchor route. It is an exterior
regularizer, not a teacher target, and should not replace the standard task
loss.

## Virtual Trace Compatibility

Experiential recall training aligns a corrupted input's internal recall trace to the clean input's latent output. Recall is treated as a layer-internal trace of prior signal processing, not as an external task memory.

```python
from arti.training import experiential_recall_alignment_loss

task_loss = criterion(head(clean_out.pooled), labels)

recall_loss, clean_out, corrupt_out = experiential_recall_alignment_loss(
    layer,
    clean_x,
    corrupt_x,
    coord=coord,
    mask=mask,
    epoch=epoch,
    align_start_epoch=2,
)

loss = task_loss + 0.1 * recall_loss
loss.backward()
```

Use this when the model should learn an internal recovery path for noisy, incomplete, or partially phase-corrupted tensors.

`virtual_recall_alignment_loss` remains available as a backward-compatible alias.

This helper remains available for virtual-trace experiments and artifact
compatibility. It is not the default objective for teaching a bank-first Recall
adapter.

After positive traces are established, an optional selectivity stage can enable
learned recognition and include unrelated inputs in the auxiliary objective:

```python
from arti import experiential_recall_selectivity_loss
from arti.legacy import ARTILayer

layer = ARTILayer(
    input_dim=hidden_dim,
    recall_recognition_mode="alignment",
)

loss, clean_out, corrupt_out, unseen_out = experiential_recall_selectivity_loss(
    layer,
    clean_x,
    corrupt_x,
    unseen_x,
    mask=mask,
    unseen_mask=unseen_mask,
    epoch=epoch,
)
```

The positive branch aligns a corrupted view of an experienced signal to its
complete processing trace. The negative branch drives `unseen_out.recall_influence`
toward zero. No identity labels or future query targets are required. This
second stage is opt-in. Use `recall_recognition_mode="explicit"` when a fixed
trace-agreement rule is preferred, or leave the default `"none"` when the task
does not require abstention.
