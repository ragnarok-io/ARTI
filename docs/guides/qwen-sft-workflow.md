# Qwen Supervised Fine-Tuning Workflow

This recipe connects real conversations, the host model's causal-LM objective,
checkpoint continuation, independent evaluation and a portable inference bundle.
It uses the same `ARTITrainingSession` as classification, native programs and
model attachments. It does not create another training engine.

[Recover a Task Run](recipe-recovery.md) covers restoring the original optimizer
target and finishing evaluation/export without repeating completed training.
Use [Prepare Qwen Training Data](qwen-data-preparation.md) to reuse prepared
supervision across plan, training and recovery without loading host weights.
Use [Import Local SFT Data](qwen-data-import.md) for explicitly split chat or
prompt/completion exports and portable source lineage.
Use [Evaluate Generated Answers](qwen-evaluation.md) for declarative generation
checks, answer-based checkpoint selection and offline receipt scoring.
Use [Independent Asset Evaluation](qwen-independent-evaluation.md) for new
test-only data without rebuilding training state.
Use [Recoverable Prediction Runs](qwen-prediction-runs.md) to commit large
inference jobs by batch and avoid regenerating completed outputs after interruption.
Use [Hand Off a Qwen Deployment](qwen-deployment.md) to inspect dependencies and
move a native-base package without relying on the training machine's cache.

Install the optional model and serving integrations:

```bash
pip install "arti-fit[qwen,serve]"
```

## Data And Supervision

Use JSONL records with globally unique IDs and explicit split membership:

```json
{"id":"train-001","split":"train","messages":[{"role":"user","content":"What is 2 plus 2?"},{"role":"assistant","content":"4"}]}
{"id":"val-001","split":"validation","messages":[{"role":"user","content":"What is 3 plus 4?"},{"role":"assistant","content":"7"}]}
```

The entire conversation is input, including any previous turns. By default only
the final assistant answer and its native EOS contribute to loss. User/system
tokens, earlier assistant turns and padding have label `-100`. A real EOS still
has a target when PAD and EOS share an ID. The recipe never substitutes tensor
alignment or invented responses for the task's answers.

Training and generation use the saved native chat template and identical
template kwargs. The final template rendering must preserve the generation
prefix. Prompt/completion tokenization is separate so BPE cannot move the last
prompt token across the target boundary. The tokenizer vocabulary, special
tokens and template are fingerprinted and shipped with the model.

Preparation is once per dataset, not once per optimizer update. Batches use
length-grouped deterministic shuffling and dynamic right padding rounded to a
configured multiple. `max_length` is a declared input-size policy, not a limit on
a teacher's thinking. Its default behavior is an error; explicit `drop` records
the dropped IDs. Answers are never silently truncated.

This first recipe supports text-only system/user/assistant messages and final
answer supervision. Tool calls, images and explicit reasoning traces require a
different task adapter, rather than silently dropping their fields.

## One Resolved Recipe

```toml
task = "causal-lm"
seed = 17
max_grad_norm = 1.0

[model]
id = "Qwen/Qwen3-0.6B"
revision = "c1899de289a04d12100db370d81485cdf75e47ca"
dtype = "bfloat16"
local_files_only = true
attn_implementation = "sdpa"

[data]
path = "conversations.jsonl"
cache_dir = "prepared-chat"
batch_size = 16
max_length = 256
overlength = "error"
pad_to_multiple_of = 8
group_by_length = true

[data.template_kwargs]
enable_thinking = false

[adapter]
rank = 32
activation = "silu"
last_layers = 2

[training]
device = "cuda"
steps = 200
learning_rate = 0.001
mixed_precision = "bf16"
gradient_accumulation_steps = 1

[optimizer]
weight_decay = 0.01
scheduler = "constant"
implementation = "auto"

[generation]
max_new_tokens = 64
batch_size = 8

[run]
output_dir = "runs/first"
checkpoint_steps = 25
checkpoint_seconds = 30
evaluation_steps = 50
```

The default model adaptation is an explicit native Formula program: projection,
activation, projection, residual addition. Its operands are FP32 trainable
tensors; the pretrained Qwen host is frozen and may run in BF16. Zero output
initialization preserves initial host outputs while connecting gradients. This
fixed profile is not a claim of architecture-search or full-library superiority.
`adapter.layers` can select exact paths instead of `last_layers`.

The CLI plan uses the native configuration and a meta-device model. It inspects
structure and preprocessing without loading base weights, creating an optimizer
or allocating CUDA memory. `check` is an explicit real gradient preflight.

The program is lowered once to a shared tensor-only plan. Default training does
not build graph views or query equality diagnostics inside every CUDA call.
CUDA uses fused AdamW by default; CPU uses foreach AdamW. Gradient clipping,
mixed precision and optional native gradient checkpointing are explicit.
Scheduler state is saved; linear/cosine schedules require `total_steps` for the
whole planned lifecycle, including subsequent continuation. Changing stages
does not silently restart a scheduler.

Advanced Python callers can supply a preloaded host/tokenizer and an ARTILayer
factory. Their program remains subject to its real graph/serialization contract;
arbitrary Python model code is not automatically packaged or lowered.
Such assets declare `adapter_factory="caller-supplied"` and require the same
factory at `CausalLMPredictor.from_pretrained(..., layer_factory=...)`. The loader
reports a missing factory before loading host weights; CLI deployment does not
guess a default program. Custom forward-mutated state is caller-managed, not
implicitly covered by the default read-only adapter serving profile.

## Train And Continue

```bash
arti recipe prepare-data recipe.toml --output prepared.json
arti recipe plan recipe.toml --output plan.json
arti recipe check recipe.toml
arti recipe train recipe.toml --run-dir outputs/first
arti recipe train recipe.toml --resume-from-checkpoint outputs/first/checkpoints/step-00000200/model.arti.st --steps 40 --run-dir outputs/continued
```

`--steps` means additional updates. The resolved configuration, split IDs,
tokenizer and plan are retained before training. The first completed update is
saved, followed by periodic and final checkpoints. Resume restores ARTI weights,
optimizer, scheduler, RNG/scaler, epoch order, cursor and token/sample counters.
Model revision, template, task labels and optimizer/data-order contracts must
match. Use the returned checkpoint path when a budget ends early.

### Validation Selection

Default export still uses the last model; automatic early stopping is off.
Declare an optional policy alongside `run.evaluation_steps`:

```toml
[run.validation]
monitor = "loss"
mode = "min"
selection = "best"
patience = 6
min_evaluations = 4
min_delta = 0.0001
```

Built-in validation includes target-token NLL (`loss`), perplexity and
`teacher_forced_token_accuracy`. They are not generation correctness. Python
callers can pass `validation_fn=lambda session: recipe.evaluate("validation",
generate_samples=32)` and monitor `generation_exact_match` with `mode="max"`;
this explicitly adds generation cost. No teacher/reasoning trace is introduced.

`checkpoint` remains the last complete training state. `selected_checkpoint`
identifies the chosen inference source, and `selection` records its metric/step.
Final validation/export use that source while the live recipe remains at last.
`recipe.evaluate("test", checkpoint=result["selected_checkpoint"],
generate_samples=32)` evaluates the exported model's source. Deployment metadata
contains selection provenance, not source machine paths. Resume with the same
policy/cadence; see [Training Workflow](training-workflow.md#select-and-stop-on-validation).

Microbatch losses are weighted by their actual target-token counts before
gradient accumulation. Unequal answer lengths therefore match a single joint
token-mean task batch; they are not mistakenly equal-weight microbatch means.

```python
import arti

recipe = arti.CausalLMRecipe.from_config("recipe.toml")
assert recipe.check()["ok"]
result = recipe.fit()
metrics = recipe.evaluate("test", generate_samples=16, output="outputs/test.json")
```

Workflow status distinguishes training, evaluation, export and completed. A
failed export cannot report a deployed model even when its training checkpoint
has completed. The local recipe currently qualifies single-process Torch; a
sharded checkpoint/distributed launcher is a separate execution profile.

## Evaluate The Actual Task

```bash
arti recipe evaluate recipe.toml --model outputs/continued/model --split test --generate-samples 16 --output outputs/test.json
arti recipe evaluate outputs/continued/model --data new-labeled.jsonl --device cuda --generate-samples 16 --output outputs/fresh-test.json
```

Evaluation reports target-token mean NLL/perplexity and clearly labeled
teacher-forced token accuracy. Generation receives only the conversation before
the final gold answer. Its raw text, token IDs, stop/length reason and exact text
match are written against the original record IDs. A generation hitting its
token budget is not reported as EOS completion.

NLL reduction alone is not evidence of correct answers. Exact text match can
also differ from a task-specific semantic metric; downstream applications should
evaluate their declared answer schema and retain the raw predictions.

`session.evaluation_context()` preserves modes/RNG and ARTI state while callers
compute additional metrics. Custom task metrics do not require an additional
forward loop inside the core trainer.

## Deploy The Same Asset

For variable conversation lengths, [Token Budget Batching](qwen-token-batching.md)
keeps independent rows within an explicit padded-token budget across training,
teacher-forced evaluation and exact continuation.

The same recipe can use `training.engine="accelerate"` for the qualified local
single-process lifecycle. See [Train With Local Accelerate](accelerate-training.md)
for accumulation, checkpoint/continuation and precision boundaries.

Use [A Custom Qwen Program](qwen-custom-programs.md) to declare an installed
layer factory across the same train/resume/evaluate/export/CLI deployment workflow.

Use [Compare Qwen Evaluation Results](qwen-evaluation-comparison.md) to pair
candidate/baseline replies and identify regressions from stored receipts without
running generation again.

The exported directory contains `arti-hub.json`, `model.arti.st` plus bound
sidecars/native graphs, `tokenizer/`, and `deployment.json`. It carries canonical
base model ID + pinned revision, not the original local cache path. Base weights
remain separate, as in normal adapter deployment. No optimizer or training data
is required for inference.

```python
predictor = arti.CausalLMPredictor.from_pretrained("outputs/continued/model", device="cuda")
answers = predictor.predict_records([
    {"id": "request-1", "messages": [{"role": "user", "content": "What is 2 plus 2?"}]},
    {"id": "request-2", "messages": [{"role": "user", "content": "What is 3 plus 4?"}]},
])
```

Generation batches use native left padding, attention masks, KV caching and
greedy decoding. It is not equivalent to independently processing padded tokens
or concatenating unrelated conversations into shared attention.

```bash
arti recipe predict outputs/continued/model --data requests.jsonl --device cuda --output predictions.jsonl
arti recipe serve outputs/continued/model --device cuda --port 8080
```

The optional FastAPI/Uvicorn worker loads one asset once. `POST /predict` accepts
`{"records": [...]}` for actual batch inference. `POST /chat` accepts
`{"messages": [...], "max_new_tokens": 64}`. `/health`, `/docs` and `/openapi.json`
expose readiness and the contract. One worker serializes model use; multiple
workers each load their own model. Compatible requests now share dynamic batches
by default for the builtin read-only profile. Different generation limits/effective
batch sizes stay separate; histories remain independent batch rows, never one
concatenated conversation. This is not OpenAI API compatibility, authentication,
vLLM/SGLang/NIM/TensorRT-LLM support or session-isolated forward plasticity.

```bash
arti recipe serve outputs/continued/model --device cuda --batching auto --max-batch-size 32 --batch-wait-ms 4 --max-pending-records 1024
```

`max_batch_size` caps dynamic dispatch rows. The saved generation batch size (or
a request override) can select a smaller limit. A larger server cap alone does
not override the saved profile's memory choice. `batching=off` keeps per-request
dispatch; `on` is the developer's explicit assertion of independent row semantics
for a custom predictor. `auto` leaves caller-managed mutable state per-request.
The worker does not invent session storage or share a session's mutable tensor
between unrelated clients.

`/metrics` reports counts/aggregate time only: `predictor_calls` counts dispatches,
not autoregressive token forwards. Full queues return 503/Retry-After; an ASGI
disconnect skips queued work. Closing rejects queued work and awaits the active
model call, which cannot be interrupted by Python cancelling the HTTP task.
Read [Task Recipes](task-recipes.md#cross-request-serving) for Python/lifecycle use.

## Reproduce The Real Data Workflow

For a native Hugging Face engine rather than this recipe's local Torch lifecycle,
see [Transformers and TRL Training](transformers-training.md). It retains ARTI
parameter/asset ownership while delegating data preparation and the loop to the
chosen native Trainer; the two preprocessing/resume policies are explicit.

```bash
python examples/qwen_sft_workflow.py --prepare-only --output .tmp/qwen-sft
python examples/qwen_sft_workflow.py --output .tmp/qwen-sft --max-seconds 300
```

Preparation fetches pinned original GSM8K JSONL and extracts human final numeric
answers; calculation annotations/CoT are not supervised. Validation is held out
from source train records; test uses source test records. Data revision/file
digests, split IDs and all prediction records are saved locally.

The example freezes Qwen, trains 200 updates, restores and continues 40, evaluates
paired base/adapted test outputs, then reloads the deployed model and verifies
logits parity. The budget includes its bounded workflow and reserves saving and
evaluation time; an optimizer boundary may still require its final save tail.
Short sample subsets establish workflow validity, not a full GSM8K score.

Design references: [native chat templates](https://huggingface.co/docs/transformers/chat_templating),
[TRL assistant/completion loss semantics](https://huggingface.co/docs/trl/sft_trainer),
[Qwen3 model documentation](https://huggingface.co/Qwen/Qwen3-0.6B),
[GSM8K source](https://github.com/openai/grade-school-math),
[FastAPI](https://fastapi.tiangolo.com/tutorial/).
