# Independent Task Episodes

An application can declare interactive episodes on the existing prediction run.
The exported TaskPredictor and TrainingBinding still own the native model;
the application supplies policy, environment and terminal evaluation callbacks.
Completed results use the same atomic parts, inspection, scoring and export
services as file and JSON prediction.

```python
episodes = arti.EpisodeRunConfig(
    adapter="my_application.episodes:build_adapter",
    policy_id="my-policy-revision",
    environment_id="my-environment-revision",
    evaluator_id="my-outcome-revision",
    initial_state_field="my_initial_identity",
    seed_field="my_seed",
    trial_field="my_trial",
    recovery="snapshot",
    effects="local",
    policy_state_lifetime="trial",
    environment_state_lifetime="trial",
    native_model_state_lifetime="read-only-policy-step",
    independent_batch=True,
    max_steps=my_step_allowance,
    max_calls=my_reported_call_allowance,
    field="my_episode_result",
)
run = arti.PredictionRunConfig("outputs/episodes", batch_size=my_trial_batch, episodes=episodes)
result = arti.run_prediction(
    task_package, requests, run, task_factory=my_task_factory,
)
```

Names, seeds, state identities, observation/action meaning, task termination,
resource units and evaluator rules belong to the application. A task-owned
initial-state identity and nonnegative integer seed are required by this local
profile; an unknown physical initial state is not established by a supplied
label. Unseedable or unknown external state needs separate evidence rather than
a fabricated seed or paired-trial claim.

## Caller State And Model Lifetime

The installed factory returns an `EpisodeAdapter` with `start`, `observe`,
`policy`, `step` and `evaluate` callbacks; optional `initial_policy`, `snapshot`
and `restore` callbacks declare supported application-state recovery. The policy
receives the native model, observations and separate states for the declared
dispatch: one lane or an explicitly independent batch.
Each environment and policy state belongs to one trial. Explicit
`independent_batch=True` declares that the caller's policy computation is also
independent across those lanes. The default `False` dispatches singleton policy
callbacks; state isolation alone does not prove computational independence.
The plan/result receipts retain the declaration and actual dispatch scope.
The active set can shrink as trials terminate; the final partial batch remains
independent under the caller's declared policy semantics.

This profile supports finite JSON application state and structured observations,
actions, responses and outcomes. Policy state can live for one step or the trial;
environment state lives for the trial. It does not silently serialize tensors,
unregistered Python state or external objects. Immutable file inputs can remain
declared task resources. Dynamic binary episode-state codecs and native mutable
model persistence are unsupported and reported explicitly.

The native model lifetime is separately declared as `read-only-policy-step`.
TrainingBinding restores owned model parameters/resources/buffers/modes and
supported RNG after each policy callback and at the whole episode batch boundary.
The application trial state evolves independently. Declaring another native
model lifetime fails before model loading; JSON state is not proof of native
state recovery. Training objectives, parameter ownership, preprocessing and
training checkpoint semantics stay with the original task.

## Traces, Completion And Software Allowances

Each logical ID retains its original initial-state identity/seed/trial, ordered
observations, proposed actions, executed responses, actual outcome and termination
reason. Policy assertions of success are raw policy output. The environment's
terminal evaluator supplies outcome evidence; these are distinct authorities.

Task termination and host truncation have separate booleans and reasons. Step,
reported-call, requested-stop and elapsed allowances stop at episode step starts.
An active callback can overrun its allowance. They are software controls, not
memory/network/process-tree caps. The actual runtime host's direct-worker deadline
is a separate receipt and cannot be relabeled task termination.

Counters disclose their scope: policy batch callbacks are not model forwards;
completed environment step callbacks are library-observed returned callbacks;
underlying call/resource consumption returned by the application is attributed
to that application. Local replay calls are separate from new execution. Original
and recovered attempts/trace phases remain visible, including an action proposed
before interruption whose response was executed after recovery.

## Recovery And Ambiguous Effects

Incomplete state lives inside the current prediction journal's work directory.
Before initialization or an environment action, the runner durably records the
intent; response, next state and trace are saved afterward. The existing atomic
result part remains the output publication boundary.

- `snapshot` restores explicitly supported local environment and policy JSON
  state at the saved boundary. It does not undo physical external effects.
- `replay` reconstructs a caller-declared local simulation and verifies original
  observations, responses, termination, reason and resource/call evidence.
  Recorded policy actions are reused; policy inference is not rerun.
- `none` reports initialized unfinished work as non-resumable. It never claims
  recovery from state that was not saved or supported.

A pending external initialization/action is ambiguous even when snapshots were
declared. Recovery returns `not_run` before adapter/model import and performs
zero automatic repetitions. Caller reconciliation or a separately authorized
new lifecycle is required. The library promises neither exactly-once physical
effects nor automatic real-world tool authority.

Completed parts require no model, adapter or evaluator to inspect/resume. Fully
finished but unpublished trial results can also be adopted into the existing
part without those dependencies. Their original execution remains unchanged;
publication is separately attributed. Undurable prior timing stays unrecorded,
rather than being replaced by a fictitious fresh inference time.

## Evaluation, Handoff And Export

Configure `EvaluationTrialFields` with the original declared initial-state/seed/
trial fields. Suites retain original per-trial coverage, unavailable reference
views, caller criteria/aggregation, actual row execution and original/recovered
trace evidence. Paired comparisons verify matching original inputs and declared
trial context and the actual committed episode state/seed/trial projection from
each arm, even when their configuration selects different input field aliases.
Execution outcomes keep a prior interrupted attempt explicitly unavailable with
no terminal outcome; the observed recovered outcome belongs to its own attempt.
Imported external evidence retains its producer/protocol identity
and is not described as locally reproduced environment behavior.

The current episode profile commits structured traces; a simultaneous staged
file producer or ordinary `predict_kwargs` profile is unsupported. Existing
independent file prediction is unchanged. The application result writer may
format committed episode data into its own files through `ResultExportConfig`;
this does not replay episodes. Include original journals and required input/
suite resources explicitly for portable resumed execution.

A selected packaged runtime entrypoint can reconstruct `EpisodeRunConfig` from
its options and invoke the same prediction/validation service in a fresh process.
The neutral `examples/episode_task.py` callbacks exercise the native regression
model, independent numeric environments and separately attributed outcomes.
These checks qualify local lifecycle behavior, not task quality or real-world
effects.

```bash
arti recipe predict-run outputs/package --data trials.jsonl --run-dir outputs/episodes --episode-config episodes.json --task-factory my_application.task:build_task
arti recipe inspect-prediction outputs/episodes
arti recipe predict-run --run-dir outputs/episodes --resume
arti recipe validate-deployment outputs/package --data trials.jsonl --run-dir outputs/episodes --resume --episode-config episodes.json --evaluation-suite suite.json --targets references.jsonl --task-factory my_application.task:build_task
```

Episode JSON/TOML uses `format="arti.episode-run-config"` and the Python field
names. No environment/tool integration, dependency installation or external
execution is implied by a declaration.
