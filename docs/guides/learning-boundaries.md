# Learning Boundaries

ARTI learning boundaries separate forward data connectivity from backward
credit connectivity. A `CreditBoundary` is exactly identity in the forward
data lane. It belongs on a named `Connection`, where it controls credit from
that destination port back through the declared relation.

```python
import torch
from arti import CreditBoundary, CreditBoundaryMode, Connection, ResourcePort

boundary = CreditBoundary(mode=CreditBoundaryMode.MEAN)
connection = Connection(
    "decode",
    ResourcePort("middle"),
    ResourcePort("output"),
    credit_boundary=boundary,
)
```

`OPEN`, `CLOSED`, `MEAN`, and `BERNOULLI` are explicit runtime modes. Calling
`train()` or `eval()` does not alter them. `CLOSED` preserves the forward
TensorView exactly while supplying a zero VJP through that connection.

For `BERNOULLI`, a graph call can receive one explicit boolean mask per
connection. Reusing that same tensor during a checkpoint recomputation
reproduces the declared VJP without resampling on the host:

```python
mask = torch.tensor([[[True], [False]]], device=source.device)
graph.execute(("decode",), credit_masks={"decode": mask})
```

The same `credit_masks` mapping is accepted by serial, functional, parallel,
program, and bounded-loop graph execution. It is an execution input rather
than a new forward feature or resource payload.

Static `ResourceGraphExecutionPlan` lowering exposes each Bernoulli mask as a
separate tensor input as well. This keeps replay compatible with
`torch.compile`; it establishes compiled correctness, not an automatic claim
that a dynamically gated backward path saves kernels or wall-clock time.

## Structure Windows

Only `CLOSED` represents a committed credit cut. A `MEAN` probability is not a
structural decision. At an optimizer-window boundary, graph declarations can
be projected into logical optimizer domains:

```python
from arti import OptimizerExecutionPlan, ProgramParameterBinding

plan = OptimizerExecutionPlan.from_program_graph(
    graph,
    (
        ProgramParameterBinding("encoder_weight", "encode", encoder.weight),
        ProgramParameterBinding("decoder_weight", "decode", decoder.weight),
    ),
    learning_sources={"encode": ("task",)},
    learning_rate=1e-4,
)

# After an explicit structure decision changes a connection to CLOSED:
next_plan, receipt = plan.commit_program_graph_structure(
    graph,
    bindings,
    learning_sources={"encode": ("task",)},
)
```

The commit derives regions from declared graph edges, preserves state for
unchanged parameter identities, and gives parameters used by multiple regions
one `shared` owner domain. `domain_overrides` can assign distinct AdamW
settings to committed domains.

## Region-Specific Objectives

Different regions can learn from different scalar losses in the same forward
and optimizer transaction. Bind each objective to the region that owns its
parameters; a shared parameter receives the sum of the objectives from its
use regions. An optional first argument supplies a task loss common to all
active domains.

```python
task_region = plan.ownership.owner_for("task_weight").owner_domain_id
exit_region = plan.ownership.owner_for("exit_weight").owner_domain_id

receipt = plan.step_with_losses(local_objectives={
    task_region: answer_loss,
    exit_region: exit_loss,
})
```

For a learned stop signal, `exit_loss` can be defined on its continuous score;
the hard stop request alone has no ordinary gradient. `gradients_from_losses`
exposes the same per-owner batched VJP without committing an update. Both
methods leave `parameter.grad` untouched and use the existing grouped AdamW
plan, so omitting a region's objective skips its parameters rather than
silently applying weight decay or old momentum. `vjp_batch_size` bounds the
number of objective gradients materialized together when many regions learn
in one step.

## Learning Rules

The ordinary forward loss does not produce a standard gradient for a boundary
permeability. Use an explicit credit-gradient field and evaluate the resulting
candidate update with a later query objective:

```python
from arti import credit_gradient_field

field = credit_gradient_field(
    {"task": support_loss},
    {"task": boundary.permeability},
    {"encoder_weight": encoder.weight},
)
plan.step(gradients=field.as_dict())
```

`CreditStructureChoice` learns a separate direct-versus-boundary structural
probability from *paired candidate-update* query losses. It does not infer a
structure from two identical current forward outputs and does not threshold or
commit a connection by itself.

For a first-order SGD trial, `paired_credit_update` constructs both candidates
from the same live parameter snapshot without mutating it. The query evaluator
receives a functional parameter mapping, so a normal module can be evaluated
with `torch.func.functional_call`:

```python
from torch.func import functional_call
from arti import CreditStructureChoice, paired_credit_update

paired = paired_credit_update(
    {"stable": stable_support_loss, "conditional": conditional_support_loss},
    direct_scales={"stable": torch.ones(()), "conditional": torch.ones(())},
    boundary_scales={
        "stable": torch.ones(()),
        "conditional": boundary.permeability,
    },
    parameters=dict(model.named_parameters()),
    query_loss=lambda trial: task_loss(functional_call(model, trial, query_inputs)),
    learning_rate=1e-3,
)
structure_loss = paired.structure_objective(CreditStructureChoice())
```

The caller owns the data split: support losses supply the candidate update and
the query evaluator supplies only its subsequent evaluation. Neither trial is
committed by this helper; ordinary training chooses and applies its own support
update after the outer rule has been evaluated.

For one Bernoulli boundary, evaluate the replayed `False` and `True` masks as
two paired trials and pass their query losses to
`bernoulli_expected_query_loss(closed, open, boundary.permeability)`. This is
an exact two-outcome objective for one gate. It is not the same as applying a
mean credit scale once, and it does not yet enumerate a large correlated mask
field. For a sampled multi-gate mask, `bernoulli_score_function_objective`
provides the corresponding REINFORCE policy term with an optional scalar
baseline. Its query loss is detached by design: it trains the mask
distribution, not the ordinary model parameters through query data.

When a candidate must retain AdamW semantics, an `OptimizerExecutionPlan` can
produce a `functional_adamw_trial(gradients=...)`. The result includes trial
parameters, per-parameter moments, and domain clocks while leaving the live
parameters and `OptimizerStateArena` unchanged. A later `plan.step(...)` is
still the only ordinary committed update.

For a committed, single-device topology, `prepare_static_step()` binds the
same parameters and state arena into a fixed tensor-only update module. It can
be lowered by `torch.compile`; its runtime inputs are a boolean domain update
mask, a boolean parameter-presence mask, and one dense gradient tensor per
declared parameter:

```python
prepared = plan.prepare_static_step()
compiled_step = torch.compile(prepared, fullgraph=True)
parameter_updates, domain_updates = compiled_step(
    domain_update_mask,
    gradient_present,
    *ordered_gradients,
)
```

The prepared module borrows the plan's parameter and moment tensors. Call
`prepared.synchronize_control_state()` at a control-plane boundary before
returning to eager `plan.step(...)` or committing a new topology. Dynamic
masks preserve state for inactive entries; they do not by themselves prove
that the backend removes their update arithmetic or improves wall-clock time.

At a completed structure window, use the paired trial result to make the
discrete graph decision. `CreditStructureChoice.specialize(...)` and
`paired.structure_decision(...)` compare the actual candidate query losses;
they intentionally do not threshold `beta`. A numerical tie resolves to
direct connectivity by default, with an explicit option to retain a declared
region boundary. `CreditStructureCandidate` binds that evidence to one
declared graph connection. At the end of a training window,
`plan.commit_credit_structure_window(...)` validates all candidates, applies
their chosen `OPEN` or declared boundary modes together, then rebuilds the
optimizer domains from the same graph snapshot. A candidate only creates a
sealed region when it explicitly selects `CreditBoundaryMode.CLOSED` with
`seal_on_boundary=True`; ordinary `MEAN` or `BERNOULLI` selection does not
silently sever a learning region.

The rebuild can receive a complete `domain_overrides` mapping keyed by the
fresh committed region ids (and `shared`, when ownership creates a shared
domain). This is the control-plane point for independent learning rates,
schedules, or cadence; it does not create one Python optimizer per region.
Overrides must cover every new domain, and a region without a declared learning
source cannot be marked active.

## Local Credit Lowering

Registered Fabric layers declare a local pullback alongside their named ABI.
The default `local_vjp="autograd"` derives that pullback from the ordinary
PyTorch module. A layer with a specialized numerical rule can instead provide
a callable returning `LocalVJPResult`; ARTI validates its input and parameter
cotangent counts and shapes when the static graph is traversed.

```python
@arti.fabric_layer(
    inputs=("source",),
    outputs={"value": "source"},
    local_vjp="autograd",
)
class Scale(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return source * self.scale
```

For a fixed, direct `ProgramGraph`, `StaticProgramGraphExecutionPlan` can
then execute a separate local-reverse program. Each node sees detached local
input activations, so its VJP cannot silently traverse another node. Terminal
cotangents are composed backwards over the same SSA publication order as the
forward plan, and shared parameter contributions are accumulated under stable
`node_id.parameter_name` identifiers:

```python
credit = plan.credit_gradient(
    *resource_values,
    terminal_cotangents={"output": output_cotangent},
)
gradients = credit.parameter_cotangents
```

Formula nodes use the same local autograd rule. This is currently a
correctness-oriented lowering, not a claim of compiled backward throughput.
`StaticProgramJoinExecutionPlan.credit_gradient(...)` returns its per-row
`ready`, remaining-arrival, and publication receipts. A row that did not fire
is an identity carry in reverse; only a fired row traverses the join node's
local VJP. `StaticProgramLoopExecutionPlan.credit_gradient(...)` analogously
returns bounded per-iteration activity masks and treats every continuation
comparison as discrete control evidence rather than inventing a derivative
through it. `StaticDataflowProgramExecutionPlan.credit_gradient(...)` composes
the same ordinary-node and join receipts over a fixed operation schedule.
Unbounded routes and mixed connection/program regions remain outside this
lowering until they expose an equivalent combined receipt.

Direct static `Connection` fragments expose the same `credit_gradient(...)`
entry point. Their ordinary transfer is locally differentiated with autograd,
then any declared `CreditBoundary` is applied to that connection's outgoing
cotangent exactly once. This is important: a connection boundary is a port
credit rule, not a hidden operation inside the producer module. Conditional
connections treat their declared context as a local relation input. Their
credit result includes `context_cotangents[connection_id]`, so a caller can
compose it into the producer of that context rather than silently dropping it.
The port boundary is still applied once, before the connection local VJP; it
does not gate the receipt bookkeeping itself.

`StructureWindowPolicy` supplies that control-plane rule. Before reading a
window's query targets, it receives only declared support-side evidence and
samples whether to pay for a structure evaluation. Its score-function
objective consumes the later observed query cost plus any declared evaluation
cost, so the policy changes the *next* window rather than selecting a graph
after seeing the current target:

```python
policy = arti.StructureWindowPolicy(evidence_dim=4, min_steps=8)
proposal = policy.propose(support_evidence, steps_since_window=age)

# The control-plane loop may inspect proposal.sampled_open between windows.
# If it opens a window, run paired candidate trials and commit as usual.
policy_loss = policy.score_function_objective(observed_query_cost, proposal)
```

Optional `max_steps` is a host budget bound. A forced or ineligible action is
not treated as a policy sample and therefore contributes no policy gradient.
This policy does not make structure commits inside a compiled batch or use a
probability threshold as an implicit topology decision.

The current API supplies boundary execution, local VJP lowering for fixed
registered/Formula graphs, region derivation, optimizer-state migration,
grouped AdamW updates, paired structure-window commits, and a learned
pre-query window-opening policy. Fixed graph credit lowering and window-policy
correctness are established separately from compiled backward throughput.
