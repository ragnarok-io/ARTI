# K=2 Branch Harness

`arti.mechanisms.K2BranchHarness` is a host-only, CPU reference coordinator for two
private proposals rooted at the same volatile snapshot. It does not execute
Formula, Refine, Recall, or selection logic.

Each branch runs existing ARTI tensor modules independently, materializes its
complete next-state values as `ExternalTensorProposal@1`, and declares bounded
work in `BranchWorkReceipt@1`. A generic `OverlayProposal@1` records declared
step lineage but is not, by itself, proof that a neural executor performed the
work. The harness stages each branch into a separate private transaction
overlay. The host must then explicitly select one branch or discard both.
There is no merge and no automatic winner.

`abort()` is the unconditional cleanup path for incomplete or invalid runs. It
rolls back every private overlay without requiring complete proposals or
matched work. It never publishes a candidate.

`BranchBatchSpec@1` binds both branches to the same parent root, fixed Formula
program, fixed hard route, input, RNG, and exogenous future-tape fingerprints.
The future tape is provenance only; the harness never passes it to Query,
route, stop, or commit code.

`execute_fixed_formula_branch()` is the narrow integration path for mechanism
tests. It invokes the existing `FormulaFabricCompute` with a fixed hard route,
derives step lineage and work counters from actual `FormulaFabricTrace`
objects, and rejects a workspace, program, route, or budget that does not match
the batch spec. Its Formula overlay accepts exactly one complete candidate and
requires that candidate to equal the executed final workspace. The deterministic
executor records a canonical no-RNG receipt. `score_formula_branches()` is a
parameter-free MSE scorer over one finite, fingerprinted future tensor. The
score receipt binds both execution receipts and both final candidate values.
`select_scored_formula_branch()` is the explicit host authorization that applies
the deterministic lower-score rule; the scorer itself has no commit authority.

These are software execution receipts, not physical cost receipts. They do not
establish GPU work, latency, causal benefit, agency, or task-quality gain.

The first version is deliberately limited to exactly two branches, a volatile
single-writer CPU runtime, complete next-state proposals, and host-authorized
commit. It is not an Agent API, persistent world, causal experiment, Top-K
search, branch merge, GPU runtime, or evidence of task-quality improvement.
