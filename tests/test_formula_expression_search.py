import pytest
import torch

from benchmarks.train_formula_expression_search import (
    build_graph, capture, data, evaluate, live_panel, make_run, reference, route_rows,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def members(query):
    return {node.candidate_id: node for node in query.candidates}


def test_real_atomic_sources_share_bank_and_completed_answer_stops():
    query = build_graph()
    nodes = members(query)
    assert len(nodes) == 12 and not hasattr(query, "network")
    assert sum(p.numel() for p in query.parameters()) == 113
    assert nodes["swish.0"].candidate.operand_store is nodes["swish.1"].candidate.operand_store
    assert nodes["swish.0"].candidate.operand_store.tensor("beta").requires_grad
    assert {nodes[f"swish.{i}"].input_slots["value"] for i in range(2)} == {"scaled", "z"}
    values, _ = data(29, 4)
    panel = capture(query, values)
    assert len(panel.branches) == 49
    routes = [[row["candidate"] for row in route_rows(b)] for b in panel.branches]
    assert all(route[-1] == "stop" and route[-2].startswith("read.") for route in routes)
    assert {len(route) - 1 for route in routes} == {2, 3, 4}
    assert any(route == ["entry", "swish.0", "add.1", "read.2", "stop"] for route in routes)


def test_all_completed_outputs_match_independent_tensor_math_and_joint_gradients():
    query = build_graph()
    values, target = data(29, 4)
    values = {name: value.requires_grad_() for name, value in values.items()}
    panel = capture(query, values)
    runs, _, losses, _ = live_panel(query, values, target, panel)
    expected = [reference(query, values, b) for b in panel.branches]
    for run, value in zip(runs, expected, strict=True):
        torch.testing.assert_close(run.outputs["answer"], value)
    nodes = members(query)
    leaves = (*values.values(), nodes["entry"].candidate.operand_store.tensor("gain"),
              nodes["swish.0"].candidate.operand_store.tensor("beta"))
    actual_grad = torch.autograd.grad(losses.sum(), leaves)
    expected_grad = torch.autograd.grad(sum((value-target).square().mean() for value in expected), leaves)
    for actual, expected in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_finite_panel_surrogate_matches_exact_risk_and_has_choice_credit():
    query = build_graph()
    values, target = data(29, 4)
    panel = capture(query, values)
    _, energies, losses, coefficients = live_panel(query, values, target, panel)
    parameters = tuple(query.parameters())
    actual = torch.autograd.grad(coefficients.surrogate(losses, energies), parameters, retain_graph=True)
    exact_risk = (energies.double().softmax(0) * losses.double()).sum()
    expected = torch.autograd.grad(exact_risk, parameters, retain_graph=True)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(torch.cat([g.flatten() for g in actual])).all()
    numerical = torch.autograd.grad((coefficients.numerical * losses).sum(), parameters,
                                    retain_graph=True, allow_unused=True)
    choice = torch.autograd.grad((coefficients.choice * energies).sum(), parameters, allow_unused=True)
    assert sum(float(g.square().sum()) for g in choice if g is not None) > 1e-6
    for total, n, c in zip(actual, numerical, choice, strict=True):
        torch.testing.assert_close(total, (0 if n is None else n) + (0 if c is None else c), rtol=1e-5, atol=1e-6)


def test_one_response_only_step_reduces_current_panel_risk():
    query = build_graph()
    values, target = data(29, 4)
    panel = capture(query, values)
    _, energies, losses, coefficients = live_panel(query, values, target, panel)
    stores = {id(node.candidate.operand_store): node.candidate.operand_store for node in query.candidates}
    response = [store.tensor(f"response.{action}.{part}") for store in stores.values()
                for action in query.continuations.get("entry", {}) for part in ("slope", "bias")
                if store is not members(query)["read.0"].candidate.operand_store]
    gradients = torch.autograd.grad((coefficients.choice * energies).sum(), response)
    with torch.no_grad():
        for parameter, gradient in zip(response, gradients, strict=True):
            parameter.add_(gradient, alpha=-.01)
    fresh = capture(query, values)
    _, _, _, after = live_panel(query, values, target, fresh)
    assert after.risk < coefficients.risk


def test_managed_asset_restores_optimizer_shared_bank_and_hard_result(tmp_path):
    from accelerate.state import AcceleratorState

    AcceleratorState._reset_state(reset_partial_state=True)
    try:
        run = make_run(tmp_path)
        values, target = data(29, 4)
        _, energies, losses, coefficients = live_panel(run.trainable, values, target, capture(run.trainable, values))
        assert run.backward_and_step(coefficients.surrogate(losses, energies)) is not None
        run.record_step(examples=4, tokens=0)
        checkpoint = run.checkpoint(reason="unit-roundtrip")
        before = evaluate(run.trainable, values, target)
        restored = make_run(tmp_path)
        assert restored.resume() == checkpoint.resolve()
        assert restored.progress.step == 1
        torch.testing.assert_close(restored.trainable.state_dict(), run.trainable.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(restored.optimizer.state_dict(), run.optimizer.state_dict(), rtol=0, atol=0)
        nodes = members(restored.trainable)
        assert nodes["swish.0"].candidate.operand_store is nodes["swish.1"].candidate.operand_store
        assert evaluate(restored.trainable, values, target) == before
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)


def test_soft_budget_counts_periodic_save_and_prioritizes_final_asset(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from benchmarks import train_formula_expression_search as demo

    clock = [0.]
    saves, evaluations = [], []

    def spend(seconds, result=None):
        clock[0] += seconds
        return result

    class Run:
        trainable = torch.nn.Linear(1, 1)
        optimizer = SimpleNamespace(zero_grad=lambda **kw: None)
        progress = SimpleNamespace(step=0)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def checkpoint(self, *, reason):
            saves.append(reason)
            return spend(2, tmp_path / reason)

        def maybe_checkpoint(self):
            return spend(20, tmp_path / "interval")

        def backward_and_step(self, loss):
            return spend(1, 1.)

        def record_step(self, **kw):
            self.progress.step += 1

    run = Run()
    monkeypatch.setattr(demo.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(demo, "make_run", lambda *args, **kw: spend(3, run))
    monkeypatch.setattr(demo, "capture", lambda *args: spend(1, SimpleNamespace(scored_expansions=1, executed_expansions=1)))
    coefficient = SimpleNamespace(risk=torch.tensor(1.), surrogate=lambda *args: torch.tensor(1.))
    monkeypatch.setattr(demo, "live_panel", lambda *args: spend(1, ((), None, None, coefficient)))

    def evaluate(*args):
        evaluations.append(clock[0])
        return spend(2, {"mse": 1.})

    monkeypatch.setattr(demo, "evaluate", evaluate)
    demo.main(SimpleNamespace(output=tmp_path, seed=17, learning_rate=.03, resume=False, steps=3, seconds=30.))
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert run.progress.step == 1 and len(evaluations) == 1
    assert saves == ["segment-start", "segment-end"]
    assert next(row for row in rows if row["kind"] == "update")["checkpoint_seconds"] == 20
    assert next(row for row in rows if row["kind"] == "final")["validation"]["status"] == "skipped_due_to_budget"
    assert rows[-1]["total_seconds"] == 32 and rows[-1]["overrun_seconds"] == 2
