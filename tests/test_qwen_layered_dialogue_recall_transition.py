from __future__ import annotations

import copy
from contextlib import contextmanager
import random
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

import arti
from benchmarks import run_qwen_dialogue_recall_recurrence_curriculum as recurrence
from arti._recall_state import RecallValueUpdater
from benchmarks.build_counterfactual_dialogues import counterfactual_dialogues
from benchmarks.generate_qwen_counterfactual_recall import generated_token_metrics
from benchmarks.run_qwen_layered_dialogue_recall_transition import (
    CurrentTokenBatch,
    _LiveBoundaryCapture,
    LayeredRecallRuntime,
    build_token_batch,
    evenly_spaced_layers,
    layered_artifact_layers,
    layered_artifact_updater_capacity,
    load_runtime_updater_state,
    load_runtime_state,
    optimizer_parameter_groups,
    overlay_runtime_state,
    run_current,
    sample_compatible_indices,
    save_runtime,
)
from benchmarks.run_qwen_dialogue_recall_transition import (
    DialogueBatch,
    DialogueExample,
    build_examples,
    dialogue_token_coverage,
    token_coverage_sampling_weights,
)
from benchmarks.evaluate_qwen_vocab_recall_generalization import (
    sequence_token_ids,
    update_sequence_values,
)
from benchmarks.run_qwen_vocab_recall_composition_curriculum import (
    curriculum_condition,
    curriculum_length,
)
from benchmarks.run_qwen_dialogue_recall_recurrence_curriculum import (
    DialogueTokenBatch,
    audit_low_rank_recall_formula,
    bank_value_alignment_loss,
    collate_dialogues,
    group_dialogues_by_turn_count,
    history_bank_flow_loss,
    history_bank_token_rank_loss,
    history_effect_loss,
    history_turn_sampling_weights,
    prefix_dialogue_batch,
    read_refine_metrics,
    route_candidate_rank_loss,
)
from benchmarks.run_qwen_vocab_recall_recurrence_curriculum import (
    build_prefix_probe_batch,
    quality_flow_loss,
)
from benchmarks.run_qwen_vocab_recall_updater_pretraining import (
    VocabularySequenceStream,
    bank_prefix_alignment_loss,
    prefix_bank_rank_loss,
    select_candidate_groups,
    share_final_token_within_pairs,
    swap_adjacent_batch,
)
from benchmarks.run_qwen_recall_paired_final_training import (
    FinalTeacher,
    candidate_score,
    gather_pair_logits,
    paired_history_final_loss,
    paired_topk_indices,
)
from benchmarks.evaluate_qwen_vocab_recall_history_causality import (
    bank_update_metrics,
    same_tail_sequences,
)
from benchmarks.run_qwen_vocab_recall_curriculum import (
    choose_probe_ids,
    embedding_collision_report,
    identity_loss,
    project_logits,
    vocabulary_permutation,
)


def test_frozen_first_groups_reuses_candidates_without_claiming_route_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = SimpleNamespace()
    examples = [object(), object()]
    candidate_groups = torch.tensor([[[[0, 2]]], [[[1, 3]]]])
    calls: list[dict[str, object]] = []

    def fake_online_dialogue_state(
        model: nn.Module,
        runtime: object,
        final_capture: object,
        site_capture: object,
        current_batch: object,
        *,
        recall_steps: int,
        candidate_groups: torch.Tensor | None = None,
        return_candidate_groups: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del model, runtime, final_capture, site_capture
        assert current_batch is batch
        calls.append(
            {
                "recall_steps": recall_steps,
                "candidate_groups": candidate_groups,
                "return_candidate_groups": return_candidate_groups,
            }
        )
        values = torch.full((2, 1, 3), float(recall_steps))
        if return_candidate_groups:
            return values, selected_groups
        return values

    selected_groups = candidate_groups
    monkeypatch.setattr(recurrence, "collate_dialogues", lambda _: batch)
    monkeypatch.setattr(
        recurrence,
        "teacher_dialogue_targets",
        lambda *args: torch.zeros(2, 1, 3),
    )
    monkeypatch.setattr(recurrence, "online_dialogue_state", fake_online_dialogue_state)
    monkeypatch.setattr(
        recurrence,
        "run_dialogue_current",
        lambda *args: args[-1],
    )
    monkeypatch.setattr(
        recurrence,
        "dialogue_metrics",
        lambda *args, **kwargs: {
            "tokens": 2,
            "hidden_rmse": 1.0,
            "distribution_kl": 0.0,
            "target_token_nll": 0.0,
            "teacher_top1_agreement": 1.0,
        },
    )
    monkeypatch.setattr(
        recurrence,
        "history_effect_metrics",
        lambda *args: {"examples": 2},
    )
    runtime = SimpleNamespace(
        fields=[SimpleNamespace(bank=object())],
        initial_values=lambda batch_size, bank: torch.zeros(batch_size, 1, 3),
    )

    recurrence.evaluate_dialogues(
        nn.Linear(3, 3),
        runtime,
        object(),
        object(),
        examples,
        recall_steps=6,
        state_mode="frozen_first_groups",
    )

    assert len(calls) == 2
    assert calls[0]["recall_steps"] == 1
    assert calls[0]["candidate_groups"] is None
    assert calls[0]["return_candidate_groups"] is True
    assert calls[1]["recall_steps"] == 6
    assert calls[1]["candidate_groups"] is candidate_groups
    assert calls[1]["return_candidate_groups"] is False


def test_read_refine_metrics_reports_real_decoder_boundary_steps() -> None:
    runtime = SimpleNamespace(
        read_diagnostics=lambda: (
            {
                "recall_steps_attempted": torch.tensor([3, 3]),
                "recall_steps_committed": torch.tensor([3, 2]),
                "recall_step_committed": torch.tensor(
                    [[True, True, True], [True, True, False]]
                ),
                "recall_step_update_ratio": torch.tensor(
                    [[0.5, 0.25, 0.125], [0.4, 0.2, 0.0]]
                ),
                "recall_step_route_change": torch.tensor(
                    [[0.0, 0.3, 0.1], [0.0, 0.2, 0.0]]
                ),
                "recall_step_effective_read_change": torch.tensor(
                    [[0.0, 0.6, 0.2], [0.0, 0.4, 0.0]]
                ),
            },
            {
                "recall_steps_attempted": torch.tensor([3, 3]),
                "recall_steps_committed": torch.tensor([1, 0]),
                "recall_step_committed": torch.tensor(
                    [[True, False, False], [False, False, False]]
                ),
                "recall_step_update_ratio": torch.tensor(
                    [[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                "recall_step_route_change": torch.zeros(2, 3),
                "recall_step_effective_read_change": torch.zeros(2, 3),
            },
        )
    )

    metrics = read_refine_metrics(runtime)

    assert metrics["sites"] == 2
    assert metrics["examples"] == 4
    assert metrics["attempted_mean"] == pytest.approx(3.0)
    assert metrics["committed_mean"] == pytest.approx(1.5)
    assert metrics["update_ratio_mean"] == pytest.approx(0.2625)
    assert metrics["route_change_mean"] == pytest.approx(0.2)
    assert metrics["effective_read_change_mean"] == pytest.approx(0.4)
    assert metrics["per_site"][0]["committed_mean"] == pytest.approx(2.5)
    assert metrics["per_site"][1]["committed_mean"] == pytest.approx(0.5)


def test_read_refine_metrics_is_compatible_with_non_refining_test_runtime() -> None:
    assert read_refine_metrics(SimpleNamespace()) == {
        "sites": 0,
        "examples": 0,
        "attempted_mean": 0.0,
        "committed_mean": 0.0,
        "update_ratio_mean": 0.0,
        "route_change_mean": 0.0,
        "effective_read_change_mean": 0.0,
        "token_steps_mean": 0.0,
        "kernel_steps_mean": 0.0,
        "logical_token_steps": 0,
        "per_site": [],
    }


def test_frozen_first_route_reuses_the_complete_versioned_route_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = SimpleNamespace()
    examples = [object(), object()]
    plan = arti.RecallRoutePlan(
        schema_version=1,
        routing="grouped",
        value_composition="single",
        slots=2,
        composition_factor=1,
        group_size=1,
        layout_fingerprint="test-layout",
        weights=torch.ones(2, 1, 1, 1),
        indices=torch.zeros(2, 1, 1, 1, dtype=torch.long),
        route=torch.ones(2, 1, 1, 1),
    )
    route_stack = arti.RecallRouteStack(
        axis="turn",
        items=(
            arti.RecallRouteStack(
                axis="site",
                items=(arti.RecallRouteStack(axis="block", items=(plan,)),),
            ),
        ),
    )
    calls: list[dict[str, object]] = []

    def fake_online_dialogue_state(
        *args: object,
        recall_steps: int,
        route_stack: arti.RecallRouteStack | None = None,
        return_route_stack: bool = False,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, arti.RecallRouteStack]:
        calls.append(
            {
                "recall_steps": recall_steps,
                "route_stack": route_stack,
                "return_route_stack": return_route_stack,
            }
        )
        values = torch.full((2, 1, 3), float(recall_steps))
        if return_route_stack:
            return values, captured_stack
        return values

    captured_stack = route_stack
    monkeypatch.setattr(recurrence, "collate_dialogues", lambda _: batch)
    monkeypatch.setattr(
        recurrence,
        "teacher_dialogue_targets",
        lambda *args: torch.zeros(2, 1, 3),
    )
    monkeypatch.setattr(recurrence, "online_dialogue_state", fake_online_dialogue_state)
    monkeypatch.setattr(recurrence, "run_dialogue_current", lambda *args: args[-1])
    monkeypatch.setattr(
        recurrence,
        "dialogue_metrics",
        lambda *args, **kwargs: {
            "tokens": 2,
            "hidden_rmse": 1.0,
            "distribution_kl": 0.0,
            "target_token_nll": 0.0,
            "teacher_top1_agreement": 1.0,
        },
    )
    monkeypatch.setattr(
        recurrence,
        "history_effect_metrics",
        lambda *args: {"examples": 2},
    )
    monkeypatch.setattr(
        recurrence,
        "bank_identity_metrics",
        lambda *args: {"examples": 2},
    )
    runtime = SimpleNamespace(
        fields=[SimpleNamespace(bank=object())],
        initial_values=lambda batch_size, bank: torch.zeros(batch_size, 1, 3),
    )

    recurrence.evaluate_dialogues(
        nn.Linear(3, 3),
        runtime,
        object(),
        object(),
        examples,
        recall_steps=6,
        state_mode="frozen_first_route",
    )

    assert len(calls) == 2
    assert calls[0] == {
        "recall_steps": 1,
        "route_stack": None,
        "return_route_stack": True,
    }
    assert calls[1]["recall_steps"] == 6
    assert calls[1]["route_stack"] is route_stack
    assert calls[1]["return_route_stack"] is False


def test_bank_identity_metrics_detect_shared_and_distinct_updates() -> None:
    initial = torch.zeros(4, 2, 3)
    shared = torch.ones_like(initial)
    distinct = shared.clone()
    distinct[:, 0, 0] = torch.arange(4, dtype=distinct.dtype)

    shared_metrics = recurrence.bank_identity_metrics(shared, initial)
    distinct_metrics = recurrence.bank_identity_metrics(distinct, initial)

    assert shared_metrics["state_update_rms"] > 0.0
    assert shared_metrics["shuffled_state_rms"] == 0.0
    assert shared_metrics["state_identity_ratio"] == 0.0
    assert shared_metrics["state_cosine_to_shuffled"] == pytest.approx(1.0)
    assert distinct_metrics["shuffled_state_rms"] > 0.0
    assert distinct_metrics["state_identity_ratio"] > 0.0
    assert distinct_metrics["state_cosine_to_shuffled"] < 1.0


def test_paired_final_topk_uses_the_union_for_both_histories() -> None:
    teacher = torch.tensor(
        [
            [9.0, 1.0, 0.0, -1.0],
            [0.0, 8.0, 2.0, -1.0],
            [0.0, 1.0, 7.0, -1.0],
            [6.0, 0.0, 1.0, -1.0],
        ]
    )

    indices = paired_topk_indices(teacher, topk=1)
    selected = gather_pair_logits(teacher, indices)

    assert indices.tolist() == [[0, 1], [2, 0]]
    assert selected.shape == (2, 2, 2)
    torch.testing.assert_close(selected[0, 0], torch.tensor([9.0, 1.0]))
    torch.testing.assert_close(selected[0, 1], torch.tensor([0.0, 8.0]))


def test_paired_history_final_loss_is_finite_and_differentiable() -> None:
    generator = torch.Generator().manual_seed(71)
    teacher_hidden = torch.randn(4, 6, generator=generator)
    teacher_logits = torch.randn(4, 11, generator=generator)
    correct_hidden = teacher_hidden.clone().requires_grad_()
    correct_logits = teacher_logits.clone().requires_grad_()
    loss, parts = paired_history_final_loss(
        correct_hidden,
        correct_logits,
        FinalTeacher(teacher_hidden, teacher_logits, torch.empty(0)),
        temperature=1.5,
        topk=4,
        shape_weight=0.1,
        difference_weight=0.2,
        hidden_difference_weight=0.03,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())
    assert correct_hidden.grad is not None
    assert correct_logits.grad is not None
    assert torch.isfinite(correct_hidden.grad).all()
    assert torch.isfinite(correct_logits.grad).all()


def test_candidate_score_matches_bfloat16_model_dtype() -> None:
    class TinyCandidateModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.norm = nn.RMSNorm(6, dtype=torch.bfloat16)
            self.lm_head = nn.Linear(6, 11, bias=False, dtype=torch.bfloat16)

    model = TinyCandidateModel()
    teacher = FinalTeacher(
        torch.randn(4, 6),
        torch.randn(4, 11),
        torch.empty(0),
    )

    score = candidate_score(model, torch.randn(4, 6), teacher, topk=4)
    repeated_score = candidate_score(model, torch.randn(8, 6), teacher, topk=4)

    assert score.shape == (1, 4)
    assert repeated_score.shape == (2, 4)
    assert torch.isfinite(score).all()


def make_field() -> arti.ARTILatentRecallField:
    return arti.ARTILatentRecallField(
        4,
        3,
        routing="grouped",
        key_dim=2,
        query_mode="fixed",
        query_seed=7349,
        group_size=1,
        group_topk=1,
        project_external=False,
    ).requires_grad_(False)


def test_counterfactual_dialogues_hold_query_constant_and_vary_history() -> None:
    rows = counterfactual_dialogues(8, seed=17)

    queries = {row["messages"][-2]["content"] for row in rows}
    answers = {row["messages"][-1]["content"] for row in rows}
    premises = {row["messages"][0]["content"] for row in rows}

    assert queries == {"Return only the opaque code."}
    assert len(answers) == len(rows)
    assert len(premises) == len(rows)


def test_free_rollout_token_metrics_keep_exact_and_partial_matches_separate() -> None:
    teacher = torch.tensor([[1, 2, 3], [4, 5, 6]])
    prediction = torch.tensor([[1, 2, 3], [4, 7, 6]])

    metrics = generated_token_metrics(prediction, teacher, eos_token_id=99)

    assert metrics["exact_match"] == pytest.approx(0.5)
    assert metrics["token_agreement"] == pytest.approx(5 / 6)
    assert 0.5 < metrics["edit_similarity"] < 1.0
    eos_metrics = generated_token_metrics(
        torch.tensor([[1, 7, 99, 99]]),
        torch.tensor([[1, 2, 99, 99]]),
        eos_token_id=99,
    )
    assert eos_metrics["token_agreement"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match=r"\[B, T\]"):
        generated_token_metrics(prediction[:, :2], teacher, eos_token_id=99)


def test_prefix_dialogue_batch_rebases_current_positions() -> None:
    batch = DialogueTokenBatch(
        history_ids=torch.tensor([[[1, 2, 0], [3, 4, 5], [6, 0, 0]]]),
        history_mask=torch.tensor(
            [[[True, True, False], [True, True, True], [True, False, False]]]
        ),
        history_positions=torch.tensor([[[0, 1, 0], [2, 3, 4], [5, 0, 0]]]),
        current_ids=torch.tensor([[7, 8]]),
        current_mask=torch.ones(1, 2, dtype=torch.bool),
        current_positions=torch.tensor([[6, 7]]),
        prediction_positions=torch.tensor([[0]]),
        prediction_mask=torch.ones(1, 1, dtype=torch.bool),
        target_ids=torch.tensor([[8]]),
    )

    prefix = prefix_dialogue_batch(batch, row=0, turns=2)

    assert prefix.history_ids.shape == (1, 2, 3)
    assert prefix.current_positions.tolist() == [[5, 6]]


def test_ttt_host_formula_audit_rejects_dense_fallbacks() -> None:
    class RoutedLowRankFormula(nn.Module):
        recall_formula_contract = arti.RecallFormulaContract(
            factors=tuple(
                factor
                for route in range(4)
                for factor in (
                    arti.FactorSpec(f"route_{route}_down_00", route=f"route_{route}"),
                    arti.FactorSpec(f"route_{route}_up_00", route=f"route_{route}"),
                )
            ),
            identity_preserving=True,
        )

        def forward(self, state: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
            down = factors[..., 0::2, :]
            up = factors[..., 1::2, :]
            coefficient = torch.sum(down * state.unsqueeze(-2), dim=-1)
            return state + torch.sum(up * coefficient.unsqueeze(-1), dim=-2)

    audit = audit_low_rank_recall_formula("test/routed-low-rank@1", RoutedLowRankFormula())

    assert audit["equivalent_rank"] == 4
    assert audit["independent_routes"] == 4
    assert audit["numerically_verified"] is True
    with pytest.raises(ValueError, match="down/up"):
        audit_low_rank_recall_formula(
            "test/dense@1",
            type(
                "DenseFallback",
                (nn.Module,),
                {
                    "recall_formula_contract": arti.RecallFormulaContract(
                        factors=(arti.FactorSpec("content"),)
                    ),
                    "forward": lambda self, state, factors: state + factors[..., 0, :],
                },
            )(),
        )


def test_dialogue_builder_rejects_invalid_minimum_history(tmp_path) -> None:
    with pytest.raises(ValueError, match="minimum_history_messages"):
        build_examples(
            tmp_path / "unused.jsonl",
            object(),
            train_examples=2,
            eval_examples=2,
            examples_per_dialogue=1,
            max_history_messages=4,
            max_message_tokens=8,
            max_context_tokens=32,
            target_positions=None,
            source_dialogue_offset=0,
            minimum_source_offset=None,
            seed=1,
            minimum_history_messages=3,
        )


def make_updater() -> RecallValueUpdater:
    return RecallValueUpdater(
        4,
        3,
        workspace_dim=8,
        depth=1,
        interface_slots=2,
    )


def test_layered_hook_uses_core_next_state_and_read_side_refine() -> None:
    class ConstantStateFormula(nn.Module):
        recall_formula_contract = arti.RecallFormulaContract(
            factors=(arti.FactorSpec("state"),),
            identity_preserving=False,
        )

        def forward(self, state: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
            del factors
            return torch.full_like(state, 2.0)

    field = arti.ARTILatentRecallField(
        4,
        2,
        formula=ConstantStateFormula(),
        factor_activation="none",
        project_external=False,
    ).requires_grad_(False)
    runtime = LayeredRecallRuntime(
        (0,),
        (field,),
        (RecallValueUpdater(4, 2, workspace_dim=4, depth=1, recall_slots=2),),
        read_policy=arti.RefinePolicy.fixed(3, trace_level="full"),
    )
    state_keys = tuple(runtime.state_dict())
    x = torch.randn(2, 3, 4)
    memory = torch.randn(2, 1, 2, 4)
    mask = torch.ones(2, 3, dtype=torch.bool)

    with runtime.use(memory, mask):
        output = runtime._make_hook(0)(nn.Identity(), (), x)

    assert isinstance(output, torch.Tensor)
    torch.testing.assert_close(output, torch.full_like(x, 2.0), rtol=0, atol=0)
    diagnostics = runtime.read_diagnostics()[0]
    assert diagnostics is not None
    assert diagnostics["recall_steps_committed"].tolist() == [3, 3]
    assert tuple(runtime.state_dict()) == state_keys
    assert not any("_read_states" in key for key in state_keys)


def test_layered_hook_accepts_token_adaptive_read_policy_and_reports_work() -> None:
    runtime = LayeredRecallRuntime(
        (0,),
        (make_field(),),
        (make_updater(),),
        read_policy=arti.RefinePolicy.adaptive(
            max_steps=3,
            min_steps=3,
            scope="token",
            relative_tolerance=1e-6,
            trace_level="summary",
            executor="early_break",
        ),
    ).eval()
    x = torch.randn(2, 4, 4)
    memory = torch.randn(2, 1, 3, 4)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])

    with runtime.use(memory, mask):
        output = runtime._make_hook(0)(nn.Identity(), (), x)

    assert isinstance(output, torch.Tensor)
    diagnostics = runtime.read_diagnostics()[0]
    assert diagnostics is not None
    assert diagnostics["recall_token_steps_attempted"].tolist() == [
        [3, 3, 0, 0],
        [3, 3, 3, 0],
    ]
    assert diagnostics["recall_kernel_steps"].item() == 3
    assert diagnostics["recall_logical_token_steps"].item() == 15
    metrics = read_refine_metrics(runtime)
    assert metrics["token_steps_mean"] == pytest.approx(15 / 8)
    assert metrics["kernel_steps_mean"] == pytest.approx(3.0)


def test_layered_read_activation_checkpoint_preserves_output_and_memory_gradient() -> None:
    torch.manual_seed(1103)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = LayeredRecallRuntime(
        (0,),
        (make_field(),),
        (make_updater(),),
        read_policy=arti.RefinePolicy.fixed(3, trace_level="summary"),
    ).to(device).train()
    x = torch.randn(2, 4, 4, device=device)
    mask = torch.ones(2, 4, dtype=torch.bool, device=device)
    plain_memory = torch.randn(2, 1, 3, 4, device=device, requires_grad=True)

    with runtime.use(plain_memory, mask):
        plain = runtime._make_hook(0)(nn.Identity(), (), x)
    assert isinstance(plain, torch.Tensor)
    plain.square().mean().backward()
    plain_grad = plain_memory.grad.detach().clone()

    checkpoint_memory = plain_memory.detach().clone().requires_grad_(True)
    runtime.set_read_activation_checkpoint(True)
    with runtime.use(checkpoint_memory, mask):
        recomputed = runtime._make_hook(0)(nn.Identity(), (), x)
    assert isinstance(recomputed, torch.Tensor)
    assert runtime._active_values is None
    recomputed.square().mean().backward()

    torch.testing.assert_close(recomputed, plain, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        checkpoint_memory.grad,
        plain_grad,
        rtol=1e-5,
        atol=1e-6,
    )
    diagnostics = runtime.read_diagnostics()[0]
    assert diagnostics is not None
    assert diagnostics["recall_steps_attempted"].tolist() == [3, 3]


def test_layered_read_refine_zero_is_identity_and_one_matches_legacy_delta_read() -> None:
    class AffineStateFormula(nn.Module):
        recall_formula_contract = arti.RecallFormulaContract(
            factors=(arti.FactorSpec("state"),),
            identity_preserving=False,
        )

        def forward(self, state: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
            return 0.5 * state + factors[..., 0, :]

    torch.manual_seed(1973)
    field = arti.ARTILatentRecallField(
        4,
        3,
        formula=AffineStateFormula(),
        factor_activation="none",
        project_external=False,
    ).requires_grad_(False)
    updater = RecallValueUpdater(4, 3, workspace_dim=8, depth=1, recall_steps=2)
    runtime = LayeredRecallRuntime((0,), (field,), (updater,))
    x = torch.randn(2, 4, 4)
    memory = torch.randn(2, 1, 3, 4)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])

    with runtime.use(memory, mask, read_policy=arti.RefinePolicy.fixed(0)):
        zero = runtime._make_hook(0)(nn.Identity(), (), x)
    zero_diagnostics = runtime.read_diagnostics()[0]
    with runtime.use(memory, mask, read_policy=arti.RefinePolicy.fixed(1)):
        one = runtime._make_hook(0)(nn.Identity(), (), (x, "tail"))

    legacy = x + field.read_context(x, mask, memory=memory[:, 0])
    assert isinstance(zero, torch.Tensor)
    torch.testing.assert_close(zero, x, rtol=0, atol=0)
    assert zero_diagnostics is not None
    assert zero_diagnostics["recall_steps_attempted"].tolist() == [0, 0]
    assert isinstance(one, tuple)
    assert one[1] == "tail"
    torch.testing.assert_close(one[0], legacy, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(one[0][~mask], x[~mask], rtol=0, atol=0)
    assert updater.recall_steps == 2
    assert runtime.read_diagnostics()[0]["recall_steps_attempted"].tolist() == [1, 1]


def test_layered_read_wrapper_does_not_duplicate_registered_field_parameters() -> None:
    field = make_field()
    runtime = LayeredRecallRuntime((0,), (field,), (make_updater(),))

    parameter_ids = [id(parameter) for parameter in runtime.parameters()]
    state_keys = tuple(runtime.state_dict())

    assert len(parameter_ids) == len(set(parameter_ids))
    assert not any("_read_states" in key for key in state_keys)
    assert sum(key.startswith("fields.0.") for key in state_keys) == len(field.state_dict())


def test_layered_matched_and_frozen_read_modes_are_equal_at_one_step() -> None:
    torch.manual_seed(2039)
    field = make_field()
    runtime = LayeredRecallRuntime(
        (0,),
        (field,),
        (make_updater(),),
        read_policy=arti.RefinePolicy.fixed(1, trace_level="full"),
    ).eval()
    x = torch.randn(2, 4, 4)
    memory = torch.randn(2, 1, 3, 4)
    mask = torch.ones(2, 4, dtype=torch.bool)
    bank_before = field.bank.detach().clone()

    torch.manual_seed(811)
    with runtime.use(memory, mask, read_ablation_mode="reroute_with_capture_control"):
        dynamic = runtime._make_hook(0)(nn.Identity(), (), x)
    torch.manual_seed(811)
    with runtime.use(memory, mask, read_ablation_mode="reuse_first_route"):
        frozen = runtime._make_hook(0)(nn.Identity(), (), x)

    assert isinstance(dynamic, torch.Tensor)
    assert isinstance(frozen, torch.Tensor)
    torch.testing.assert_close(dynamic, frozen, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(field.bank, bank_before, rtol=0, atol=0)
    assert runtime.read_policy.max_steps == 1
    assert runtime.updaters[0].recall_steps == 0
    with pytest.raises(ValueError, match="read ablation mode"):
        runtime.set_read_ablation_mode("unknown")


def test_layered_reroute_capture_control_has_no_effect_at_three_steps() -> None:
    torch.manual_seed(2131)
    runtime = LayeredRecallRuntime(
        (0,),
        (make_field(),),
        (make_updater(),),
        read_policy=arti.RefinePolicy.fixed(3, trace_level="full"),
    ).eval()
    x = torch.randn(2, 4, 4)
    memory = torch.randn(2, 1, 3, 4)
    mask = torch.ones(2, 4, dtype=torch.bool)

    torch.manual_seed(977)
    with runtime.use(memory, mask, read_ablation_mode="reroute_each_step"):
        dynamic = runtime._make_hook(0)(nn.Identity(), (), x)
    torch.manual_seed(977)
    with runtime.use(memory, mask, read_ablation_mode="reroute_with_capture_control"):
        control = runtime._make_hook(0)(nn.Identity(), (), x)

    assert isinstance(dynamic, torch.Tensor)
    assert isinstance(control, torch.Tensor)
    torch.testing.assert_close(control, dynamic, rtol=1e-6, atol=1e-6)
    runtime.train()
    with pytest.raises(RuntimeError, match="evaluation-only"):
        with runtime.use(memory, mask, read_ablation_mode="reuse_first_route"):
            runtime._make_hook(0)(nn.Identity(), (), x)


def make_recall_updater() -> RecallValueUpdater:
    return RecallValueUpdater(
        4,
        3,
        workspace_dim=8,
        depth=1,
        interface_slots=2,
        recall_slots=2,
        recall_steps=1,
    )


def test_vocab_curriculum_uses_deterministic_complete_token_permutation() -> None:
    first = vocabulary_permutation(17, seed=41)
    second = vocabulary_permutation(17, seed=41)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert sorted(first.tolist()) == list(range(17))


def test_vocab_sequence_stream_covers_each_token_before_repeating() -> None:
    stream = VocabularySequenceStream(17, seed=41)
    first = stream.next(1, 17)
    second = stream.next(1, 4)

    assert sorted(first.flatten().tolist()) == list(range(17))
    assert torch.equal(first, VocabularySequenceStream(17, seed=41).next(1, 17))
    assert len(set(second.flatten().tolist())) == 4
    assert stream.report()["completed_passes"] == 1


def test_vocab_sequence_stream_resumes_without_replaying_the_prefix() -> None:
    uninterrupted = VocabularySequenceStream(17, seed=41)
    expected_prefix = uninterrupted.next(1, 7)
    expected_suffix = uninterrupted.next(1, 5)
    resumed = VocabularySequenceStream(17, seed=41)
    resumed.skip(7)

    actual_suffix = resumed.next(1, 5)

    assert torch.equal(actual_suffix, expected_suffix)
    assert not torch.isin(actual_suffix, expected_prefix).any()
    assert resumed.report()["start_offset"] == 7
    assert resumed.report()["run_tokens_emitted"] == 5
    assert resumed.report()["absolute_tokens_emitted"] == 12


def test_bank_prefix_alignment_compares_every_prefix_v_state() -> None:
    one_shot = torch.tensor([[[[[1.0]]], [[[2.0]]], [[[3.0]]]]])
    aligned = one_shot.clone().requires_grad_()
    shifted = one_shot.clone()
    shifted[:, 1] += 1.0
    shifted.requires_grad_()
    initial = torch.zeros(1, 1, 1, 1)

    aligned_loss, aligned_raw = bank_prefix_alignment_loss(aligned, one_shot, initial)
    shifted_loss, shifted_raw = bank_prefix_alignment_loss(shifted, one_shot, initial)
    shifted_loss.backward()

    assert aligned_loss == 0
    assert aligned_raw == 0
    assert shifted_loss > 0
    assert shifted_raw > 0
    assert shifted.grad is not None
    assert shifted.grad[:, 0].abs().sum() == 0
    assert shifted.grad[:, 1].abs().sum() > 0
    assert shifted.grad[:, 2].abs().sum() == 0


def test_bank_prefix_alignment_normalizes_each_position_independently() -> None:
    initial = torch.zeros(1, 1, 1, 1)
    one_shot = torch.tensor([[[[[1.0]]], [[[100.0]]]]])
    recurrent = torch.tensor([[[[[2.0]]], [[[200.0]]]]])

    normalized, _ = bank_prefix_alignment_loss(recurrent, one_shot, initial)

    # Both prefixes have the same relative squared error despite a 100x scale gap.
    torch.testing.assert_close(normalized, torch.tensor(1.0))


def test_prefix_bank_rank_loss_prefers_the_matching_sequence_state() -> None:
    teacher = torch.tensor([[[1.0, -1.0]], [[-1.0, 1.0]]])
    matching = teacher.clone().requires_grad_()
    mismatch = torch.roll(teacher, 1, dims=0)
    regressed = mismatch.clone().requires_grad_()

    matching_loss = prefix_bank_rank_loss(
        matching,
        mismatch,
        teacher,
        improvement_rate=0.01,
    )
    regressed_loss = prefix_bank_rank_loss(
        regressed,
        matching.detach(),
        teacher,
        improvement_rate=0.01,
    )
    regressed_loss.backward()

    assert matching_loss == 0
    assert regressed_loss > 0
    assert regressed.grad is not None
    assert torch.isfinite(regressed.grad).all()


def test_vocab_candidate_groups_select_a_winner_per_sequence() -> None:
    scores = torch.tensor(
        [
            [2.0, 1.0, 3.0],
            [1.0, 4.0, 2.0],
            [3.0, 0.5, 4.0],
        ]
    )
    candidate_group_sets = torch.arange(3 * 3 * 2).reshape(3, 3, 2)

    winners, selected = select_candidate_groups(scores, candidate_group_sets)

    assert winners.tolist() == [1, 2, 1]
    torch.testing.assert_close(
        selected,
        candidate_group_sets[winners, torch.arange(3)],
    )


def test_same_tail_sequences_hold_current_token_constant_within_pairs() -> None:
    sequences = same_tail_sequences(101, pairs=4, length=5, seed=13).reshape(4, 2, 5)

    torch.testing.assert_close(sequences[:, 0, -1], sequences[:, 1, -1])
    assert not bool(torch.any(sequences[:, 0, :-1].eq(sequences[:, 1, :-1])))


def test_bank_update_metrics_remove_the_shared_formula_template() -> None:
    initial = torch.full((4, 1, 3), 100.0)
    update = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    ).unsqueeze(1)

    metrics = bank_update_metrics(initial + update, initial)

    assert metrics["whole_state_pair_cosine"] > 0.999
    assert metrics["history_pair_update_difference_rms"] > 0
    assert metrics["history_update_pair_cosine"] < metrics["whole_state_pair_cosine"]


def test_vocab_paired_history_batch_only_shares_the_current_token() -> None:
    tokens = torch.arange(24).reshape(4, 6)

    paired = share_final_token_within_pairs(tokens)

    torch.testing.assert_close(paired[::2, -1], paired[1::2, -1])
    torch.testing.assert_close(paired[:, :-1], tokens[:, :-1])


def test_vocab_adjacent_swap_is_an_involution() -> None:
    values = torch.arange(30).reshape(6, 5)

    torch.testing.assert_close(swap_adjacent_batch(swap_adjacent_batch(values)), values)


def test_vocab_generalization_builds_unique_unseen_combinations() -> None:
    first = sequence_token_ids(31, length=3, examples=7, seed=19)
    second = sequence_token_ids(31, length=3, examples=7, seed=19)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.shape == (7, 3)
    assert first.unique().numel() == first.numel()


def test_composition_curriculum_cycles_the_full_condition_grid() -> None:
    conditions = [curriculum_condition(step, (1, 2), (0, 1, 2)) for step in range(6)]

    assert conditions == [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    assert curriculum_condition(6, (1, 2), (0, 1, 2)) == (1, 0)
    assert [curriculum_length(step, (1, 2, 4)) for step in range(4)] == [1, 2, 4, 1]


def test_recurrence_teacher_batch_contains_every_prefix_probe_pair() -> None:
    histories = torch.tensor([[10, 11], [20, 21]])

    batch = build_prefix_probe_batch(histories, (7, 8))

    assert batch.input_ids.tolist() == [
        [10, 7, 0],
        [10, 8, 0],
        [10, 11, 7],
        [10, 11, 8],
        [20, 7, 0],
        [20, 8, 0],
        [20, 21, 7],
        [20, 21, 8],
    ]
    assert batch.target_positions.tolist() == [1, 1, 2, 2, 1, 1, 2, 2]
    assert batch.attention_mask.sum(dim=1).tolist() == [2, 2, 3, 3, 2, 2, 3, 3]
    assert (batch.batch_size, batch.turns, batch.probes) == (2, 2, 2)


def test_dialogue_recurrence_collation_preserves_absolute_positions() -> None:
    examples = [
        DialogueExample(
            history_segments=(torch.tensor([10, 11]), torch.tensor([12])),
            history_prefix_ids=torch.tensor([10, 11, 12]),
            current_ids=torch.tensor([20, 21, 22]),
            current_prompt_ids=torch.tensor([20]),
            prediction_positions=torch.tensor([1]),
            target_ids=torch.tensor([22]),
            reference="first",
        ),
        DialogueExample(
            history_segments=(torch.tensor([30]), torch.tensor([31, 32])),
            history_prefix_ids=torch.tensor([30, 31, 32]),
            current_ids=torch.tensor([40, 41]),
            current_prompt_ids=torch.tensor([40]),
            prediction_positions=torch.tensor([0]),
            target_ids=torch.tensor([41]),
            reference="second",
        ),
    ]

    batch = collate_dialogues(examples)

    assert batch.history_mask.sum(dim=-1).tolist() == [[2, 1], [1, 2]]
    assert batch.history_positions.tolist() == [
        [[0, 1], [2, 0]],
        [[0, 0], [1, 2]],
    ]
    assert batch.current_positions.tolist() == [[3, 4, 5], [3, 4, 0]]
    assert batch.prediction_positions.tolist() == [[1], [0]]


def test_history_effect_loss_prefers_the_matching_context_delta() -> None:
    batch = DialogueTokenBatch(
        history_ids=torch.zeros(2, 1, 1, dtype=torch.long),
        history_mask=torch.ones(2, 1, 1, dtype=torch.bool),
        history_positions=torch.zeros(2, 1, 1, dtype=torch.long),
        current_ids=torch.zeros(2, 1, dtype=torch.long),
        current_mask=torch.ones(2, 1, dtype=torch.bool),
        current_positions=torch.zeros(2, 1, dtype=torch.long),
        prediction_positions=torch.zeros(2, 1, dtype=torch.long),
        prediction_mask=torch.ones(2, 1, dtype=torch.bool),
        target_ids=torch.zeros(2, 1, dtype=torch.long),
    )
    reset = torch.zeros(2, 1, 2)
    teacher = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    aligned = teacher.clone().requires_grad_()
    swapped = teacher.flip(0)

    aligned_direction, aligned_identity = history_effect_loss(
        aligned,
        reset,
        teacher,
        batch,
        temperature=0.1,
    )
    swapped_direction, swapped_identity = history_effect_loss(
        swapped,
        reset,
        teacher,
        batch,
        temperature=0.1,
    )
    (aligned_direction + aligned_identity).backward()

    assert aligned_direction + aligned_identity < swapped_direction + swapped_identity
    assert aligned.grad is not None
    assert torch.isfinite(aligned.grad).all()


def test_history_pair_centered_loss_cancels_shared_updates() -> None:
    batch = DialogueTokenBatch(
        history_ids=torch.zeros(2, 1, 1, dtype=torch.long),
        history_mask=torch.ones(2, 1, 1, dtype=torch.bool),
        history_positions=torch.zeros(2, 1, 1, dtype=torch.long),
        current_ids=torch.zeros(2, 1, dtype=torch.long),
        current_mask=torch.ones(2, 1, dtype=torch.bool),
        current_positions=torch.zeros(2, 1, dtype=torch.long),
        prediction_positions=torch.zeros(2, 1, dtype=torch.long),
        prediction_mask=torch.ones(2, 1, dtype=torch.bool),
        target_ids=torch.zeros(2, 1, dtype=torch.long),
    )
    reset = torch.zeros(2, 1, 2)
    teacher = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    aligned = teacher.clone().requires_grad_()
    shared = torch.full_like(teacher, 0.5).requires_grad_()

    aligned_loss = recurrence.history_pair_centered_loss(
        aligned,
        reset,
        teacher,
        batch,
    )
    shared_loss = recurrence.history_pair_centered_loss(
        shared,
        reset,
        teacher,
        batch,
    )
    shared_loss.backward()

    assert aligned_loss < 1e-7
    assert shared_loss > aligned_loss
    assert shared.grad is not None
    assert torch.isfinite(shared.grad).all()


def test_matched_current_prompt_validation_rejects_mixed_queries() -> None:
    matched = [
        SimpleNamespace(
            current_prompt_ids=torch.tensor([1, 2]),
            history_prefix_ids=torch.tensor([3, 4]),
            history_segments=(torch.tensor([3]), torch.tensor([4])),
        ),
        SimpleNamespace(
            current_prompt_ids=torch.tensor([1, 2]),
            history_prefix_ids=torch.tensor([5, 6]),
            history_segments=(torch.tensor([5]), torch.tensor([6])),
        ),
    ]
    recurrence.validate_matched_current_prompts(matched)

    mismatched = [
        *matched[:1],
        SimpleNamespace(
            current_prompt_ids=torch.tensor([1, 3]),
            history_prefix_ids=torch.tensor([5, 6]),
            history_segments=(torch.tensor([5]), torch.tensor([6])),
        ),
    ]
    with pytest.raises(ValueError, match="identical latest prompt"):
        recurrence.validate_matched_current_prompts(mismatched)

    with pytest.raises(ValueError, match="equal history lengths"):
        recurrence.validate_matched_current_prompts(
            [
                matched[0],
                SimpleNamespace(
                    current_prompt_ids=torch.tensor([1, 2]),
                    history_prefix_ids=torch.tensor([5]),
                    history_segments=(torch.tensor([5]),),
                ),
            ]
        )
    with pytest.raises(ValueError, match="equal per-turn history lengths"):
        recurrence.validate_matched_current_prompts(
            [
                matched[0],
                SimpleNamespace(
                    current_prompt_ids=torch.tensor([1, 2]),
                    history_prefix_ids=torch.tensor([5, 6]),
                    history_segments=(torch.tensor([5, 6]), torch.tensor([])),
                ),
            ]
        )
    with pytest.raises(ValueError, match="distinct histories"):
        recurrence.validate_matched_current_prompts([matched[0], matched[0]])


def test_position_matched_dialogue_groups_are_valid_sampling_units() -> None:
    def row(current: tuple[int, ...], history: tuple[int, ...]):
        return SimpleNamespace(
            current_prompt_ids=torch.tensor(current),
            history_prefix_ids=torch.tensor(history),
            history_segments=(torch.tensor(history[:1]), torch.tensor(history[1:])),
        )

    examples = [
        row((1, 2), (3, 4)),
        row((1, 2), (5, 6)),
        row((1, 2), (7, 8)),
        row((1, 3), (9, 10)),
        row((1, 2), (3, 4)),
    ]

    grouped = recurrence.group_position_matched_dialogues(examples)

    assert sorted(len(bucket) for bucket in grouped[2]) == [1, 3]
    for bucket in grouped[2]:
        if len(bucket) >= 2:
            recurrence.validate_matched_current_prompts(bucket)


def test_deterministic_half_evaluation_restores_runtime_configuration() -> None:
    class Runtime(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.activation = arti.Half(stochastic=True)

        def stacked_groups(self) -> tuple[()]:
            return ()

    runtime = Runtime()
    with recurrence.deterministic_half_evaluation(runtime, enabled=True):
        assert runtime.activation.stochastic is False
    assert runtime.activation.stochastic is True


def test_history_bank_flow_loss_requires_correct_bank_to_beat_mismatch() -> None:
    batch = DialogueTokenBatch(
        history_ids=torch.zeros(2, 1, 1, dtype=torch.long),
        history_mask=torch.ones(2, 1, 1, dtype=torch.bool),
        history_positions=torch.zeros(2, 1, 1, dtype=torch.long),
        current_ids=torch.zeros(2, 1, dtype=torch.long),
        current_mask=torch.ones(2, 1, dtype=torch.bool),
        current_positions=torch.zeros(2, 1, dtype=torch.long),
        prediction_positions=torch.zeros(2, 1, dtype=torch.long),
        prediction_mask=torch.ones(2, 1, dtype=torch.bool),
        target_ids=torch.zeros(2, 1, dtype=torch.long),
    )
    teacher = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    shuffled = torch.zeros_like(teacher)
    improved = (teacher * 0.75).requires_grad_()
    regressed = (-teacher).requires_grad_()

    improved_rank, improved_direction = history_bank_flow_loss(
        improved,
        shuffled,
        teacher,
        batch,
        improvement_rate=0.01,
    )
    regressed_rank, _ = history_bank_flow_loss(
        regressed,
        shuffled,
        teacher,
        batch,
        improvement_rate=0.01,
    )
    (improved_rank + improved_direction).backward()

    assert improved_rank < regressed_rank
    assert improved_direction < 1e-6
    assert improved.grad is not None
    assert torch.isfinite(improved.grad).all()


def test_history_bank_token_rank_loss_prefers_matching_bank_logits() -> None:
    batch = DialogueTokenBatch(
        history_ids=torch.zeros(2, 1, 1, dtype=torch.long),
        history_mask=torch.ones(2, 1, 1, dtype=torch.bool),
        history_positions=torch.zeros(2, 1, 1, dtype=torch.long),
        current_ids=torch.zeros(2, 1, dtype=torch.long),
        current_mask=torch.ones(2, 1, dtype=torch.bool),
        current_positions=torch.zeros(2, 1, dtype=torch.long),
        prediction_positions=torch.zeros(2, 1, dtype=torch.long),
        prediction_mask=torch.ones(2, 1, dtype=torch.bool),
        target_ids=torch.tensor([[0], [1]]),
    )
    model = nn.Module()
    model.lm_head = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.lm_head.weight.copy_(torch.eye(2))
    mismatch = torch.zeros(2, 1, 2)
    matching = torch.tensor([[[3.0, 0.0]], [[0.0, 3.0]]], requires_grad=True)
    wrong = matching.detach().flip(0).requires_grad_()

    matching_loss = history_bank_token_rank_loss(
        model,
        matching,
        mismatch,
        batch,
        improvement_rate=0.01,
    )
    wrong_loss = history_bank_token_rank_loss(
        model,
        wrong,
        mismatch,
        batch,
        improvement_rate=0.01,
    )
    matching_loss.backward()

    assert matching_loss < wrong_loss
    assert matching.grad is not None
    assert torch.isfinite(matching.grad).all()


def test_history_bank_token_rank_loss_uses_later_supervised_tokens() -> None:
    batch = DialogueTokenBatch(
        history_ids=torch.zeros(2, 1, 1, dtype=torch.long),
        history_mask=torch.ones(2, 1, 1, dtype=torch.bool),
        history_positions=torch.zeros(2, 1, 1, dtype=torch.long),
        current_ids=torch.zeros(2, 2, dtype=torch.long),
        current_mask=torch.ones(2, 2, dtype=torch.bool),
        current_positions=torch.zeros(2, 2, dtype=torch.long),
        prediction_positions=torch.tensor([[0, 1], [0, 1]]),
        prediction_mask=torch.ones(2, 2, dtype=torch.bool),
        target_ids=torch.tensor([[0, 0], [0, 1]]),
    )
    model = nn.Module()
    model.lm_head = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.lm_head.weight.copy_(torch.eye(2))
    mismatch = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]
    )
    matching = torch.tensor(
        [[[3.0, 0.0], [0.5, 0.0]], [[3.0, 0.0], [0.0, 0.5]]],
        requires_grad=True,
    )
    wrong = matching.detach().clone()
    wrong[:, 1] = wrong.flip(0)[:, 1]
    wrong.requires_grad_()

    matching_loss = history_bank_token_rank_loss(
        model,
        matching,
        mismatch,
        batch,
        improvement_rate=0.01,
    )
    wrong_loss = history_bank_token_rank_loss(
        model,
        wrong,
        mismatch,
        batch,
        improvement_rate=0.01,
    )
    matching_loss.backward()

    assert matching_loss < wrong_loss
    assert matching.grad is not None
    assert torch.count_nonzero(matching.grad[:, 1]) > 0


def test_bank_value_alignment_loss_matches_complete_v_state() -> None:
    previous = torch.tensor([[[[2.0, -1.0]]]])
    token_value = torch.tensor([[[[3.0, 1.0]]]])
    aligned = token_value.clone().requires_grad_()
    shifted = torch.tensor([[[[2.5, 0.0]]]], requires_grad=True)

    aligned_loss = bank_value_alignment_loss(aligned, token_value, previous)
    shifted_loss = bank_value_alignment_loss(shifted, token_value, previous)
    shifted_loss.backward()

    assert aligned_loss == 0
    assert shifted_loss > aligned_loss
    assert shifted.grad is not None
    assert torch.isfinite(shifted.grad).all()


def test_route_candidate_rank_loss_is_scale_free_and_only_penalizes_regression() -> None:
    deterministic = torch.tensor([0.4, 0.1], requires_grad=True)
    winner = torch.tensor([0.2, 0.2])
    explored = torch.tensor([True, False])

    loss = route_candidate_rank_loss(
        deterministic,
        winner,
        explored,
        improvement_margin=0.01,
    )
    scaled = route_candidate_rank_loss(
        deterministic.detach() * 7.0,
        winner * 7.0,
        explored,
        improvement_margin=0.01,
    )
    loss.backward()

    torch.testing.assert_close(loss.detach(), scaled, rtol=1e-6, atol=1e-7)
    assert deterministic.grad is not None
    assert deterministic.grad[0] > 0
    assert deterministic.grad[1] == 0


def test_dialogue_recurrence_groups_only_by_online_turn_count() -> None:
    def example(turns: int, reference: str) -> DialogueExample:
        segments = tuple(torch.tensor([10 + turn]) for turn in range(turns))
        prefix = torch.cat(segments)
        return DialogueExample(
            history_segments=segments,
            history_prefix_ids=prefix,
            current_ids=torch.tensor([20, 21]),
            current_prompt_ids=torch.tensor([20]),
            prediction_positions=torch.tensor([0]),
            target_ids=torch.tensor([21]),
            reference=reference,
        )

    grouped = group_dialogues_by_turn_count(
        [example(4, "four"), example(2, "two-a"), example(2, "two-b")]
    )

    assert list(grouped) == [2, 4]
    assert [item.reference for item in grouped[2]] == ["two-a", "two-b"]
    assert history_turn_sampling_weights(grouped, "balanced") == (1.0, 1.0)
    assert history_turn_sampling_weights(grouped, "proportional") == (2.0, 1.0)


def test_recurrence_update_uses_explicit_previous_values_without_reprocessing() -> None:
    runtime = LayeredRecallRuntime(
        (0, 1),
        [make_field(), make_field()],
        [make_updater(), make_updater()],
    )
    runtime.pack_updaters()
    traces = torch.randn(2, 2, 3, 4)
    previous = torch.randn(2, 2, 3, 4)
    mask = torch.ones(2, 2, 3, dtype=torch.bool)

    expected = runtime.stacked_updater(traces, previous, mask=mask, recall_steps=0)
    actual = update_sequence_values(
        runtime,
        traces,
        recall_steps=0,
        previous_values=previous,
        traces_include_recall=True,
    )

    torch.testing.assert_close(actual, expected)


def test_recurrence_update_preserves_variable_length_mask() -> None:
    runtime = LayeredRecallRuntime(
        (0, 1),
        [make_field(), make_field()],
        [make_updater(), make_updater()],
    )
    runtime.pack_updaters()
    traces = torch.randn(2, 2, 3, 4)
    previous = torch.randn(2, 2, 3, 4)
    token_mask = torch.tensor([[True, True, False], [True, False, False]])
    site_mask = token_mask.unsqueeze(1).expand(-1, 2, -1)

    expected = runtime.stacked_updater(
        traces,
        previous,
        mask=site_mask,
        recall_steps=0,
    )
    actual = update_sequence_values(
        runtime,
        traces,
        recall_steps=0,
        previous_values=previous,
        traces_include_recall=True,
        mask=token_mask,
    )

    torch.testing.assert_close(actual, expected)


def test_streaming_bank_recomputes_each_trace_after_the_previous_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Capture:
        def __init__(self) -> None:
            self.value: torch.Tensor | None = None

        def pop(self) -> torch.Tensor:
            assert self.value is not None
            value = self.value
            self.value = None
            return value

    class Runtime:
        def __init__(self) -> None:
            self.fields = [SimpleNamespace(bank=torch.zeros(1, 1))]
            self.active: torch.Tensor | None = None

        def initial_values(self, batch_size: int, _reference: torch.Tensor) -> torch.Tensor:
            return torch.zeros(batch_size, 1, 1, 1)

        @contextmanager
        def use(self, values: torch.Tensor, _mask: torch.Tensor):
            self.active = values
            try:
                yield
            finally:
                self.active = None

    class Backbone(nn.Module):
        def __init__(self, runtime: Runtime, final: Capture, sites: Capture) -> None:
            super().__init__()
            self.runtime = runtime
            self.final = final
            self.sites = sites
            self.cache_inputs: list[object | None] = []

        def forward(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
            past_key_values: object | None,
            use_cache: bool,
        ) -> object:
            assert use_cache
            assert attention_mask.shape[1] == len(self.cache_inputs) + 1
            assert position_ids.shape == input_ids.shape
            assert self.runtime.active is not None
            self.cache_inputs.append(past_key_values)
            trace = input_ids.float().unsqueeze(-1)
            trace = trace + self.runtime.active[:, 0, 0, 0].reshape(-1, 1, 1)
            self.final.value = trace
            self.sites.value = trace.unsqueeze(1)
            return SimpleNamespace(past_key_values=len(self.cache_inputs))

    class Model(nn.Module):
        def __init__(self, backbone: Backbone) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.model = backbone

    final = Capture()
    sites = Capture()
    runtime = Runtime()
    backbone = Backbone(runtime, final, sites)
    model = Model(backbone)

    def update(
        _runtime: object,
        traces: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        previous = kwargs["previous_values"]
        assert isinstance(previous, torch.Tensor)
        return previous + traces

    monkeypatch.setattr(recurrence, "update_sequence_values", update)
    state = recurrence.stream_sequence_values(
        model,
        runtime,  # type: ignore[arg-type]
        final,  # type: ignore[arg-type]
        sites,  # type: ignore[arg-type]
        torch.tensor([[1, 1]]),
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([[0, 1]]),
        recall_steps=1,
    )

    # First trace is 1 + Bank(0); second is recomputed as 1 + Bank(1).
    torch.testing.assert_close(state.values, torch.tensor([[[[3.0]]]]))
    assert backbone.cache_inputs == [None, 1]
    assert state.attention_mask.tolist() == [[True, True]]


def test_quality_flow_rewards_deeper_state_that_moves_toward_teacher() -> None:
    teacher = torch.tensor([[[[2.0, -2.0]]]])
    shallow = torch.zeros_like(teacher)
    improved = torch.tensor([[[[1.5, -1.5]]]], requires_grad=True)

    rank, direction = quality_flow_loss(
        improved,
        shallow,
        teacher,
        improvement_rate=0.01,
    )
    (rank + direction).backward()

    assert rank == 0
    assert direction == 0
    assert improved.grad is not None
    assert torch.isfinite(improved.grad).all()


def test_read_depth_cycle_is_deterministic_and_preserves_duplicate_coverage() -> None:
    depths = (16, 32, 8, 32, 64)
    rng = random.Random(17)

    observed = [
        recurrence.scheduled_read_depth(step, depths, mode="cycle", rng=rng)
        for step in range(8)
    ]

    assert observed == [16, 32, 8, 32, 64, 16, 32, 8]


def test_read_quality_anchor_scales_with_deep_depth() -> None:
    assert recurrence.shallow_read_depth(8, min_steps=2) == 2
    assert recurrence.shallow_read_depth(16, min_steps=2) == 4
    assert recurrence.shallow_read_depth(32, min_steps=2) == 8
    assert recurrence.shallow_read_depth(64, min_steps=2) == 16


def test_forced_continuation_uses_one_based_training_interval() -> None:
    assert [
        recurrence.scheduled_read_flag(step, interval=4)
        for step in range(8)
    ] == [False, False, False, True, False, False, False, True]


def test_read_force_pattern_can_mix_fixed_and_adaptive_at_same_depth() -> None:
    pattern = (1, 0, 1, 0)
    assert [
        recurrence.scheduled_read_flag(step, pattern=pattern, interval=0)
        for step in range(6)
    ] == [True, False, True, False, True, False]


def test_vocab_curriculum_probe_selection_uses_only_single_token_encodings() -> None:
    class Tokenizer:
        eos_token_id = 9

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert not add_special_tokens
            return {"?": [3], "\n": [4], " the": [5, 6], "0": [3]}[text]

    assert choose_probe_ids(Tokenizer()) == (3, 4)


def test_embedding_collision_audit_reports_only_exact_aliases() -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(4, 3)
            with torch.no_grad():
                self.embedding.weight.copy_(
                    torch.tensor(
                        [
                            [0.0, 0.0, 0.0],
                            [1.0, 2.0, 3.0],
                            [1.0, 2.0, 3.0],
                            [1.0, 2.0, 3.001],
                        ]
                    )
                )

        def get_input_embeddings(self) -> nn.Embedding:
            return self.embedding

    report = embedding_collision_report(Model(), token_count=3)

    assert report["zero_rows"] == 1
    assert report["duplicate_rows"] == 1
    assert report["duplicate_group_count"] == 1
    assert report["token_rows_audited"] == 3
    assert report["model_embedding_rows"] == 4
    assert report["padding_rows_excluded"] == 1
    assert report["duplicate_groups"] == [{"size": 2, "sample_token_ids": [1, 2]}]


def test_vocab_projection_restores_frozen_model_dtype() -> None:
    class Decoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(3, dtype=torch.float64)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Decoder()
            self.lm_head = nn.Linear(3, 5, bias=False, dtype=torch.float64)

    result = project_logits(Model(), torch.randn(2, 3, dtype=torch.float32))

    assert result.shape == (2, 5)
    assert result.dtype == torch.float32


def test_vocab_identity_loss_rewards_distinct_bidirectional_matches() -> None:
    teacher = torch.eye(4)
    aligned = teacher.clone().requires_grad_()
    collapsed = torch.ones_like(teacher)

    aligned_loss, _ = identity_loss(
        nn.Module(),
        aligned,
        teacher,
        temperature=0.1,
        identity_weight=0.25,
        centered_weight=1.0,
        kl_weight=0.25,
        include_logits=False,
    )
    collapsed_loss, _ = identity_loss(
        nn.Module(),
        collapsed,
        teacher,
        temperature=0.1,
        identity_weight=0.25,
        centered_weight=1.0,
        kl_weight=0.25,
        include_logits=False,
    )
    aligned_loss.backward()

    assert aligned_loss < collapsed_loss
    assert aligned.grad is not None
    assert torch.isfinite(aligned.grad).all()


def test_evenly_spaced_qwen_sites_include_the_final_layer() -> None:
    assert evenly_spaced_layers(28, 1) == (27,)
    assert evenly_spaced_layers(28, 4) == (6, 13, 20, 27)


def test_new_layered_sites_start_as_identity_while_final_site_preserves_asset() -> None:
    template = make_updater()
    with torch.no_grad():
        template.value_weight.normal_(std=0.1)
        template.value_bias.normal_(std=0.01)
    updaters = [copy.deepcopy(template) for _ in range(4)]
    for updater in updaters[:-1]:
        with torch.no_grad():
            updater.value_weight.zero_()
            updater.value_bias.zero_()
    runtime = LayeredRecallRuntime(
        (0, 1, 2, 3),
        [make_field() for _ in range(4)],
        updaters,
    )
    trace = torch.randn(2, 5, 4)
    previous = torch.randn(2, 4, 3, 4)
    mask = torch.ones(2, 5, dtype=torch.bool)

    next_values = torch.stack(
        [
            updater(trace, previous[:, site], mask=mask)
            for site, updater in enumerate(runtime.updaters)
        ],
        dim=1,
    )

    torch.testing.assert_close(next_values[:, :3], previous[:, :3], rtol=0, atol=0)
    assert not torch.equal(next_values[:, 3], previous[:, 3])


def test_optimizer_groups_apply_new_site_rate_only_to_new_sites() -> None:
    runtime = LayeredRecallRuntime(
        (0, 1, 2, 3),
        [make_field() for _ in range(4)],
        [make_updater() for _ in range(4)],
    )

    groups = optimizer_parameter_groups(
        runtime,
        restored_sites=(1, 3),
        learning_rate=1e-3,
        new_site_learning_rate_multiplier=0.02,
        recall_learning_rate_multiplier=20.0,
    )
    by_name = {str(group["group_name"]): group for group in groups}

    assert set(by_name) == {"new/base", "restored/base"}
    assert by_name["new/base"]["lr"] == 2e-5
    assert by_name["restored/base"]["lr"] == 1e-3
    new_ids = {id(parameter) for parameter in by_name["new/base"]["params"]}
    restored_ids = {id(parameter) for parameter in by_name["restored/base"]["params"]}
    assert id(runtime.updaters[0].value_weight) in new_ids
    assert id(runtime.updaters[2].value_weight) in new_ids
    assert id(runtime.updaters[1].value_weight) in restored_ids
    assert id(runtime.updaters[3].value_weight) in restored_ids
    assert new_ids.isdisjoint(restored_ids)


def test_stacked_updaters_match_serial_forward_and_gradients() -> None:
    torch.manual_seed(1729)
    serial = LayeredRecallRuntime(
        (0, 1, 2),
        [make_field() for _ in range(3)],
        [make_updater() for _ in range(3)],
    )
    with torch.no_grad():
        for updater in serial.updaters:
            updater.value_weight.normal_(std=0.05)
            updater.value_bias.normal_(std=0.01)
    stacked = copy.deepcopy(serial)
    stacked.pack_updaters()
    history = torch.randn(2, 2, 5, 4)
    history_mask = torch.tensor(
        [
            [[True, True, True, False, False], [True, True, True, True, False]],
            [[True, True, False, False, False], [True, True, True, True, True]],
        ]
    )
    batch = DialogueBatch(
        history=history,
        history_mask=history_mask,
        current_local=torch.zeros(2, 1, 4),
        current_mask=torch.ones(2, 1, dtype=torch.bool),
        prediction_positions=torch.zeros(2, 1, dtype=torch.long),
        prediction_mask=torch.ones(2, 1, dtype=torch.bool),
        teacher_hidden=torch.zeros(2, 1, 4),
        target_ids=torch.zeros(2, 1, dtype=torch.long),
    )
    target = torch.randn(2, 3, 3, 4)

    serial_output = serial.unroll_history(batch)
    (serial_output * target).sum().backward()
    stacked_output = stacked.unroll_history(batch)
    (stacked_output * target).sum().backward()

    torch.testing.assert_close(stacked_output, serial_output, rtol=1e-5, atol=1e-6)
    assert stacked.stacked_updater is not None
    for site, updater in enumerate(serial.updaters):
        for name, serial_parameter in updater.named_parameters():
            stacked_parameter = stacked.stacked_updater.parameter_for(name)
            assert stacked_parameter.requires_grad == serial_parameter.requires_grad
            if serial_parameter.grad is None:
                continue
            assert stacked_parameter.grad is not None, name
            torch.testing.assert_close(
                stacked_parameter.grad[site],
                serial_parameter.grad,
                rtol=1e-5,
                atol=1e-6,
            )


def test_heterogeneous_recall_capacities_pack_by_compatible_shape() -> None:
    torch.manual_seed(1731)
    small = RecallValueUpdater(
        4,
        3,
        workspace_dim=8,
        depth=1,
        interface_slots=2,
        recall_slots=2,
        recall_group_topk=1,
        recall_steps=1,
    )
    large = RecallValueUpdater(
        4,
        3,
        workspace_dim=8,
        depth=1,
        interface_slots=2,
        recall_slots=4,
        recall_group_topk=2,
        recall_steps=1,
    )
    with torch.no_grad():
        small.value_weight.normal_(std=0.05)
        large.value_weight.normal_(std=0.05)
        small.workspace[0].layer.state.recall.bank.normal_(std=0.1)
        large.workspace[0].layer.state.recall.bank.normal_(std=0.1)
    serial = LayeredRecallRuntime(
        (0, 1),
        [make_field(), make_field()],
        [small, large],
    )
    packed = copy.deepcopy(serial)
    packed.pack_updaters()
    batch = DialogueBatch(
        history=torch.randn(2, 2, 5, 4),
        history_mask=torch.ones(2, 2, 5, dtype=torch.bool),
        current_local=torch.zeros(2, 1, 4),
        current_mask=torch.ones(2, 1, dtype=torch.bool),
        prediction_positions=torch.zeros(2, 1, dtype=torch.long),
        prediction_mask=torch.ones(2, 1, dtype=torch.bool),
        teacher_hidden=torch.zeros(2, 1, 4),
        target_ids=torch.zeros(2, 1, dtype=torch.long),
    )

    torch.manual_seed(1777)
    expected = serial.unroll_history(batch, recall_steps=1)
    torch.manual_seed(1777)
    actual = packed.unroll_history(batch, recall_steps=1)

    assert packed.stacked_updater is None
    assert tuple(sites for sites, _group in packed.stacked_groups()) == ((0,), (1,))
    assert packed.recall_slots_by_site == (2, 4)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert set(packed.updater_state_dict(0)) == set(small.state_dict())
    assert set(packed.updater_state_dict(1)) == set(large.state_dict())


def test_heterogeneous_packed_candidate_groups_restrict_search_with_gradients() -> None:
    torch.manual_seed(1733)
    small = RecallValueUpdater(
        4,
        3,
        workspace_dim=8,
        depth=1,
        interface_slots=2,
        recall_slots=2,
        recall_group_topk=1,
        recall_route_exploration=1.0,
        recall_steps=1,
    )
    large = RecallValueUpdater(
        4,
        3,
        workspace_dim=8,
        depth=1,
        interface_slots=2,
        recall_slots=4,
        recall_group_topk=2,
        recall_route_exploration=1.0,
        recall_steps=1,
    )
    with torch.no_grad():
        for updater in (small, large):
            updater.value_weight.normal_(std=0.05)
            updater.workspace[0].layer.state.recall.bank.normal_(std=0.1)
    runtime = LayeredRecallRuntime(
        (0, 1),
        [make_field(), make_field()],
        [small, large],
    ).train()
    runtime.pack_updaters()
    processed = torch.randn(2, 2, 5, 4)
    previous = torch.randn(2, 2, 3, 4)
    mask = torch.ones(2, 5, dtype=torch.bool)

    with torch.no_grad():
        candidate, candidate_groups = runtime.update_packed_values(
            processed,
            previous,
            mask=mask,
            recall_steps=1,
            return_candidate_groups=True,
        )
    restricted, restricted_groups = runtime.update_packed_values(
        processed,
        previous,
        mask=mask,
        recall_steps=1,
        candidate_groups=candidate_groups,
        return_candidate_groups=True,
    )
    restricted.square().mean().backward()

    assert candidate.shape == restricted.shape == previous.shape
    assert candidate_groups.shape[:3] == (2, 2, 1)
    assert candidate_groups.shape[-1] == 2
    assert torch.count_nonzero(candidate_groups[:, 0, ..., 1]) == 0
    torch.testing.assert_close(restricted_groups, candidate_groups, rtol=0, atol=0)
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(restricted).all()
    gradients = [
        parameter.grad
        for parameter in runtime.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_packed_runtime_captures_and_reuses_full_site_route_stack() -> None:
    torch.manual_seed(1831)
    runtime = LayeredRecallRuntime(
        (0, 1),
        [make_field(), make_field()],
        [make_recall_updater(), make_recall_updater()],
    ).eval()
    runtime.pack_updaters()
    processed = torch.randn(2, 2, 5, 4)
    previous = torch.randn(2, 2, 3, 4)
    mask = torch.ones(2, 5, dtype=torch.bool)

    captured, route_stack = runtime.update_route_planned_values(
        processed,
        previous,
        mask=mask,
        recall_steps=1,
        return_route_stack=True,
    )
    replayed = runtime.update_route_planned_values(
        processed,
        previous,
        mask=mask,
        recall_steps=3,
        route_stack=route_stack,
    )

    assert route_stack.axis == "site"
    assert len(route_stack.items) == runtime.site_count
    assert all(item.axis == "block" for item in route_stack.items)
    assert captured.shape == replayed.shape == previous.shape
    assert torch.isfinite(replayed).all()


def test_stacked_artifact_state_excludes_runtime_only_buffers() -> None:
    updaters = [make_recall_updater(), make_recall_updater()]
    runtime = LayeredRecallRuntime(
        (0, 1),
        [make_field(), make_field()],
        updaters,
    )
    expected_keys = set(updaters[0].state_dict())
    runtime.pack_updaters()
    assert runtime.stacked_updater is not None

    state = runtime.updater_state_dict(0)

    assert set(state) == expected_keys
    assert not any("_factor_route" in name for name in state)
    assert not any("recognition_" in name for name in state)


def test_legacy_runtime_buffers_are_ignored_but_unknown_keys_remain_strict() -> None:
    source = make_recall_updater()
    target = make_recall_updater()
    state = dict(source.state_dict())
    runtime_only = {name: value for name, value in source.named_buffers() if name not in state}
    assert runtime_only
    state.update(runtime_only)

    load_runtime_updater_state(target, state)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value)

    state["not_a_runtime_buffer"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_runtime_updater_state(target, state)


class AddOne(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1.0


class TinyTrunk(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        self.layers = nn.ModuleList([AddOne(), AddOne()])

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        use_cache: bool,
    ) -> None:
        del attention_mask, use_cache
        value = self.embedding(input_ids) + position_ids.unsqueeze(-1)
        for layer in self.layers:
            value = layer(value)


class TinyBatchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = TinyTrunk()


def test_runtime_only_changes_explicitly_activated_forward() -> None:
    layers = nn.ModuleList([AddOne(), AddOne()])
    field = make_field()
    with torch.no_grad():
        field.bank.normal_(std=0.2)
    runtime = LayeredRecallRuntime((1,), [field], [make_updater()])
    runtime.install(layers)
    x = torch.randn(2, 3, 4)
    plain = layers[1](layers[0](x))
    values = field.bank.detach().unsqueeze(0).expand(2, -1, -1).clone()
    mask = torch.ones(2, 3, dtype=torch.bool)

    with runtime.use(values.unsqueeze(1), mask):
        recalled = layers[1](layers[0](x))
    restored = layers[1](layers[0](x))
    runtime.close()

    assert not torch.equal(recalled, plain)
    torch.testing.assert_close(restored, plain, rtol=0, atol=0)


def test_token_batch_preserves_history_offset_positions() -> None:
    example = DialogueExample(
        history_segments=(torch.tensor([1, 2]),),
        history_prefix_ids=torch.tensor([1, 2]),
        current_ids=torch.tensor([3, 4, 5]),
        current_prompt_ids=torch.tensor([3, 4]),
        prediction_positions=torch.tensor([1]),
        target_ids=torch.tensor([5]),
        reference="sample",
    )
    batch = build_token_batch((example,), pad_token_id=0)

    assert torch.equal(batch.input_ids, torch.tensor([[3, 4, 5]]))
    assert torch.equal(batch.attention_mask, torch.tensor([[True, True, True]]))
    assert torch.equal(batch.position_ids, torch.tensor([[2, 3, 4]]))


def test_run_current_batches_padded_rows_with_explicit_positions() -> None:
    model = TinyBatchModel()
    runtime = LayeredRecallRuntime(
        (0,),
        [make_field()],
        [make_updater()],
    )
    runtime.install(model.model.layers)
    capture = _LiveBoundaryCapture(model.model.layers[-1])
    tokens = CurrentTokenBatch(
        input_ids=torch.tensor([[1, 2, 3], [4, 5, 0]]),
        attention_mask=torch.tensor([[True, True, True], [True, True, False]]),
        position_ids=torch.tensor([[7, 8, 9], [20, 21, 0]]),
    )
    values = torch.randn(2, 1, 3, 4)

    output = run_current(model, runtime, capture, tokens, values)

    assert output.shape == (2, 3, 4)
    assert runtime._active_values is None
    assert runtime._active_mask is None
    capture.close()
    runtime.close()


def test_compatible_sampler_returns_one_exact_length_bucket() -> None:
    generator = torch.Generator().manual_seed(19)
    lengths = torch.tensor([2, 3, 2, 4, 3, 3])
    weights = torch.tensor([1.0, 1.0, 2.0, 1.0, 3.0, 1.0])

    indices = sample_compatible_indices(
        lengths,
        weights,
        3,
        generator=generator,
    )

    assert 1 <= indices.numel() <= 3
    assert lengths.index_select(0, indices).unique().numel() == 1
    assert indices.unique().numel() == indices.numel()


def make_dialogue_example(input_ids: list[int], target_id: int) -> DialogueExample:
    values = torch.tensor(input_ids)
    return DialogueExample(
        history_segments=(values[:1],),
        history_prefix_ids=values[:1],
        current_ids=values[1:],
        current_prompt_ids=values[1:],
        prediction_positions=torch.tensor([0, 1]),
        target_ids=torch.tensor([target_id]),
        reference="sample",
    )


def test_token_coverage_weights_emphasize_rare_literal_inputs() -> None:
    examples = [
        make_dialogue_example([1, 1, 1], 2),
        make_dialogue_example([1, 1, 3], 2),
        make_dialogue_example([8, 9], 8),
    ]

    weights = token_coverage_sampling_weights(
        examples,
        vocabulary_size=10,
        alpha=0.5,
    )
    coverage = dialogue_token_coverage(
        examples,
        train_examples=2,
        vocabulary_size=10,
        special_token_ids=(0,),
    )

    assert weights.shape == (3,)
    assert weights[2] > weights[:2].max()
    assert coverage["output_distribution_supervision"] == "full_vocabulary"
    assert coverage["train"]["context_tokens_max"] == 3
    assert coverage["evaluation"]["unique_input_tokens"] == 2


def test_layered_artifact_fresh_reload_restores_every_updater(tmp_path) -> None:
    source = LayeredRecallRuntime(
        (0, 1),
        [make_field(), make_field()],
        [make_updater(), make_updater()],
    ).eval()
    with torch.no_grad():
        for site, updater in enumerate(source.updaters):
            updater.value_weight.normal_(std=0.01 * (site + 1))
            updater.value_bias.fill_(0.1 * (site + 1))
    trace = torch.randn(2, 4, 4)
    previous = torch.randn(2, 2, 3, 4)
    mask = torch.ones(2, 4, dtype=torch.bool)
    expected = torch.stack(
        [
            updater(trace, previous[:, site], mask=mask)
            for site, updater in enumerate(source.updaters)
        ],
        dim=1,
    )
    source.pack_updaters()
    artifact = tmp_path / "layered.recall.arti.st"
    save_runtime(
        artifact,
        source,
        expected[0],
        model_id="tiny",
        formula_reference="test/formula@1",
    )
    restored = LayeredRecallRuntime(
        (0, 1),
        [make_field(), make_field()],
        [make_updater(), make_updater()],
    ).eval()

    load_runtime_state(restored, load_file(str(artifact)))
    with safe_open(str(artifact), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    actual = torch.stack(
        [
            updater(trace, previous[:, site], mask=mask)
            for site, updater in enumerate(restored.updaters)
        ],
        dim=1,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert metadata["runtime_schema"] == "arti/benchmark-layered-recall-runtime@1"
    assert metadata["host_read_component"] == "arti/recall@2"
    assert metadata["host_read_policy_component"] == "arti/refine-policy@1"
    assert metadata["host_read_semantics"] == "core_next_state_refine"
    assert metadata["host_read_activation"] == "none"
    assert metadata["host_read_retention"] == "0"


def test_layered_artifact_preserves_per_site_updater_capacity(tmp_path) -> None:
    small = RecallValueUpdater(
        4,
        3,
        workspace_dim=8,
        depth=1,
        interface_slots=2,
        recall_slots=2,
        recall_group_topk=1,
        recall_steps=1,
    )
    large = RecallValueUpdater(
        4,
        3,
        workspace_dim=8,
        depth=1,
        interface_slots=2,
        recall_slots=4,
        recall_group_topk=2,
        recall_steps=1,
    )
    runtime = LayeredRecallRuntime(
        (0, 1),
        [make_field(), make_field()],
        [small, large],
    )
    artifact = tmp_path / "graded.recall.arti.st"
    save_runtime(
        artifact,
        runtime,
        torch.zeros(2, 3, 4),
        model_id="tiny",
        formula_reference="test/formula@1",
    )

    slots, topk = layered_artifact_updater_capacity(artifact)

    assert slots == (2, 4)
    assert topk == (1, 2)


def test_layered_artifact_expands_into_nested_topology_without_touching_new_sites(
    tmp_path,
) -> None:
    source_layers = (1, 3)
    source = LayeredRecallRuntime(
        source_layers,
        [make_field(), make_field()],
        [make_updater(), make_updater()],
    ).eval()
    with torch.no_grad():
        source.updaters[0].value_bias.fill_(0.25)
        source.updaters[1].value_bias.fill_(0.75)
    artifact = tmp_path / "source.recall.arti.st"
    save_runtime(
        artifact,
        source,
        torch.zeros(2, 3, 4),
        model_id="tiny",
        formula_reference="test/formula@1",
    )
    target = LayeredRecallRuntime(
        (0, 1, 2, 3),
        [make_field() for _ in range(4)],
        [make_updater() for _ in range(4)],
    ).eval()
    with torch.no_grad():
        for updater in target.updaters:
            updater.value_weight.zero_()
            updater.value_bias.zero_()

    layers = layered_artifact_layers(artifact)
    restored = overlay_runtime_state(target, load_file(str(artifact)), layers)

    assert layers == source_layers
    assert restored == (1, 3)
    assert torch.count_nonzero(target.updaters[0].value_bias) == 0
    assert torch.count_nonzero(target.updaters[2].value_bias) == 0
    torch.testing.assert_close(
        target.updaters[1].value_bias,
        source.updaters[0].value_bias,
    )
    torch.testing.assert_close(
        target.updaters[3].value_bias,
        source.updaters[1].value_bias,
    )
