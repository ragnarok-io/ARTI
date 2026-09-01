from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest
import torch
from torch import Tensor, nn

import arti
from arti import mechanisms
from arti.component_registry import component_spec


def digest(name: str) -> str:
    return hashlib.sha256(name.encode("ascii")).hexdigest()


def schema(*dimensions: int | str, axes: tuple[str, ...]) -> mechanisms.TensorSchema:
    return mechanisms.TensorSchema(
        dtype="floating",
        device_class="any",
        dimensions=dimensions,
        semantic_axes=axes,
    )


def terminal_abi() -> mechanisms.TerminalOutputABI:
    return mechanisms.TerminalOutputABI(
        fields=(
            mechanisms.TerminalField(
                "value",
                schema("B", 4, axes=("batch", "feature")),
                "terminal-value",
            ),
            mechanisms.TerminalField(
                "validity",
                mechanisms.TensorSchema(
                    dtype="boolean",
                    device_class="any",
                    dimensions=("B",),
                    semantic_axes=("batch",),
                ),
                "terminal-validity",
            ),
            mechanisms.TerminalField(
                "score",
                schema("B", axes=("batch",)),
                "terminal-score",
            ),
        ),
        factor_order=(),
        validity_contract="one validity bit per terminal row",
        packing_contract="named dense terminal tensors",
        score_contract="one calibrated terminal score per row",
        consumer_contract="hard one-winner consumes one valid record",
        gradient_contract=mechanisms.GradientContract.autograd(),
    )


def signature(
    bank_id: str,
    *,
    input_schema: mechanisms.TensorSchema,
    output_schema: mechanisms.TensorSchema,
    abi: mechanisms.TerminalOutputABI,
) -> mechanisms.BankExecutionSignature:
    return mechanisms.BankExecutionSignature(
        program_ref=f"arti/test-{bank_id}@1",
        program_config_fingerprint=digest(f"{bank_id}-config"),
        program_state_fingerprint=digest(f"{bank_id}-state"),
        input_schema=input_schema,
        output_schema=output_schema,
        shape_relation=mechanisms.ShapeRelation.arbitrary_to_terminal(
            "explicit test Bank transition followed by a terminal adapter"
        ),
        query_contract={
            "algorithm": "fixed-identity",
            "fixed": True,
            "deterministic": True,
            "trainable": False,
        },
        local_normalization_contract={"scope": "bank_local", "kind": "topk"},
        local_formula_ref="arti/formula-fabric@2",
        local_refine_ref="arti/refine-policy@2",
        terminal_adapter_ref="arti/test-terminal-adapter@1",
        terminal_abi_ref="arti/terminal-output-abi@1",
        terminal_abi_fingerprint=abi.fingerprint,
        score_contract="scores are calibrated across the two test Banks",
        gradient_contract=mechanisms.GradientContract.autograd(),
        execution_capabilities=("eager",),
    )


def outputs(value: Tensor, *, score: float) -> dict[str, Tensor]:
    return {
        "value": value,
        "validity": torch.ones(value.shape[0], dtype=torch.bool, device=value.device),
        "score": value.new_full((value.shape[0],), score),
    }


class RootBank(mechanisms.AutonomousBankProgram):
    def __init__(self, abi: mechanisms.TerminalOutputABI) -> None:
        super().__init__(
            bank_id="root",
            signature=signature(
                "root",
                input_schema=schema("B", 4, axes=("batch", "feature")),
                output_schema=schema("B", 2, 3, axes=("batch", "row", "feature")),
                abi=abi,
            ),
        )
        self.register_buffer("fixed_query", torch.eye(4), persistent=True)

    def forward(self, value: Tensor, *, max_candidates: int) -> mechanisms.FederalBankStep:
        wrong = mechanisms.FederalCandidate.terminal(
            "wrong",
            local_log_score=value.new_tensor(-0.01),
            outputs=outputs(torch.zeros_like(value), score=0.0),
        )
        child_value = torch.cat((value, value[:, :2]), dim=-1).reshape(1, 2, 3)
        correct = mechanisms.FederalCandidate.child(
            "descend",
            local_log_score=value.new_tensor(-0.05),
            next_bank_id="child",
            next_value=child_value,
        )
        return mechanisms.FederalBankStep((wrong, correct)[:max_candidates])


class ChildBank(mechanisms.AutonomousBankProgram):
    def __init__(self, abi: mechanisms.TerminalOutputABI) -> None:
        super().__init__(
            bank_id="child",
            signature=signature(
                "child",
                input_schema=schema(
                    "B", 2, 3, axes=("batch", "row", "feature")
                ),
                output_schema=schema("B", 6, axes=("batch", "feature")),
                abi=abi,
            ),
        )
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, value: Tensor, *, max_candidates: int) -> mechanisms.FederalBankStep:
        flat = value.reshape(value.shape[0], -1)
        terminal = mechanisms.FederalCandidate.terminal(
            "answer",
            local_log_score=value.new_tensor(-0.02),
            outputs=outputs(flat[:, :4] * self.scale, score=1.0),
        )
        return mechanisms.FederalBankStep((terminal,)[:max_candidates])


class UnusedBank(mechanisms.AutonomousBankProgram):
    def __init__(self, abi: mechanisms.TerminalOutputABI) -> None:
        super().__init__(
            bank_id="unused",
            signature=signature(
                "unused",
                input_schema=schema("B", 3, axes=("batch", "feature")),
                output_schema=schema("B", 9, axes=("batch", "feature")),
                abi=abi,
            ),
        )

    def forward(self, value: Tensor, *, max_candidates: int) -> mechanisms.FederalBankStep:
        raise AssertionError("the unrelated Bank must never execute")


def make_federal(*, with_unused: bool = False) -> mechanisms.FederalRecall:
    abi = terminal_abi()
    banks: dict[str, mechanisms.AutonomousBankProgram] = {
        "root": RootBank(abi),
        "child": ChildBank(abi),
    }
    if with_unused:
        banks["unused"] = UnusedBank(abi)
    return mechanisms.FederalRecall(
        banks,
        terminal_abi=abi,
        root_bank_ids=("root",),
        max_levels=2,
        max_k=32,
    )


def test_federal_requery_crosses_heterogeneous_bank_shapes() -> None:
    federal = make_federal()
    value = torch.randn(5, 4)

    result, trace = federal(value, max_k=2, return_trace=True)

    torch.testing.assert_close(result["value"], value)
    assert result["validity"].all()
    assert trace.maximum_kept_paths == 2
    assert all("child" in path for path in trace.winner_paths)
    assert {step.depth for step in trace.steps} == {0, 1}
    assert all(step.kept_count <= 2 for step in trace.steps)


def test_k1_loses_ambiguous_route_that_k2_preserves() -> None:
    federal = make_federal()
    value = torch.randn(3, 4)

    narrow = federal(value, max_k=1)
    wide = federal(value, max_k=2)

    torch.testing.assert_close(narrow["value"], torch.zeros_like(value))
    torch.testing.assert_close(wide["value"], value)


def test_one_level_cannot_use_unresolved_child() -> None:
    federal = make_federal()
    value = torch.randn(2, 4)

    result = federal(value, max_k=2, max_levels=1)

    torch.testing.assert_close(result["value"], torch.zeros_like(value))


def test_unrelated_hotplug_does_not_change_existing_paths() -> None:
    baseline = make_federal()
    mounted = make_federal(with_unused=True)
    mounted.load_state_dict(baseline.state_dict(), strict=False)
    value = torch.randn(4, 4)

    baseline_output, baseline_trace = baseline(value, max_k=2, return_trace=True)
    mounted_output, mounted_trace = mounted(value, max_k=2, return_trace=True)

    torch.testing.assert_close(mounted_output["value"], baseline_output["value"])
    assert mounted_trace.winner_paths == baseline_trace.winner_paths


def test_terminal_adapter_gradient_reaches_selected_child_only() -> None:
    federal = make_federal()
    value = torch.randn(4, 4)

    result = federal(value, max_k=2)
    result["value"].square().mean().backward()

    child = federal.banks["child"]
    root = federal.banks["root"]
    assert child.scale.grad is not None and torch.isfinite(child.scale.grad)
    assert root.fixed_query.requires_grad is False
    assert root.fixed_query.grad is None


def test_federal_state_dict_reload_is_exact() -> None:
    source = make_federal()
    restored = make_federal()
    with torch.no_grad():
        source.banks["child"].scale.fill_(1.75)
    restored.load_state_dict(deepcopy(source.state_dict()))
    value = torch.randn(2, 4)

    expected, expected_trace = source(value, max_k=2, return_trace=True)
    actual, actual_trace = restored(value, max_k=2, return_trace=True)

    torch.testing.assert_close(actual["value"], expected["value"])
    assert actual_trace.to_dict() == expected_trace.to_dict()


def test_unknown_child_and_abi_mismatch_fail_before_coercion() -> None:
    federal = make_federal()
    bad_candidate = mechanisms.FederalCandidate.child(
        "missing",
        local_log_score=torch.tensor(0.0),
        next_bank_id="absent",
        next_value=torch.zeros(1, 3),
    )
    with pytest.raises(mechanisms.FederalRecallError, match="unknown child"):
        original = federal.banks["root"].forward
        federal.banks["root"].forward = lambda *_args, **_kwargs: mechanisms.FederalBankStep(
            (bad_candidate,)
        )
        try:
            federal(torch.zeros(1, 4))
        finally:
            federal.banks["root"].forward = original

    changed = terminal_abi()
    payload = changed.to_dict()
    payload["score_contract"] = "incompatible score semantics"
    payload.pop("fingerprint")
    incompatible = mechanisms.TerminalOutputABI(
        fields=changed.fields,
        factor_order=changed.factor_order,
        validity_contract=changed.validity_contract,
        packing_contract=changed.packing_contract,
        score_contract="incompatible score semantics",
        consumer_contract=changed.consumer_contract,
        gradient_contract=changed.gradient_contract,
    )
    with pytest.raises(mechanisms.TerminalABIError, match="does not match"):
        mechanisms.FederalRecall(
            {"root": RootBank(changed)},
            terminal_abi=incompatible,
            root_bank_ids=("root",),
        )


def test_federal_component_identity_and_runtime_trace_boundary() -> None:
    federal = make_federal()
    value = torch.randn(1, 4)
    _result, trace = federal(value, max_k=2, return_trace=True)

    assert arti.component_ref(federal) == "arti/federal-recall@1"
    assert component_spec(federal).lifecycle == "stable"
    assert component_spec(federal).dependencies == (
        "arti/bank-execution-signature@1",
        "arti/terminal-output-abi@1",
    )
    assert not any("path" in key for key in federal.state_dict())
    assert trace.to_dict()["ref"] == "arti/federal-trace@2"


def test_bank_ids_do_not_inherit_moduledict_dot_restriction() -> None:
    abi = terminal_abi()
    dotted = RootBank(abi)
    dotted.bank_id = "root.v1"
    federal = mechanisms.FederalRecall(
        {"root.v1": dotted, "child": ChildBank(abi)},
        terminal_abi=abi,
        root_bank_ids=("root.v1",),
        max_levels=2,
        max_k=2,
    )

    result = federal(torch.randn(1, 4), max_k=1)

    assert result["value"].shape == (1, 4)


def test_empty_batch_is_rejected_explicitly() -> None:
    federal = make_federal()

    with pytest.raises(mechanisms.FederalRecallError, match="non-empty batch"):
        federal(torch.empty(0, 4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_federal_terminal_execution_preserves_cuda_and_gradients() -> None:
    federal = make_federal().cuda()
    value = torch.randn(2, 4, device="cuda")

    result = federal(value, max_k=2)
    result["value"].sum().backward()

    assert result["value"].is_cuda
    assert federal.banks["child"].scale.grad is not None
    assert federal.banks["child"].scale.grad.is_cuda
