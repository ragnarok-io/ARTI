from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import Tensor, nn

import arti
from arti import mechanisms
from arti.component_registry import (
    component_provenance,
    component_spec,
    validate_component_provenance,
)


def schema(*dimensions: int | str, axes: tuple[str, ...]) -> mechanisms.TensorSchema:
    return mechanisms.TensorSchema(
        dtype="float32",
        device_class="any",
        dimensions=dimensions,
        semantic_axes=axes,
        mask_semantics="none",
    )


def terminal_abi() -> mechanisms.TerminalOutputABI:
    return mechanisms.TerminalOutputABI(
        fields=(
            mechanisms.TerminalField(
                "value",
                schema("B", 2, axes=("batch", "feature")),
                "terminal-value",
            ),
            mechanisms.TerminalField(
                "validity",
                mechanisms.TensorSchema(
                    dtype="boolean",
                    device_class="any",
                    dimensions=("B",),
                    semantic_axes=("batch",),
                    mask_semantics="boolean-validity",
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
        validity_contract="one validity value per row",
        packing_contract="named terminal tensors",
        score_contract="one terminal score per row",
        consumer_contract="hard one winner",
        gradient_contract=mechanisms.GradientContract.autograd(),
    )


def make_query() -> mechanisms.SealedBankQuery:
    query = mechanisms.LinearBankQuery(
        schema("B", 2, axes=("batch", "feature")),
        schema("B", 1, axes=("batch", "query")),
        input_dim=2,
        query_dim=1,
        bias=False,
        retrieval_contract={"kind": "bank-local-dot-product", "top_k": 1},
        normalization_contract={"scope": "bank_local", "kind": "hard-one"},
    )
    with torch.no_grad():
        query.projection.weight.copy_(torch.tensor([[1.0, 0.0]]))
    return mechanisms.seal_bank_query(
        query,
        capabilities=("latest-state-requery",),
    )


def bank_signature(
    program: mechanisms.BankOwnedQueryProgram,
    query: mechanisms.SealedBankQuery,
    abi: mechanisms.TerminalOutputABI,
    *,
    local_refine_ref: str = "arti/refine-policy@2",
) -> mechanisms.BankExecutionSignatureV2:
    return mechanisms.BankExecutionSignatureV2.from_program(
        program,
        input_schema=query.signature.input_schema,
        output_schema=schema("B", 2, axes=("batch", "feature")),
        shape_relation=mechanisms.ShapeRelation.preserves_shape(),
        query_signature=query.signature,
        local_normalization_contract={
            "scope": "bank_local",
            "kind": "hard-one",
        },
        local_formula_ref="arti/formula-fabric@2",
        local_refine_ref=local_refine_ref,
        terminal_adapter_ref="arti/test-terminal-adapter@1",
        terminal_abi_ref="arti/terminal-output-abi@1",
        terminal_abi_fingerprint=abi.fingerprint,
        score_contract="one serial Bank-local path score",
        gradient_contract=mechanisms.GradientContract.autograd(),
        execution_capabilities=("eager", "latest-state-requery", "serial-k1"),
    )


def terminal_outputs(value: Tensor) -> dict[str, Tensor]:
    return {
        "value": value,
        "validity": torch.ones(
            value.shape[0], dtype=torch.bool, device=value.device
        ),
        "score": value.new_ones((value.shape[0],)),
    }


class RequeryBank(mechanisms.BankOwnedQueryProgram):
    def __init__(
        self,
        abi: mechanisms.TerminalOutputABI,
        *,
        bank_id: str = "memory",
        bank_local_refine: bool = False,
        min_local_steps: int = 1,
        max_local_steps: int = 3,
    ) -> None:
        query = make_query()
        local_refine = (
            mechanisms.BankLocalRefinePolicy(
                min_steps=min_local_steps,
                max_steps=max_local_steps,
            )
            if bank_local_refine
            else None
        )
        super().__init__(
            bank_id=bank_id,
            query=query,
            local_refine=local_refine,
        )
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bank_local_refine = bank_local_refine
        self.observed_queries: list[float] = []
        self.bind_signature(
            bank_signature(
                self,
                query,
                abi,
                local_refine_ref=(
                    arti.component_ref(local_refine)
                    if local_refine is not None
                    else "arti/refine-policy@2"
                ),
            )
        )

    def execute(
        self,
        value: Tensor,
        *,
        query_result: mechanisms.BankQueryResult,
        max_candidates: int,
    ) -> mechanisms.FederalBankStep:
        query_value = query_result.value.reshape(())
        self.observed_queries.append(float(query_value.detach().cpu()))
        if bool((query_value < 0).detach()):
            next_value = torch.stack((-value[:, 0], value[:, 1]), dim=-1)
            candidate = (
                mechanisms.FederalCandidate.local(
                    "rewrite",
                    local_log_score=value.new_zeros(()),
                    next_value=next_value,
                )
                if self.bank_local_refine
                else mechanisms.FederalCandidate.child(
                    "rewrite",
                    local_log_score=value.new_zeros(()),
                    next_bank_id=self.bank_id,
                    next_value=next_value,
                )
            )
        else:
            candidate = mechanisms.FederalCandidate.terminal(
                "answer",
                local_log_score=value.new_zeros(()),
                outputs=terminal_outputs(
                    query_result.value.expand_as(value) * self.scale
                ),
            )
        return mechanisms.FederalBankStep((candidate,)[:max_candidates])


class UnusedQueryBank(RequeryBank):
    def execute(
        self,
        value: Tensor,
        *,
        query_result: mechanisms.BankQueryResult,
        max_candidates: int,
    ) -> mechanisms.FederalBankStep:
        raise AssertionError("an unrelated mounted Bank must not execute")


class EmptyOwnedQueryBank(mechanisms.BankOwnedQueryProgram):
    def execute(
        self,
        value: Tensor,
        *,
        query_result: mechanisms.BankQueryResult,
        max_candidates: int,
    ) -> mechanisms.FederalBankStep:
        raise AssertionError("invalid mounted assets must fail before execution")


class TestTerminalAdapter:
    pass


arti.register_component(
    "arti/test-terminal-adapter@1",
    component_type=TestTerminalAdapter,
    lifecycle="alpha",
    constructible=False,
    config_builder=lambda _component: {},
)
arti.register_component(
    "arti/test-requery-bank@1",
    component_type=RequeryBank,
    lifecycle="alpha",
    constructible=False,
    config_builder=lambda component: {
        "bank_id": component.bank_id,
        "behavior": "sign-flip-latest-state-requery",
    },
    dependency_builder=lambda _component: (
        "arti/formula-fabric@2",
        "arti/refine-policy@2",
        "arti/sealed-bank-query@1",
        "arti/test-terminal-adapter@1",
    ),
)
arti.register_component(
    "arti/test-unused-query-bank@1",
    component_type=UnusedQueryBank,
    lifecycle="alpha",
    constructible=False,
    config_builder=lambda component: {
        "bank_id": component.bank_id,
        "behavior": "unreachable-test-bank",
    },
    dependency_builder=lambda _component: (
        "arti/formula-fabric@2",
        "arti/refine-policy@2",
        "arti/sealed-bank-query@1",
        "arti/test-terminal-adapter@1",
    ),
)


def make_federal(*, with_unused: bool = False) -> mechanisms.FederalRecallV2:
    abi = terminal_abi()
    banks: dict[str, mechanisms.BankOwnedQueryProgram] = {
        "memory": RequeryBank(abi),
    }
    if with_unused:
        banks["a-unrelated"] = UnusedQueryBank(abi, bank_id="a-unrelated")
    return mechanisms.FederalRecallV2(
        banks,
        terminal_abi=abi,
        root_bank_ids=("memory",),
        max_levels=2,
        max_k=1,
    )


def make_bank_local_federal(
    *,
    min_steps: int = 1,
    max_steps: int = 3,
) -> mechanisms.FederalRecallV2:
    abi = terminal_abi()
    return mechanisms.FederalRecallV2(
        {
            "memory": RequeryBank(
                abi,
                bank_local_refine=True,
                min_local_steps=min_steps,
                max_local_steps=max_steps,
            )
        },
        terminal_abi=abi,
        root_bank_ids=("memory",),
        max_levels=1,
        max_k=1,
    )


def test_owned_query_is_reexecuted_from_the_latest_bank_state() -> None:
    federal = make_federal()
    value = torch.tensor([[-2.0, 0.5]])

    result, trace = federal(value, return_trace=True)

    torch.testing.assert_close(result["value"], torch.tensor([[2.0, 2.0]]))
    assert federal.banks["memory"].observed_queries == [-2.0, 2.0]
    assert len(trace.steps) == 2
    assert trace.winner_paths == ("memory/rewrite/memory/answer",)


def test_bank_local_refine_requeries_without_consuming_federation_depth() -> None:
    federal = make_bank_local_federal()
    value = torch.tensor([[-2.0, 0.5]])

    result, trace = federal(value, return_trace=True)

    torch.testing.assert_close(result["value"], torch.tensor([[2.0, 2.0]]))
    assert federal.banks["memory"].observed_queries == [-2.0, 2.0]
    assert len(trace.steps) == 1
    local = trace.steps[0].local_refine
    assert [item.action for item in local] == ["continue-local", "terminal"]
    assert [item.local_step for item in local] == [1, 2]
    assert [item.input_shape for item in local] == [(1, 2), (1, 2)]
    assert all(item.query_ref == "arti/linear-bank-query@1" for item in local)
    assert all(item.formula_ref == "arti/formula-fabric@2" for item in local)
    assert local[-1].exit_reason == "formula-exit"
    assert trace.winner_paths == ("memory/answer",)


def test_bank_local_refine_enforces_minimum_and_maximum_steps() -> None:
    with pytest.raises(mechanisms.FederalRecallError, match="before min_steps"):
        make_bank_local_federal(min_steps=3)(torch.tensor([[-2.0, 0.5]]))

    federal = make_bank_local_federal(max_steps=1)
    with pytest.raises(mechanisms.FederalRecallError, match="reached max_steps"):
        federal(torch.tensor([[-2.0, 0.5]]))


def test_bank_local_refine_policy_is_versioned_and_in_provenance() -> None:
    federal = make_bank_local_federal()
    policy = federal.banks["memory"].local_refine

    assert policy is not None
    assert arti.component_ref(policy) == "arti/bank-local-refine-policy@1"
    assert arti.component_spec(policy).config == {
        "min_steps": 1,
        "max_steps": 3,
        "state_source": "latest-local-state",
        "exit_semantics": "formula-request-after-min-steps",
    }
    provenance = component_provenance(federal)
    assert validate_component_provenance(provenance) == provenance
    root = next(item for item in provenance["components"] if item["path"] == "$")
    assert "arti/bank-local-refine-policy@1" in root["dependencies"]


def test_k1_federation_matches_explicit_serial_query_execution() -> None:
    direct = make_federal()
    federated = make_federal()
    federated.load_state_dict(deepcopy(direct.state_dict()))
    value = torch.tensor([[-3.0, 0.25]])

    first = direct.banks["memory"](value, max_candidates=1).candidates[0]
    assert first.next_value is not None
    second = direct.banks["memory"](
        first.next_value,
        max_candidates=1,
    ).candidates[0]
    assert second.terminal_outputs is not None

    actual = federated(value)

    torch.testing.assert_close(actual["value"], second.terminal_outputs["value"])


def test_owned_bank_rejects_child_value_outside_its_declared_output_schema() -> None:
    bank = RequeryBank(terminal_abi())

    def invalid_execute(
        value: Tensor,
        *,
        query_result: mechanisms.BankQueryResult,
        max_candidates: int,
    ) -> mechanisms.FederalBankStep:
        del query_result, max_candidates
        return mechanisms.FederalBankStep(
            (
                mechanisms.FederalCandidate.child(
                    "rewrite",
                    local_log_score=value.new_zeros(()),
                    next_bank_id="memory",
                    next_value=value.new_zeros((1, 3)),
                ),
            )
        )

    bank.execute = invalid_execute

    with pytest.raises(mechanisms.TensorSchemaError, match="shape does not match"):
        bank(torch.tensor([[-2.0, 0.5]]), max_candidates=1)


def test_unrelated_bank_mount_does_not_dilute_local_query() -> None:
    baseline = make_federal()
    mounted = make_federal(with_unused=True)
    with torch.no_grad():
        baseline.banks["memory"].scale.fill_(1.75)
    mounted.load_state_dict(baseline.state_dict(), strict=False)
    value = torch.tensor([[-1.5, 0.75]])

    expected = baseline(value)
    actual = mounted(value)

    torch.testing.assert_close(actual["value"], expected["value"])
    assert mounted.banks["a-unrelated"].observed_queries == []
    baseline_scale_key = next(
        name for name in baseline.state_dict() if name.endswith(".scale")
    )
    assert baseline_scale_key in mounted.state_dict()


def test_owned_query_remains_frozen_while_program_parameters_train() -> None:
    federal = make_federal()
    value = torch.tensor([[-2.0, 0.5]], requires_grad=True)

    result = federal(value)
    result["value"].square().mean().backward()

    bank = federal.banks["memory"]
    assert bank.scale.grad is not None and torch.isfinite(bank.scale.grad)
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert all(not parameter.requires_grad for parameter in bank.query.parameters())
    assert all(parameter.grad is None for parameter in bank.query.parameters())
    trainable = tuple(
        parameter for parameter in bank.parameters() if parameter.requires_grad
    )
    assert len(trainable) == 1 and trainable[0] is bank.scale


def test_optimizer_updates_bank_program_without_creating_query_state() -> None:
    federal = make_federal()
    bank = federal.banks["memory"]
    query_parameters = tuple(bank.query.parameters())
    initial_query = tuple(parameter.detach().clone() for parameter in query_parameters)
    initial_scale = bank.scale.detach().clone()
    optimizer = torch.optim.AdamW(federal.parameters(), lr=0.05)
    value = torch.tensor([[-2.0, 0.5]])

    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = federal(value)["value"].square().mean()
        loss.backward()
        optimizer.step()

    assert not torch.equal(bank.scale.detach(), initial_scale)
    assert all(parameter not in optimizer.state for parameter in query_parameters)
    for actual, expected in zip(query_parameters, initial_query, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_federal_v2_state_dict_reload_is_exact() -> None:
    source = make_federal()
    restored = make_federal()
    with torch.no_grad():
        source.banks["memory"].scale.fill_(1.75)
    restored.load_state_dict(deepcopy(source.state_dict()))
    value = torch.tensor([[-2.0, 0.5]])

    expected, expected_trace = source(value, return_trace=True)
    actual, actual_trace = restored(value, return_trace=True)

    torch.testing.assert_close(actual["value"], expected["value"])
    assert actual_trace.to_dict() == expected_trace.to_dict()


def test_federal_v2_arti_st_round_trip_and_provenance(tmp_path) -> None:
    source = make_federal()
    restored = make_federal()
    with torch.no_grad():
        source.banks["memory"].scale.fill_(1.25)
    value = torch.tensor([[-2.0, 0.5]])
    expected = source(value)["value"]

    provenance = component_provenance(source)
    assert validate_component_provenance(provenance) == provenance
    saved = arti.save(source, tmp_path / "federal-query.arti.st")
    arti.load(saved.weights_path, model=restored)
    actual = restored(value)["value"]

    torch.testing.assert_close(actual, expected)
    root = next(item for item in provenance["components"] if item["path"] == "$")
    assert root["ref"] == "arti/federal-recall@2"


def test_federal_v2_has_independent_identity_and_rejects_wide_k() -> None:
    federal = make_federal()

    assert arti.component_ref(federal) == "arti/federal-recall@2"
    assert component_spec(federal).dependencies == (
        "arti/bank-execution-signature@2",
        "arti/formula-fabric@2",
        "arti/gradient-contract@1",
        "arti/linear-bank-query@1",
        "arti/query-execution-signature@1",
        "arti/refine-policy@2",
        "arti/sealed-bank-query@1",
        "arti/shape-relation@1",
        "arti/tensor-schema@1",
        "arti/terminal-output-abi@1",
        "arti/test-requery-bank@1",
        "arti/test-terminal-adapter@1",
    )
    with pytest.raises(mechanisms.FederalRecallError, match="serial K=1"):
        mechanisms.FederalRecallV2(
            {"memory": RequeryBank(terminal_abi())},
            terminal_abi=terminal_abi(),
            root_bank_ids=("memory",),
            max_k=2,
        )


def test_bank_rejects_a_query_asset_from_another_signature() -> None:
    abi = terminal_abi()
    other = make_query()
    with torch.no_grad():
        other.query.projection.weight.add_(1.0)

    with pytest.raises(mechanisms.FederalRecallError, match="asset is invalid"):
        program = EmptyOwnedQueryBank(
            bank_id="memory",
            query=other,
        )
        program.bind_signature(RequeryBank(abi).signature)


def test_mounted_query_rejects_in_place_state_mutation() -> None:
    federal = make_federal()
    with torch.no_grad():
        federal.banks["memory"].query.query.projection.weight.add_(1.0)

    with pytest.raises(mechanisms.BankQueryError, match="modified after mounting"):
        federal(torch.tensor([[-2.0, 0.5]]))


def test_parent_state_load_rejects_query_state_outside_its_signature() -> None:
    federal = make_federal()
    state = deepcopy(federal.state_dict())
    query_weight = next(
        name for name in state if name.endswith("query.projection.weight")
    )
    state[query_weight].add_(1.0)

    with pytest.raises(mechanisms.BankQueryError, match="tensor state"):
        federal.load_state_dict(state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_federal_v2_preserves_cuda_and_gradients() -> None:
    federal = make_federal().cuda()
    value = torch.tensor([[-2.0, 0.5]], device="cuda", requires_grad=True)

    result = federal(value)
    result["value"].sum().backward()

    assert result["value"].is_cuda
    assert value.grad is not None and value.grad.is_cuda
    assert federal.banks["memory"].scale.grad is not None
