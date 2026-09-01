from __future__ import annotations

from copy import deepcopy

import pytest
import torch

import arti
from arti import mechanisms
from arti.component_registry import component_spec


def tensor_schema(*dimensions: int | str, axes: tuple[str, ...]) -> mechanisms.TensorSchema:
    return mechanisms.TensorSchema(
        dtype="float32",
        device_class="any",
        dimensions=dimensions,
        semantic_axes=axes,
        mask_semantics="boolean-validity",
    )


def terminal_abi() -> mechanisms.TerminalOutputABI:
    return mechanisms.TerminalOutputABI(
        fields=(
            mechanisms.TerminalField(
                "value",
                tensor_schema("B", "T", 32, axes=("batch", "token", "feature")),
                "terminal-value",
                factor_index=0,
            ),
            mechanisms.TerminalField(
                "validity",
                mechanisms.TensorSchema(
                    dtype="boolean",
                    device_class="any",
                    dimensions=("B", "T"),
                    semantic_axes=("batch", "token"),
                    mask_semantics="boolean-validity",
                ),
                "terminal-validity",
            ),
            mechanisms.TerminalField(
                "score",
                tensor_schema("B", axes=("batch",)),
                "terminal-score",
                factor_index=0,
            ),
        ),
        factor_order=("primary",),
        validity_contract="validity masks value token positions",
        packing_contract="named dense tensors without implicit padding",
        score_contract="one raw terminal score per batch row",
        consumer_contract="consumer selects one terminal result",
        gradient_contract=mechanisms.GradientContract.autograd(),
    )


def signature(
    *,
    name: str,
    input_schema: mechanisms.TensorSchema,
    output_schema: mechanisms.TensorSchema,
    abi: mechanisms.TerminalOutputABI,
) -> mechanisms.BankExecutionSignature:
    return mechanisms.BankExecutionSignature(
        program_ref=f"arti/{name}@1",
        program_config_fingerprint="1" * 64,
        program_state_fingerprint="2" * 64,
        input_schema=input_schema,
        output_schema=output_schema,
        shape_relation=mechanisms.ShapeRelation.arbitrary_to_terminal(
            "local program maps its private shape through an explicit terminal adapter"
        ),
        query_contract={
            "algorithm": "fixed-projection",
            "fixed": True,
            "deterministic": True,
            "trainable": False,
            "seed": 17,
        },
        local_normalization_contract={
            "scope": "bank_local",
            "kind": "softmax",
        },
        local_formula_ref="arti/formula-fabric@2",
        local_refine_ref="arti/refine-policy@2",
        terminal_adapter_ref="arti/bank-terminal-adapter@1",
        terminal_abi_ref="arti/terminal-output-abi@1",
        terminal_abi_fingerprint=abi.fingerprint,
        score_contract="raw local path score remains separate from terminal score",
        gradient_contract=mechanisms.GradientContract.autograd(),
        execution_capabilities=("eager", "local-refine"),
    )


def test_tensor_schema_validates_concrete_and_shared_symbolic_dimensions() -> None:
    schema = tensor_schema("B", "N", 4, axes=("batch", "token", "feature"))
    symbols = schema.validate_tensor(torch.randn(2, 3, 4))

    assert symbols == {"B": 2, "N": 3}
    with pytest.raises(mechanisms.TensorSchemaError, match="symbolic dimension"):
        schema.validate_tensor(torch.randn(5, 3, 4), symbols=symbols)
    with pytest.raises(mechanisms.TensorSchemaError, match="shape"):
        schema.validate_tensor(torch.randn(2, 3, 5))


def test_tensor_schema_round_trip_and_tamper_rejection() -> None:
    source = tensor_schema("B", 17, 64, axes=("batch", "slot", "feature"))
    payload = source.to_dict()

    assert mechanisms.TensorSchema.from_dict(payload) == source
    tampered = deepcopy(payload)
    tampered["dimensions"][1] = 18
    with pytest.raises(mechanisms.TensorSchemaError, match="fingerprint"):
        mechanisms.TensorSchema.from_dict(tampered)


def test_preserves_shape_is_a_profile_not_a_core_requirement() -> None:
    relation = mechanisms.ShapeRelation.preserves_shape()
    source = tensor_schema("B", 8, axes=("batch", "feature"))
    relation.validate_schemas(source, source)

    changed = tensor_schema("B", 4, 2, axes=("batch", "token", "feature"))
    with pytest.raises(mechanisms.TensorSchemaError, match="identical logical shapes"):
        relation.validate_schemas(source, changed)

    mechanisms.ShapeRelation.maps_shape("[B,D] -> [B,N,D]").validate_schemas(
        source, changed
    )


def test_terminal_abi_validates_named_outputs_and_cross_field_symbols() -> None:
    abi = terminal_abi()
    outputs = {
        "value": torch.randn(2, 5, 32),
        "validity": torch.ones(2, 5, dtype=torch.bool),
        "score": torch.randn(2),
    }

    assert abi.validate_outputs(outputs) == {"B": 2, "T": 5}
    wrong = {**outputs, "validity": torch.ones(2, 4, dtype=torch.bool)}
    with pytest.raises(mechanisms.TerminalABIError, match="symbolic dimension"):
        abi.validate_outputs(wrong)


def test_terminal_abi_round_trip_and_config_identity() -> None:
    abi = terminal_abi()
    restored = mechanisms.TerminalOutputABI.from_dict(abi.to_dict())

    assert restored == abi
    assert restored.compatible_with(abi)
    changed = mechanisms.TerminalOutputABI(
        fields=abi.fields,
        factor_order=abi.factor_order,
        validity_contract=abi.validity_contract,
        packing_contract=abi.packing_contract,
        score_contract="a different terminal score contract",
        consumer_contract=abi.consumer_contract,
        gradient_contract=abi.gradient_contract,
    )
    assert not changed.compatible_with(abi)


def test_heterogeneous_bank_signatures_share_only_terminal_abi() -> None:
    abi = terminal_abi()
    bank_a = signature(
        name="test-bank-a",
        input_schema=tensor_schema("B", 17, 64, axes=("batch", "slot", "feature")),
        output_schema=tensor_schema("B", 5, 32, axes=("batch", "pulse", "feature")),
        abi=abi,
    )
    bank_b = signature(
        name="test-bank-b",
        input_schema=tensor_schema("B", 3, 4096, axes=("batch", "plane", "feature")),
        output_schema=tensor_schema(
            "B", 2, 8, 16, axes=("batch", "view", "token", "feature")
        ),
        abi=abi,
    )

    bank_a.validate_terminal_abi(abi)
    bank_b.validate_terminal_abi(abi)
    assert bank_a.input_schema.rank != bank_b.input_schema.rank or (
        bank_a.input_schema.dimensions != bank_b.input_schema.dimensions
    )
    assert bank_a.output_schema.dimensions != bank_b.output_schema.dimensions
    assert bank_a.fingerprint != bank_b.fingerprint
    assert bank_a.terminal_abi_fingerprint == bank_b.terminal_abi_fingerprint


def test_bank_signature_is_immutable_round_trippable_and_fail_closed() -> None:
    abi = terminal_abi()
    source = signature(
        name="test-bank",
        input_schema=tensor_schema("B", 7, 11, axes=("batch", "token", "feature")),
        output_schema=tensor_schema("B", 4, 13, axes=("batch", "pulse", "feature")),
        abi=abi,
    )
    restored = mechanisms.BankExecutionSignature.from_dict(source.to_dict())

    assert restored == source
    with pytest.raises(TypeError):
        source.query_contract["fixed"] = False

    bad_query = source.to_dict()
    bad_query["query_contract"]["trainable"] = True
    bad_query["fingerprint"] = "0" * 64
    with pytest.raises(mechanisms.TerminalABIError, match="non-trainable"):
        mechanisms.BankExecutionSignature.from_dict(bad_query)

    bad_normalization = source.to_dict()
    bad_normalization["local_normalization_contract"]["scope"] = "global"
    bad_normalization["fingerprint"] = "0" * 64
    with pytest.raises(mechanisms.TerminalABIError, match="Bank-local"):
        mechanisms.BankExecutionSignature.from_dict(bad_normalization)


def test_bank_signature_rejects_wrong_terminal_abi_instance() -> None:
    abi = terminal_abi()
    source = signature(
        name="test-bank",
        input_schema=tensor_schema("B", 4, axes=("batch", "feature")),
        output_schema=tensor_schema("B", 2, axes=("batch", "feature")),
        abi=abi,
    )
    changed = mechanisms.TerminalOutputABI(
        fields=abi.fields,
        factor_order=abi.factor_order,
        validity_contract=abi.validity_contract,
        packing_contract="different explicit packing",
        score_contract=abi.score_contract,
        consumer_contract=abi.consumer_contract,
        gradient_contract=abi.gradient_contract,
    )

    with pytest.raises(mechanisms.TerminalABIError, match="does not match"):
        source.validate_terminal_abi(changed)


def test_federal_contract_component_identities_are_versioned() -> None:
    abi = terminal_abi()
    source = signature(
        name="test-bank",
        input_schema=tensor_schema("B", 4, axes=("batch", "feature")),
        output_schema=tensor_schema("B", 2, axes=("batch", "feature")),
        abi=abi,
    )

    assert arti.component_ref(source.input_schema) == "arti/tensor-schema@1"
    assert arti.component_ref(source.shape_relation) == "arti/shape-relation@1"
    assert arti.component_ref(source.gradient_contract) == "arti/gradient-contract@1"
    assert arti.component_ref(abi) == "arti/terminal-output-abi@1"
    assert arti.component_ref(source) == "arti/bank-execution-signature@1"
    assert component_spec(source).lifecycle == "stable"
    assert component_spec(source).dependencies == (
        "arti/gradient-contract@1",
        "arti/shape-relation@1",
        "arti/tensor-schema@1",
        "arti/terminal-output-abi@1",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_terminal_abi_validates_cuda_without_forcing_internal_shape() -> None:
    abi = terminal_abi()
    outputs = {
        "value": torch.randn(3, 7, 32, device="cuda"),
        "validity": torch.ones(3, 7, dtype=torch.bool, device="cuda"),
        "score": torch.randn(3, device="cuda"),
    }

    assert abi.validate_outputs(outputs) == {"B": 3, "T": 7}
