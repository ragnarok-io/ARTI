from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

import arti
from arti import mechanisms
from arti.component_registry import (
    ComponentCompatibilityError,
    component_graph_fingerprint,
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


def make_query(*, query_dim: int = 3) -> mechanisms.LinearBankQuery:
    return mechanisms.LinearBankQuery(
        schema("B", 4, axes=("batch", "feature")),
        schema("B", query_dim, axes=("batch", "query")),
        input_dim=4,
        query_dim=query_dim,
        retrieval_contract={"kind": "bank-local-dot-product", "top_k": 4},
        normalization_contract={"scope": "bank_local", "kind": "softmax"},
    )


class CustomSquaredBankQuery(mechanisms.BankQuery):
    def __init__(self) -> None:
        value_schema = schema("B", 4, axes=("batch", "feature"))
        super().__init__(
            input_schema=value_schema,
            output_schema=value_schema,
            retrieval_contract={"kind": "elementwise-squared-query"},
            normalization_contract={"scope": "bank_local", "kind": "none"},
        )
        self.scale = torch.nn.Parameter(torch.ones(4))

    def forward(self, value: torch.Tensor) -> mechanisms.BankQueryResult:
        self.input_schema.validate_tensor(value, name="custom_query.input")
        result = value.square() * self.scale
        self.output_schema.validate_tensor(result, name="custom_query.output")
        return mechanisms.BankQueryResult(result)


arti.register_component(
    "arti/test-custom-squared-bank-query@1",
    component_type=CustomSquaredBankQuery,
    lifecycle="alpha",
    constructible=False,
    config_builder=lambda component: component.contract_config(),
    dependency_builder=lambda _component: (
        "arti/gradient-contract@1",
        "arti/tensor-schema@1",
    ),
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
        packing_contract="named tensors",
        score_contract="one terminal score per row",
        consumer_contract="hard one winner",
        gradient_contract=mechanisms.GradientContract.autograd(),
    )


def bank_signature_v2(
    sealed: mechanisms.SealedBankQuery,
) -> mechanisms.BankExecutionSignatureV2:
    abi = terminal_abi()
    return mechanisms.BankExecutionSignatureV2(
        program_ref="arti/test-bank-program@1",
        program_api_identity="tests.TestBankProgram",
        program_config_fingerprint="1" * 64,
        program_state_schema_fingerprint="2" * 64,
        input_schema=sealed.signature.input_schema,
        output_schema=schema("B", 4, axes=("batch", "feature")),
        shape_relation=mechanisms.ShapeRelation.preserves_shape(),
        query_signature=sealed.signature,
        local_normalization_contract={"scope": "bank_local", "kind": "softmax"},
        terminal_adapter_ref="arti/bank-terminal-adapter@1",
        terminal_abi_ref="arti/terminal-output-abi@1",
        terminal_abi_fingerprint=abi.fingerprint,
        score_contract="Bank-local route scores remain separate from terminal score",
        gradient_contract=mechanisms.GradientContract.autograd(),
        execution_capabilities=("eager", "latest-state-requery"),
    )


def test_query_trains_before_seal_and_seal_freezes_only_parameters() -> None:
    query = make_query()
    optimizer = torch.optim.AdamW(query.parameters(), lr=0.1)
    x = torch.randn(8, 4)
    target = torch.randn(8, 3)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = (query(x).value - target).square().mean()
        loss.backward()
        optimizer.step()

    sealed = mechanisms.seal_bank_query(query, capabilities=("local-retrieval",))
    value = torch.randn(5, 4, requires_grad=True)
    output = sealed(value).value
    output.sum().backward()

    assert all(not parameter.requires_grad for parameter in sealed.parameters())
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert sealed.training is False and sealed.query.training is False
    assert sealed.signature.runtime_mode == "sealed"
    assert sealed.signature.state_owner == "bank"
    assert sealed.signature.normalization_contract["scope"] == "bank_local"


def test_registered_custom_query_seals_saves_and_reloads(tmp_path) -> None:
    query = CustomSquaredBankQuery()
    with torch.no_grad():
        query.scale.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
    sealed = mechanisms.seal_bank_query(query)
    value = torch.randn(3, 4)
    expected = sealed(value).value

    saved = mechanisms.save_bank_query(sealed, tmp_path / "custom-query")
    restored = mechanisms.load_bank_query(saved.weights_path, CustomSquaredBankQuery())

    torch.testing.assert_close(restored(value).value, expected)
    assert sealed.signature.query_ref == "arti/test-custom-squared-bank-query@1"


def test_seal_clears_training_gradients_and_rejects_runtime_unfreeze() -> None:
    query = make_query()
    query(torch.randn(2, 4)).value.sum().backward()
    assert any(parameter.grad is not None for parameter in query.parameters())

    sealed = mechanisms.seal_bank_query(query)

    assert all(parameter.grad is None for parameter in sealed.parameters())
    with pytest.raises(mechanisms.BankQueryError, match="cannot be unfrozen"):
        sealed.requires_grad_(True)

    next(iter(sealed.parameters())).requires_grad_(True)
    with pytest.raises(mechanisms.BankQueryError, match="must remain frozen"):
        sealed(torch.randn(2, 4))


def test_sealed_query_ignores_parent_train_mode() -> None:
    sealed = mechanisms.seal_bank_query(make_query())

    sealed.train()

    assert sealed.training is False
    assert sealed.query.training is False


def test_sealed_query_rejects_dtype_conversion_without_resealing() -> None:
    sealed = mechanisms.seal_bank_query(make_query())

    with pytest.raises(mechanisms.BankQueryError, match="tensor state"):
        sealed.half()


def test_query_signature_round_trip_and_tamper_rejection() -> None:
    signature = mechanisms.seal_bank_query(make_query()).signature
    restored = mechanisms.QueryExecutionSignature.from_dict(signature.to_dict())

    assert restored == signature
    tampered = deepcopy(signature.to_dict())
    tampered["state_fingerprint"] = "0" * 64
    with pytest.raises(mechanisms.BankQueryError, match="fingerprint"):
        mechanisms.QueryExecutionSignature.from_dict(tampered)


def test_query_signature_binds_api_state_roles_and_behavior_contracts() -> None:
    sealed = mechanisms.seal_bank_query(make_query())
    signature = sealed.signature

    with pytest.raises(mechanisms.BankQueryError, match="API implementation"):
        replace(signature, api_identity="example.WrongQuery").validate_query(
            sealed.query
        )
    with pytest.raises(mechanisms.BankQueryError, match="tensor roles"):
        replace(signature, state_schema_fingerprint="0" * 64).validate_query(
            sealed.query
        )
    with pytest.raises(mechanisms.BankQueryError, match="retrieval contract"):
        replace(signature, retrieval_contract={"kind": "wrong"}).validate_query(
            sealed.query
        )
    with pytest.raises(mechanisms.BankQueryError, match="normalization contract"):
        replace(
            signature,
            normalization_contract={"scope": "bank_local", "kind": "wrong"},
        ).validate_query(sealed.query)


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_query_signature_rejects_non_integer_schema_version(schema_version) -> None:
    value = mechanisms.seal_bank_query(make_query()).signature.to_dict()
    value["schema_version"] = schema_version
    value["fingerprint"] = "0" * 64

    with pytest.raises(mechanisms.BankQueryError, match="version"):
        mechanisms.QueryExecutionSignature.from_dict(value)


def test_bank_query_asset_save_load_and_fresh_parity(tmp_path) -> None:
    source = make_query()
    with torch.no_grad():
        source.projection.weight.copy_(
            torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
        )
        source.projection.bias.copy_(torch.tensor([0.25, -0.5, 0.75]))
    sealed = mechanisms.seal_bank_query(source)
    value = torch.randn(7, 4)
    expected = sealed(value).value

    result = mechanisms.save_bank_query(sealed, tmp_path / "visual-query")
    asset = mechanisms.inspect_bank_query(result.weights_path)
    restored = mechanisms.load_bank_query(result.weights_path, make_query())
    actual = restored(value).value

    torch.testing.assert_close(actual, expected)
    assert asset.signature == sealed.signature
    assert set(asset.state_dict) == {"projection.bias", "projection.weight"}
    assert result.weights_path.name == "visual-query.arti.st"
    assert all(not parameter.requires_grad for parameter in restored.parameters())

    first = asset.state_dict
    first["projection.weight"].zero_()
    assert torch.count_nonzero(asset.state_dict["projection.weight"]) > 0


def test_bank_query_asset_rejects_wrong_query_config(tmp_path) -> None:
    sealed = mechanisms.seal_bank_query(make_query())
    result = mechanisms.save_bank_query(sealed, tmp_path / "query")

    with pytest.raises(mechanisms.BankQueryError, match="incompatible"):
        mechanisms.load_bank_query(result.weights_path, make_query(query_dim=5))


def test_generic_arti_save_rejects_a_modified_sealed_query(tmp_path) -> None:
    sealed = mechanisms.seal_bank_query(make_query())
    with torch.no_grad():
        sealed.query.projection.weight.add_(1.0)

    with pytest.raises(mechanisms.BankQueryError, match="modified after mounting"):
        arti.save(sealed, tmp_path / "modified-query.arti.st")


def test_bank_execution_signature_v2_binds_exact_query_and_keeps_v1_identity() -> None:
    sealed = mechanisms.seal_bank_query(make_query())
    signature = bank_signature_v2(sealed)
    restored = mechanisms.BankExecutionSignatureV2.from_dict(signature.to_dict())

    assert restored == signature
    assert arti.component_ref(signature) == "arti/bank-execution-signature@2"
    assert arti.component_ref(restored.query_signature) == "arti/query-execution-signature@1"
    assert arti.component_ref(sealed) == "arti/sealed-bank-query@1"
    assert arti.component_ref(sealed.query) == "arti/linear-bank-query@1"
    assert component_spec(signature).lifecycle == "stable"


def test_component_provenance_closes_over_the_owned_query_identity() -> None:
    sealed = mechanisms.seal_bank_query(make_query())
    provenance = component_provenance(sealed)

    assert validate_component_provenance(provenance) == provenance
    tampered = deepcopy(provenance)
    tampered["components"][0]["dependencies"] = [
        "arti/query-execution-signature@1"
    ]
    tampered["fingerprint"] = component_graph_fingerprint(tampered["components"])
    with pytest.raises(ComponentCompatibilityError, match="dependency closure"):
        validate_component_provenance(tampered)


def test_bank_execution_signature_v2_rejects_query_input_schema_mismatch() -> None:
    sealed = mechanisms.seal_bank_query(make_query())
    source = bank_signature_v2(sealed).to_dict()
    source["input_schema"] = schema(
        "B", 5, axes=("batch", "feature")
    ).to_dict()
    source["output_schema"] = schema(
        "B", 5, axes=("batch", "feature")
    ).to_dict()
    source["fingerprint"] = "0" * 64

    with pytest.raises(mechanisms.TerminalABIError, match="Query input schema"):
        mechanisms.BankExecutionSignatureV2.from_dict(source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_sealed_bank_query_preserves_cuda_and_input_gradient() -> None:
    sealed = mechanisms.seal_bank_query(make_query()).cuda()
    value = torch.randn(6, 4, device="cuda", requires_grad=True)

    output = sealed(value).value
    output.square().mean().backward()

    assert output.is_cuda
    assert value.grad is not None and value.grad.is_cuda
    assert all(parameter.grad is None for parameter in sealed.parameters())
