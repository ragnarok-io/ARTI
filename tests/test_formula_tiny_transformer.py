from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from arti import mechanisms
from benchmarks.formula_fabric_tiny_transformer import (
    TinyTransformerConfig,
    bind_bank_values,
    build_tiny_transformer_program,
    initialize_bank_values,
    reference_forward,
)


def _inputs(
    config: TinyTransformerConfig,
    *,
    batch: int,
    sequence: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(9127 + batch * 31 + sequence)
    return {
        "token_ids": torch.randint(
            0,
            config.vocab_size,
            (batch, sequence),
            generator=generator,
            dtype=torch.int64,
        ).to(device),
        "position_ids": torch.arange(sequence, dtype=torch.int64)
        .unsqueeze(0)
        .expand(batch, -1)
        .to(device),
        "input_residual": (torch.randn(batch, sequence, config.dim, generator=generator) * 0.01)
        .to(device)
        .requires_grad_(),
        "causal_mask": torch.tril(torch.ones(sequence, sequence, dtype=torch.bool))
        .unsqueeze(0)
        .expand(batch, -1, -1)
        .to(device),
    }


@pytest.mark.parametrize(("batch", "sequence"), ((1, 1), (2, 4), (3, 6)))
def test_formula_program_matches_tiny_causal_transformer_and_backpropagates(
    batch: int,
    sequence: int,
) -> None:
    config = TinyTransformerConfig()
    program = build_tiny_transformer_program(config)
    banks = initialize_bank_values(
        program,
        config,
        seed=7401,
        device=torch.device("cpu"),
        requires_grad=True,
    )
    inputs = _inputs(
        config,
        batch=batch,
        sequence=sequence,
        device=torch.device("cpu"),
    )

    actual = mechanisms.FormulaFabricV2(program)(
        inputs=inputs,
        banks=bind_bank_values(program, banks),
        return_trace=True,
    )
    expected = reference_forward(
        inputs["token_ids"],
        inputs["position_ids"],
        inputs["input_residual"],
        inputs["causal_mask"],
        banks,
    )
    for value, reference in zip(actual.values, expected, strict=True):
        torch.testing.assert_close(value, reference, atol=1e-6, rtol=1e-5)

    targets = (inputs["token_ids"] + 3) % config.vocab_size
    loss = torch.nn.functional.cross_entropy(
        actual.values[0].reshape(-1, config.vocab_size),
        targets.reshape(-1),
    )
    reference_loss = torch.nn.functional.cross_entropy(
        expected[0].reshape(-1, config.vocab_size),
        targets.reshape(-1),
    )
    torch.testing.assert_close(loss, reference_loss, atol=1e-6, rtol=1e-5)
    loss.backward()

    input_gradient = inputs["input_residual"].grad
    assert input_gradient is not None and torch.isfinite(input_gradient).all()
    assert all(
        value.grad is not None and torch.isfinite(value.grad).all()
        for value in banks.values()
    )
    assert actual.trace is not None
    assert len(actual.trace.instruction_ids) == 40


@pytest.mark.parametrize("gated", (False, True))
def test_tiny_transformer_program_and_bank_round_trip(tmp_path, gated) -> None:
    config = TinyTransformerConfig(gated_mlp=gated)
    program = build_tiny_transformer_program(config)
    banks = initialize_bank_values(
        program,
        config,
        seed=7401,
        device=torch.device("cpu"),
    )
    inputs = _inputs(
        config,
        batch=2,
        sequence=5,
        device=torch.device("cpu"),
    )
    expected = mechanisms.FormulaFabricV2(program)(
        inputs=inputs,
        banks=bind_bank_values(program, banks),
    ).values

    program_path = tmp_path / "program.json"
    bank_path = tmp_path / "bank.safetensors"
    program_path.write_text(
        json.dumps(program.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    save_file({name: value.contiguous() for name, value in banks.items()}, bank_path)
    restored_program = mechanisms.FormulaProgram.from_dict(
        json.loads(program_path.read_text(encoding="utf-8"))
    )
    restored_banks = load_file(bank_path, device="cpu")
    actual = mechanisms.FormulaFabricV2(restored_program)(
        inputs=inputs,
        banks=bind_bank_values(restored_program, restored_banks),
    ).values

    assert restored_program.fingerprint == program.fingerprint
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, atol=0.0, rtol=0.0)


def test_tiny_transformer_program_exposes_composed_atom_basis() -> None:
    program = build_tiny_transformer_program(TinyTransformerConfig())
    assert {instruction.atom_ref for instruction in program.instructions} == {
        "arti/formula-atom-add@1",
        "arti/formula-atom-broadcast@1",
        "arti/formula-atom-contract@1",
        "arti/formula-atom-lookup@1",
        "arti/formula-atom-masked-softmax@1",
        "arti/formula-atom-reduce@1",
        "arti/formula-atom-reshape@1",
        "arti/formula-atom-scalar-map@1",
        "arti/formula-atom-scale@1",
    }
    assert len(program.instructions) == 40


@pytest.mark.parametrize("backend", (None, "eager", "aot_eager"))
def test_bank_gated_transformer_all_input_and_bank_gradients_match_reference(backend):
    from arti._formula_candidate_batch import _checked_plan
    from arti._formula_grouped_training import execute_grouped_training, grouped_formula_training

    config = TinyTransformerConfig(gated_mlp=True)
    program = build_tiny_transformer_program(config)
    fabric = mechanisms.FormulaFabricV2(program)
    banks = initialize_bank_values(program, config, seed=7401, device=torch.device("cpu"), requires_grad=True)
    banks["mlp.gate_beta"] = torch.linspace(.7, 1.4, config.hidden_dim, requires_grad=True)
    inputs = [_inputs(config, batch=1, sequence=3, device=torch.device("cpu")) for _ in range(2)]
    inputs[1]["input_residual"] = (inputs[1]["input_residual"].detach() + .07).requires_grad_()
    # Fully masked rows must remain finite in both data and parameter backward.
    for values in inputs:
        values["causal_mask"][:, 0] = False
    prepared = tuple(fabric.bind_tensors(inputs=value, banks=bind_bank_values(program, banks)) for value in inputs)
    leaves = (*[value["input_residual"] for value in inputs], *banks.values())
    expected = [reference_forward(**value, banks=banks) for value in inputs]
    expected_loss = sum(v[0].square().mean() + v[1].square().mean() for v in expected)
    reference_gradients = torch.autograd.grad(expected_loss, leaves)
    if backend is None:
        actual = [fabric.execution_plan()(value) for value in prepared]
        gradients = torch.autograd.grad(sum(v[0].square().mean() + v[1].square().mean() for v in actual), leaves)
    else:
        with grouped_formula_training(backend=backend):
            actual, valid = execute_grouped_training(_checked_plan(program), prepared)
            assert valid.all()
            gradients = torch.autograd.grad(sum(v[0].square().mean() + v[1].square().mean() for v in actual), leaves)
    for left, right in zip(actual, expected, strict=True):
        for a, e in zip(left, right, strict=True):
            torch.testing.assert_close(a, e, rtol=1e-5, atol=1e-6)
    for a, e in zip(gradients, reference_gradients, strict=True):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, e, rtol=1e-5, atol=1e-6)
    beta_position = 2 + tuple(banks).index("mlp.gate_beta")
    assert gradients[beta_position].abs().sum() > 0
    refs = {instruction.atom_ref for instruction in program.instructions}
    assert "arti/formula-atom-reduce@2" in refs and "arti/formula-atom-scalar-map@2" in refs


def test_lookup_checked_rows_keep_range_validity_and_plain_errors():
    from dataclasses import replace
    from arti._formula_candidate_batch import _checked_plan
    from arti._formula_grouped_training import execute_grouped_training, grouped_formula_training

    table = mechanisms.BankBinding("table", "arti/lookup-test@1", "table",
        mechanisms.TensorType(("V", "D"), (3, 2), dtype="float32"))
    indices = mechanisms.InputBinding("ids", mechanisms.TensorType(("B", "N"), (1, 2), dtype="int64"))
    program = mechanisms.FormulaProgram.build(outputs=(mechanisms.lookup(table, indices, table_axis="V"),))
    weight = torch.arange(6.).reshape(3, 2).requires_grad_()
    fabric = mechanisms.FormulaFabricV2(program)
    def prepare(ids):
        return fabric.bind_tensors(inputs={"ids": torch.tensor([ids])}, banks={"table": table.bind(weight)})
    valid, invalid = prepare([1, 1]), prepare([-1, 3])
    plan = _checked_plan(program)
    assert plan is not None
    with pytest.raises(mechanisms.FormulaBindingError, match="outside the table") as error:
        plan(invalid)
    assert error.value.code == "FF2_INDEX_RANGE"
    with pytest.raises(mechanisms.FormulaBindingError, match="outside the table") as error:
        fabric(inputs={"ids": torch.tensor([[0, 9]])}, banks={"table": table.bind(weight)})
    assert error.value.code == "FF2_INDEX_RANGE"
    with grouped_formula_training(backend="eager"):
        rows, validity = execute_grouped_training(plan, (valid, invalid))
        valid_gradient, = torch.autograd.grad(rows[0][0].sum(), (weight,))
    assert validity.tolist() == [True, False]
    assert all(torch.isfinite(value).all() for row in rows for value in row)
    output = plan(valid)[0]
    gradient, = torch.autograd.grad(output.sum(), (weight,))
    torch.testing.assert_close(gradient, torch.tensor([[0., 0.], [2., 2.], [0., 0.]]))
    torch.testing.assert_close(valid_gradient, gradient)
    from arti._formula_candidate_batch import execute_many

    node = mechanisms.FormulaProgramTensorCandidateV4(mechanisms.FormulaProgramCandidateV3(
        "lookup", program, input_slots={"ids": "ids"}, output_slots={program.outputs[0]: "answer"}, operands={"table": weight}))
    query = mechanisms.FormulaProgramQueryV7(slot_ids=("ids", "answer"), candidates=(node,),
        terminal_slots={"y": "answer"}, entry_candidates=("lookup",), continuations={}, max_steps=1)
    requests = tuple((node, query._arena({"ids": torch.tensor([ids])}, bank_state=None)) for ids in ([1, 1], [-1, 3]))
    with torch.no_grad():
        accepted, rejected = execute_many(requests, reject_nonfinite=True)
        assert accepted is not None and rejected is None
        with pytest.raises(mechanisms.FormulaBindingError):
            execute_many(requests)
    tight = mechanisms.FormulaFabricV2(replace(program, limits=mechanisms.FormulaLimits(max_working_bytes=120)))
    with pytest.raises(mechanisms.FormulaBindingError, match="working"):
        tight.bind_tensors(inputs={"ids": torch.tensor([[1, 1]])}, banks={"table": table.bind(weight)})


def test_gated_transformer_executes_through_actual_heterogeneous_pools_on_cpu():
    from arti._formula_device_dispatch import FormulaDeviceDispatchLayout, FormulaDeviceNumericalDispatch, formula_device_dispatch_groups
    from arti._formula_device_frames import FormulaDeviceFrameKernel
    from arti._formula_device_pools import FormulaDevicePoolLayout

    config = TinyTransformerConfig(gated_mlp=True)
    program = build_tiny_transformer_program(config)
    banks = initialize_bank_values(program, config, seed=7401, device=torch.device("cpu"))
    inputs = _inputs(config, batch=1, sequence=3, device=torch.device("cpu"))
    expected = reference_forward(**inputs, banks=banks)
    output_slots = dict(zip(program.outputs, ("logits", "attention"), strict=True))
    node = mechanisms.FormulaProgramTensorCandidateV4(mechanisms.FormulaProgramCandidateV3(
        "block", program, input_slots={name: name for name in inputs}, output_slots=output_slots, operands=banks))
    query = mechanisms.FormulaProgramQueryV7(slot_ids=(*inputs, "logits", "attention"), candidates=(node,),
        terminal_slots={"y": "logits"}, entry_candidates=("block",), continuations={}, max_steps=1)
    kernel = FormulaDeviceFrameKernel.from_query(query)
    samples = {(tuple(v.shape), v.dtype): v for v in (*inputs.values(), *expected)}
    layout = FormulaDevicePoolLayout.from_samples(tuple(samples.values()), 4)
    bank_layout = FormulaDevicePoolLayout.from_samples((torch.zeros(1),), 1)
    pools, bank_pools = layout.allocate("cpu"), bank_layout.allocate("cpu")
    handles, positions = [], {}
    for value in inputs.values():
        bucket = layout.index(value)
        position = positions.get(bucket, 0)
        pools[bucket][position] = value.detach()
        handles.append(layout.offsets[bucket] + position)
        positions[bucket] = position + 1
    state = kernel.initial_state(1, torch.tensor([[*handles, -1, -1]]))
    dispatch = FormulaDeviceNumericalDispatch(query, kernel)
    dispatch.prepare_typed_pools_(layout, bank_layout)
    packet = FormulaDeviceDispatchLayout(formula_device_dispatch_groups(query)[0])(torch.tensor([[0]]))
    result = dispatch(state, packet, pools, bank_pools)
    assert result.numeric_valid.all() and not result.overflow
    for head, expected_value in enumerate(expected):
        bucket = layout.index(expected_value)
        assert result.output_present[bucket][0, head]
        torch.testing.assert_close(result.output_values[bucket][0, head], expected_value, rtol=1e-5, atol=1e-6)
    token_bucket = layout.index(inputs["token_ids"])
    pools[token_bucket][2] = torch.tensor([[-1, config.vocab_size, 0]])
    bad_handles = [layout.offsets[token_bucket] + 2, *handles[1:]]
    paired_state = kernel.initial_state(2, torch.tensor([[*handles, -1, -1], [*bad_handles, -1, -1]]))
    paired_packet = FormulaDeviceDispatchLayout(formula_device_dispatch_groups(query)[0])(torch.tensor([[0], [0]]))
    paired = dispatch(paired_state, paired_packet, pools, bank_pools)
    assert paired.numeric_valid.tolist() == [True, False]
    for head, expected_value in enumerate(expected):
        bucket = layout.index(expected_value)
        assert paired.output_present[bucket][0, head] and not paired.output_present[bucket][1, head]
        torch.testing.assert_close(paired.output_values[bucket][0, head], expected_value, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tiny_transformer_formula_program_matches_cuda() -> None:
    config = TinyTransformerConfig()
    program = build_tiny_transformer_program(config)
    cpu_banks = initialize_bank_values(
        program,
        config,
        seed=7401,
        device=torch.device("cpu"),
    )
    cuda_banks = {name: value.cuda() for name, value in cpu_banks.items()}
    cpu_inputs = _inputs(
        config,
        batch=2,
        sequence=6,
        device=torch.device("cpu"),
    )
    cuda_inputs = {name: value.cuda() for name, value in cpu_inputs.items()}

    expected = mechanisms.FormulaFabricV2(program)(
        inputs=cpu_inputs,
        banks=bind_bank_values(program, cpu_banks),
    ).values
    actual = mechanisms.FormulaFabricV2(program).cuda()(
        inputs=cuda_inputs,
        banks=bind_bank_values(program, cuda_banks),
    ).values

    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value.cpu(), reference, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tiny_transformer_execution_plan_compiles_fullgraph_on_cuda() -> None:
    config = TinyTransformerConfig()
    program = build_tiny_transformer_program(config)
    banks = initialize_bank_values(
        program,
        config,
        seed=7401,
        device=torch.device("cuda"),
    )
    inputs = _inputs(
        config,
        batch=2,
        sequence=6,
        device=torch.device("cuda"),
    )
    fabric = mechanisms.FormulaFabricV2(program).cuda()
    prepared = fabric.bind_tensors(
        inputs=inputs,
        banks=bind_bank_values(program, banks),
    )
    plan = fabric.execution_plan().cuda()

    expected = plan(prepared)
    actual = torch.compile(plan, fullgraph=True)(prepared)

    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, atol=1e-5, rtol=1e-5)
