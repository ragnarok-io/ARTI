import torch
import pytest

import arti
from arti import mechanisms


def _view(values: list[list[float]]) -> mechanisms.TensorView:
    return mechanisms.TensorView.from_tensor(
        torch.tensor(values, dtype=torch.float32).unsqueeze(-1),
        axis_names=("batch", "token", "feature"),
        axis_roles=("batch", "sequence", "feature"),
    )


def _spec(resource_id: str, lifetime: mechanisms.ResourceLifetime) -> mechanisms.TensorResourceSpec:
    return mechanisms.TensorResourceSpec(
        resource_id,
        mechanisms.TensorViewPattern(
            min_rank=3,
            max_rank=3,
            allowed_axis_roles=("batch", "sequence", "feature"),
        ),
        lifetime=lifetime,
        axis_capacity=(("token", 8),),
    )


def test_portable_program_graph_artifact_restores_graph_and_mutable_resource_state(tmp_path) -> None:
    source = mechanisms.TensorResource(_spec("source", mechanisms.ResourceLifetime.CALL), _view([[2, 7]]))
    memory = mechanisms.TensorResource(
        _spec("memory", mechanisms.ResourceLifetime.PERSISTENT), _view([[0, 0]])
    )
    transfer = mechanisms.LearnableAffineTransfer(gain=2.0, bias=1.0)
    graph = mechanisms.ProgramGraph(
        (source, memory),
        (
            mechanisms.Connection(
                "write", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("memory"), transfer=transfer
            ),
        ),
        programs={"write_memory": ("write",)},
    )
    graph.execute_program("write_memory")

    saved = arti.save_program_graph(graph, tmp_path / "memory-program")
    restored = arti.load_program_graph(saved.tensors_path)

    assert restored.contract_fingerprint == graph.contract_fingerprint
    torch.testing.assert_close(
        restored.resource("memory").resolve().view.value,
        _view([[5, 15]]).value,
    )
    restored_transfer = restored.connection("write").transfer
    assert isinstance(restored_transfer, mechanisms.LearnableAffineTransfer)
    assert restored_transfer.learnable
    torch.testing.assert_close(restored_transfer.gain, torch.tensor(2.0))
    torch.testing.assert_close(restored_transfer.bias, torch.tensor(1.0))


def test_program_graph_artifact_rejects_runtime_callable_connections(tmp_path) -> None:
    resource = mechanisms.TensorResource(_spec("source", mechanisms.ResourceLifetime.CALL), _view([[1, 2]]))
    graph = mechanisms.ProgramGraph(
        (resource,),
        (
            mechanisms.Connection(
                "runtime", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("source"), transfer=lambda value: value
            ),
        ),
    )

    with pytest.raises(mechanisms.ResourceGraphError, match="arbitrary Python callables"):
        mechanisms.save_program_graph(graph, tmp_path / "runtime")


def test_program_graph_rejects_connection_ids_reserved_by_torch_modules() -> None:
    with pytest.raises(mechanisms.ResourceGraphError, match="reserved nn.Module attribute"):
        mechanisms.Connection("double", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("source"))


def test_portable_program_graph_artifact_restores_formula_transfer(tmp_path) -> None:
    value_type = mechanisms.TensorType(
        ("batch", "token", "feature"), sizes=("B", "S", 1)
    )
    source_binding = mechanisms.InputBinding("source", value_type)
    program = mechanisms.FormulaProgram.build(
        outputs=(mechanisms.add(source_binding, source_binding),)
    )
    transfer = mechanisms.FormulaTensorViewTransfer(
        mechanisms.FormulaFabricV2(program), source_input="source"
    )
    source = mechanisms.TensorResource(_spec("source", mechanisms.ResourceLifetime.CALL), _view([[2, 7]]))
    target = mechanisms.TensorResource(_spec("target", mechanisms.ResourceLifetime.STATE), _view([[0, 0]]))
    graph = mechanisms.ProgramGraph(
        (source, target),
        (
            mechanisms.Connection(
                "double_value", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("target"), transfer=transfer
            ),
        ),
    )

    restored = arti.load_program_graph(arti.save_program_graph(graph, tmp_path / "formula").tensors_path)
    restored.execute(("double_value",))

    torch.testing.assert_close(
        restored.resource("target").resolve().view.value,
        _view([[4, 14]]).value,
    )
