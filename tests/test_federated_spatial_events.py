import pytest
import torch

from benchmarks._federated_spatial_events import SpatialEventSpec, make_spatial_events


def test_labels_are_fully_determined_by_presented_pixels_and_question():
    spec = SpatialEventSpec(height=5, width=7, markers=5, reuse_questions=8)
    batch = make_spatial_events(spec, split="train", seed=19, count=32)
    positions = batch.x0.argmax(1).reshape(32, 2, 5)
    for panel in range(2):
        assert all(len(set(row.tolist())) == 5 for row in positions[:, panel])
    torch.testing.assert_close(batch.x0.sum(1), torch.ones(32, 10))
    questions = torch.cat((batch.question[:, None], batch.readonly_questions), dim=1)
    answers = torch.cat((batch.answer[:, None], batch.readonly_answers), dim=1)
    for event in range(32):
        seen = set()
        for question, answer in zip(questions[event], answers[event], strict=True):
            i, j = int(question[:5].argmax()), int(question[5:].argmax())
            assert i != j and (i, j) not in seen
            seen.add((i, j))
            angles = []
            for coordinate, period in ((lambda p: p // 7, 5), (lambda p: p % 7, 7)):
                pi = coordinate(positions[event, 1, i]) - coordinate(positions[event, 0, i])
                pj = coordinate(positions[event, 1, j]) - coordinate(positions[event, 0, j])
                angles.append((pi - pj).double() * (2 * torch.pi / period))
            expected = torch.stack((angles[0].sin(), angles[0].cos(), angles[1].sin(), angles[1].cos())).float()
            torch.testing.assert_close(answer, expected)


def test_readonly_panel_never_enters_writing_inputs_and_count_does_not_change_event():
    small = make_spatial_events(SpatialEventSpec(reuse_questions=1), split="train", seed=20, count=4)
    large = make_spatial_events(SpatialEventSpec(reuse_questions=9), split="train", seed=20, count=4)
    assert set(large.writing_inputs()) == {"x0", "question"}
    for name in ("x0", "question", "answer"):
        torch.testing.assert_close(getattr(small, name), getattr(large, name), rtol=0, atol=0)
    torch.testing.assert_close(small.readonly_answers, large.readonly_answers[:, :1], rtol=0, atol=0)
    assert large.writing_inputs(2)["x0"].shape == (1, 64, 8)


def test_replay_split_namespaces_and_global_rng_isolation():
    rng = torch.random.get_rng_state().clone()
    rows = [make_spatial_events(split=split, seed=31, batch_index=12, count=8)
            for split in ("train", "validation", "test", "cost")]
    replay = make_spatial_events(split="train", seed=31, batch_index=12, count=8)
    torch.testing.assert_close(rng, torch.random.get_rng_state(), rtol=0, atol=0)
    torch.testing.assert_close(rows[0].x0, replay.x0, rtol=0, atol=0)
    torch.testing.assert_close(rows[0].readonly_answers, replay.readonly_answers, rtol=0, atol=0)
    assert all(not torch.equal(rows[i].x0, rows[j].x0) for i in range(4) for j in range(i))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_batch_device_transfer_preserves_event_and_final_targets():
    cpu = make_spatial_events(split="cost", seed=17, count=16)
    with torch.device("cuda"):
        gpu = make_spatial_events(split="cost", seed=17, count=16, device="cuda")
    for name in ("x0", "question", "answer", "readonly_questions", "readonly_answers"):
        assert getattr(gpu, name).is_cuda
        torch.testing.assert_close(getattr(cpu, name), getattr(gpu, name).cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [{"markers": 65}, {"markers": 1}, {"height": 1}, {"reuse_questions": 12}])
def test_invalid_task_sizes(kwargs):
    with pytest.raises(ValueError):
        SpatialEventSpec(**kwargs)
