import pytest
import torch

from benchmarks._federated_spatial_events import SpatialEventSpec
from benchmarks._federated_spatial_episodes import SpatialEpisodeSpec, make_calibrated_spatial_episodes


def visible_positions(event):
    count, size, _ = event.x0.shape
    markers = event.spec.markers
    encoded = event.x0.reshape(count, size, 2, markers)
    calibration = encoded[:, :markers, 0]
    canonical = torch.einsum("espc,ecm->espm", encoded, torch.linalg.inv(calibration))
    canonical[:, :markers] = 0
    return canonical.argmax(1), calibration


@pytest.mark.parametrize("height,width,markers", [(8, 8, 4), (5, 7, 5)])
def test_current_visible_scene_suffices_for_all_final_answers(height, width, markers):
    spec = SpatialEpisodeSpec(SpatialEventSpec(height, width, markers, reuse_questions=6), length=3)
    episode = make_calibrated_spatial_episodes(spec, split="train", family_seed=11, scene_seed=23, count=4)
    previous = None
    for event in episode.events:
        positions, calibration = visible_positions(event)
        assert event.x0.shape == (4, height * width, 2 * markers)
        assert positions.min() >= markers
        assert all(len(set(panel.tolist())) == markers for row in positions for panel in row)
        torch.testing.assert_close(calibration @ calibration.transpose(-1, -2),
                                   torch.eye(markers).expand(4, -1, -1), rtol=1e-5, atol=1e-6)
        if previous is not None:
            torch.testing.assert_close(calibration, previous, rtol=0, atol=0)
        previous = calibration
        questions = torch.cat((event.question[:, None], event.readonly_questions), 1)
        answers = torch.cat((event.answer[:, None], event.readonly_answers), 1)
        for row in range(4):
            for question, answer in zip(questions[row], answers[row], strict=True):
                i, j = int(question[:markers].argmax()), int(question[markers:].argmax())
                dy = positions[row, 1] // width - positions[row, 0] // width
                dx = positions[row, 1] % width - positions[row, 0] % width
                ay, ax = (dy[i] - dy[j]).double() * (2 * torch.pi / height), (dx[i] - dx[j]).double() * (2 * torch.pi / width)
                expected = torch.stack((ay.sin(), ay.cos(), ax.sin(), ax.cos())).float()
                torch.testing.assert_close(answer, expected)
        assert set(event.writing_inputs()) == {"x0", "question"}


def test_family_change_preserves_positions_questions_and_targets():
    a = make_calibrated_spatial_episodes(split="test", family_seed=1, scene_seed=2, count=4)
    b = make_calibrated_spatial_episodes(split="test", family_seed=3, scene_seed=2, count=4)
    for left, right in zip(a.events, b.events, strict=True):
        assert not torch.equal(left.x0, right.x0)
        torch.testing.assert_close(visible_positions(left)[0], visible_positions(right)[0], rtol=0, atol=0)
        for name in ("question", "answer", "readonly_questions", "readonly_answers"):
            torch.testing.assert_close(getattr(left, name), getattr(right, name), rtol=0, atol=0)


def test_scene_change_is_not_a_new_code_and_has_no_global_rng_side_effect():
    before = torch.random.get_rng_state().clone()
    a = make_calibrated_spatial_episodes(split="train", family_seed=7, scene_seed=9, count=4)
    b = make_calibrated_spatial_episodes(split="train", family_seed=7, scene_seed=10, count=4)
    torch.testing.assert_close(before, torch.random.get_rng_state(), rtol=0, atol=0)
    for left, right in zip(a.events, b.events, strict=True):
        lp, lc = visible_positions(left)
        rp, rc = visible_positions(right)
        torch.testing.assert_close(lc, rc, rtol=0, atol=0)
        assert not torch.equal(lp, rp)
    assert not torch.equal(visible_positions(a.events[0])[0], visible_positions(a.events[1])[0])


def test_batching_scene_resume_and_readonly_count_do_not_change_existing_material():
    spec = SpatialEpisodeSpec(length=4)
    many = make_calibrated_spatial_episodes(spec, split="cost", family_seed=12, scene_seed=19,
                                           episode_index=5, count=3)
    one = make_calibrated_spatial_episodes(SpatialEpisodeSpec(length=2), split="cost", family_seed=12,
                                          scene_seed=19, episode_index=6, scene_start=2)
    short = make_calibrated_spatial_episodes(SpatialEpisodeSpec(SpatialEventSpec(reuse_questions=1), length=4),
                                            split="cost", family_seed=12, scene_seed=19, episode_index=5, count=3)
    for offset, event in enumerate(one.events):
        for name in ("x0", "question", "answer", "readonly_questions", "readonly_answers"):
            torch.testing.assert_close(getattr(event, name), getattr(many.events[offset + 2], name)[1:2], rtol=0, atol=0)
    for left, right in zip(many.events, short.events, strict=True):
        for name in ("x0", "question", "answer"):
            torch.testing.assert_close(getattr(left, name), getattr(right, name), rtol=0, atol=0)


def test_splits_separate_calibration_families_and_scenes():
    rows = [make_calibrated_spatial_episodes(split=split, family_seed=5, scene_seed=6)
            for split in ("train", "validation", "test")]
    codes = [visible_positions(row.events[0])[1] for row in rows]
    assert all(not torch.equal(codes[i], codes[j]) for i in range(3) for j in range(i))


def test_calibration_requires_room_separate_from_real_markers():
    with pytest.raises(ValueError, match="separate"):
        SpatialEpisodeSpec(SpatialEventSpec(2, 3, 4))
