from benchmarks._rcc_training import CorpusCursor, RCCCurriculum


def test_rcc_curriculum_and_cursor_resume_at_optimizer_boundary():
    curriculum = RCCCurriculum(boundary_step=2, early_depths=(1,), late_depths=(2,))
    cursor = CorpusCursor(
        ([1, 2, 3, 4], [5, 6, 7, 8]),
        length=4,
        batch_size=1,
        curriculum=curriculum,
    )
    first = cursor.peek()
    state = cursor.state_dict()
    restored = CorpusCursor(
        ([1, 2, 3, 4], [5, 6, 7, 8]),
        length=4,
        batch_size=1,
        curriculum=curriculum,
    )
    restored.load_state_dict(state)
    assert restored.peek() == first
    assert curriculum.depth_for(0) == 1
    assert curriculum.depth_for(2) == 2


def test_rcc_training_always_includes_fully_expanded_base_path():
    curriculum = RCCCurriculum(boundary_step=2, early_depths=(1,), late_depths=(2,))

    assert curriculum.training_depths_for(0) == (0, 1)
    assert curriculum.training_depths_for(2) == (0, 2)

    base_only = RCCCurriculum.from_depths((0,))
    assert base_only.training_depths_for(0) == (0,)
