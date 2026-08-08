import torch

from new_mobile_ov.training.neodragon_opsd import (
    adaptive_base_trust_loss,
    balanced_joint_position,
    balanced_opsd_position,
    build_privileged_units,
    teacher_advantage_gate,
)


def test_balanced_positions_cover_only_context_informative_calls() -> None:
    positions = [
        balanced_opsd_position(
            step,
            min_unit=2,
            num_units=6,
            num_stages=3,
        )
        for step in range(1, 13)
    ]
    assert len(set(positions)) == 12
    assert positions[0] == (2, 0)
    assert positions[-1] == (5, 2)


def test_joint_positions_cover_all_deployed_calls() -> None:
    positions = [
        balanced_joint_position(step, num_units=6, num_stages=3)
        for step in range(1, 19)
    ]
    assert len(set(positions)) == 18
    assert positions[:3] == [(0, 0), (0, 1), (0, 2)]
    assert positions[-1] == (5, 2)


def test_privileged_history_replaces_old_units_but_keeps_latest() -> None:
    clean = torch.arange(5.0).view(1, 1, 5, 1, 1)
    generated = [torch.full((1, 1, 1, 1, 1), 100.0 + index) for index in range(4)]
    privileged = build_privileged_units(
        clean,
        generated,
        keep_recent_generated=1,
    )
    assert [float(value.item()) for value in privileged] == [0.0, 1.0, 2.0, 103.0]
    assert privileged[-1] is generated[-1]


def test_adaptive_trust_allows_teacher_justified_change() -> None:
    base = torch.zeros(1, 1, 1, 1, 4)
    teacher = torch.full_like(base, 0.2)
    inside = torch.full_like(base, 0.1)
    outside = torch.full_like(base, 0.4)
    inside_loss, _, _, margin = adaptive_base_trust_loss(
        inside,
        base,
        teacher,
        scale=1.0,
        minimum_margin=0.0,
        maximum_margin=1.0,
    )
    outside_loss, _, _, _ = adaptive_base_trust_loss(
        outside,
        base,
        teacher,
        scale=1.0,
        minimum_margin=0.0,
        maximum_margin=1.0,
    )
    assert torch.allclose(margin, torch.tensor(0.2))
    assert torch.allclose(inside_loss, torch.tensor(0.0))
    assert outside_loss > 0


def test_teacher_gate_rejects_harmful_privileged_context() -> None:
    useful_gate, useful_gain = teacher_advantage_gate(
        torch.tensor(1.0),
        torch.tensor(0.95),
        ramp=0.05,
    )
    harmful_gate, harmful_gain = teacher_advantage_gate(
        torch.tensor(1.0),
        torch.tensor(1.10),
        ramp=0.05,
    )
    assert torch.allclose(useful_gain, torch.tensor(0.05))
    assert torch.allclose(useful_gate, torch.tensor(1.0))
    assert harmful_gain < 0
    assert torch.allclose(harmful_gate, torch.tensor(0.0))
