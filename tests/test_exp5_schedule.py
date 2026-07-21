import pytest

from new_mobile_ov.training.exp5_schedule import Exp5Schedule


def make_schedule() -> Exp5Schedule:
    return Exp5Schedule(
        total_steps=255_000,
        phase_a_steps=10_000,
        phase_b_steps=120_000,
        dit_warmup_steps=2_000,
        final_cooldown_steps=20_000,
        bridge_warmup_steps=2_000,
        bridge_cooldown_steps=10_000,
        flow_start_weight=0.05,
        flow_peak_weight=0.3,
        flow_final_weight=0.1,
    )


def test_exp5_phase_boundaries() -> None:
    schedule = make_schedule()

    assert schedule.phase(1).name == "A_dit_warmup"
    assert schedule.phase(10_000).name == "A_dit_warmup"
    assert schedule.phase(10_001).name == "B_joint_refinement"
    assert schedule.phase(130_000).name == "B_joint_refinement"
    assert schedule.phase(130_001).name == "C_dit_consolidation"
    assert schedule.phase(255_000).name == "C_dit_consolidation"


def test_exp5_lr_and_flow_schedules() -> None:
    schedule = make_schedule()

    assert schedule.dit_lr_scale(1) == pytest.approx(1 / 2_000)
    assert schedule.dit_lr_scale(2_000) == pytest.approx(1.0)
    assert schedule.dit_lr_scale(255_000) == pytest.approx(0.1)

    assert schedule.bridge_lr_scale(10_000) == 0.0
    assert schedule.bridge_lr_scale(10_001) == pytest.approx(1 / 2_000)
    assert schedule.bridge_lr_scale(12_000) == pytest.approx(1.0)
    assert schedule.bridge_lr_scale(120_000) == pytest.approx(1.0)
    assert schedule.bridge_lr_scale(130_000) == pytest.approx(0.0)
    assert schedule.bridge_lr_scale(130_001) == 0.0

    assert schedule.flow_weight(1) > 0.05
    assert schedule.flow_weight(2_000) == pytest.approx(0.3)
    assert schedule.flow_weight(235_000) == pytest.approx(0.3)
    assert schedule.flow_weight(255_000) == pytest.approx(0.1)


def test_exp5_rejects_overlapping_phases() -> None:
    with pytest.raises(ValueError, match="Phase C"):
        Exp5Schedule(
            total_steps=100,
            phase_a_steps=10,
            phase_b_steps=90,
            dit_warmup_steps=1,
            final_cooldown_steps=10,
            bridge_warmup_steps=1,
            bridge_cooldown_steps=1,
            flow_start_weight=0.05,
            flow_peak_weight=0.3,
            flow_final_weight=0.1,
        )
