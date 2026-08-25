from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from new_mobile_ov.training.neodragon_pyramidal_dmd import (
    build_stage_pair,
    cauchy_endpoint_loss,
    dmd_sample_weight,
    dmd_surrogate_loss,
    linear_weight_decay,
    motion_residual_anchor_loss,
    rollout_history_probability,
    stage_noisy_student_endpoint,
    stage_timestep,
    student_probe_sigmas,
)
from tools.train_neodragon_pyramidal_dmd import (
    ALL_NATIVE_SCHEDULE,
    ANCHOR_ALT_SCHEDULE,
    LEGACY_VIDEO_ONLY_SCHEDULE,
    ROLLOUT_AWARE_V3_SCHEDULE,
    assert_fp32_trainable_parameters,
    protocol_metadata,
    rollout_student_state_to_position,
    select_native_unit_stage,
)


class DummyScheduler:
    config = SimpleNamespace(stages=3, num_train_timesteps=1000, gamma=1.0 / 3.0)
    orig_start_sigmas = {0: 1.0, 1: 2.0 / 3.0, 2: 1.0 / 3.0}
    end_sigmas = {0: 2.0 / 3.0, 1: 1.0 / 3.0, 2: 0.0}
    timestep_ratios = {0: (0.0, 0.4), 1: (0.4, 0.7), 2: (0.7, 1.0)}

    def get_stage_timesteps(self, _num_steps: int, stage: int, device: torch.device):
        values = {0: (1000.0, 600.0), 1: (600.0, 300.0), 2: (300.0, 0.0)}
        return torch.tensor(values[stage], device=device)


class ZeroFlowDiT(torch.nn.Module):
    def forward(self, *, sample, **_kwargs):
        return [torch.zeros_like(sample[0][-1])]


def test_stage_pair_matches_pyramidal_endpoint_construction() -> None:
    scheduler = DummyScheduler()
    clean = torch.ones(2, 3, 1, 8, 12)
    noise = torch.full_like(clean, 2.0)

    coarse_pair = build_stage_pair(clean=clean, scheduler=scheduler, stage=0, noise=noise)
    assert torch.equal(coarse_pair.start, noise)
    assert torch.allclose(coarse_pair.end, clean / 3.0 + noise * (2.0 / 3.0))
    assert torch.allclose(coarse_pair.flow_target, coarse_pair.start - coarse_pair.end)

    fine_pair = build_stage_pair(clean=clean, scheduler=scheduler, stage=2, noise=noise)
    assert torch.allclose(fine_pair.start, clean * (2.0 / 3.0) + noise / 3.0)
    assert torch.equal(fine_pair.end, clean)


def test_stage_noising_and_time_mapping_are_finite() -> None:
    scheduler = DummyScheduler()
    endpoint = torch.randn(2, 3, 1, 8, 12)
    probe, start, end = stage_noisy_student_endpoint(
        endpoint=endpoint,
        scheduler=scheduler,
        stage=1,
        local_sigma=0.25,
        noise=torch.randn_like(endpoint),
    )
    assert torch.allclose(probe, 0.75 * end + 0.25 * start)
    assert stage_timestep(scheduler, stage=1, local_sigma=1.0, device=endpoint.device).item() == 600.0
    assert stage_timestep(scheduler, stage=1, local_sigma=0.0, device=endpoint.device).item() == 300.0
    assert student_probe_sigmas() == (0.125, 0.375, 0.625, 0.875)


def test_dmd_surrogate_has_endpoint_gradient_and_cauchy_is_positive() -> None:
    endpoint = torch.randn(2, 3, 1, 4, 4, requires_grad=True)
    teacher_flow = torch.full_like(endpoint, 2.0)
    fake_flow = torch.full_like(endpoint, 0.5)
    weights = dmd_sample_weight(
        teacher_flow=teacher_flow,
        stage_flow_target=torch.ones_like(endpoint),
        maximum=10.0,
    )
    surrogate, direction_rms = dmd_surrogate_loss(
        endpoint=endpoint,
        teacher_flow=teacher_flow,
        fake_flow=fake_flow,
        sample_weight=weights,
    )
    loss = surrogate + 0.5 * cauchy_endpoint_loss(endpoint, torch.zeros_like(endpoint))
    loss.backward()
    assert endpoint.grad is not None
    assert torch.isfinite(endpoint.grad).all()
    assert direction_rms.item() > 0


def test_dmd_requires_fp32_master_parameters() -> None:
    assert_fp32_trainable_parameters(torch.nn.Linear(2, 2).float(), name="student")
    with pytest.raises(RuntimeError, match="FP32 master parameters"):
        assert_fp32_trainable_parameters(
            torch.nn.Linear(2, 2).bfloat16(),
            name="student",
        )


def test_anchor_alternative_has_a_distinct_checkpoint_contract() -> None:
    assert protocol_metadata(
        include_first_unit=True,
        external_anchor_alternative=False,
    )[0] == ALL_NATIVE_SCHEDULE
    assert protocol_metadata(
        include_first_unit=False,
        external_anchor_alternative=False,
    )[0] == LEGACY_VIDEO_ONLY_SCHEDULE
    schedule, objective = protocol_metadata(
        include_first_unit=False,
        external_anchor_alternative=True,
    )
    assert schedule == ANCHOR_ALT_SCHEDULE
    assert objective == "pyramidal_dmd_v2_alt_external_anchor_video_units"


def test_anchor_alternative_cycles_only_six_video_units() -> None:
    positions = [
        select_native_unit_stage(step, include_first_unit=False)
        for step in range(1, 19)
    ]
    assert positions == [(unit, stage) for unit in range(1, 7) for stage in range(3)]


def test_anchor_alternative_rejects_first_unit_optimization() -> None:
    try:
        protocol_metadata(
            include_first_unit=True,
            external_anchor_alternative=True,
        )
    except ValueError as error:
        assert "--no-include-first-unit" in str(error)
    else:
        raise AssertionError("Conflicting DMD protocols must be rejected")


def test_v3_has_distinct_rollout_aware_contract() -> None:
    schedule, objective = protocol_metadata(
        include_first_unit=False,
        external_anchor_alternative=True,
        rollout_aware_v3=True,
    )
    assert schedule == ROLLOUT_AWARE_V3_SCHEDULE
    assert objective == "pyramidal_dmd_v3_rollout_aware_external_anchor_video_units"


def test_v3_curriculum_and_cauchy_schedule_hit_declared_boundaries() -> None:
    kwargs = {
        "warmup_steps": 1000,
        "midpoint_step": 4000,
        "final_step": 10000,
        "midpoint_probability": 0.5,
        "final_probability": 0.75,
    }
    assert rollout_history_probability(1000, **kwargs) == 0.0
    assert rollout_history_probability(4000, **kwargs) == 0.5
    assert rollout_history_probability(10000, **kwargs) == 0.75
    assert linear_weight_decay(1, initial=0.5, final=0.1, decay_steps=4000) == 0.5
    assert linear_weight_decay(4000, initial=0.5, final=0.1, decay_steps=4000) == pytest.approx(0.1)
    assert linear_weight_decay(10000, initial=0.5, final=0.1, decay_steps=4000) == pytest.approx(0.1)


def test_motion_residual_anchor_rewards_correct_temporal_change() -> None:
    previous = torch.zeros(2, 3, 1, 4, 4)
    clean = torch.ones_like(previous)
    exact = motion_residual_anchor_loss(clean, clean, previous)
    static = motion_residual_anchor_loss(previous, clean, previous)
    opposite = motion_residual_anchor_loss(-clean, clean, previous)
    assert exact.item() < 1e-6
    assert static.item() > exact.item()
    assert opposite.item() > static.item()


def test_v3_student_rollout_reaches_exact_unit_stage_position() -> None:
    condition = SimpleNamespace(
        tokens=torch.zeros(1, 1, 1),
        mask=torch.ones(1, 1),
        pooled=torch.zeros(1, 1),
        uses_guidance=False,
    )
    current, history, calls = rollout_student_state_to_position(
        student=ZeroFlowDiT(),
        scheduler=DummyScheduler(),
        anchor=torch.zeros(1, 1, 1, 16, 16),
        full_noise=torch.randn(1, 1, 7, 16, 16),
        condition=condition,
        target_unit=2,
        target_stage=1,
        generator=torch.Generator().manual_seed(7),
    )
    assert current.shape == (1, 1, 1, 8, 8)
    assert history
    assert calls == 4


def test_anchor_submit_preserves_v2_hyperparameters_and_excludes_unit_zero() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "reproduce_neodragon_pyramidal_dmd_v2_anchor_alt_1node8gpu.sbatch"
    ).read_text(encoding="utf-8")

    assert '--steps "${STEPS:-10000}"' in script
    assert '--student-lr "${STUDENT_LR:-1e-6}"' in script
    assert '--fake-updates "${FAKE_UPDATES:-2}"' in script
    assert '--cauchy-weight "${CAUCHY_WEIGHT:-0.5}"' in script
    assert "--no-include-first-unit" in script
    assert "--external-anchor-alternative" in script


def test_v3_submit_is_fresh_rollout_aware_and_storage_bounded() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "reproduce_neodragon_pyramidal_dmd_v3_1node8gpu.sbatch"
    ).read_text(encoding="utf-8")

    assert '--resume "${RESUME:-none}"' in script
    assert '--steps "${STEPS:-10000}"' in script
    assert "--no-include-first-unit" in script
    assert "--external-anchor-alternative" in script
    assert "--rollout-aware-v3" in script
    assert '--history-midpoint-probability "${HISTORY_MIDPOINT_PROBABILITY:-0.5}"' in script
    assert '--history-final-probability "${HISTORY_FINAL_PROBABILITY:-0.75}"' in script
    assert '--cauchy-final-weight "${CAUCHY_FINAL_WEIGHT:-0.1}"' in script
    assert '--motion-residual-weight "${MOTION_RESIDUAL_WEIGHT:-0.05}"' in script
    assert '--save-every "${SAVE_EVERY:-1000}"' in script
    assert '--archive-every "${ARCHIVE_EVERY:-5000}"' in script
    assert "--no-save-resume" in script
    assert "--trainable-dtype fp32" in script
