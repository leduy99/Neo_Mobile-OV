from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from new_mobile_ov.training.neodragon_hybrid_recovery import (
    DiTCondition,
    StageUnitScaleEMA,
    balanced_position,
    curriculum_probabilities,
    hybrid_trust_region_loss,
    normalized_charbonnier,
    rollout_state_to_position,
    run_stage_endpoint,
    teacher_forced_state_to_position,
)


class ToyScheduler:
    def __init__(self) -> None:
        self.orig_start_sigmas = {0: 1.0, 1: 2 / 3, 2: 1 / 3}
        self.config = SimpleNamespace(gamma=1 / 3)

    def get_stage_timesteps(self, steps: int, stage: int, device=None):
        return torch.linspace(1.0, 0.1, steps, device=device) + stage

    def get_stage_sigmas(self, steps: int, stage: int, device=None):
        del stage
        return torch.linspace(1.0, 0.0, steps + 1, device=device)

    def step(self, model_output, sigma, sigma_next, sample):
        return SimpleNamespace(
            prev_sample=sample.float().add(
                model_output.float(),
                alpha=float(sigma_next - sigma),
            ).to(model_output.dtype)
        )


class ToyDiT(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))

    def forward(
        self,
        *,
        sample,
        encoder_hidden_states,
        encoder_attention_mask,
        pooled_projections,
        timestep_ratio,
    ):
        del encoder_hidden_states, encoder_attention_mask, pooled_projections
        current = sample[0][-1]
        value = self.value.to(dtype=current.dtype)
        timestep = timestep_ratio.view(-1, 1, 1, 1, 1) * 0.0
        return [torch.ones_like(current) * value + timestep]


def condition(batch: int = 1) -> DiTCondition:
    return DiTCondition(
        tokens=torch.zeros(batch, 2, 4),
        mask=torch.ones(batch, 2, dtype=torch.bool),
        pooled=torch.zeros(batch, 3),
    )


def test_balanced_position_covers_all_hybrid_calls() -> None:
    positions = [balanced_position(step) for step in range(1, 19)]
    assert len(set(positions)) == 18
    assert positions[0] == (0, 0)
    assert positions[-1] == (5, 2)
    assert balanced_position(19) == (0, 0)
    assert balanced_position(1, offset=17) == (5, 2)


def test_curriculum_probabilities_are_normalized() -> None:
    for step in [1, 501, 5001]:
        values = curriculum_probabilities(
            step,
            parity_steps=500,
            map_end_step=5000,
        )
        assert abs(sum(values.values()) - 1.0) < 1e-8


def test_stage_endpoint_is_exact_euler_map() -> None:
    scheduler = ToyScheduler()
    start = torch.ones(1, 1, 1, 2, 2)
    endpoint, first = run_stage_endpoint(
        dit=ToyDiT(0.25),
        scheduler=scheduler,
        current=start,
        history=(),
        condition=condition(),
        stage=0,
        num_steps=4,
    )
    assert torch.allclose(first, torch.full_like(first, 0.25))
    assert torch.allclose(endpoint, torch.full_like(endpoint, 0.75))


def test_rollout_stops_before_selected_call() -> None:
    scheduler = ToyScheduler()
    anchor = torch.zeros(1, 1, 1, 8, 8)
    noise = torch.ones(1, 1, 7, 8, 8)
    state = rollout_state_to_position(
        actor=ToyDiT(0.1),
        scheduler=scheduler,
        anchor=anchor,
        full_noise=noise,
        condition=condition(),
        target_unit=0,
        target_stage=0,
        actor_steps=1,
        generator=torch.Generator().manual_seed(0),
    )
    assert state.unit == 0
    assert state.stage == 0
    assert state.start.shape == (1, 1, 1, 2, 2)
    assert len(state.history) == 1


def test_normalization_and_trust_margin() -> None:
    start = torch.zeros(1, 1, 1, 2, 2)
    hybrid = torch.ones_like(start)
    monolithic = torch.full_like(start, 1.2)
    student = torch.full_like(start, 1.1)
    loss, student_gap, margin = hybrid_trust_region_loss(
        student,
        hybrid,
        monolithic,
        start=start,
    )
    assert student_gap < margin
    assert loss.item() == 0.0
    assert normalized_charbonnier(student, monolithic, scale=0.2).item() > 0.0


def test_teacher_forcing_uses_real_history() -> None:
    scheduler = ToyScheduler()
    clean = torch.arange(7.0).view(1, 1, 7, 1, 1).expand(-1, -1, -1, 8, 8)
    noise = torch.ones_like(clean)
    state = teacher_forced_state_to_position(
        actor=ToyDiT(0.1),
        scheduler=scheduler,
        clean_latents=clean,
        full_noise=noise,
        condition=condition(),
        target_unit=2,
        target_stage=0,
        actor_steps=1,
        generator=torch.Generator().manual_seed(0),
    )
    assert len(state.history) >= 1
    assert torch.isclose(state.history[-1].mean(), torch.tensor(2.0))


def test_stage_unit_ema_round_trip() -> None:
    ema = StageUnitScaleEMA(decay=0.5)
    assert ema.update(2, 1, 2.0) == 2.0
    assert ema.update(2, 1, 4.0) == 3.0
    restored = StageUnitScaleEMA(decay=0.5)
    restored.load_state_dict(ema.state_dict())
    assert restored.get(2, 1) == 3.0
