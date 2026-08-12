from __future__ import annotations

from types import SimpleNamespace

import torch

from new_mobile_ov.training.neodragon_pyramidal_dmd import (
    build_stage_pair,
    cauchy_endpoint_loss,
    dmd_sample_weight,
    dmd_surrogate_loss,
    stage_noisy_student_endpoint,
    stage_timestep,
    student_probe_sigmas,
)


class DummyScheduler:
    config = SimpleNamespace(stages=3, num_train_timesteps=1000)
    orig_start_sigmas = {0: 1.0, 1: 2.0 / 3.0, 2: 1.0 / 3.0}
    end_sigmas = {0: 2.0 / 3.0, 1: 1.0 / 3.0, 2: 0.0}
    timestep_ratios = {0: (0.0, 0.4), 1: (0.4, 0.7), 2: (0.7, 1.0)}

    def get_stage_timesteps(self, _num_steps: int, stage: int, device: torch.device):
        values = {0: (1000.0, 600.0), 1: (600.0, 300.0), 2: (300.0, 0.0)}
        return torch.tensor(values[stage], device=device)


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
