import pytest
import torch

from new_mobile_ov.training.distributed import DistributedContext, scalar_mean
from new_mobile_ov.training.neodragon_pyramid_flow import (
    build_pyramid_flow_state,
    correlated_pyramid_noise,
    corrupt_history,
    pyramid_latents,
    stage_from_ratio_slot,
)


def test_pyramid_shapes_are_low_to_high():
    value = torch.randn(2, 4, 1, 16, 24)
    levels = pyramid_latents(value, stages=3)
    assert [tuple(level.shape[-2:]) for level in levels] == [(4, 6), (8, 12), (16, 24)]


def test_stage_endpoints_follow_official_definition():
    clean = torch.randn(2, 4, 1, 16, 24)
    noise = torch.randn_like(clean)
    levels = pyramid_latents(clean, stages=3)
    noise_levels = correlated_pyramid_noise(noise, stages=3)

    stage0 = build_pyramid_flow_state(
        clean,
        stage=0,
        local_sigma=torch.ones(2),
        start_sigma=1.0,
        end_sigma=2.0 / 3.0,
        noise_high=noise,
    )
    assert torch.equal(stage0.noisy, noise_levels[0])

    stage2 = build_pyramid_flow_state(
        clean,
        stage=2,
        local_sigma=torch.zeros(2),
        start_sigma=0.4,
        end_sigma=0.0,
        noise_high=noise,
    )
    assert torch.equal(stage2.noisy, levels[2])
    assert torch.allclose(stage2.target, stage2.start - levels[2])


def test_history_corruption_endpoints():
    history = [torch.ones(2, 4, 1, 4, 6)]
    clean = corrupt_history(history, sigma=torch.zeros(2))[0]
    assert torch.equal(clean, history[0])

    generator = torch.Generator().manual_seed(7)
    noisy = corrupt_history(history, sigma=torch.ones(2), generator=generator)[0]
    assert not torch.equal(noisy, history[0])


def test_stage_ratio_slots_are_configurable():
    assert [stage_from_ratio_slot(i) for i in range(6)] == [0, 1, 2, 0, 1, 2]
    assert [stage_from_ratio_slot(i, (1, 2, 1)) for i in range(8)] == [
        0,
        1,
        1,
        2,
        0,
        1,
        1,
        2,
    ]


def test_scalar_mean_accepts_per_sample_metrics():
    ctx = DistributedContext(rank=0, local_rank=0, world_size=1, device=torch.device("cpu"))
    assert scalar_mean(torch.tensor([0.2, 0.4]), ctx) == pytest.approx(0.3)
