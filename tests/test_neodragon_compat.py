from __future__ import annotations

import torch

from new_mobile_ov.generation.neodragon_compat import stable_block_noise


def test_stable_block_noise_supports_semidefinite_gamma_boundary() -> None:
    torch.manual_seed(7)
    noise = stable_block_noise(
        1.0 / 3.0,
        batch_size=2,
        channels=3,
        temporal=2,
        height=8,
        width=10,
    )

    assert noise.shape == (2, 3, 2, 8, 10)
    assert torch.isfinite(noise).all()

    blocks = noise.reshape(2, 3, 2, 4, 2, 5, 2).permute(0, 1, 2, 3, 5, 4, 6)
    block_sums = blocks.reshape(-1, 4).sum(dim=-1)
    assert torch.allclose(block_sums, torch.zeros_like(block_sums), atol=1e-5)


def test_stable_block_noise_matches_target_covariance() -> None:
    torch.manual_seed(11)
    gamma = 0.2
    noise = stable_block_noise(
        gamma,
        batch_size=1,
        channels=1,
        temporal=1,
        height=400,
        width=400,
    )
    blocks = noise.reshape(1, 1, 1, 200, 2, 200, 2).permute(0, 1, 2, 3, 5, 4, 6)
    samples = blocks.reshape(-1, 4)
    covariance = torch.cov(samples.T)
    expected = torch.eye(4) * (1.0 + gamma) - torch.ones(4, 4) * gamma

    assert torch.allclose(covariance, expected, atol=0.025, rtol=0.05)
