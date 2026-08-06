from __future__ import annotations

import math

import torch
from einops import rearrange


def stable_block_noise(
    gamma: float,
    batch_size: int,
    channels: int,
    temporal: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Sample NeoDragon's 2x2 corrective noise without a Cholesky factor.

    At gamma=1/3 the covariance is positive semidefinite, not positive
    definite. PyTorch's MultivariateNormal rejects that valid boundary case.
    This samples the same Gaussian through its parallel/perpendicular
    eigenspaces and supports the zero eigenvalue exactly.
    """
    if min(batch_size, channels, temporal, height, width) <= 0:
        raise ValueError("Block-noise dimensions must be positive.")
    if height % 2 or width % 2:
        raise ValueError(f"Block-noise resolution must be even, got {height}x{width}.")

    gamma = float(gamma)
    perpendicular_variance = 1.0 + gamma
    parallel_variance = 1.0 - 3.0 * gamma
    tolerance = 1e-6
    if perpendicular_variance < -tolerance or parallel_variance < -tolerance:
        raise ValueError(
            "NeoDragon block-noise covariance is not positive semidefinite: "
            f"gamma={gamma}, eigenvalues=({parallel_variance}, "
            f"{perpendicular_variance})."
        )

    block_count = (
        batch_size
        * channels
        * temporal
        * (height // 2)
        * (width // 2)
    )
    iid = torch.randn(block_count, 4, dtype=torch.float32)
    parallel = iid.mean(dim=-1, keepdim=True)
    perpendicular = iid - parallel
    samples = (
        math.sqrt(max(perpendicular_variance, 0.0)) * perpendicular
        + math.sqrt(max(parallel_variance, 0.0)) * parallel
    )
    return rearrange(
        samples,
        "(b c t h w) (p q) -> b c t (h p) (w q)",
        b=batch_size,
        c=channels,
        t=temporal,
        h=height // 2,
        w=width // 2,
        p=2,
        q=2,
    )


def install_neodragon_generation_patches() -> None:
    """Install runtime fixes without modifying the downloaded NeoDragon clone."""
    from neodragon.utils import generation_utils

    if getattr(generation_utils, "_mobile_ov_block_noise_patch", False):
        return
    generation_utils._sample_block_noise = stable_block_noise
    generation_utils._mobile_ov_block_noise_patch = True
