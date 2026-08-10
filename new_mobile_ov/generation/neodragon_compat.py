from __future__ import annotations

import math
from typing import Union

import torch
from einops import rearrange


def _stable_block_noise(
    gamma: float,
    batch_size: int,
    channels: int,
    temporal: int,
    height: int,
    width: int,
    *,
    device: torch.device | None,
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
    iid = torch.randn(block_count, 4, dtype=torch.float32, device=device)
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


def stable_block_noise(
    gamma: float,
    batch_size: int,
    channels: int,
    temporal: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """CPU reference sampler matching NeoDragon's 2x2 noise covariance."""
    return _stable_block_noise(
        gamma,
        batch_size,
        channels,
        temporal,
        height,
        width,
        device=None,
    )


def install_neodragon_generation_patches(
    *,
    device: Union[str, torch.device, None] = None,
) -> None:
    """Install the valid 2x2 noise sampler without editing the upstream clone.

    Passing a CUDA device generates the corrective noise directly on that device.
    The original code samples thousands of tiny CPU tensors in Python and then
    copies them to CUDA for every pyramid transition. The CUDA path keeps the
    identical Gaussian covariance while removing that host-side bottleneck.
    """
    from neodragon.utils import generation_utils

    runtime_device = None if device is None else torch.device(device)
    patch_key = "cpu" if runtime_device is None else str(runtime_device)
    if getattr(generation_utils, "_mobile_ov_block_noise_patch", None) == patch_key:
        return

    if runtime_device is None:
        sampler = stable_block_noise
    else:
        def sampler(
            gamma: float,
            batch_size: int,
            channels: int,
            temporal: int,
            height: int,
            width: int,
        ) -> torch.Tensor:
            return _stable_block_noise(
                gamma,
                batch_size,
                channels,
                temporal,
                height,
                width,
                device=runtime_device,
            )

    generation_utils._sample_block_noise = sampler
    generation_utils._mobile_ov_block_noise_patch = patch_key
