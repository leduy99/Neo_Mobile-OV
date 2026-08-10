# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Qualcomm Innovation Center, Inc.
# Adapted from https://github.com/quic/aimet-model-zoo
"""Minimal Qualcomm QuickSRNet loader for reproducible 2x video post-processing.

The architecture below is adapted from Qualcomm's AIMET Model Zoo QuickSRNet
implementation (BSD-3-Clause).  It deliberately has no AIMET dependency: the
public float checkpoints load directly into regular PyTorch modules.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Literal

import torch
from torch import nn


QuickSRVariant = Literal["small", "medium", "large"]

_VARIANT_CONFIG: dict[str, tuple[int, int, bool]] = {
    "small": (32, 2, False),
    "medium": (32, 5, False),
    "large": (64, 11, True),
}
_BASE_URL = (
    "https://github.com/quic/aimet-model-zoo/releases/download/"
    "phase_2_january_artifacts/quicksrnet_{variant}_{scale}x_checkpoint_float32.pth.tar"
)


class _AnchorOp(nn.Module):
    """Learned input-to-output residual used by the public Large variant."""

    def __init__(self, scale: int, in_channels: int = 3) -> None:
        super().__init__()
        self.net = nn.Conv2d(in_channels, in_channels * scale**2, kernel_size=1)
        with torch.no_grad():
            self.net.weight.zero_()
            self.net.bias.zero_()
            for channel in range(in_channels):
                self.net.weight[channel * scale**2 : (channel + 1) * scale**2, channel, 0, 0] = 1.0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class QuickSRNet(nn.Module):
    """QuickSRNet compatible with Qualcomm's released float PyTorch checkpoints."""

    def __init__(self, variant: QuickSRVariant = "medium", scale: int = 2) -> None:
        super().__init__()
        if variant not in _VARIANT_CONFIG:
            raise ValueError(f"Unsupported QuickSRNet variant={variant!r}; choose from {sorted(_VARIANT_CONFIG)}")
        if scale not in {2, 3, 4}:
            raise ValueError("This lightweight loader supports integer QuickSRNet scales 2, 3, and 4.")

        channels, intermediate_layers, use_ito_connection = _VARIANT_CONFIG[variant]
        self.variant = variant
        self.scale = scale
        self.use_ito_connection = use_ito_connection

        layers: list[nn.Module] = [
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.Hardtanh(min_val=0.0, max_val=1.0),
        ]
        for _ in range(intermediate_layers):
            layers.extend(
                [
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.Hardtanh(min_val=0.0, max_val=1.0),
                ]
            )
        self.cnn = nn.Sequential(*layers)
        self.conv_last = nn.Conv2d(channels, 3 * scale**2, kernel_size=3, padding=1)
        if use_ito_connection:
            self.anchor = _AnchorOp(scale=scale)
        self.clip_output = nn.Hardtanh(min_val=0.0, max_val=1.0)
        self.depth_to_space = nn.PixelShuffle(scale)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.cnn(inputs)
        outputs = self.conv_last(outputs)
        if self.use_ito_connection:
            outputs = self.anchor(inputs) + outputs
        return self.depth_to_space(self.clip_output(outputs))


def checkpoint_url(variant: QuickSRVariant, scale: int) -> str:
    if variant not in _VARIANT_CONFIG:
        raise ValueError(f"Unsupported QuickSRNet variant={variant!r}")
    return _BASE_URL.format(variant=variant, scale=scale)


def ensure_checkpoint(
    *,
    variant: QuickSRVariant = "medium",
    scale: int = 2,
    cache_dir: str | Path = "checkpoints/quicksrnet",
) -> Path:
    """Download a public Qualcomm float checkpoint once, then reuse it locally."""

    directory = Path(cache_dir)
    destination = directory / f"quicksrnet_{variant}_{scale}x_float32.pth.tar"
    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    directory.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    print(f"Downloading public QuickSRNet checkpoint: {checkpoint_url(variant, scale)}", flush=True)
    try:
        urllib.request.urlretrieve(checkpoint_url(variant, scale), temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_quicksrnet(
    *,
    variant: QuickSRVariant = "medium",
    scale: int = 2,
    checkpoint: str | Path | None = None,
    cache_dir: str | Path = "checkpoints/quicksrnet",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[QuickSRNet, Path]:
    """Return an eval-mode public QuickSRNet and the exact weight file used."""

    checkpoint_path = Path(checkpoint) if checkpoint else ensure_checkpoint(
        variant=variant,
        scale=scale,
        cache_dir=cache_dir,
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"QuickSRNet checkpoint does not exist: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unexpected QuickSRNet checkpoint payload type: {type(payload)!r}")
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    model = QuickSRNet(variant=variant, scale=scale)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"QuickSRNet state mismatch: missing={missing}, unexpected={unexpected}")
    return model.to(device=device, dtype=dtype).eval(), checkpoint_path


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
