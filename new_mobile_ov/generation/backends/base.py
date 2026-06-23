from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AnchorGenerationBackend(nn.Module, ABC):
    """Backend that converts text conditioning into one spatial anchor latent."""

    latent_channels: int

    @abstractmethod
    def generate_anchor(
        self,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor | None = None,
        *,
        height: int = 480,
        width: int = 832,
        num_steps: int = 24,
        cfg_scale: float = 3.0,
        seed: int = 0,
    ) -> torch.Tensor:
        """Return anchor latent as [B, C, H_lat, W_lat]."""
        raise NotImplementedError
