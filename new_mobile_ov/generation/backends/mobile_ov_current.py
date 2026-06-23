from __future__ import annotations

import torch

from new_mobile_ov.generation.backends.base import AnchorGenerationBackend


class MobileOVCurrentBackend(AnchorGenerationBackend):
    """Current Mobile-OV/SANA-video backend boundary.

    This option represents the existing 135k-style branch. For the new project,
    it should be used mainly as a reference backend and teacher; the lightweight
    branch should target ``z0_gt`` first through LatentMotionWeaver.
    """

    latent_channels = 16

    def __init__(self, checkpoint_path: str | None = None, device: torch.device | None = None):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def generate_anchor(
        self,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor | None = None,
        *,
        height: int = 480,
        width: int = 832,
        num_steps: int = 24,
        cfg_scale: float = 6.0,
        seed: int = 0,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Current Mobile-OV backend is kept as a reference option. Use the original "
            "SANA-video sampler or implement an anchor-only sampler before using it here."
        )
