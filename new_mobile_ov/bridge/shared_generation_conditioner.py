"""One-forward SmolVLM2 conditioning for DreamLite and NeoDragon.

The shared encoder follows the already deployed Exp1 NeoDragon token contract.
It is intentionally generation-only: DreamLite editing also consumes source-image
tokens, which cannot be represented by a text-only NeoDragon condition.
"""

from __future__ import annotations

from typing import List, NamedTuple

import torch
import torch.nn as nn

from new_mobile_ov.bridge.dreamlite_image_bridge import (
    DreamLiteCondition,
    MobileOVDreamLiteImageBridge,
)
from new_mobile_ov.bridge.neodragon_text_bridge import MobileOVNeodragonTextBridge


class SharedGenerationConditions(NamedTuple):
    """Task-specific conditions produced from one canonical SmolVLM2 forward."""

    image: DreamLiteCondition
    video_prompt_embeds: torch.Tensor
    video_prompt_mask: torch.Tensor
    video_pooled: torch.Tensor


class SharedMobileOVGenerationConditioner(nn.Module):
    """Fan out one Exp1-compatible SmolVLM2 feature sequence to two heads.

    The NeoDragon branch remains numerically identical to
    ``MobileOVNeodragonTextBridge.encode``. The DreamLite branch must use an
    image head distilled on this canonical feature sequence; legacy V1-V7 image
    checkpoints used a separate ``[Generate]:`` input contract and are not
    quality-compatible with this module.
    """

    def __init__(
        self,
        *,
        image_bridge: MobileOVDreamLiteImageBridge,
        video_bridge: MobileOVNeodragonTextBridge,
        prompt_suffix: str = "",
    ) -> None:
        super().__init__()
        if image_bridge.feature_provider is not None:
            raise ValueError(
                "Shared conditioner requires an image bridge constructed with "
                "load_feature_provider=False to avoid a second SmolVLM2 instance."
            )
        self.image_bridge = image_bridge
        self.video_bridge = video_bridge
        self.prompt_suffix = str(prompt_suffix)

    def encode_generation(self, prompts: List[str]) -> SharedGenerationConditions:
        canonical_prompts = [str(prompt) + self.prompt_suffix for prompt in prompts]
        hidden_states, source_mask, hidden_layers = self.video_bridge.encode_smolvlm2_features(
            canonical_prompts
        )
        if hidden_layers is None:
            raise RuntimeError(
                "The NeoDragon MCP bridge must return hidden layers for shared conditioning."
            )
        image = self.image_bridge(
            prompts,
            mode="generate",
            shared_hidden_layers=hidden_layers,
            shared_source_mask=source_mask,
        )
        video_prompt_embeds, video_prompt_mask, video_pooled = (
            self.video_bridge.encode_from_smolvlm2_features(
                hidden_states,
                source_mask,
                hidden_layers,
            )
        )
        return SharedGenerationConditions(
            image=image,
            video_prompt_embeds=video_prompt_embeds,
            video_prompt_mask=video_prompt_mask,
            video_pooled=video_pooled,
        )
