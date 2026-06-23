from __future__ import annotations

import sys
from pathlib import Path

import torch

from new_mobile_ov.generation.backends.base import AnchorGenerationBackend


class MobileOSana05BBackend(AnchorGenerationBackend):
    """Mobile-O 0.5B image generator backend for anchor-latent generation.

    This backend vendors Mobile-O source under ``third_party/mobileo``. It keeps
    the current implementation boundary simple: Mobile-O produces a DC-AE image
    latent/image, and a later adapter can map that anchor into the WanVAE video
    latent space if needed.
    """

    latent_channels = 32

    def __init__(self, model_path: str, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        repo_root = Path(__file__).resolve().parents[3]
        third_party = repo_root / "third_party"
        if str(third_party) not in sys.path:
            sys.path.insert(0, str(third_party))
        from mobileo.model.builder import load_pretrained_model
        from mobileo.constants import IMAGE_TOKEN_INDEX
        from mobileo.conversation import conv_templates
        from mobileo.mm_utils import tokenizer_image_token

        self.device = device
        self.dtype = dtype
        self.tokenizer, self.model, _ = load_pretrained_model(model_path)
        self.image_token_index = IMAGE_TOKEN_INDEX
        self.conv_templates = conv_templates
        self.tokenizer_image_token = tokenizer_image_token
        try:
            self.model.to(device)
        except RuntimeError as exc:
            # Mobile-O loads with device_map="auto"; accelerate-dispatched modules
            # cannot always be moved again. That is fine as long as generate works.
            if "dispatch" not in str(exc).lower() and "device_map" not in str(exc).lower():
                raise
        self.model.eval()

    @torch.no_grad()
    def generate_image_from_prompt(self, prompt: str):
        """Generate a PIL image using Mobile-O's native text prompt path."""
        query = "Please generate image based on the following caption: " + str(prompt)
        conv = self.conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        text = conv.get_prompt()
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        input_ids = self.tokenizer_image_token(
            text,
            self.tokenizer,
            self.image_token_index,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)
        return self.model.generate_image(input_ids, pixel_values=None)[0]

    @torch.no_grad()
    def generate_anchor(
        self,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor | None = None,
        *,
        height: int = 480,
        width: int = 832,
        num_steps: int = 20,
        cfg_scale: float = 1.5,
        seed: int = 0,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Mobile-O currently exposes image generation through tokenized prompts, not direct "
            "Mobile-OV bridge tokens. The next step is adding a small bridge-token-to-Mobile-O "
            "connector or training a direct anchor head into WanVAE z0_gt."
        )
