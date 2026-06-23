from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from new_mobile_ov.checkpoints import ensure_neodragon_assets
from new_mobile_ov.generation.backends.base import AnchorGenerationBackend


class _PrecomputedNeodragonTextEncoder(nn.Module):
    """Tiny callable matching Neodragon TextEncoderBundle.forward."""

    def __init__(self, prompt_embeds: torch.Tensor, prompt_mask: torch.Tensor, pooled: torch.Tensor):
        super().__init__()
        self.register_buffer("prompt_embeds", prompt_embeds.detach(), persistent=False)
        self.register_buffer("prompt_mask", prompt_mask.detach(), persistent=False)
        self.register_buffer("pooled", pooled.detach(), persistent=False)

    def forward(self, input_prompts, device: torch.device):
        return (
            self.prompt_embeds.to(device),
            self.prompt_mask.to(device),
            self.pooled.to(device),
        )


class MobileOVNeodragonBackend(AnchorGenerationBackend):
    """Neodragon video generation backend for the Mobile-OV text side.

    This backend deliberately keeps two boundaries separate:

    - Mobile-OV understanding branch + bridge still runs and can be trained/aligned.
    - Neodragon native generation currently consumes raw text prompts through its own
      CLIP/T5 text stack.

    The training path in this repo adds a bridge-to-Neodragon-condition adapter so
    later inference can replace Neodragon's text stack with Mobile-OV bridge outputs.
    """

    latent_channels = 16

    def __init__(
        self,
        *,
        repo_path: str,
        cache_dir: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        mode: str = "hybrid",
        model_id: str = "karnewar/Neodragon",
        repo_url: str | None = None,
    ):
        super().__init__()
        repo_path, cache_dir, _ = ensure_neodragon_assets(
            repo_path=repo_path,
            cache_dir=cache_dir,
            model_id=model_id,
            repo_url=repo_url,
        )
        self.repo_path = Path(repo_path).expanduser().resolve()
        if not self.repo_path.exists():
            raise FileNotFoundError(f"Neodragon repo_path does not exist: {self.repo_path}")
        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))

        from neodragon import NeodragonPipeline

        self.cache_dir = str(Path(cache_dir).expanduser())
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.model_id = model_id
        self.pipeline = NeodragonPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            mode=mode,
            cache_dir=self.cache_dir,
        ).to(device)
        for module_name in ["text_encoder_bundle", "context_adapter", "dit", "vae", "first_frame_gen_pipeline"]:
            module = getattr(self.pipeline, module_name, None)
            if hasattr(module, "eval"):
                module.eval()

    @torch.no_grad()
    def generate_video_from_prompt(
        self,
        prompt: str,
        *,
        height: int = 320,
        width: int = 512,
        num_frames: int = 49,
        prompt_modifier: str | None = None,
        profile: bool = False,
        **kwargs: Any,
    ):
        """Generate a PIL-frame video using Neodragon's native text path."""
        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda", dtype=self.dtype):
            return self.pipeline(
                prompt=str(prompt),
                height=int(height),
                width=int(width),
                num_frames=int(num_frames),
                prompt_modifier=prompt_modifier,
                profile=profile,
                **kwargs,
            )

    @torch.no_grad()
    def generate_video_from_bridge_condition(
        self,
        prompt: str,
        *,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        height: int = 320,
        width: int = 512,
        num_frames: int = 49,
        prompt_modifier: str | None = None,
        negative_prompt: str | None = None,
        profile: bool = False,
        **kwargs: Any,
    ):
        """Generate video with Mobile-OV bridge outputs replacing Neodragon text conditioning.

        In hybrid mode, Neodragon still uses its SSD first-frame generator from raw text.
        The autoregressive video DiT receives bridge-conditioned tensors directly,
        without Neodragon's TextEncoderBundle or ContextAdapter.
        """
        if self.mode != "hybrid":
            raise NotImplementedError("Bridge-conditioned smoke path currently supports Neodragon hybrid mode.")
        from neodragon.utils.generation_utils import DEFAULT_PROMPT_MODIFIER, generate

        prompt_modifier = DEFAULT_PROMPT_MODIFIER if prompt_modifier is None else prompt_modifier
        text_encoder = _PrecomputedNeodragonTextEncoder(
            prompt_embeds.to(device=self.device, dtype=self.dtype),
            prompt_mask.to(device=self.device),
            pooled_prompt_embeds.to(device=self.device, dtype=self.dtype),
        )
        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda", dtype=self.dtype):
            first_frame = self.pipeline.first_frame_gen_pipeline(
                prompt=str(prompt) + prompt_modifier,
                num_images_per_prompt=1,
            ).images[0]
            return generate(
                text_encoder_bundle=text_encoder,
                dit=self.pipeline.dit,
                context_adapter=nn.Identity(),
                vae=self.pipeline.vae,
                scheduler=self.pipeline.scheduler,
                prompt=str(prompt),
                image=first_frame,
                height=int(height),
                width=int(width),
                num_frames=int(num_frames),
                prompt_modifier=prompt_modifier,
                negative_prompt=negative_prompt,
                frames_per_unit=self.pipeline.config.frames_per_unit,
                num_stages=len(self.pipeline.config.stages),
                output_type="pil",
                profile=profile,
                device=self.device,
                dtype=self.dtype,
                **{**self.pipeline.config.gen_confs["hybrid"], **kwargs},
            )

    @torch.no_grad()
    def encode_neodragon_text(self, prompts: list[str]):
        """Return Neodragon native text conditions before context adaptation."""
        return self.pipeline.text_encoder_bundle(prompts, self.device)

    @torch.no_grad()
    def encode_neodragon_context(self, prompts: list[str]):
        """Return Neodragon text conditions after the context adapter."""
        prompt_embeds, prompt_mask, pooled = self.encode_neodragon_text(prompts)
        prompt_embeds = self.pipeline.context_adapter(prompt_embeds)
        return prompt_embeds, prompt_mask, pooled

    @torch.no_grad()
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
        raise NotImplementedError(
            "Neodragon is a full video generation backend, not a WanVAE anchor backend. "
            "Use generate_video_from_prompt() for native Neodragon inference, or train "
            "the bridge-to-Neodragon-condition adapter before bypassing Neodragon text encoders."
        )
