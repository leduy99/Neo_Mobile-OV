from __future__ import annotations

from copy import deepcopy
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from new_mobile_ov.bridge.dreamlite_image_bridge import DreamLiteCondition
from new_mobile_ov.checkpoints import ensure_dreamlite_checkpoint
from new_mobile_ov.config import DreamLiteConfig


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
) -> float:
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return image_seq_len * slope + base_shift - slope * base_seq_len


class DreamLiteMobileBackend(nn.Module):
    """Self-contained DreamLite-mobile denoiser driven by external conditions."""

    def __init__(
        self,
        cfg: DreamLiteConfig,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        load_vae: bool = True,
    ) -> None:
        super().__init__()
        try:
            from diffusers import AutoencoderTiny, DreamLiteUNetModel, FlowMatchEulerDiscreteScheduler
            from diffusers.image_processor import VaeImageProcessor
        except ImportError as exc:
            raise ImportError(
                "DreamLite requires diffusers==0.39.0. Run "
                "bash scripts/install_dreamlite_dependencies.sh first."
            ) from exc

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = ensure_dreamlite_checkpoint(
            cfg.checkpoint_dir,
            model_id=cfg.model_id,
            revision=cfg.revision,
        )
        self.cfg = cfg
        self.checkpoint = checkpoint
        self.unet = DreamLiteUNetModel.from_pretrained(
            checkpoint,
            subfolder="unet",
            torch_dtype=dtype,
            local_files_only=True,
        ).to(device=device, dtype=dtype)
        self.unet.eval().requires_grad_(False)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            checkpoint,
            subfolder="scheduler",
            local_files_only=True,
        )
        self.vae = None
        self.image_processor = None
        self.vae_scale_factor = 8
        if load_vae:
            self.vae = AutoencoderTiny.from_pretrained(
                checkpoint,
                subfolder="vae",
                torch_dtype=dtype,
                local_files_only=True,
            ).to(device=device, dtype=dtype)
            self.vae.eval().requires_grad_(False)
            if hasattr(self.vae.config, "encoder_block_out_channels"):
                self.vae_scale_factor = 2 ** (len(self.vae.config.encoder_block_out_channels) - 1)
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

    @property
    def device(self) -> torch.device:
        return next(self.unet.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.unet.parameters()).dtype

    def train(self, mode: bool = True):
        super().train(False)
        self.unet.eval()
        if self.vae is not None:
            self.vae.eval()
        return self

    def make_scheduler(self):
        return self.scheduler.__class__.from_config(deepcopy(dict(self.scheduler.config)))

    def prepare_schedule(
        self,
        scheduler,
        latent: torch.Tensor,
        *,
        num_steps: int,
    ) -> torch.Tensor:
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        image_seq_len = latent.shape[-2] * latent.shape[-1] // 4
        config = scheduler.config
        mu = calculate_shift(
            image_seq_len,
            config.get("base_image_seq_len", 256),
            config.get("max_image_seq_len", 4096),
            config.get("base_shift", 0.5),
            config.get("max_shift", 1.16),
        )
        scheduler.set_timesteps(sigmas=sigmas, device=latent.device, mu=mu)
        return scheduler.timesteps

    def random_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        channels = int(getattr(getattr(self.vae, "config", None), "latent_channels", 4))
        return torch.randn(
            batch_size,
            channels,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )

    def encode_source_images(
        self,
        images: Sequence[Image.Image],
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if self.vae is None or self.image_processor is None:
            raise RuntimeError("DreamLite VAE was not loaded.")
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
            retrieve_latents,
        )

        processed = self.image_processor.preprocess(
            [image.convert("RGB") for image in images],
            height=height,
            width=width,
        ).to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            return retrieve_latents(self.vae.encode(processed), sample_mode="argmax")

    def predict(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        condition: DreamLiteCondition,
        *,
        source_latents: torch.Tensor | None = None,
        height: int,
        width: int,
        time_id_height: int | None = None,
        time_id_width: int | None = None,
    ) -> torch.Tensor:
        if source_latents is None:
            source_latents = torch.zeros_like(latents)
        model_input = torch.cat([latents, source_latents], dim=3)
        time_id_height = int(time_id_height or height)
        time_id_width = int(time_id_width or width)
        time_ids = torch.tensor(
            [[time_id_width, time_id_height]],
            device=latents.device,
            dtype=latents.dtype,
        ).expand(latents.shape[0], -1)
        prediction = self.unet(
            model_input,
            timestep=timestep.expand(model_input.shape[0]).to(latents.dtype),
            encoder_hidden_states=condition.prompt_embeds.to(dtype=latents.dtype),
            encoder_attention_mask=condition.attention_mask,
            added_cond_kwargs={"time_ids": time_ids},
            return_dict=False,
        )[0]
        return prediction[..., : latents.shape[-1]]

    def decode(self, latents: torch.Tensor) -> list[Image.Image]:
        if self.vae is None or self.image_processor is None:
            raise RuntimeError("DreamLite VAE was not loaded.")
        shift = getattr(self.vae.config, "shift_factor", 0.0)
        scaled = (latents / self.vae.config.scaling_factor) + shift
        with torch.no_grad():
            image = self.vae.decode(scaled, return_dict=False)[0]
        return self.image_processor.postprocess(image, output_type="pil")

    @torch.no_grad()
    def generate_images(
        self,
        condition: DreamLiteCondition,
        *,
        source_images: Sequence[Image.Image] | None = None,
        height: int | None = None,
        width: int | None = None,
        time_id_height: int | None = None,
        time_id_width: int | None = None,
        num_steps: int | None = None,
        seed: int = 0,
    ) -> list[Image.Image]:
        height = int(height or self.cfg.height)
        width = int(width or self.cfg.width)
        num_steps = int(num_steps or self.cfg.num_inference_steps)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        latents = self.random_latents(
            condition.prompt_embeds.shape[0],
            height,
            width,
            generator=generator,
        )
        source_latents = None
        if source_images is not None:
            source_latents = self.encode_source_images(
                source_images,
                height=height,
                width=width,
            )
        scheduler = self.make_scheduler()
        timesteps = self.prepare_schedule(scheduler, latents, num_steps=num_steps)
        for timestep in timesteps:
            prediction = self.predict(
                latents,
                timestep,
                condition,
                source_latents=source_latents,
                height=height,
                width=width,
                time_id_height=time_id_height,
                time_id_width=time_id_width,
            )
            latents = scheduler.step(prediction, timestep, latents, return_dict=False)[0]
        return self.decode(latents)
