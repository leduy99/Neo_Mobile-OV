from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from new_mobile_ov.bridge.ssd1b_image_bridge import SSD1BImageCondition
from new_mobile_ov.checkpoints import ensure_neodragon_assets
from new_mobile_ov.training.neodragon_objectives import (
    flat_cosine_distance,
    masked_token_cosine,
    masked_token_mse,
    pooled_cosine,
    token_norm_alignment,
)


SSD1B_TIMESTEPS = (999, 749, 499, 249)
SSD1B_HEIGHT = 640
SSD1B_WIDTH = 1024


class SSD1BTeacherCondition(NamedTuple):
    clip_l_tokens: torch.Tensor
    clip_big_g_tokens: torch.Tensor
    pooled: torch.Tensor

    @property
    def prompt_embeds(self) -> torch.Tensor:
        return torch.cat([self.clip_l_tokens, self.clip_big_g_tokens], dim=-1)


@dataclass
class SSD1BFunctionalResult:
    mse: torch.Tensor
    cosine: torch.Tensor
    timestep: int


@dataclass
class SSD1BRolloutResult:
    prediction_mse: torch.Tensor
    prediction_cosine: torch.Tensor
    transition_mse: torch.Tensor
    per_step_mse: tuple[torch.Tensor, ...]
    calls: int


def _resolve_ssd1b_model_path(cfg) -> tuple[Path, Path]:
    repo_path, _, model_path = ensure_neodragon_assets(
        repo_path=cfg.backend.extra.get("repo_path"),
        cache_dir=cfg.backend.extra.get("cache_dir"),
        model_id=cfg.backend.extra.get("model_id", "karnewar/Neodragon"),
        repo_url=cfg.backend.extra.get("repo_url"),
    )
    repo_path = Path(repo_path).expanduser().resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    return repo_path, Path(model_path)


class SSD1BFrozenTeacher:
    """Native SSD1B CLIP-L and CLIP-bigG encoders used only during training."""

    def __init__(self, cfg, device: torch.device, dtype: torch.dtype) -> None:
        _, model_path = _resolve_ssd1b_model_path(cfg)
        from neodragon.first_frame_gen import (
            SSD1B_TEXT_ENCODER_2_ID,
            SSD1B_TEXT_ENCODER_ID,
            SSD1B_TOKENIZER_2_ID,
            SSD1B_TOKENIZER_ID,
        )
        from transformers import (
            CLIPTextModel,
            CLIPTextModelWithProjection,
            CLIPTokenizer,
        )

        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path / SSD1B_TOKENIZER_ID)
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(model_path / SSD1B_TOKENIZER_2_ID)
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_path / SSD1B_TEXT_ENCODER_ID,
            torch_dtype=dtype,
        ).to(device).eval()
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_path / SSD1B_TEXT_ENCODER_2_ID,
            torch_dtype=dtype,
        ).to(device).eval()
        for module in (self.text_encoder, self.text_encoder_2):
            module.requires_grad_(False)

    @torch.no_grad()
    def encode(self, prompts: list[str]) -> SSD1BTeacherCondition:
        token_tensors: list[torch.Tensor] = []
        pooled = None
        for tokenizer, encoder in (
            (self.tokenizer, self.text_encoder),
            (self.tokenizer_2, self.text_encoder_2),
        ):
            encoded = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            output = encoder(
                encoded.input_ids.to(self.device),
                output_hidden_states=True,
                return_dict=True,
            )
            token_tensors.append(output.hidden_states[-2])
            if output[0].dim() == 2:
                pooled = output[0]
        if pooled is None:
            raise RuntimeError("SSD1B CLIP-bigG teacher did not return pooled text embeddings.")
        return SSD1BTeacherCondition(
            clip_l_tokens=token_tensors[0],
            clip_big_g_tokens=token_tensors[1],
            pooled=pooled,
        )


class SSD1BFrozenUNetController:
    """Frozen SSD1B UNet used for one-step and four-step functional distillation."""

    def __init__(self, cfg, device: torch.device, dtype: torch.dtype) -> None:
        _, model_path = _resolve_ssd1b_model_path(cfg)
        from diffusers import LCMScheduler, UNet2DConditionModel
        from neodragon.first_frame_gen import SSD1B_UNET_ID

        self.device = device
        self.dtype = dtype
        self.unet = UNet2DConditionModel.from_pretrained(
            model_path / SSD1B_UNET_ID,
            torch_dtype=dtype,
        ).to(device).eval()
        self.unet.requires_grad_(False)
        self.scheduler_class = LCMScheduler

    def _scheduler(self):
        scheduler = self.scheduler_class(
            set_alpha_to_one=False,
            original_inference_steps=len(SSD1B_TIMESTEPS),
            steps_offset=1,
        )
        scheduler.set_timesteps(
            timesteps=list(SSD1B_TIMESTEPS),
            device=self.device,
        )
        return scheduler

    def time_ids(self, batch_size: int) -> torch.Tensor:
        values = [
            SSD1B_HEIGHT,
            SSD1B_WIDTH,
            0,
            0,
            SSD1B_HEIGHT,
            SSD1B_WIDTH,
        ]
        return torch.tensor(values, device=self.device, dtype=self.dtype).unsqueeze(0).repeat(batch_size, 1)

    def predict(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        condition: SSD1BImageCondition | SSD1BTeacherCondition,
    ) -> torch.Tensor:
        return self.unet(
            latent,
            timestep,
            encoder_hidden_states=condition.prompt_embeds,
            added_cond_kwargs={
                "text_embeds": condition.pooled,
                "time_ids": self.time_ids(latent.shape[0]),
            },
            return_dict=False,
        )[0]

    def sample_latent(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return torch.randn(
            batch_size,
            4,
            SSD1B_HEIGHT // 8,
            SSD1B_WIDTH // 8,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )

    def functional_loss(
        self,
        student: SSD1BImageCondition,
        teacher: SSD1BTeacherCondition,
        *,
        batch_size: int,
        generator: torch.Generator | None = None,
    ) -> SSD1BFunctionalResult:
        effect_batch = min(batch_size, student.pooled.shape[0])
        latent = self.sample_latent(effect_batch, generator=generator)
        timestep_index = int(
            torch.randint(
                0,
                len(SSD1B_TIMESTEPS),
                (1,),
                device=self.device,
                generator=generator,
            ).item()
        )
        timestep_value = SSD1B_TIMESTEPS[timestep_index]
        timestep = torch.full(
            (effect_batch,),
            timestep_value,
            device=self.device,
            dtype=torch.long,
        )
        student_slice = slice_condition(student, effect_batch)
        teacher_slice = slice_condition(teacher, effect_batch)
        with torch.no_grad():
            teacher_prediction = self.predict(latent, timestep, teacher_slice)
        student_prediction = self.predict(latent, timestep, student_slice)
        return SSD1BFunctionalResult(
            mse=F.mse_loss(student_prediction.float(), teacher_prediction.float()),
            cosine=flat_cosine_distance(student_prediction, teacher_prediction),
            timestep=timestep_value,
        )

    def rollout_loss(
        self,
        student: SSD1BImageCondition,
        teacher: SSD1BTeacherCondition,
        *,
        batch_size: int,
        generator: torch.Generator | None = None,
    ) -> SSD1BRolloutResult:
        """Backpropagate through all four student-conditioned SSD1B UNet calls."""
        effect_batch = min(batch_size, student.pooled.shape[0])
        student_condition = slice_condition(student, effect_batch)
        teacher_condition = slice_condition(teacher, effect_batch)
        student_scheduler = self._scheduler()
        teacher_scheduler = self._scheduler()
        current = self.sample_latent(effect_batch, generator=generator)

        prediction_mse: list[torch.Tensor] = []
        prediction_cosine: list[torch.Tensor] = []
        transition_mse: list[torch.Tensor] = []
        for timestep in student_scheduler.timesteps:
            timestep_batch = timestep.expand(effect_batch)
            model_input = student_scheduler.scale_model_input(current, timestep)
            generator_state = generator.get_state() if generator is not None else None
            with torch.no_grad():
                teacher_prediction = self.predict(
                    model_input.detach(),
                    timestep_batch,
                    teacher_condition,
                )
                teacher_next = teacher_scheduler.step(
                    teacher_prediction,
                    timestep,
                    current.detach(),
                    generator=generator,
                    return_dict=False,
                )[0]
            if generator is not None and generator_state is not None:
                # LCM injects scheduler noise. Rewind so teacher and student
                # transitions receive exactly the same random sample.
                generator.set_state(generator_state)

            student_prediction = self.predict(
                model_input,
                timestep_batch,
                student_condition,
            )
            student_next = student_scheduler.step(
                student_prediction,
                timestep,
                current,
                generator=generator,
                return_dict=False,
            )[0]
            prediction_mse.append(
                F.mse_loss(student_prediction.float(), teacher_prediction.float())
            )
            prediction_cosine.append(
                flat_cosine_distance(student_prediction, teacher_prediction)
            )
            transition_mse.append(
                F.mse_loss(student_next.float(), teacher_next.float())
            )
            current = student_next

        return SSD1BRolloutResult(
            prediction_mse=torch.stack(prediction_mse).mean(),
            prediction_cosine=torch.stack(prediction_cosine).mean(),
            transition_mse=torch.stack(transition_mse).mean(),
            per_step_mse=tuple(prediction_mse),
            calls=len(prediction_mse),
        )


def slice_condition(
    condition: SSD1BImageCondition | SSD1BTeacherCondition,
    batch_size: int,
):
    values = (
        condition.clip_l_tokens[:batch_size],
        condition.clip_big_g_tokens[:batch_size],
        condition.pooled[:batch_size],
    )
    if isinstance(condition, SSD1BImageCondition):
        return SSD1BImageCondition(*values)
    return SSD1BTeacherCondition(*values)


def pooled_relational_cosine(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape[0] < 2:
        return prediction.new_zeros(())
    prediction = F.normalize(prediction.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    return F.mse_loss(prediction @ prediction.T, target @ target.T)


def _gather_for_geometry(value: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    from torch.distributed.nn.functional import all_gather

    return torch.cat(tuple(all_gather(value)), dim=0)


def global_relational_cosine(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Preserve prompt geometry over the global DDP batch with gradient support."""
    prediction = _gather_for_geometry(prediction)
    target = _gather_for_geometry(target)
    return pooled_relational_cosine(prediction, target)


def ssd1b_representation_losses(
    student: SSD1BImageCondition,
    teacher: SSD1BTeacherCondition,
) -> dict[str, torch.Tensor]:
    batch, sequence_length = teacher.clip_l_tokens.shape[:2]
    mask = torch.ones(
        batch,
        sequence_length,
        device=teacher.clip_l_tokens.device,
        dtype=torch.long,
    )
    return {
        "clip_l_normalized_mse": masked_token_mse(
            student.clip_l_tokens,
            teacher.clip_l_tokens,
            mask,
            normalize_tokens=True,
        ),
        "clip_l_cosine": masked_token_cosine(
            student.clip_l_tokens,
            teacher.clip_l_tokens,
            mask,
        ),
        "clip_l_norm": token_norm_alignment(
            student.clip_l_tokens,
            teacher.clip_l_tokens,
            mask,
        ),
        "clip_big_g_normalized_mse": masked_token_mse(
            student.clip_big_g_tokens,
            teacher.clip_big_g_tokens,
            mask,
            normalize_tokens=True,
        ),
        "clip_big_g_cosine": masked_token_cosine(
            student.clip_big_g_tokens,
            teacher.clip_big_g_tokens,
            mask,
        ),
        "clip_big_g_norm": token_norm_alignment(
            student.clip_big_g_tokens,
            teacher.clip_big_g_tokens,
            mask,
        ),
        "pooled_mse": F.mse_loss(student.pooled.float(), teacher.pooled.float()),
        "pooled_cosine": pooled_cosine(student.pooled, teacher.pooled),
        "clip_l_geometry": global_relational_cosine(
            student.clip_l_tokens.float().mean(dim=1),
            teacher.clip_l_tokens.float().mean(dim=1),
        ),
        "clip_big_g_geometry": global_relational_cosine(
            student.clip_big_g_tokens.float().mean(dim=1),
            teacher.clip_big_g_tokens.float().mean(dim=1),
        ),
        "pooled_geometry": global_relational_cosine(student.pooled, teacher.pooled),
    }
