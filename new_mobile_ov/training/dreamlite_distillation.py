from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

from new_mobile_ov.bridge.dreamlite_image_bridge import DreamLiteCondition, DreamLiteMode
from new_mobile_ov.checkpoints import ensure_dreamlite_checkpoint
from new_mobile_ov.config import DreamLiteConfig
from new_mobile_ov.generation.backends.dreamlite import DreamLiteMobileBackend
from new_mobile_ov.training.neodragon_objectives import (
    flat_cosine_distance,
    pooled_cosine,
)


@dataclass
class DreamLiteFunctionalResult:
    relative_mse: torch.Tensor
    cosine: torch.Tensor
    call_index: int


@dataclass
class DreamLiteClosedLoopResult:
    prediction_relative_mse: torch.Tensor
    prediction_cosine: torch.Tensor
    transition_relative_mse: torch.Tensor
    transition_cosine: torch.Tensor
    terminal_relative_mse: torch.Tensor
    calls: int


def relative_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = (prediction.float() - target.float()).pow(2).mean()
    denominator = target.float().pow(2).mean().clamp_min(1e-6)
    return numerator / denominator


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _resize_valid_tokens(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    output_length: int,
) -> torch.Tensor:
    resized = []
    for item, item_mask in zip(tokens, mask):
        valid = item[item_mask.bool()]
        if valid.shape[0] == 0:
            valid = item[:1]
        value = F.interpolate(
            valid.float().transpose(0, 1).unsqueeze(0),
            size=output_length,
            mode="linear",
            align_corners=False,
        )
        resized.append(value.squeeze(0).transpose(0, 1))
    return torch.stack(resized, dim=0).to(dtype=tokens.dtype)


def dreamlite_representation_losses(
    student: DreamLiteCondition,
    teacher: DreamLiteCondition,
) -> dict[str, torch.Tensor]:
    """Align unlike tokenizers without pretending their token indices are equal."""

    target = _resize_valid_tokens(
        teacher.prompt_embeds,
        teacher.attention_mask,
        student.prompt_embeds.shape[1],
    )
    student_fp32 = student.prompt_embeds.float()
    target_fp32 = target.float()
    student_normalized = F.layer_norm(student_fp32, (student_fp32.shape[-1],))
    target_normalized = F.layer_norm(target_fp32, (target_fp32.shape[-1],))
    student_pool = student_fp32.mean(dim=1)
    teacher_pool = _masked_mean(
        teacher.prompt_embeds.float(),
        teacher.attention_mask,
    )
    if student_fp32.shape[0] > 1:
        student_rel = F.normalize(student_pool, dim=-1) @ F.normalize(student_pool, dim=-1).T
        teacher_rel = F.normalize(teacher_pool, dim=-1) @ F.normalize(teacher_pool, dim=-1).T
        geometry = F.mse_loss(student_rel, teacher_rel)
    else:
        geometry = student_fp32.new_zeros(())
    student_std = student_pool.std(dim=0, unbiased=False).mean()
    teacher_std = teacher_pool.std(dim=0, unbiased=False).mean()
    return {
        "token_normalized_mse": F.mse_loss(student_normalized, target_normalized),
        "token_cosine": 1.0
        - F.cosine_similarity(student_fp32, target_fp32, dim=-1).mean(),
        "token_norm": (
            (
                student_fp32.norm(dim=-1).mean()
                - target_fp32.norm(dim=-1).mean()
            )
            / target_fp32.norm(dim=-1).mean().clamp_min(1e-4)
        ).pow(2),
        "pooled_normalized_mse": F.mse_loss(
            F.layer_norm(student_pool, (student_pool.shape[-1],)),
            F.layer_norm(teacher_pool, (teacher_pool.shape[-1],)),
        ),
        "pooled_cosine": pooled_cosine(student_pool, teacher_pool),
        "geometry": geometry,
        "variance": (
            (student_std - teacher_std) / teacher_std.clamp_min(1e-4)
        ).pow(2),
    }


class DreamLiteFrozenQwenTeacher:
    """Native BF16 Qwen3-VL condition used only while distilling the bridge."""

    def __init__(
        self,
        cfg: DreamLiteConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

        checkpoint = Path(
            ensure_dreamlite_checkpoint(
                cfg.checkpoint_dir,
                model_id=cfg.model_id,
                revision=cfg.revision,
            )
        )
        self.device = device
        self.dtype = dtype
        self.text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            checkpoint / "text_encoder",
            torch_dtype=dtype,
            local_files_only=True,
        ).to(device=device, dtype=dtype)
        self.text_encoder.eval().requires_grad_(False)
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            checkpoint / "tokenizer",
            local_files_only=True,
        )
        self.processor = Qwen3VLProcessor.from_pretrained(
            checkpoint / "processor",
            local_files_only=True,
        )

    @staticmethod
    def _condition_from_outputs(outputs, attention_mask: torch.Tensor, drop_idx: int, dtype):
        hidden = outputs.hidden_states[-1]
        valid = [row[row_mask.bool()][drop_idx:] for row, row_mask in zip(hidden, attention_mask)]
        if any(item.shape[0] == 0 for item in valid):
            raise RuntimeError("DreamLite teacher prefix consumed the full condition sequence.")
        prompt_embeds = pad_sequence(valid, batch_first=True, padding_value=0).to(dtype=dtype)
        mask = torch.zeros(
            prompt_embeds.shape[:2],
            dtype=torch.long,
            device=prompt_embeds.device,
        )
        for index, item in enumerate(valid):
            mask[index, : item.shape[0]] = 1
        return DreamLiteCondition(prompt_embeds, mask)

    @torch.no_grad()
    def encode(
        self,
        prompts: Sequence[str],
        *,
        mode: DreamLiteMode,
        images: Sequence[Image.Image] | None = None,
    ) -> DreamLiteCondition:
        prompts = [" ".join(str(value).strip().split()) for value in prompts]
        if mode == "generate":
            template = (
                "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
                "texture, quantity, text, spatial relationships of the objects and background:"
                "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            )
            text = [template.format(f"[Generate]: {prompt}") for prompt in prompts]
            encoded = self.tokenizer(text=text, padding=True, return_tensors="pt").to(self.device)
            outputs = self.text_encoder(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                output_hidden_states=True,
            )
            return self._condition_from_outputs(
                outputs,
                encoded.attention_mask,
                34,
                self.dtype,
            )
        if mode != "edit":
            raise ValueError(f"Unsupported DreamLite teacher mode: {mode}")
        if images is None or len(images) != len(prompts):
            raise ValueError("Edit teacher requires one source image per prompt.")
        template = (
            "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, "
            "texture, objects, background), then explain how the user's text instruction should alter "
            "or modify the image. Generate a new image that meets the user's requirements while "
            "maintaining consistency with the original input where appropriate.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        instructions = [
            (
                "[Edit]: A diptych with two side-by-side images of the same scene. Compared to the "
                f"right side, the left one has {prompt}"
            )
            for prompt in prompts
        ]
        encoded = self.processor(
            text=[template.format(value) for value in instructions],
            images=[image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS) for image in images],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.text_encoder(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            pixel_values=encoded.pixel_values,
            image_grid_thw=encoded.image_grid_thw,
            output_hidden_states=True,
        )
        return self._condition_from_outputs(
            outputs,
            encoded.attention_mask,
            64,
            self.dtype,
        )


class DreamLiteFrozenController:
    """Frozen four-call DreamLite denoiser used for functional supervision."""

    def __init__(
        self,
        cfg: DreamLiteConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.backend = DreamLiteMobileBackend(
            cfg,
            device=device,
            dtype=dtype,
            load_vae=True,
        )
        self.num_steps = int(cfg.num_inference_steps)

    def source_latents(
        self,
        images: Sequence[Image.Image] | None,
        *,
        batch_size: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if images is None:
            return self.backend.random_latents(batch_size, height, width).zero_()
        return self.backend.encode_source_images(images, height=height, width=width)

    def functional_loss(
        self,
        student: DreamLiteCondition,
        teacher: DreamLiteCondition,
        *,
        source_images: Sequence[Image.Image] | None,
        height: int,
        width: int,
        batch_size: int,
    ) -> DreamLiteFunctionalResult:
        effect_batch = min(batch_size, student.prompt_embeds.shape[0])
        student = DreamLiteCondition(
            student.prompt_embeds[:effect_batch],
            student.attention_mask[:effect_batch],
        )
        teacher = DreamLiteCondition(
            teacher.prompt_embeds[:effect_batch],
            teacher.attention_mask[:effect_batch],
        )
        images = None if source_images is None else list(source_images[:effect_batch])
        initial = self.backend.random_latents(effect_batch, height, width)
        source = self.source_latents(
            images,
            batch_size=effect_batch,
            height=height,
            width=width,
        )
        scheduler = self.backend.make_scheduler()
        timesteps = self.backend.prepare_schedule(
            scheduler,
            initial,
            num_steps=self.num_steps,
        )
        call_index = int(torch.randint(len(timesteps), (1,), device=initial.device).item())
        timestep = timesteps[call_index]
        with torch.no_grad():
            teacher_prediction = self.backend.predict(
                initial,
                timestep,
                teacher,
                source_latents=source,
                height=height,
                width=width,
            )
        student_prediction = self.backend.predict(
            initial,
            timestep,
            student,
            source_latents=source,
            height=height,
            width=width,
        )
        return DreamLiteFunctionalResult(
            relative_mse=relative_mse(student_prediction, teacher_prediction),
            cosine=flat_cosine_distance(student_prediction, teacher_prediction),
            call_index=call_index,
        )

    def closed_loop_loss(
        self,
        student: DreamLiteCondition,
        teacher: DreamLiteCondition,
        *,
        source_images: Sequence[Image.Image] | None,
        height: int,
        width: int,
        batch_size: int,
    ) -> DreamLiteClosedLoopResult:
        effect_batch = min(batch_size, student.prompt_embeds.shape[0])
        student = DreamLiteCondition(
            student.prompt_embeds[:effect_batch],
            student.attention_mask[:effect_batch],
        )
        teacher = DreamLiteCondition(
            teacher.prompt_embeds[:effect_batch],
            teacher.attention_mask[:effect_batch],
        )
        images = None if source_images is None else list(source_images[:effect_batch])
        initial = self.backend.random_latents(effect_batch, height, width)
        source = self.source_latents(
            images,
            batch_size=effect_batch,
            height=height,
            width=width,
        )
        teacher_scheduler = self.backend.make_scheduler()
        student_scheduler = self.backend.make_scheduler()
        teacher_timesteps = self.backend.prepare_schedule(
            teacher_scheduler,
            initial,
            num_steps=self.num_steps,
        )
        student_timesteps = self.backend.prepare_schedule(
            student_scheduler,
            initial,
            num_steps=self.num_steps,
        )
        teacher_current = initial.detach().clone()
        student_current = initial.clone()
        prediction_mse = initial.new_zeros((), dtype=torch.float32)
        prediction_cosine = initial.new_zeros((), dtype=torch.float32)
        transition_mse = initial.new_zeros((), dtype=torch.float32)
        transition_cosine = initial.new_zeros((), dtype=torch.float32)
        for teacher_timestep, student_timestep in zip(teacher_timesteps, student_timesteps):
            with torch.no_grad():
                teacher_prediction = self.backend.predict(
                    teacher_current,
                    teacher_timestep,
                    teacher,
                    source_latents=source,
                    height=height,
                    width=width,
                )
                teacher_next = teacher_scheduler.step(
                    teacher_prediction,
                    teacher_timestep,
                    teacher_current,
                    return_dict=False,
                )[0]
            student_prediction = self.backend.predict(
                student_current,
                student_timestep,
                student,
                source_latents=source,
                height=height,
                width=width,
            )
            student_next = student_scheduler.step(
                student_prediction,
                student_timestep,
                student_current,
                return_dict=False,
            )[0]
            prediction_mse = prediction_mse + relative_mse(
                student_prediction,
                teacher_prediction,
            )
            prediction_cosine = prediction_cosine + flat_cosine_distance(
                student_prediction,
                teacher_prediction,
            )
            transition_mse = transition_mse + relative_mse(student_next, teacher_next)
            transition_cosine = transition_cosine + flat_cosine_distance(
                student_next,
                teacher_next,
            )
            teacher_current = teacher_next
            student_current = student_next
        calls = len(teacher_timesteps)
        return DreamLiteClosedLoopResult(
            prediction_relative_mse=prediction_mse / calls,
            prediction_cosine=prediction_cosine / calls,
            transition_relative_mse=transition_mse / calls,
            transition_cosine=transition_cosine / calls,
            terminal_relative_mse=relative_mse(student_current, teacher_current),
            calls=calls,
        )
