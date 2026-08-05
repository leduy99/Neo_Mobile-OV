from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

from new_mobile_ov.bridge.dreamlite_image_bridge import (
    DreamLiteCondition,
    DreamLiteMode,
    dreamlite_generation_texts,
)
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
    transition_relative_mse: torch.Tensor
    transition_cosine: torch.Tensor
    call_index: int
    state_source: str


@dataclass
class DreamLiteClosedLoopResult:
    prediction_relative_mse: torch.Tensor
    prediction_cosine: torch.Tensor
    transition_relative_mse: torch.Tensor
    transition_cosine: torch.Tensor
    terminal_relative_mse: torch.Tensor
    calls: int


@dataclass(frozen=True)
class DreamLiteResolutionBucket:
    width: int
    height: int
    time_id_width: int
    time_id_height: int
    weight: float

    @property
    def label(self) -> str:
        return (
            f"{self.width}x{self.height}"
            f"@{self.time_id_width}x{self.time_id_height}"
        )


def parse_dreamlite_resolution_buckets(value: str) -> list[DreamLiteResolutionBucket]:
    """Parse `actual_widthxheight@logical_widthxheight:weight` buckets."""

    buckets: list[DreamLiteResolutionBucket] = []
    for raw_item in str(value).split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        try:
            sizes, raw_weight = item.rsplit(":", 1)
            actual, logical = sizes.split("@", 1)
            width, height = (int(part) for part in actual.split("x", 1))
            time_id_width, time_id_height = (
                int(part) for part in logical.split("x", 1)
            )
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid DreamLite resolution bucket "
                f"{raw_item!r}; expected WIDTHxHEIGHT@TIME_WIDTHxTIME_HEIGHT:WEIGHT"
            ) from exc
        dimensions = (width, height, time_id_width, time_id_height)
        if any(dimension <= 0 or dimension % 8 != 0 for dimension in dimensions):
            raise ValueError(
                f"DreamLite resolution dimensions must be positive multiples of 8: {raw_item!r}"
            )
        if weight <= 0:
            raise ValueError(f"DreamLite resolution weight must be positive: {raw_item!r}")
        buckets.append(
            DreamLiteResolutionBucket(
                width=width,
                height=height,
                time_id_width=time_id_width,
                time_id_height=time_id_height,
                weight=weight,
            )
        )
    if not buckets:
        raise ValueError("At least one DreamLite resolution bucket is required.")
    return buckets


def relative_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = (prediction.float() - target.float()).pow(2).mean()
    denominator = target.float().pow(2).mean().clamp_min(1e-6)
    return numerator / denominator


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _masked_token_average(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _masked_feature_moments(
    tokens: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
    count = weights.sum(dim=1).clamp_min(1.0)
    mean = (tokens * weights).sum(dim=1) / count
    variance = ((tokens - mean.unsqueeze(1)).pow(2) * weights).sum(dim=1) / count
    return mean, variance.clamp_min(1e-8).sqrt()


def _relative_feature_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).pow(2).mean() / target.pow(2).mean().clamp_min(1e-6)


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


def dreamlite_direct_representation_losses(
    student: DreamLiteCondition,
    teacher: DreamLiteCondition,
) -> dict[str, torch.Tensor]:
    """Match native Qwen condition positions without sequence interpolation."""

    common_length = min(student.prompt_embeds.shape[1], teacher.prompt_embeds.shape[1])
    if common_length < 1:
        raise RuntimeError("DreamLite conditions must contain at least one token.")
    student_tokens = student.prompt_embeds[:, :common_length].float()
    teacher_tokens = teacher.prompt_embeds[:, :common_length].float()
    student_mask = student.attention_mask[:, :common_length].bool()
    teacher_mask = teacher.attention_mask[:, :common_length].bool()
    common_mask = student_mask & teacher_mask
    if not bool(common_mask.any()):
        raise RuntimeError("Student and teacher DreamLite conditions share no valid tokens.")

    student_normalized = F.layer_norm(student_tokens, (student_tokens.shape[-1],))
    teacher_normalized = F.layer_norm(teacher_tokens, (teacher_tokens.shape[-1],))
    token_mse = (student_normalized - teacher_normalized).pow(2).mean(dim=-1)
    token_cosine = 1.0 - F.cosine_similarity(student_tokens, teacher_tokens, dim=-1)
    student_norm = student_tokens.norm(dim=-1)
    teacher_norm = teacher_tokens.norm(dim=-1)
    token_norm = ((student_norm - teacher_norm) / teacher_norm.clamp_min(1e-4)).pow(2)

    student_pool = _masked_mean(student_tokens, common_mask)
    teacher_pool = _masked_mean(teacher_tokens, common_mask)
    student_mean, student_std = _masked_feature_moments(student_tokens, common_mask)
    teacher_mean, teacher_std = _masked_feature_moments(teacher_tokens, common_mask)
    if student_tokens.shape[0] > 1:
        student_rel = F.normalize(student_pool, dim=-1) @ F.normalize(student_pool, dim=-1).T
        teacher_rel = F.normalize(teacher_pool, dim=-1) @ F.normalize(teacher_pool, dim=-1).T
        geometry = F.mse_loss(student_rel, teacher_rel)
        prompt_variance = _relative_feature_mse(
            student_pool.std(dim=0, unbiased=False),
            teacher_pool.std(dim=0, unbiased=False),
        )
    else:
        geometry = student_tokens.new_zeros(())
        prompt_variance = student_tokens.new_zeros(())
    return {
        "token_normalized_mse": _masked_token_average(token_mse, common_mask),
        "token_cosine": _masked_token_average(token_cosine, common_mask),
        "token_norm": _masked_token_average(token_norm, common_mask),
        "pooled_normalized_mse": F.mse_loss(
            F.layer_norm(student_pool, (student_pool.shape[-1],)),
            F.layer_norm(teacher_pool, (teacher_pool.shape[-1],)),
        ),
        "pooled_cosine": pooled_cosine(student_pool, teacher_pool),
        "geometry": geometry,
        "variance": prompt_variance,
        "token_mean": _relative_feature_mse(student_mean, teacher_mean),
        "token_std": _relative_feature_mse(student_std, teacher_std),
        "mask_agreement": (student_mask == teacher_mask).float().mean(),
    }


def _content_and_wrapper_masks(
    common_mask: torch.Tensor,
    *,
    prefix_tokens: int,
    suffix_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split DreamLite's retained Qwen sequence into prompt and template tokens."""

    content_mask = torch.zeros_like(common_mask)
    for row_index, row_mask in enumerate(common_mask):
        valid_length = int(row_mask.sum().item())
        content_start = min(max(int(prefix_tokens), 0), valid_length)
        content_end = max(content_start, valid_length - max(int(suffix_tokens), 0))
        if content_end == content_start and valid_length > 0:
            content_start = min(content_start, valid_length - 1)
            content_end = content_start + 1
        content_mask[row_index, content_start:content_end] = True
    content_mask &= common_mask
    return content_mask, common_mask & ~content_mask


def _masked_direct_token_losses(
    student_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not bool(mask.any()):
        zero = student_tokens.new_zeros(())
        return zero, zero
    student_normalized = F.layer_norm(student_tokens, (student_tokens.shape[-1],))
    teacher_normalized = F.layer_norm(teacher_tokens, (teacher_tokens.shape[-1],))
    token_mse = (student_normalized - teacher_normalized).pow(2).mean(dim=-1)
    token_cosine = 1.0 - F.cosine_similarity(student_tokens, teacher_tokens, dim=-1)
    return (
        _masked_token_average(token_mse, mask),
        _masked_token_average(token_cosine, mask),
    )


def _semantic_contrastive_loss(
    student_pool: torch.Tensor,
    teacher_pool: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if student_pool.shape[0] < 2:
        return student_pool.new_zeros(())
    temperature = max(float(temperature), 1e-4)
    logits = F.normalize(student_pool, dim=-1) @ F.normalize(teacher_pool, dim=-1).T
    logits = logits / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.T, labels)
    )


def dreamlite_content_aware_representation_losses(
    student: DreamLiteCondition,
    teacher: DreamLiteCondition,
    *,
    prefix_tokens: int = 3,
    suffix_tokens: int = 5,
    contrastive_temperature: float = 0.07,
) -> dict[str, torch.Tensor]:
    """Prioritize prompt semantics while retaining weak template supervision."""

    common_length = min(student.prompt_embeds.shape[1], teacher.prompt_embeds.shape[1])
    if common_length < 1:
        raise RuntimeError("DreamLite conditions must contain at least one token.")
    student_tokens = student.prompt_embeds[:, :common_length].float()
    teacher_tokens = teacher.prompt_embeds[:, :common_length].float()
    student_mask = student.attention_mask[:, :common_length].bool()
    teacher_mask = teacher.attention_mask[:, :common_length].bool()
    common_mask = student_mask & teacher_mask
    if not bool(common_mask.any()):
        raise RuntimeError("Student and teacher DreamLite conditions share no valid tokens.")
    content_mask, wrapper_mask = _content_and_wrapper_masks(
        common_mask,
        prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
    )
    content_mse, content_cosine = _masked_direct_token_losses(
        student_tokens,
        teacher_tokens,
        content_mask,
    )
    wrapper_mse, wrapper_cosine = _masked_direct_token_losses(
        student_tokens,
        teacher_tokens,
        wrapper_mask,
    )
    student_content_pool = _masked_mean(student_tokens, content_mask)
    teacher_content_pool = _masked_mean(teacher_tokens, content_mask)
    return {
        "content_token_normalized_mse": content_mse,
        "content_token_cosine": content_cosine,
        "wrapper_token_normalized_mse": wrapper_mse,
        "wrapper_token_cosine": wrapper_cosine,
        "content_pooled_cosine": pooled_cosine(
            student_content_pool,
            teacher_content_pool,
        ),
        "semantic_contrastive": _semantic_contrastive_loss(
            student_content_pool,
            teacher_content_pool,
            temperature=contrastive_temperature,
        ),
        "content_fraction": content_mask.float().sum() / common_mask.float().sum().clamp_min(1.0),
        "mask_agreement": (student_mask == teacher_mask).float().mean(),
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
            text = dreamlite_generation_texts(prompts)
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

    def project_condition(self, condition: DreamLiteCondition) -> DreamLiteCondition:
        projected = self.backend.unet.process_encoder_hidden_states(
            condition.prompt_embeds.to(dtype=self.backend.dtype),
            added_cond_kwargs={},
        )
        return DreamLiteCondition(projected, condition.attention_mask)

    def functional_loss(
        self,
        student: DreamLiteCondition,
        teacher: DreamLiteCondition,
        *,
        source_images: Sequence[Image.Image] | None,
        height: int,
        width: int,
        time_id_height: int | None = None,
        time_id_width: int | None = None,
        batch_size: int,
        call_index: int | None = None,
        state_source: str = "teacher",
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
        teacher_scheduler = self.backend.make_scheduler()
        student_scheduler = self.backend.make_scheduler()
        timesteps = self.backend.prepare_schedule(
            teacher_scheduler,
            initial,
            num_steps=self.num_steps,
        )
        student_timesteps = self.backend.prepare_schedule(
            student_scheduler,
            initial,
            num_steps=self.num_steps,
        )
        if len(student_timesteps) != len(timesteps):
            raise RuntimeError("DreamLite teacher/student schedulers produced different call counts.")
        if call_index is None:
            call_index = int(
                torch.randint(len(timesteps), (1,), device=initial.device).item()
            )
        if not 0 <= call_index < len(timesteps):
            raise ValueError(
                f"call_index={call_index} is outside the {len(timesteps)}-call schedule"
            )
        if state_source not in {"teacher", "student"}:
            raise ValueError(f"Unsupported functional state_source={state_source!r}")
        prefix_condition = teacher if state_source == "teacher" else student
        # Both schedulers consume the same detached prefix predictions. At call
        # k teacher and student therefore see one identical state, including for
        # the on-policy student-state branch.
        shared_state = initial
        with torch.no_grad():
            for prefix_timestep in timesteps[:call_index]:
                prefix_prediction = self.backend.predict(
                    shared_state,
                    prefix_timestep,
                    prefix_condition,
                    source_latents=source,
                    height=height,
                    width=width,
                    time_id_height=time_id_height,
                    time_id_width=time_id_width,
                )
                next_state = teacher_scheduler.step(
                    prefix_prediction,
                    prefix_timestep,
                    shared_state,
                    return_dict=False,
                )[0]
                student_scheduler.step(
                    prefix_prediction,
                    prefix_timestep,
                    shared_state,
                    return_dict=False,
                )
                shared_state = next_state
        shared_state = shared_state.detach()
        timestep = timesteps[call_index]
        with torch.no_grad():
            teacher_prediction = self.backend.predict(
                shared_state,
                timestep,
                teacher,
                source_latents=source,
                height=height,
                width=width,
                time_id_height=time_id_height,
                time_id_width=time_id_width,
            )
        student_prediction = self.backend.predict(
            shared_state,
            timestep,
            student,
            source_latents=source,
            height=height,
            width=width,
            time_id_height=time_id_height,
            time_id_width=time_id_width,
        )
        with torch.no_grad():
            teacher_next = teacher_scheduler.step(
                teacher_prediction,
                timestep,
                shared_state,
                return_dict=False,
            )[0]
        student_next = student_scheduler.step(
            student_prediction,
            student_timesteps[call_index],
            shared_state,
            return_dict=False,
        )[0]
        return DreamLiteFunctionalResult(
            relative_mse=relative_mse(student_prediction, teacher_prediction),
            cosine=flat_cosine_distance(student_prediction, teacher_prediction),
            transition_relative_mse=relative_mse(student_next, teacher_next),
            transition_cosine=flat_cosine_distance(student_next, teacher_next),
            call_index=call_index,
            state_source=state_source,
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
