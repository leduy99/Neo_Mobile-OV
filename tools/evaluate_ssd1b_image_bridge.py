#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, NamedTuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.bridge import MobileOVSSD1BImageBridge, SSD1BImageCondition  # noqa: E402
from new_mobile_ov.checkpoints import ensure_neodragon_assets  # noqa: E402
from new_mobile_ov.config import load_config  # noqa: E402
from new_mobile_ov.training.ssd1b_distillation import (  # noqa: E402
    SSD1BFrozenTeacher,
    SSD1BTeacherCondition,
    SSD1B_TIMESTEPS,
)


class ConditionPair(NamedTuple):
    student: SSD1BImageCondition
    native: SSD1BTeacherCondition


def dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def read_prompts(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return list(dict.fromkeys(value for value in values if value))


def prompt_bucket(prompt: str) -> str:
    words = len(prompt.split())
    if words <= 5:
        return "short"
    if words <= 12:
        return "medium"
    return "long"


def stratified_sample(
    prompts: list[str],
    count: int,
    *,
    seed: int,
) -> list[str]:
    if count < 0 or count >= len(prompts):
        return list(prompts)
    rng = random.Random(seed)
    buckets: dict[str, list[str]] = {"short": [], "medium": [], "long": []}
    for prompt in prompts:
        buckets[prompt_bucket(prompt)].append(prompt)
    for values in buckets.values():
        rng.shuffle(values)

    selected: list[str] = []
    target_per_bucket = count // len(buckets)
    for values in buckets.values():
        selected.extend(values[:target_per_bucket])
    remaining = count - len(selected)
    pool = [value for values in buckets.values() for value in values[target_per_bucket:]]
    rng.shuffle(pool)
    selected.extend(pool[:remaining])
    rng.shuffle(selected)
    return selected


def safe_stem(text: str, max_len: int = 70) -> str:
    value = re.sub(r"[^a-zA-Z0-9._ -]+", "_", text).strip().replace(" ", "_")
    return (value[:max_len] or "prompt").strip("_")


def tensor_stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().reshape(-1).cpu()
    quantiles = torch.quantile(
        values,
        torch.tensor([0.1, 0.5, 0.9, 0.95]),
    )
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "p10": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "max": float(values.max()),
    }


def parameter_summary(module: torch.nn.Module) -> dict[str, float | int]:
    parameters = list(module.parameters())
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    total_numel = sum(parameter.numel() for parameter in parameters)
    trainable_numel = sum(parameter.numel() for parameter in trainable)
    total_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    trainable_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in trainable
    )
    return {
        "total_parameters": total_numel,
        "trainable_parameters": trainable_numel,
        "parameter_mib": total_bytes / 2**20,
        "trainable_parameter_mib": trainable_bytes / 2**20,
    }


def to_cpu_condition(condition):
    cls = type(condition)
    return cls(*(value.detach().cpu() for value in condition))


def slice_condition(condition, indices: list[int], device: torch.device, dtype: torch.dtype):
    index = torch.tensor(indices, dtype=torch.long)
    cls = type(condition)
    return cls(
        *(
            value.index_select(0, index).to(device=device, dtype=dtype)
            for value in condition
        )
    )


def cat_conditions(conditions: list):
    cls = type(conditions[0])
    return cls(*(torch.cat([value[index] for value in conditions], dim=0) for index in range(3)))


@torch.inference_mode()
def encode_all_conditions(
    bridge: MobileOVSSD1BImageBridge,
    teacher: SSD1BFrozenTeacher,
    prompts: list[str],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ConditionPair:
    student_batches: list[SSD1BImageCondition] = []
    native_batches: list[SSD1BTeacherCondition] = []
    for offset in range(0, len(prompts), batch_size):
        batch = prompts[offset : offset + batch_size]
        with torch.autocast("cuda", dtype=dtype):
            student_batches.append(to_cpu_condition(bridge(batch)))
            native_batches.append(to_cpu_condition(teacher.encode(batch)))
        print(
            f"Condition encoding: {min(offset + len(batch), len(prompts))}/{len(prompts)}",
            flush=True,
        )
    return ConditionPair(
        student=cat_conditions(student_batches),
        native=cat_conditions(native_batches),
    )


def retrieval_metrics(student: torch.Tensor, native: torch.Tensor) -> dict[str, float]:
    student = F.normalize(student.float(), dim=-1)
    native = F.normalize(native.float(), dim=-1)
    similarity = student @ native.T
    ranking = similarity.argsort(dim=1, descending=True)
    target = torch.arange(similarity.shape[0]).unsqueeze(1)
    reciprocal_rank = 1.0 / (
        (ranking == target).float().argmax(dim=1).float() + 1.0
    )
    return {
        "top1": float((ranking[:, :1] == target).any(dim=1).float().mean()),
        "top5": float((ranking[:, :5] == target).any(dim=1).float().mean()),
        "mean_reciprocal_rank": float(reciprocal_rank.mean()),
    }


def linear_cka(student: torch.Tensor, native: torch.Tensor) -> float:
    student = student.float() - student.float().mean(dim=0, keepdim=True)
    native = native.float() - native.float().mean(dim=0, keepdim=True)
    student_gram = student @ student.T
    native_gram = native @ native.T
    numerator = (student_gram * native_gram).sum()
    denominator = student_gram.square().sum().sqrt() * native_gram.square().sum().sqrt()
    return float((numerator / denominator.clamp_min(1e-12)).cpu())


def effective_rank(features: torch.Tensor) -> dict[str, float]:
    features = features.float() - features.float().mean(dim=0, keepdim=True)
    gram = features @ features.T
    eigenvalues = torch.linalg.eigvalsh(gram.double()).clamp_min(0)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    entropy_rank = torch.exp(-(probabilities * probabilities.log()).sum())
    participation = eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-12)
    return {
        "entropy_effective_rank": float(entropy_rank),
        "participation_ratio": float(participation),
        "top1_variance_ratio": float(eigenvalues[-1] / eigenvalues.sum()),
    }


def geometry_metrics(student: torch.Tensor, native: torch.Tensor) -> dict[str, float]:
    student = F.normalize(student.float(), dim=-1)
    native = F.normalize(native.float(), dim=-1)
    student_geometry = student @ student.T
    native_geometry = native @ native.T
    upper = torch.triu_indices(student.shape[0], student.shape[0], offset=1)
    student_offdiag = student_geometry[upper[0], upper[1]]
    native_offdiag = native_geometry[upper[0], upper[1]]
    student_centered = student_offdiag - student_offdiag.mean()
    native_centered = native_offdiag - native_offdiag.mean()
    correlation = (student_centered * native_centered).sum() / (
        student_centered.square().sum().sqrt()
        * native_centered.square().sum().sqrt()
    ).clamp_min(1e-12)
    return {
        "pairwise_cosine_mae": float(
            (student_offdiag - native_offdiag).abs().mean()
        ),
        "pairwise_cosine_pearson": float(correlation),
        "student_offdiag_mean": float(student_offdiag.mean()),
        "student_offdiag_std": float(student_offdiag.std(unbiased=False)),
        "native_offdiag_mean": float(native_offdiag.mean()),
        "native_offdiag_std": float(native_offdiag.std(unbiased=False)),
    }


def stream_condition_metrics(
    student_tokens: torch.Tensor,
    native_tokens: torch.Tensor,
) -> dict[str, object]:
    student = student_tokens.float()
    native = native_tokens.float()
    token_cosine_distance = 1.0 - F.cosine_similarity(student, native, dim=-1)
    normalized_mse = (
        F.normalize(student, dim=-1) - F.normalize(native, dim=-1)
    ).square().mean(dim=-1)
    raw_mse = (student - native).square().mean(dim=-1)
    student_norm = student.norm(dim=-1)
    native_norm = native.norm(dim=-1)
    student_global = student.mean(dim=1) if student.dim() == 3 else student
    native_global = native.mean(dim=1) if native.dim() == 3 else native
    result: dict[str, object] = {
        "cosine_distance": tensor_stats(token_cosine_distance),
        "normalized_mse": tensor_stats(normalized_mse),
        "raw_mse": tensor_stats(raw_mse),
        "student_norm": tensor_stats(student_norm),
        "native_norm": tensor_stats(native_norm),
        "norm_ratio": tensor_stats(student_norm / native_norm.clamp_min(1e-8)),
        "retrieval": retrieval_metrics(student_global, native_global),
        "linear_cka": linear_cka(student_global, native_global),
        "student_effective_rank": effective_rank(student_global),
        "native_effective_rank": effective_rank(native_global),
        "geometry": geometry_metrics(student_global, native_global),
    }
    if student.dim() == 3:
        result["position_cosine_distance"] = (
            token_cosine_distance.mean(dim=0).tolist()
        )
    return result


@torch.inference_mode()
def benchmark_encoder(
    function: Callable[[list[str]], object],
    prompts: list[str],
    *,
    batch_sizes: list[int],
    repeats: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for batch_size in batch_sizes:
        batch = [prompts[index % len(prompts)] for index in range(batch_size)]
        for _ in range(2):
            output = function(batch)
            del output
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        start = time.perf_counter()
        for _ in range(repeats):
            output = function(batch)
            del output
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(device)
        results[str(batch_size)] = {
            "batch_ms": elapsed * 1000.0 / repeats,
            "ms_per_prompt": elapsed * 1000.0 / (repeats * batch_size),
            "temporary_peak_mib": max(0, peak - baseline) / 2**20,
        }
    return results


def prediction_metrics(student: torch.Tensor, native: torch.Tensor) -> dict[str, torch.Tensor]:
    reduce_dims = tuple(range(1, student.dim()))
    mse = (student.float() - native.float()).square().mean(dim=reduce_dims)
    native_rms = native.float().square().mean(dim=reduce_dims).sqrt()
    student_flat = student.float().flatten(1)
    native_flat = native.float().flatten(1)
    return {
        "mse": mse,
        "relative_rmse": mse.sqrt() / native_rms.clamp_min(1e-8),
        "cosine_distance": 1.0
        - F.cosine_similarity(student_flat, native_flat, dim=-1),
        "norm_ratio": student_flat.norm(dim=-1)
        / native_flat.norm(dim=-1).clamp_min(1e-8),
    }


def append_metric(
    target: dict[str, list[torch.Tensor]],
    values: dict[str, torch.Tensor],
) -> None:
    for name, value in values.items():
        target.setdefault(name, []).append(value.detach().cpu())


def finalize_metric_lists(values: dict[str, list[torch.Tensor]]) -> dict[str, object]:
    return {
        name: tensor_stats(torch.cat(parts))
        for name, parts in values.items()
    }


def unet_predict(pipe, latent, timestep, condition, time_ids):
    return pipe.unet(
        latent,
        timestep,
        encoder_hidden_states=condition.prompt_embeds,
        added_cond_kwargs={
            "text_embeds": condition.pooled,
            "time_ids": time_ids,
        },
        return_dict=False,
    )[0]


@torch.inference_mode()
def evaluate_functional_and_rollout(
    pipe,
    conditions: ConditionPair,
    indices: list[int],
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    functional: dict[int, dict[str, list[torch.Tensor]]] = {
        timestep: {} for timestep in SSD1B_TIMESTEPS
    }
    rollout_on_policy: dict[int, dict[str, list[torch.Tensor]]] = {
        index: {} for index in range(len(SSD1B_TIMESTEPS))
    }
    rollout_closed_loop: dict[int, dict[str, list[torch.Tensor]]] = {
        index: {} for index in range(len(SSD1B_TIMESTEPS))
    }

    for batch_offset in range(0, len(indices), batch_size):
        batch_indices = indices[batch_offset : batch_offset + batch_size]
        student = slice_condition(conditions.student, batch_indices, device, dtype)
        native = slice_condition(conditions.native, batch_indices, device, dtype)
        current_batch = len(batch_indices)
        generator = torch.Generator(device=device).manual_seed(seed + batch_offset)
        initial = torch.randn(
            current_batch,
            4,
            80,
            128,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        time_ids = torch.tensor(
            [640, 1024, 0, 0, 640, 1024],
            device=device,
            dtype=dtype,
        ).unsqueeze(0).repeat(current_batch, 1)

        with torch.autocast("cuda", dtype=dtype):
            for timestep_value in SSD1B_TIMESTEPS:
                timestep = torch.full(
                    (current_batch,),
                    timestep_value,
                    device=device,
                    dtype=torch.long,
                )
                native_prediction = unet_predict(
                    pipe,
                    initial,
                    timestep,
                    native,
                    time_ids,
                )
                student_prediction = unet_predict(
                    pipe,
                    initial,
                    timestep,
                    student,
                    time_ids,
                )
                append_metric(
                    functional[timestep_value],
                    prediction_metrics(student_prediction, native_prediction),
                )

            native_scheduler = copy.deepcopy(pipe.scheduler)
            student_scheduler = copy.deepcopy(pipe.scheduler)
            teacher_on_policy_scheduler = copy.deepcopy(pipe.scheduler)
            for scheduler in (
                native_scheduler,
                student_scheduler,
                teacher_on_policy_scheduler,
            ):
                scheduler.set_timesteps(
                    timesteps=list(SSD1B_TIMESTEPS),
                    device=device,
                )

            native_current = initial.clone()
            student_current = initial.clone()
            native_generator = torch.Generator(device=device).manual_seed(
                seed + 100000 + batch_offset
            )
            student_generator = torch.Generator(device=device).manual_seed(
                seed + 100000 + batch_offset
            )
            teacher_on_policy_generator = torch.Generator(device=device).manual_seed(
                seed + 100000 + batch_offset
            )
            for call_index, timestep_scalar in enumerate(native_scheduler.timesteps):
                timestep = timestep_scalar.expand(current_batch)
                native_input = native_scheduler.scale_model_input(
                    native_current,
                    timestep_scalar,
                )
                student_input = student_scheduler.scale_model_input(
                    student_current,
                    timestep_scalar,
                )
                native_prediction = unet_predict(
                    pipe,
                    native_input,
                    timestep,
                    native,
                    time_ids,
                )
                student_prediction = unet_predict(
                    pipe,
                    student_input,
                    timestep,
                    student,
                    time_ids,
                )
                teacher_on_student = unet_predict(
                    pipe,
                    student_input,
                    timestep,
                    native,
                    time_ids,
                )

                native_next = native_scheduler.step(
                    native_prediction,
                    timestep_scalar,
                    native_current,
                    generator=native_generator,
                    return_dict=False,
                )[0]
                student_next = student_scheduler.step(
                    student_prediction,
                    timestep_scalar,
                    student_current,
                    generator=student_generator,
                    return_dict=False,
                )[0]
                teacher_on_student_next = teacher_on_policy_scheduler.step(
                    teacher_on_student,
                    timestep_scalar,
                    student_current,
                    generator=teacher_on_policy_generator,
                    return_dict=False,
                )[0]

                on_policy_values = prediction_metrics(
                    student_prediction,
                    teacher_on_student,
                )
                transition_values = prediction_metrics(
                    student_next,
                    teacher_on_student_next,
                )
                append_metric(
                    rollout_on_policy[call_index],
                    {
                        **{
                            f"prediction_{name}": value
                            for name, value in on_policy_values.items()
                        },
                        **{
                            f"transition_{name}": value
                            for name, value in transition_values.items()
                        },
                    },
                )
                append_metric(
                    rollout_closed_loop[call_index],
                    prediction_metrics(student_next, native_next),
                )
                native_current = native_next
                student_current = student_next

        print(
            f"UNet/rollout evaluation: "
            f"{min(batch_offset + current_batch, len(indices))}/{len(indices)}",
            flush=True,
        )

    return {
        "num_prompts": len(indices),
        "functional_same_state": {
            str(timestep): finalize_metric_lists(values)
            for timestep, values in functional.items()
        },
        "rollout_teacher_on_student_state": {
            str(call_index + 1): finalize_metric_lists(values)
            for call_index, values in rollout_on_policy.items()
        },
        "rollout_free_running_native_vs_student": {
            str(call_index + 1): finalize_metric_lists(values)
            for call_index, values in rollout_closed_loop.items()
        },
    }


def image_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def psnr(native: Image.Image, student: Image.Image) -> float:
    error = np.mean((image_array(native) - image_array(student)) ** 2)
    return float(-10.0 * np.log10(max(error, 1e-12)))


def ssim(native: Image.Image, student: Image.Image) -> float:
    x = torch.from_numpy(image_array(native)).permute(2, 0, 1).unsqueeze(0)
    y = torch.from_numpy(image_array(student)).permute(2, 0, 1).unsqueeze(0)
    kernel = 11
    padding = kernel // 2
    mu_x = F.avg_pool2d(x, kernel, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, kernel, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(x * x, kernel, stride=1, padding=padding) - mu_x.square()
    sigma_y = F.avg_pool2d(y * y, kernel, stride=1, padding=padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(x * y, kernel, stride=1, padding=padding) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1)
        * (sigma_x + sigma_y + c2)
    ).clamp_min(1e-12)
    return float(score.mean())


def image_quality(image: Image.Image) -> dict[str, float]:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return {
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "luminance_mean": float(gray.mean() / 255.0),
        "luminance_std": float(gray.std() / 255.0),
        "saturation_mean": float(hsv[..., 1].mean() / 255.0),
    }


def make_pair(native: Image.Image, student: Image.Image, prompt: str) -> Image.Image:
    target_size = (512, 320)
    native = native.convert("RGB").resize(target_size)
    student = student.convert("RGB").resize(target_size)
    header = 58
    canvas = Image.new("RGB", (1024, 320 + header), "white")
    canvas.paste(native, (0, header))
    canvas.paste(student, (512, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), prompt[:150], fill="black")
    draw.text((6, 31), "Native dual-CLIP", fill="black")
    draw.text((518, 31), "Mobile-OV Image Bridge", fill="black")
    return canvas


@torch.inference_mode()
def generate_images(
    pipe,
    conditions: ConditionPair,
    prompts: list[str],
    indices: list[int],
    *,
    output_dir: Path,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[Image.Image], list[Image.Image], list[dict[str, object]]]:
    native_images: list[Image.Image] = []
    student_images: list[Image.Image] = []
    rows: list[dict[str, object]] = []
    pair_images: list[Image.Image] = []
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for output_index, condition_index in enumerate(indices):
        student = slice_condition(conditions.student, [condition_index], device, dtype)
        native = slice_condition(conditions.native, [condition_index], device, dtype)
        current_seed = seed + output_index
        with torch.autocast("cuda", dtype=dtype):
            native_start = time.perf_counter()
            native_image = pipe(
                prompt_embeds=native.prompt_embeds,
                pooled_prompt_embeds=native.pooled,
                generator=torch.Generator(device=device).manual_seed(current_seed),
            ).images[0]
            native_seconds = time.perf_counter() - native_start
            student_start = time.perf_counter()
            student_image = pipe(
                prompt_embeds=student.prompt_embeds,
                pooled_prompt_embeds=student.pooled,
                generator=torch.Generator(device=device).manual_seed(current_seed),
            ).images[0]
            student_seconds = time.perf_counter() - student_start

        prompt = prompts[condition_index]
        stem = f"{output_index + 1:03d}_{safe_stem(prompt)}"
        native_path = image_dir / f"{stem}_native.png"
        student_path = image_dir / f"{stem}_bridge.png"
        native_image.save(native_path)
        student_image.save(student_path)
        pair_images.append(make_pair(native_image, student_image, prompt))
        native_images.append(native_image)
        student_images.append(student_image)
        rows.append(
            {
                "index": output_index + 1,
                "condition_index": condition_index,
                "prompt": prompt,
                "bucket": prompt_bucket(prompt),
                "seed": current_seed,
                "native_seconds": native_seconds,
                "bridge_seconds": student_seconds,
                "psnr": psnr(native_image, student_image),
                "ssim": ssim(native_image, student_image),
                "native_quality": image_quality(native_image),
                "bridge_quality": image_quality(student_image),
                "native_path": str(native_path),
                "bridge_path": str(student_path),
            }
        )
        print(
            f"Image generation: {output_index + 1}/{len(indices)} "
            f"PSNR={rows[-1]['psnr']:.2f} SSIM={rows[-1]['ssim']:.3f}",
            flush=True,
        )

    contact_sheet = Image.new(
        "RGB",
        (1024, sum(image.height for image in pair_images[:12])),
        "white",
    )
    offset = 0
    for pair in pair_images[:12]:
        contact_sheet.paste(pair, (0, offset))
        offset += pair.height
    contact_sheet.save(output_dir / "native_vs_bridge_contact_sheet.jpg", quality=92)
    return native_images, student_images, rows


@torch.inference_mode()
def clip_semantic_metrics(
    prompts: list[str],
    native_images: list[Image.Image],
    student_images: list[Image.Image],
    *,
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> tuple[dict[str, object], list[dict[str, float]]]:
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    text_features: list[torch.Tensor] = []
    native_features: list[torch.Tensor] = []
    student_features: list[torch.Tensor] = []
    for offset in range(0, len(prompts), batch_size):
        batch_prompts = prompts[offset : offset + batch_size]
        text_inputs = processor(
            text=batch_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        text_features.append(model.get_text_features(**text_inputs).float().cpu())
        for images, target in (
            (native_images, native_features),
            (student_images, student_features),
        ):
            image_inputs = processor(
                images=images[offset : offset + batch_size],
                return_tensors="pt",
            ).to(device)
            image_inputs["pixel_values"] = image_inputs["pixel_values"].to(dtype=dtype)
            target.append(model.get_image_features(**image_inputs).float().cpu())

    text = F.normalize(torch.cat(text_features), dim=-1)
    native = F.normalize(torch.cat(native_features), dim=-1)
    student = F.normalize(torch.cat(student_features), dim=-1)
    native_scores = (text * native).sum(dim=-1)
    student_scores = (text * student).sum(dim=-1)
    paired_similarity = (native * student).sum(dim=-1)
    native_retrieval = (text @ native.T).argmax(dim=1)
    student_retrieval = (text @ student.T).argmax(dim=1)
    target = torch.arange(len(prompts))
    per_prompt = [
        {
            "native_clip_cosine": float(native_scores[index]),
            "bridge_clip_cosine": float(student_scores[index]),
            "clip_delta_bridge_minus_native": float(
                student_scores[index] - native_scores[index]
            ),
            "native_bridge_image_cosine": float(paired_similarity[index]),
        }
        for index in range(len(prompts))
    ]
    summary = {
        "model_id": model_id,
        "native_prompt_image_cosine": tensor_stats(native_scores),
        "bridge_prompt_image_cosine": tensor_stats(student_scores),
        "bridge_minus_native": tensor_stats(student_scores - native_scores),
        "native_bridge_image_cosine": tensor_stats(paired_similarity),
        "native_prompt_to_image_top1": float((native_retrieval == target).float().mean()),
        "bridge_prompt_to_image_top1": float((student_retrieval == target).float().mean()),
        "bridge_semantic_win_rate": float(
            (student_scores >= native_scores).float().mean()
        ),
        "bridge_within_0p02_of_native_rate": float(
            (student_scores >= native_scores - 0.02).float().mean()
        ),
    }
    return summary, per_prompt


def aggregate_image_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    fields = ("psnr", "ssim", "native_seconds", "bridge_seconds")
    result: dict[str, object] = {
        field: tensor_stats(torch.tensor([float(row[field]) for row in rows]))
        for field in fields
    }
    for side in ("native_quality", "bridge_quality"):
        keys = rows[0][side].keys()
        result[side] = {
            key: tensor_stats(
                torch.tensor([float(row[side][key]) for row in rows])
            )
            for key in keys
        }
    return result


def training_audit(checkpoint: dict, world_size: int) -> dict[str, object]:
    args = checkpoint.get("args", {})
    total_steps = int(checkpoint.get("step", args.get("target_step", 0)))
    functional_start = int(args.get("functional_start_step", 5001))
    rollout_start = int(args.get("rollout_start_step", 10001))
    rollout_every = int(args.get("rollout_every", 8))
    batch_size = int(args.get("batch_size", 4))
    functional_batch = int(args.get("functional_batch_size", 1))
    rollout_batch = int(args.get("rollout_batch_size", 1))
    functional_updates = 0
    rollout_updates = 0
    representation_only_updates = 0
    for step in range(1, total_steps + 1):
        run_rollout = step >= rollout_start and (step - rollout_start) % rollout_every == 0
        if run_rollout:
            rollout_updates += 1
        elif step >= functional_start:
            functional_updates += 1
        else:
            representation_only_updates += 1

    history = checkpoint.get("history", [])
    logged_modes = Counter(str(row.get("mode", "unknown")) for row in history)
    return {
        "total_updates": total_steps,
        "representation_loss_updates": total_steps,
        "representation_only_updates": representation_only_updates,
        "functional_updates": functional_updates,
        "rollout_updates": rollout_updates,
        "estimated_representation_prompt_exposures": total_steps
        * batch_size
        * world_size,
        "estimated_functional_prompt_exposures": functional_updates
        * functional_batch
        * world_size,
        "estimated_rollout_prompt_trajectories": rollout_updates
        * rollout_batch
        * world_size,
        "estimated_rollout_student_unet_calls": rollout_updates
        * rollout_batch
        * world_size
        * len(SSD1B_TIMESTEPS),
        "history_rows": len(history),
        "logged_modes": dict(logged_modes),
        "rollout_rows_logged": logged_modes.get("rollout", 0),
        "rollout_observability_gap": (
            rollout_updates > 0 and logged_modes.get("rollout", 0) == 0
        ),
        "observability_explanation": (
            "rollout steps are 10001+8k while log_every=20; these schedules never intersect"
        ),
    }


def write_image_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "index",
        "condition_index",
        "prompt",
        "bucket",
        "seed",
        "native_seconds",
        "bridge_seconds",
        "psnr",
        "ssim",
        "native_clip_cosine",
        "bridge_clip_cosine",
        "clip_delta_bridge_minus_native",
        "native_bridge_image_cosine",
        "native_path",
        "bridge_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def make_summary_plot(summary: dict, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    condition = summary["condition"]
    functional = summary["unet_and_rollout"]["functional_same_state"]
    rollout = summary["unet_and_rollout"][
        "rollout_free_running_native_vs_student"
    ]
    image = summary["clip_semantic"]
    speed = summary["efficiency"]["latency"]

    figure, axes = plt.subplots(2, 3, figsize=(17, 9))
    names = ["CLIP-L", "CLIP-bigG", "Pooled"]
    condition_values = [
        condition["clip_l"]["cosine_distance"]["mean"],
        condition["clip_big_g"]["cosine_distance"]["mean"],
        condition["pooled"]["cosine_distance"]["mean"],
    ]
    axes[0, 0].bar(names, condition_values)
    axes[0, 0].set_title("Condition cosine distance (lower is better)")

    timesteps = [str(value) for value in SSD1B_TIMESTEPS]
    axes[0, 1].plot(
        timesteps,
        [functional[value]["relative_rmse"]["mean"] for value in timesteps],
        marker="o",
    )
    axes[0, 1].set_title("UNet response relative RMSE")
    axes[0, 1].set_xlabel("Timestep")

    calls = [str(index) for index in range(1, len(SSD1B_TIMESTEPS) + 1)]
    axes[0, 2].plot(
        calls,
        [rollout[value]["relative_rmse"]["mean"] for value in calls],
        marker="o",
    )
    axes[0, 2].set_title("Free-running latent drift")
    axes[0, 2].set_xlabel("LCM call")

    axes[1, 0].bar(
        ["Native", "Bridge"],
        [
            image["native_prompt_image_cosine"]["mean"],
            image["bridge_prompt_image_cosine"]["mean"],
        ],
    )
    axes[1, 0].set_title("CLIP prompt-image cosine")

    batch_sizes = sorted(speed["bridge"], key=int)
    axes[1, 1].plot(
        batch_sizes,
        [speed["native"][value]["ms_per_prompt"] for value in batch_sizes],
        marker="o",
        label="Native dual-CLIP",
    )
    axes[1, 1].plot(
        batch_sizes,
        [speed["bridge"][value]["ms_per_prompt"] for value in batch_sizes],
        marker="o",
        label="Mobile-OV Bridge",
    )
    axes[1, 1].set_title("Conditioning latency")
    axes[1, 1].set_xlabel("Batch size")
    axes[1, 1].set_ylabel("ms/prompt")
    axes[1, 1].legend()

    geometry_names = ["CLIP-L", "CLIP-bigG", "Pooled"]
    axes[1, 2].bar(
        geometry_names,
        [
            condition["clip_l"]["geometry"]["pairwise_cosine_pearson"],
            condition["clip_big_g"]["geometry"]["pairwise_cosine_pearson"],
            condition["pooled"]["geometry"]["pairwise_cosine_pearson"],
        ],
    )
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_title("Prompt-geometry correlation")

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive SSD1B Image Bridge deployment audit."
    )
    parser.add_argument("--config", default="configs/mobile_ov_ssd1b_image_bridge.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--prompt-file",
        default="checkpoints/neodragon_repo/prompts/vbench_prompts.txt",
    )
    parser.add_argument(
        "--output-dir",
        default="output/ssd1b_image_bridge_comprehensive_eval",
    )
    parser.add_argument("--condition-prompts", type=int, default=384)
    parser.add_argument("--functional-prompts", type=int, default=96)
    parser.add_argument("--image-prompts", type=int, default=30)
    parser.add_argument("--condition-batch-size", type=int, default=16)
    parser.add_argument("--unet-batch-size", type=int, default=4)
    parser.add_argument("--clip-batch-size", type=int, default=8)
    parser.add_argument("--latency-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--training-world-size", type=int, default=8)
    parser.add_argument("--clip-model-id", default="openai/clip-vit-large-patch14")
    parser.add_argument(
        "--append-prompt-modifier",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Comprehensive SSD1B evaluation requires a CUDA allocation.")
    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    raw_prompt_pool = read_prompts(Path(args.prompt_file))
    raw_prompts = stratified_sample(
        raw_prompt_pool,
        args.condition_prompts,
        seed=args.seed,
    )

    repo_path, _, model_path = ensure_neodragon_assets(
        repo_path=cfg.backend.extra.get("repo_path"),
        cache_dir=cfg.backend.extra.get("cache_dir"),
        model_id=cfg.backend.extra.get("model_id", "karnewar/Neodragon"),
        repo_url=cfg.backend.extra.get("repo_url"),
    )
    repo_path = Path(repo_path).resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    from neodragon.first_frame_gen import SSD1B_FirstFrameGeneratorPipeline
    from neodragon.utils.generation_utils import DEFAULT_PROMPT_MODIFIER

    conditioned_prompts = [
        prompt + DEFAULT_PROMPT_MODIFIER if args.append_prompt_modifier else prompt
        for prompt in raw_prompts
    ]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    audit = training_audit(checkpoint, args.training_world_size)

    bridge = MobileOVSSD1BImageBridge(
        cfg.bridge,
        cfg.image_bridge,
        device=device,
        dtype=dtype,
    ).eval()
    bridge.load_trainable_state_dict(checkpoint["image_bridge"])
    non_finite = [
        name
        for name, value in checkpoint["image_bridge"].items()
        if torch.is_tensor(value) and not torch.isfinite(value).all()
    ]
    if non_finite:
        raise RuntimeError(f"Non-finite checkpoint tensors: {non_finite}")
    teacher = SSD1BFrozenTeacher(cfg, device, dtype)

    bridge_parameters = parameter_summary(bridge)
    native_parameters = {
        "clip_l": parameter_summary(teacher.text_encoder),
        "clip_big_g": parameter_summary(teacher.text_encoder_2),
    }
    native_parameters["combined_total_parameters"] = (
        native_parameters["clip_l"]["total_parameters"]
        + native_parameters["clip_big_g"]["total_parameters"]
    )
    native_parameters["combined_parameter_mib"] = (
        native_parameters["clip_l"]["parameter_mib"]
        + native_parameters["clip_big_g"]["parameter_mib"]
    )

    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        latency = {
            "bridge": benchmark_encoder(
                bridge,
                conditioned_prompts,
                batch_sizes=[1, 4, 16],
                repeats=args.latency_repeats,
                device=device,
            ),
            "native": benchmark_encoder(
                teacher.encode,
                conditioned_prompts,
                batch_sizes=[1, 4, 16],
                repeats=args.latency_repeats,
                device=device,
            ),
        }
        conditions = encode_all_conditions(
            bridge,
            teacher,
            conditioned_prompts,
            batch_size=args.condition_batch_size,
            device=device,
            dtype=dtype,
        )

    condition_summary = {
        "num_prompts": len(raw_prompts),
        "length_buckets": dict(Counter(prompt_bucket(prompt) for prompt in raw_prompts)),
        "clip_l": stream_condition_metrics(
            conditions.student.clip_l_tokens,
            conditions.native.clip_l_tokens,
        ),
        "clip_big_g": stream_condition_metrics(
            conditions.student.clip_big_g_tokens,
            conditions.native.clip_big_g_tokens,
        ),
        "pooled": stream_condition_metrics(
            conditions.student.pooled,
            conditions.native.pooled,
        ),
    }
    del bridge, teacher
    torch.cuda.empty_cache()

    pipe = SSD1B_FirstFrameGeneratorPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    functional_indices = list(
        range(min(args.functional_prompts, len(raw_prompts)))
    )
    unet_and_rollout = evaluate_functional_and_rollout(
        pipe,
        conditions,
        functional_indices,
        batch_size=args.unet_batch_size,
        seed=args.seed,
        device=device,
        dtype=dtype,
    )
    image_indices = list(range(min(args.image_prompts, len(raw_prompts))))
    native_images, bridge_images, image_rows = generate_images(
        pipe,
        conditions,
        raw_prompts,
        image_indices,
        output_dir=output_dir,
        seed=args.seed + 50000,
        device=device,
        dtype=dtype,
    )
    image_summary = aggregate_image_rows(image_rows)
    del pipe
    torch.cuda.empty_cache()

    selected_image_prompts = [raw_prompts[index] for index in image_indices]
    clip_summary, clip_rows = clip_semantic_metrics(
        selected_image_prompts,
        native_images,
        bridge_images,
        model_id=args.clip_model_id,
        device=device,
        dtype=dtype,
        batch_size=args.clip_batch_size,
    )
    for row, clip_row in zip(image_rows, clip_rows):
        row.update(clip_row)
    write_image_csv(image_rows, output_dir / "image_metrics.csv")

    summary = {
        "status": "ok",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "checkpoint_target": checkpoint.get("target"),
        "prompt_source": args.prompt_file,
        "append_prompt_modifier": args.append_prompt_modifier,
        "training_audit": audit,
        "condition": condition_summary,
        "unet_and_rollout": unet_and_rollout,
        "image_pair": image_summary,
        "clip_semantic": clip_summary,
        "efficiency": {
            "parameters": {
                "bridge_full_stack": bridge_parameters,
                "native_dual_clip": native_parameters,
            },
            "latency": latency,
        },
        "artifacts": {
            "contact_sheet": str(output_dir / "native_vs_bridge_contact_sheet.jpg"),
            "image_metrics_csv": str(output_dir / "image_metrics.csv"),
            "summary_plot": str(output_dir / "summary_plot.png"),
        },
    }
    summary_path = output_dir / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    make_summary_plot(summary, output_dir / "summary_plot.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
