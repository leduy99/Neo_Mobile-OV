#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.utils import export_to_video
from PIL import Image, ImageDraw
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.bridge import MobileOVNeodragonTextBridge  # noqa: E402
from new_mobile_ov.config import load_config  # noqa: E402
from new_mobile_ov.generation import build_generation_backend  # noqa: E402
from new_mobile_ov.training.neodragon_hybrid_recovery import (  # noqa: E402
    DiTCondition,
    downsample_noise_2x,
    prepare_past_conditions,
    run_stage_endpoint,
    upsample_pyramidal_latent,
)
from tools.train_neodragon_dit_bridge import (  # noqa: E402
    load_bridge,
    load_neodragon_train_modules,
)
from tools.train_neodragon_hybrid_recovery import (  # noqa: E402
    load_monolithic_teacher,
    native_condition,
)


CONFIG_NAMES = {
    "A": "Released Hybrid + native Hybrid condition",
    "B": "Released Hybrid + rollout bridge 80K",
    "C": "Exp6 2K + native Hybrid condition",
    "D": "Exp6 2K + rollout bridge 80K",
    "E": "Released Hybrid + Exp1 bridge 64K",
}
FULL_ORDER = ("A", "B", "C", "D", "E")


def dtype_from_name(name: str) -> torch.dtype:
    value = str(name).lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
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


def stratified_sample(prompts: list[str], count: int, *, seed: int) -> list[str]:
    if count < 0 or count >= len(prompts):
        return list(prompts)
    rng = random.Random(seed)
    buckets: dict[str, list[str]] = {"short": [], "medium": [], "long": []}
    for prompt in prompts:
        buckets[prompt_bucket(prompt)].append(prompt)
    for values in buckets.values():
        rng.shuffle(values)
    selected: list[str] = []
    per_bucket = count // len(buckets)
    for values in buckets.values():
        selected.extend(values[:per_bucket])
    remaining = count - len(selected)
    pool = [value for values in buckets.values() for value in values[per_bucket:]]
    rng.shuffle(pool)
    selected.extend(pool[:remaining])
    rng.shuffle(selected)
    return selected


def selected_prompts(args: argparse.Namespace, output_dir: Path) -> list[str]:
    destination = output_dir / "prompts_96.txt"
    if destination.is_file():
        prompts = read_prompts(destination)
    else:
        prompts = stratified_sample(
            read_prompts(Path(args.prompt_file)),
            args.num_prompts,
            seed=args.prompt_seed,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]
    if not prompts:
        raise ValueError("No evaluation prompts were selected.")
    return prompts


def safe_stem(text: str, max_len: int = 68) -> str:
    value = re.sub(r"[^a-zA-Z0-9._ -]+", "_", text).strip().replace(" ", "_")
    return (value[:max_len] or "prompt").strip("_")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_state(path: Path, key: str) -> tuple[dict[str, torch.Tensor], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if key not in payload:
        raise KeyError(f"Checkpoint {path} does not contain `{key}`.")
    state = payload[key]
    metadata = {
        "path": str(path),
        "step": int(payload.get("step", -1)),
        "tensors": len(state),
        "parameters": int(
            sum(value.numel() for value in state.values() if torch.is_tensor(value))
        ),
    }
    return state, metadata


def condition_to_cpu(condition: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(value.detach().cpu() for value in condition)


def condition_slice(
    condition: tuple[torch.Tensor, ...],
    start: int,
    end: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    values = []
    for value in condition:
        target_dtype = dtype if value.is_floating_point() else value.dtype
        values.append(value[start:end].to(device=device, dtype=target_dtype))
    return tuple(values)


@torch.inference_mode()
def encode_native_conditions(
    backend,
    prompts: list[str],
    *,
    modifier: str,
    batch_size: int,
) -> tuple[torch.Tensor, ...]:
    batches = [[], [], []]
    for start in range(0, len(prompts), batch_size):
        values = backend.encode_neodragon_context(
            [prompt + modifier for prompt in prompts[start : start + batch_size]]
        )
        for index, value in enumerate(values):
            batches[index].append(value.detach().cpu())
    return tuple(torch.cat(values, dim=0) for values in batches)


@torch.inference_mode()
def encode_bridge_conditions(
    cfg,
    checkpoint: Path,
    prompts: list[str],
    *,
    modifier: str,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> tuple[tuple[torch.Tensor, ...], dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    bridge = MobileOVNeodragonTextBridge(cfg.bridge, device=device, dtype=dtype).eval()
    state = payload.get("bridge", payload.get("student_state", payload))
    missing, unexpected = bridge.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Bridge mismatch for {checkpoint}: missing={missing[:8]} "
            f"unexpected={unexpected[:8]}"
        )
    outputs = [[], [], []]
    for start in range(0, len(prompts), batch_size):
        values = bridge.encode(
            [prompt + modifier for prompt in prompts[start : start + batch_size]]
        )
        for index, value in enumerate(values):
            outputs[index].append(value.detach().cpu())
    metadata = {
        "path": str(checkpoint),
        "step": int(payload.get("step", -1)),
        "tensors": len(state),
    }
    del bridge, payload, state
    torch.cuda.empty_cache()
    return tuple(torch.cat(values, dim=0) for values in outputs), metadata


@torch.inference_mode()
def generate_native_anchors(
    backend,
    prompts: list[str],
    *,
    modifier: str,
    output_dir: Path,
    seed: int,
    device: torch.device,
) -> tuple[list[Image.Image], list[dict]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = backend.pipeline.first_frame_gen_pipeline
    pipeline.set_progress_bar_config(disable=True)
    images: list[Image.Image] = []
    rows = []
    for index, prompt in enumerate(tqdm(prompts, desc="Native SSD1B anchors")):
        current_seed = seed + index
        started = time.perf_counter()
        image = pipeline(
            prompt=prompt + modifier,
            num_images_per_prompt=1,
            generator=torch.Generator(device=device).manual_seed(current_seed),
        ).images[0].convert("RGB")
        seconds = time.perf_counter() - started
        path = output_dir / f"{index + 1:03d}_{safe_stem(prompt)}.png"
        image.save(path)
        images.append(image)
        rows.append({"index": index + 1, "seed": current_seed, "path": str(path), "seconds": seconds})
    return images, rows


def load_native_anchors(output_dir: Path, prompts: list[str]) -> list[Image.Image]:
    paths = sorted(output_dir.glob("*.png"))
    if len(paths) < len(prompts):
        raise RuntimeError(
            f"Expected {len(prompts)} native anchors in {output_dir}, found {len(paths)}."
        )
    return [Image.open(path).convert("RGB") for path in paths[: len(prompts)]]


@torch.inference_mode()
def generate_configuration(
    backend,
    prompts: list[str],
    anchors: list[Image.Image],
    condition: tuple[torch.Tensor, ...],
    *,
    config_id: str,
    output_dir: Path,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    fps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict]:
    video_dir = output_dir / "videos" / config_id
    video_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (prompt, anchor) in enumerate(
        tqdm(zip(prompts, anchors), total=len(prompts), desc=f"Config {config_id}")
    ):
        prompt_embeds, mask, pooled = condition_slice(
            condition,
            index,
            index + 1,
            device=device,
            dtype=dtype,
        )
        current_seed = seed + index
        set_seed(current_seed)
        started = time.perf_counter()
        frames = backend.generate_video_from_bridge_condition(
            prompt,
            prompt_embeds=prompt_embeds,
            prompt_mask=mask,
            pooled_prompt_embeds=pooled,
            first_frame=anchor,
            height=height,
            width=width,
            num_frames=num_frames,
        )
        seconds = time.perf_counter() - started
        path = video_dir / f"{index + 1:03d}_{safe_stem(prompt)}.mp4"
        export_to_video(frames, path, fps=fps)
        rows.append(
            {
                "config": config_id,
                "index": index + 1,
                "prompt": prompt,
                "bucket": prompt_bucket(prompt),
                "seed": current_seed,
                "seconds": seconds,
                "path": str(path),
            }
        )
    return rows


def read_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"Could not decode video: {path}")
    return frames


def pixel_video_metrics(frames: list[np.ndarray]) -> dict[str, float]:
    rgb = np.stack(frames).astype(np.float32) / 255.0
    adjacent = np.abs(np.diff(rgb, axis=0)).mean(axis=(1, 2, 3))
    first_last = float(np.abs(rgb[-1] - rgb[0]).mean())
    sharpness = []
    saturation = []
    small_gray = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        saturation.append(float(hsv[..., 1].mean() / 255.0))
        small_gray.append(cv2.resize(gray, (256, 160), interpolation=cv2.INTER_AREA))
    flow = []
    for previous, current in zip(small_gray[:-1], small_gray[1:]):
        values = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        flow.append(float(np.linalg.norm(values, axis=-1).mean()))
    if len(small_gray) >= 3:
        gray_stack = np.stack(small_gray).astype(np.float32) / 255.0
        flicker = float(
            np.abs(
                gray_stack[1:-1]
                - 0.5 * (gray_stack[:-2] + gray_stack[2:])
            ).mean()
        )
    else:
        flicker = 0.0
    return {
        "adjacent_rgb_mae": float(adjacent.mean()),
        "first_last_rgb_mae": first_last,
        "optical_flow": float(np.mean(flow)),
        "mean_sharpness": float(np.mean(sharpness)),
        "last_frame_sharpness": sharpness[-1],
        "flicker_proxy": flicker,
        "mean_saturation": float(np.mean(saturation)),
    }


@torch.inference_mode()
def clip_video_metrics(
    rows: list[dict],
    *,
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    for row in tqdm(rows, desc="CLIP video metrics"):
        frames = read_video(Path(row["path"]))
        indices = np.linspace(0, len(frames) - 1, 8, dtype=int).tolist()
        sampled = [Image.fromarray(frames[index]) for index in indices]
        inputs = processor(
            text=[str(row["prompt"])],
            images=sampled,
            return_tensors="pt",
            padding=True,
        )
        pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        image_features = F.normalize(model.get_image_features(pixel_values), dim=-1)
        text_features = F.normalize(
            model.get_text_features(input_ids=input_ids, attention_mask=attention_mask),
            dim=-1,
        )
        text_scores = (image_features @ text_features.T).squeeze(-1).float().cpu()
        subject_scores = (image_features @ image_features[:1].T).squeeze(-1).float().cpu()
        adjacent_scores = F.cosine_similarity(
            image_features[:-1], image_features[1:], dim=-1
        ).float().cpu()
        row.update(
            {
                "prompt_clip_mean": float(text_scores.mean()),
                "prompt_clip_min": float(text_scores.min()),
                "prompt_clip_last": float(text_scores[-1]),
                "subject_clip_mean": float(subject_scores.mean()),
                "subject_clip_min": float(subject_scores.min()),
                "subject_clip_last": float(subject_scores[-1]),
                "temporal_clip_adjacent": float(adjacent_scores.mean()),
                **pixel_video_metrics(frames),
            }
        )
    del model
    torch.cuda.empty_cache()


def mean_std(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def aggregate_full(rows: list[dict]) -> dict:
    keys = [
        "seconds",
        "prompt_clip_mean",
        "prompt_clip_min",
        "prompt_clip_last",
        "subject_clip_mean",
        "subject_clip_min",
        "subject_clip_last",
        "temporal_clip_adjacent",
        "adjacent_rgb_mae",
        "first_last_rgb_mae",
        "optical_flow",
        "mean_sharpness",
        "last_frame_sharpness",
        "flicker_proxy",
        "mean_saturation",
    ]
    output = {}
    for config_id in FULL_ORDER:
        selected = [row for row in rows if row["config"] == config_id]
        output[config_id] = {
            "name": CONFIG_NAMES[config_id],
            "num_videos": len(selected),
            **{key: mean_std(row[key] for row in selected) for key in keys},
        }
    return output


def make_video_contact_sheet(rows: list[dict], output_path: Path) -> None:
    selected_indices = [0, 1, 2, 3, 4, 5]
    width, height = 256, 160
    header = 28
    canvas = Image.new(
        "RGB",
        (width * 3, (height + header) * len(FULL_ORDER) * len(selected_indices)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    y = 0
    for prompt_index in selected_indices:
        for config_id in FULL_ORDER:
            matches = [
                row
                for row in rows
                if row["config"] == config_id and row["index"] == prompt_index + 1
            ]
            if not matches:
                continue
            row = matches[0]
            frames = read_video(Path(row["path"]))
            sample_indices = [0, len(frames) // 2, len(frames) - 1]
            draw.text(
                (4, y + 4),
                f"{config_id}: {row['prompt'][:86]}",
                fill="black",
            )
            for column, frame_index in enumerate(sample_indices):
                image = Image.fromarray(frames[frame_index]).resize((width, height))
                canvas.paste(image, (column * width, y + header))
            y += height + header
    canvas = canvas.crop((0, 0, canvas.width, y))
    canvas.save(output_path, quality=90)


def run_full(args: argparse.Namespace, output_dir: Path, prompts: list[str]) -> dict:
    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    cfg = load_config(args.config)
    backend = build_generation_backend(cfg.backend, device=device)
    from neodragon.utils.generation_utils import DEFAULT_PROMPT_MODIFIER

    anchor_dir = output_dir / "native_ssd1b_anchors"
    if len(list(anchor_dir.glob("*.png"))) >= len(prompts):
        anchors = load_native_anchors(anchor_dir, prompts)
        anchor_rows = [{"path": str(path)} for path in sorted(anchor_dir.glob("*.png"))[: len(prompts)]]
    else:
        anchors, anchor_rows = generate_native_anchors(
            backend,
            prompts,
            modifier=DEFAULT_PROMPT_MODIFIER,
            output_dir=anchor_dir,
            seed=args.seed,
            device=device,
        )
    native = encode_native_conditions(
        backend,
        prompts,
        modifier=DEFAULT_PROMPT_MODIFIER,
        batch_size=args.condition_batch_size,
    )
    bridge80, bridge80_meta = encode_bridge_conditions(
        cfg,
        Path(args.bridge80),
        prompts,
        modifier=DEFAULT_PROMPT_MODIFIER,
        device=device,
        dtype=dtype,
        batch_size=args.condition_batch_size,
    )
    bridge64, bridge64_meta = encode_bridge_conditions(
        cfg,
        Path(args.bridge64),
        prompts,
        modifier=DEFAULT_PROMPT_MODIFIER,
        device=device,
        dtype=dtype,
        batch_size=args.condition_batch_size,
    )

    rows = []
    for config_id, condition in (("A", native), ("B", bridge80), ("E", bridge64)):
        rows.extend(
            generate_configuration(
                backend,
                prompts,
                anchors,
                condition,
                config_id=config_id,
                output_dir=output_dir,
                seed=args.seed,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                fps=args.fps,
                device=device,
                dtype=dtype,
            )
        )

    exp6_state, exp6_meta = checkpoint_state(Path(args.exp6), "dit")
    missing, unexpected = backend.pipeline.dit.load_state_dict(exp6_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Exp6 DiT mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    del exp6_state
    torch.cuda.empty_cache()
    for config_id, condition in (("C", native), ("D", bridge80)):
        rows.extend(
            generate_configuration(
                backend,
                prompts,
                anchors,
                condition,
                config_id=config_id,
                output_dir=output_dir,
                seed=args.seed,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                fps=args.fps,
                device=device,
                dtype=dtype,
            )
        )
    del backend, anchors, native, bridge80, bridge64
    torch.cuda.empty_cache()

    clip_video_metrics(rows, model_id=args.clip_model, device=device, dtype=dtype)
    summary = {
        "status": "ok",
        "protocol": {
            "num_prompts": len(prompts),
            "prompt_file": str(output_dir / "prompts_96.txt"),
            "prompt_seed": args.prompt_seed,
            "generation_seed": args.seed,
            "native_ssd1b_anchor": True,
            "shared_anchor": True,
            "shared_generation_noise": True,
            "prompt_modifier": True,
            "dtype": str(dtype),
            "schedule": "Hybrid 1-1-1 (6 units x 3 stages)",
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "fps": args.fps,
        },
        "checkpoints": {
            "bridge80": bridge80_meta,
            "bridge64": bridge64_meta,
            "exp6": exp6_meta,
        },
        "anchors": anchor_rows,
        "aggregate": aggregate_full(rows),
        "rows": rows,
    }
    (output_dir / "full_rollout_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    make_video_contact_sheet(rows, output_dir / "full_rollout_contact_sheet.jpg")
    return summary


def image_batch_to_anchor(
    images: list[Image.Image],
    *,
    vae,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    from neodragon.utils.generation_utils import VAE_SCALE_FACTOR, VAE_SHIFT_FACTOR

    arrays = []
    for image in images:
        value = np.asarray(
            image.resize((width, height), resample=Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
        arrays.append((value / 255.0) * 2.0 - 1.0)
    tensor = torch.from_numpy(np.stack(arrays)).to(device=device, dtype=dtype)
    tensor = tensor.permute(0, 3, 1, 2).unsqueeze(2)
    latent = vae.encode(tensor).latent_dist.sample()
    return (latent - VAE_SHIFT_FACTOR) * VAE_SCALE_FACTOR


def per_sample_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    start: torch.Tensor,
) -> torch.Tensor:
    numerator = (prediction.float() - target.float()).flatten(1).norm(dim=-1)
    denominator = (target.float() - start.float()).flatten(1).norm(dim=-1)
    return numerator / denominator.clamp_min(1e-6)


def per_sample_transition_cosine(
    prediction: torch.Tensor,
    target: torch.Tensor,
    start: torch.Tensor,
) -> torch.Tensor:
    return F.cosine_similarity(
        (prediction.float() - start.float()).flatten(1),
        (target.float() - start.float()).flatten(1),
        dim=-1,
    )


@torch.inference_mode()
def evaluate_state_bank(
    *,
    name: str,
    released,
    exp6,
    monolithic,
    scheduler,
    anchors: torch.Tensor,
    full_noise: torch.Tensor,
    bank_condition: DiTCondition,
    teacher_condition: DiTCondition,
    prompt_offset: int,
    prompts: list[str],
    generator: torch.Generator,
    evaluate_exp6: bool,
) -> list[dict]:
    rows = []
    generated = [anchors]
    low_noise = downsample_noise_2x(full_noise, 2)
    for unit in range(6):
        stage_histories = prepare_past_conditions(generated, 3)
        current = low_noise[:, :, unit + 1 : unit + 2]
        for stage in range(3):
            if stage > 0:
                current = upsample_pyramidal_latent(
                    current,
                    orig_sigma=1 - scheduler.orig_start_sigmas[stage],
                    gamma=scheduler.config.gamma,
                    generator=generator,
                )
            history = tuple(stage_histories[stage])
            released_endpoint, _ = run_stage_endpoint(
                dit=released,
                scheduler=scheduler,
                current=current,
                history=history,
                condition=bank_condition,
                stage=stage,
                num_steps=1,
            )
            monolithic_endpoint, _ = run_stage_endpoint(
                dit=monolithic,
                scheduler=scheduler,
                current=current,
                history=history,
                condition=teacher_condition,
                stage=stage,
                num_steps=10,
            )
            exp6_endpoint = None
            if evaluate_exp6:
                exp6_endpoint, _ = run_stage_endpoint(
                    dit=exp6,
                    scheduler=scheduler,
                    current=current,
                    history=history,
                    condition=bank_condition,
                    stage=stage,
                    num_steps=1,
                )
            released_l2 = per_sample_relative_l2(
                released_endpoint, monolithic_endpoint, current
            )
            released_cos = per_sample_transition_cosine(
                released_endpoint, monolithic_endpoint, current
            )
            if exp6_endpoint is not None:
                exp6_l2 = per_sample_relative_l2(
                    exp6_endpoint, monolithic_endpoint, current
                )
                exp6_cos = per_sample_transition_cosine(
                    exp6_endpoint, monolithic_endpoint, current
                )
                exp6_gap = per_sample_relative_l2(
                    exp6_endpoint, released_endpoint, current
                )
            for index in range(current.shape[0]):
                row = {
                    "bank": name,
                    "prompt_index": prompt_offset + index + 1,
                    "prompt": prompts[index],
                    "unit": unit,
                    "stage": stage,
                    "released_target_relative_l2": float(released_l2[index]),
                    "released_target_transition_cosine": float(released_cos[index]),
                }
                if exp6_endpoint is not None:
                    row.update(
                        {
                            "exp6_target_relative_l2": float(exp6_l2[index]),
                            "exp6_target_transition_cosine": float(exp6_cos[index]),
                            "exp6_released_relative_l2": float(exp6_gap[index]),
                        }
                    )
                rows.append(row)
            current = released_endpoint
        generated.append(current)
    return rows


def dit_condition(values: tuple[torch.Tensor, ...]) -> DiTCondition:
    return DiTCondition(tokens=values[0], mask=values[1], pooled=values[2])


def aggregate_local(rows: list[dict]) -> dict:
    output = {}
    for bank in ("native", "bridge80", "bridge64"):
        bank_rows = [row for row in rows if row["bank"] == bank]
        positions = {}
        for unit in range(6):
            for stage in range(3):
                values = [
                    row
                    for row in bank_rows
                    if row["unit"] == unit and row["stage"] == stage
                ]
                item = {
                    "released_target_relative_l2": mean_std(
                        row["released_target_relative_l2"] for row in values
                    ),
                    "released_target_transition_cosine": mean_std(
                        row["released_target_transition_cosine"] for row in values
                    ),
                }
                if bank != "bridge64":
                    item.update(
                        {
                            "exp6_target_relative_l2": mean_std(
                                row["exp6_target_relative_l2"] for row in values
                            ),
                            "exp6_target_transition_cosine": mean_std(
                                row["exp6_target_transition_cosine"] for row in values
                            ),
                            "exp6_released_relative_l2": mean_std(
                                row["exp6_released_relative_l2"] for row in values
                            ),
                        }
                    )
                    released_mean = item["released_target_relative_l2"]["mean"]
                    exp6_mean = item["exp6_target_relative_l2"]["mean"]
                    item["relative_l2_improvement_percent"] = float(
                        100.0 * (released_mean - exp6_mean) / max(released_mean, 1e-8)
                    )
                positions[f"u{unit}_s{stage}"] = item
        result = {"num_rows": len(bank_rows), "positions": positions}
        if bank != "bridge64":
            improvements = [
                item["relative_l2_improvement_percent"]
                for item in positions.values()
            ]
            late = [
                positions[f"u{unit}_s{stage}"]["relative_l2_improvement_percent"]
                for unit in (4, 5)
                for stage in range(3)
            ]
            result.update(
                {
                    "improved_positions": int(sum(value > 0.0 for value in improvements)),
                    "positions_over_10_percent": int(
                        sum(value >= 10.0 for value in improvements)
                    ),
                    "mean_improvement_percent": float(np.mean(improvements)),
                    "late_unit_improvement_percent": float(np.mean(late)),
                    "danger_position_u5_s2_percent": positions["u5_s2"][
                        "relative_l2_improvement_percent"
                    ],
                }
            )
        output[bank] = result
    return output


def make_local_heatmaps(summary: dict, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    all_values = [
        summary[bank]["positions"][f"u{unit}_s{stage}"][
            "relative_l2_improvement_percent"
        ]
        for bank in ("native", "bridge80")
        for unit in range(6)
        for stage in range(3)
    ]
    color_limit = max(20.0, 5.0 * np.ceil(max(abs(value) for value in all_values) / 5.0))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for axis, bank in zip(axes, ("native", "bridge80")):
        matrix = np.zeros((6, 3), dtype=np.float32)
        for unit in range(6):
            for stage in range(3):
                matrix[unit, stage] = summary[bank]["positions"][f"u{unit}_s{stage}"][
                    "relative_l2_improvement_percent"
                ]
        image = axis.imshow(
            matrix,
            cmap="RdYlGn",
            vmin=-color_limit,
            vmax=color_limit,
            aspect="auto",
        )
        axis.set_title(f"Exp6 relative-L2 improvement: {bank}")
        axis.set_xlabel("Stage")
        axis.set_ylabel("Unit")
        axis.set_xticks(range(3))
        axis.set_yticks(range(6))
        for unit in range(6):
            for stage in range(3):
                axis.text(stage, unit, f"{matrix[unit, stage]:.1f}%", ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axes.ravel().tolist(), label="Improvement (%)")
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def run_local(args: argparse.Namespace, output_dir: Path, prompts: list[str]) -> dict:
    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    cfg = load_config(args.config)
    anchors = load_native_anchors(output_dir / "native_ssd1b_anchors", prompts)

    backend = build_generation_backend(cfg.backend, device=device)
    from neodragon.utils.generation_utils import DEFAULT_PROMPT_MODIFIER

    native_all = encode_native_conditions(
        backend,
        prompts,
        modifier=DEFAULT_PROMPT_MODIFIER,
        batch_size=args.condition_batch_size,
    )
    bridge80_all, _ = encode_bridge_conditions(
        cfg,
        Path(args.bridge80),
        prompts,
        modifier=DEFAULT_PROMPT_MODIFIER,
        device=device,
        dtype=dtype,
        batch_size=args.condition_batch_size,
    )
    bridge64_all, _ = encode_bridge_conditions(
        cfg,
        Path(args.bridge64),
        prompts,
        modifier=DEFAULT_PROMPT_MODIFIER,
        device=device,
        dtype=dtype,
        batch_size=args.condition_batch_size,
    )
    released = backend.pipeline.dit
    scheduler = backend.pipeline.scheduler
    vae = backend.pipeline.vae

    exp6, _, _, _ = load_neodragon_train_modules(
        cfg, device, dtype, load_vae=False
    )
    exp6_state, exp6_meta = checkpoint_state(Path(args.exp6), "dit")
    exp6.load_state_dict(exp6_state, strict=True)
    exp6.eval().requires_grad_(False)
    del exp6_state

    (
        monolithic_text,
        monolithic_adapter,
        monolithic,
        _,
        negative_prompt,
    ) = load_monolithic_teacher(cfg, device=device, dtype=dtype)
    monolithic.eval()

    rows = []
    for start in tqdm(
        range(0, len(prompts), args.local_batch_size),
        desc="Local 18-transition banks",
    ):
        end = min(start + args.local_batch_size, len(prompts))
        batch_prompts = prompts[start:end]
        set_seed(args.seed + start)
        anchor_latents = image_batch_to_anchor(
            anchors[start:end],
            vae=vae,
            height=args.height,
            width=args.width,
            device=device,
            dtype=dtype,
        )
        noises = []
        for index in range(start, end):
            generator = torch.Generator(device=device).manual_seed(
                args.seed + 100_000 + index
            )
            noises.append(
                torch.randn(
                    (1, 16, 7, anchor_latents.shape[-2], anchor_latents.shape[-1]),
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
            )
        full_noise = torch.cat(noises, dim=0)
        teacher = native_condition(
            text_bundle=monolithic_text,
            context_adapter=monolithic_adapter,
            prompts=[prompt + DEFAULT_PROMPT_MODIFIER for prompt in batch_prompts],
            negative_prompt=negative_prompt,
            device=device,
            guidance_scale=5.0,
        )
        banks = (
            ("native", native_all, True),
            ("bridge80", bridge80_all, True),
            ("bridge64", bridge64_all, False),
        )
        for bank_index, (name, values, evaluate_exp6) in enumerate(banks):
            condition = dit_condition(
                condition_slice(values, start, end, device=device, dtype=dtype)
            )
            generator = torch.Generator(device=device).manual_seed(
                args.seed + 900_000 + start
            )
            rows.extend(
                evaluate_state_bank(
                    name=name,
                    released=released,
                    exp6=exp6,
                    monolithic=monolithic,
                    scheduler=scheduler,
                    anchors=anchor_latents,
                    full_noise=full_noise,
                    bank_condition=condition,
                    teacher_condition=teacher,
                    prompt_offset=start,
                    prompts=batch_prompts,
                    generator=generator,
                    evaluate_exp6=evaluate_exp6,
                )
            )

    aggregate = aggregate_local(rows)
    summary = {
        "status": "ok",
        "protocol": {
            "num_prompts": len(prompts),
            "state_actor": "released Hybrid separately for each condition bank",
            "shared_native_ssd1b_anchor": True,
            "shared_initial_and_corrective_noise": True,
            "positions": "6 units x 3 stages",
            "target": "native-CFG Monolithic 10-step stage endpoint",
            "released_and_exp6_steps": 1,
            "dtype": str(dtype),
        },
        "exp6": exp6_meta,
        "aggregate": aggregate,
        "rows": rows,
    }
    (output_dir / "local_transition_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    make_local_heatmaps(aggregate, output_dir / "local_transition_heatmaps.png")
    return summary


def metric_delta(aggregate: dict, left: str, right: str, metric: str) -> float:
    return float(
        aggregate[left][metric]["mean"] - aggregate[right][metric]["mean"]
    )


def metric_delta_percent(aggregate: dict, left: str, right: str, metric: str) -> float:
    baseline = float(aggregate[right][metric]["mean"])
    return 100.0 * metric_delta(aggregate, left, right, metric) / max(abs(baseline), 1e-8)


def run_report(output_dir: Path) -> None:
    full = json.loads((output_dir / "full_rollout_metrics.json").read_text())
    local = json.loads((output_dir / "local_transition_metrics.json").read_text())
    full_agg = full["aggregate"]
    local_agg = local["aggregate"]
    metrics = [
        ("Prompt CLIP", "prompt_clip_mean"),
        ("Subject CLIP", "subject_clip_mean"),
        ("Temporal CLIP", "temporal_clip_adjacent"),
        ("Adjacent MAE", "adjacent_rgb_mae"),
        ("Optical flow", "optical_flow"),
        ("First-last MAE", "first_last_rgb_mae"),
        ("Mean sharpness", "mean_sharpness"),
        ("Last sharpness", "last_frame_sharpness"),
        ("Flicker proxy", "flicker_proxy"),
        ("Saturation", "mean_saturation"),
    ]
    lines = [
        "# Exp6 Decision Matrix",
        "",
        f"Prompts: **{full['protocol']['num_prompts']}** stratified VBench prompts. ",
        "All configurations use the same native SSD1B anchor, prompt modifier, BF16, generation seed, and Hybrid 1-1-1 schedule.",
        "",
        "## Configurations",
        "",
        "| ID | Configuration |",
        "| --- | --- |",
    ]
    lines.extend(f"| {key} | {CONFIG_NAMES[key]} |" for key in FULL_ORDER)
    lines.extend(
        [
            "",
            "## Full-Rollout Metrics",
            "",
            "| Config | " + " | ".join(name for name, _ in metrics) + " |",
            "| --- | " + " | ".join("---:" for _ in metrics) + " |",
        ]
    )
    for config_id in FULL_ORDER:
        values = [full_agg[config_id][key]["mean"] for _, key in metrics]
        lines.append(
            f"| {config_id} | " + " | ".join(f"{value:.5f}" for value in values) + " |"
        )
    lines.extend(
        [
            "",
            "![Full-rollout examples](full_rollout_contact_sheet.jpg)",
            "",
            "## Paired DiT Effects",
            "",
            "Positive deltas below mean Exp6 is numerically larger than Released Hybrid. For flicker, lower is preferable; motion metrics require visual interpretation.",
            "",
            "| Comparison | Prompt CLIP | Subject CLIP | Temporal CLIP | Adjacent MAE | Optical flow | First-last MAE | Sharpness | Flicker |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for exp6_id, released_id, label in (("C", "A", "Native condition"), ("D", "B", "MCP 80K condition")):
        values = [
            metric_delta(full_agg, exp6_id, released_id, key)
            for key in (
                "prompt_clip_mean",
                "subject_clip_mean",
                "temporal_clip_adjacent",
                "adjacent_rgb_mae",
                "optical_flow",
                "first_last_rgb_mae",
                "mean_sharpness",
                "flicker_proxy",
            )
        ]
        lines.append(
            f"| {label}: {exp6_id}-{released_id} | "
            + " | ".join(f"{value:+.5f}" for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Condition-Contract Effects",
            "",
            "Positive deltas mean the left-hand condition is numerically larger. A-B isolates the rollout-80K condition penalty on Released Hybrid; C-D measures the same condition shift after Exp6.",
            "",
            "| Comparison | Prompt CLIP | Subject CLIP | Temporal CLIP | Adjacent MAE | Optical flow | First-last MAE | Sharpness | Flicker |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for left, right, label in (
        ("A", "B", "Released condition effect"),
        ("C", "D", "Exp6 condition effect"),
        ("E", "A", "Exp1-64K vs native reference"),
    ):
        values = [
            metric_delta(full_agg, left, right, key)
            for key in (
                "prompt_clip_mean",
                "subject_clip_mean",
                "temporal_clip_adjacent",
                "adjacent_rgb_mae",
                "optical_flow",
                "first_last_rgb_mae",
                "mean_sharpness",
                "flicker_proxy",
            )
        ]
        lines.append(
            f"| {label}: {left}-{right} | "
            + " | ".join(f"{value:+.5f}" for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Local 18-Transition Diagnostic",
            "",
            "| State bank | Released rel-L2 | Exp6 rel-L2 | Released cosine | Exp6 cosine | Exp6/Released gap | Improved positions | >=10% positions | Mean improvement | Late units 4-5 | Unit 5/stage 2 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for bank in ("native", "bridge80"):
        item = local_agg[bank]
        bank_rows = [row for row in local["rows"] if row["bank"] == bank]

        def row_mean(key: str) -> float:
            return float(np.mean([row[key] for row in bank_rows]))

        lines.append(
            f"| {bank} | {row_mean('released_target_relative_l2'):.5f} | "
            f"{row_mean('exp6_target_relative_l2'):.5f} | "
            f"{row_mean('released_target_transition_cosine'):.5f} | "
            f"{row_mean('exp6_target_transition_cosine'):.5f} | "
            f"{row_mean('exp6_released_relative_l2'):.5f} | "
            f"{item['improved_positions']}/18 | "
            f"{item['positions_over_10_percent']}/18 | "
            f"{item['mean_improvement_percent']:.2f}% | "
            f"{item['late_unit_improvement_percent']:.2f}% | "
            f"{item['danger_position_u5_s2_percent']:.2f}% |"
        )

    native_flow = metric_delta_percent(full_agg, "C", "A", "optical_flow")
    mcp_flow = metric_delta_percent(full_agg, "D", "B", "optical_flow")
    native_change = metric_delta_percent(full_agg, "C", "A", "first_last_rgb_mae")
    mcp_change = metric_delta_percent(full_agg, "D", "B", "first_last_rgb_mae")
    local_pass = all(
        local_agg[bank]["improved_positions"] >= 12
        and local_agg[bank]["late_unit_improvement_percent"] >= 10.0
        for bank in ("native", "bridge80")
    )
    motion_collapse = all(
        value <= -20.0 for value in (native_flow, mcp_flow, native_change, mcp_change)
    )
    if local_pass and motion_collapse:
        decision = (
            "**Outcome 2: local transitions improve, but the full rollout does not.** "
            "The evidence supports a short-horizon credit-assignment problem; do not extend "
            "one-step Exp6 training. The next allowed training attempt is the proposed "
            "three-call truncated-BPTT pilot."
        )
    elif not local_pass:
        decision = (
            "**Outcome 4: Exp6 lacks a robust local improvement signal.** Stop Exp6 and keep "
            "the released Hybrid policy as the generator mainline."
        )
    else:
        decision = (
            "**The matrix is not cleanly classified by the pre-registered rules.** Inspect "
            "the paired metrics and videos before selecting Outcome 1 or Outcome 3."
        )
    lines.extend(
        [
            "",
            "![Local transition heatmaps](local_transition_heatmaps.png)",
            "",
            "## Decision Readout",
            "",
            decision,
            "",
            f"- Native-condition Exp6 optical flow: **{native_flow:+.2f}%**; first-last change: **{native_change:+.2f}%**.",
            f"- MCP-80K Exp6 optical flow: **{mcp_flow:+.2f}%**; first-last change: **{mcp_change:+.2f}%**.",
            f"- Local criterion passed: **{local_pass}**; severe full-rollout motion reduction detected: **{motion_collapse}**.",
            "",
            "## Interpretation Guardrails",
            "",
            "- Prompt CLIP is the prompt-adherence proxy; first-frame CLIP preservation is the subject-preservation proxy.",
            "- Optical flow and RGB change measure motion magnitude, not motion quality.",
            "- The flicker score is a second-order luminance residual and remains motion-confounded.",
            "- Local state banks are generated by Released Hybrid under each condition contract; full rollout captures Student state drift.",
            "- A training decision must use both local and full-rollout evidence, not endpoint loss alone.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Exp6 A-E decision matrix.")
    parser.add_argument("--mode", choices=["all", "full", "local", "report"], default="all")
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--exp6", required=True)
    parser.add_argument("--bridge80", required=True)
    parser.add_argument("--bridge64", required=True)
    parser.add_argument(
        "--prompt-file",
        default="checkpoints/neodragon_repo/prompts/vbench_prompts.txt",
    )
    parser.add_argument("--output-dir", default="output/exp6_decision_matrix_20260731")
    parser.add_argument("--num-prompts", type=int, default=96)
    parser.add_argument("--max-prompts", type=int, default=-1)
    parser.add_argument("--prompt-seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--condition-batch-size", type=int, default=4)
    parser.add_argument("--local-batch-size", type=int, default=2)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = selected_prompts(args, output_dir)
    if args.mode != "report" and not torch.cuda.is_available():
        raise RuntimeError("Exp6 decision matrix requires a CUDA allocation.")
    if args.mode in {"all", "full"}:
        run_full(args, output_dir, prompts)
    if args.mode in {"all", "local"}:
        run_local(args, output_dir, prompts)
    if args.mode in {"all", "report"}:
        run_report(output_dir)
    print(f"Exp6 decision matrix complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
