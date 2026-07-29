#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.bridge import MobileOVSSD1BImageBridge, SSD1BImageCondition  # noqa: E402
from new_mobile_ov.checkpoints import ensure_neodragon_assets  # noqa: E402
from new_mobile_ov.config import load_config  # noqa: E402
from new_mobile_ov.training.ssd1b_distillation import SSD1B_TIMESTEPS  # noqa: E402


DEFAULT_PROMPTS = (
    "A red panda eating bamboo.",
    "A surfer riding a large ocean wave.",
    "A golden retriever runs through a field of yellow flowers.",
    "A young adult dances gracefully on a sunlit beach.",
    "An astronaut explores a crystalline cave illuminated by blue light, cinematic photography.",
    "A chef prepares fresh pasta in a warm rustic kitchen while afternoon sunlight enters through the window.",
)


def dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def safe_stem(text: str, max_len: int = 64) -> str:
    value = re.sub(r"[^a-zA-Z0-9._ -]+", "_", text).strip().replace(" ", "_")
    return (value[:max_len] or "prompt").strip("_")


def cosine_distance(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.float().reshape(prediction.shape[0], -1)
    target = target.float().reshape(target.shape[0], -1)
    return float((1.0 - F.cosine_similarity(prediction, target, dim=-1)).mean().cpu())


def normalized_mse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = F.normalize(prediction.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    return float(F.mse_loss(prediction, target).cpu())


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt:
        return [value.strip() for value in args.prompt if value.strip()]
    if args.prompt_file:
        values = Path(args.prompt_file).read_text(encoding="utf-8").splitlines()
        return [value.strip() for value in values if value.strip()]
    return list(DEFAULT_PROMPTS)


def resolve_neodragon(cfg) -> tuple[Path, Path]:
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


def load_bridge(
    cfg,
    checkpoint_path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[MobileOVSSD1BImageBridge, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "image_bridge" not in checkpoint:
        raise KeyError(f"Checkpoint has no `image_bridge` state: {checkpoint_path}")
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
        raise RuntimeError(f"Checkpoint contains non-finite tensors: {non_finite}")
    metadata = {
        "step": int(checkpoint.get("step", -1)),
        "target": checkpoint.get("target"),
        "architecture": checkpoint.get("architecture"),
        "distillation": checkpoint.get("distillation"),
        "state_tensors": len(checkpoint["image_bridge"]),
        "state_numel": sum(
            value.numel()
            for value in checkpoint["image_bridge"].values()
            if torch.is_tensor(value)
        ),
    }
    return bridge, metadata


def split_native_condition(
    prompt_embeds: torch.Tensor,
    pooled: torch.Tensor,
) -> SSD1BImageCondition:
    return SSD1BImageCondition(
        clip_l_tokens=prompt_embeds[..., :768],
        clip_big_g_tokens=prompt_embeds[..., 768:],
        pooled=pooled,
    )


def condition_metrics(
    student: SSD1BImageCondition,
    native: SSD1BImageCondition,
) -> dict[str, float]:
    return {
        "clip_l_normalized_mse": normalized_mse(student.clip_l_tokens, native.clip_l_tokens),
        "clip_l_cosine_distance": cosine_distance(student.clip_l_tokens, native.clip_l_tokens),
        "clip_big_g_normalized_mse": normalized_mse(
            student.clip_big_g_tokens,
            native.clip_big_g_tokens,
        ),
        "clip_big_g_cosine_distance": cosine_distance(
            student.clip_big_g_tokens,
            native.clip_big_g_tokens,
        ),
        "pooled_mse": float(F.mse_loss(student.pooled.float(), native.pooled.float()).cpu()),
        "pooled_cosine_distance": cosine_distance(student.pooled, native.pooled),
        "student_pooled_norm": float(student.pooled.float().norm(dim=-1).mean().cpu()),
        "native_pooled_norm": float(native.pooled.float().norm(dim=-1).mean().cpu()),
    }


@torch.inference_mode()
def unet_parity_metrics(
    pipe,
    student: SSD1BImageCondition,
    native: SSD1BImageCondition,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    generator = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn(
        1,
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
    ).unsqueeze(0)
    results: list[dict[str, float | int]] = []
    for timestep_value in SSD1B_TIMESTEPS:
        timestep = torch.tensor([timestep_value], device=device, dtype=torch.long)
        common = {
            "sample": latent,
            "timestep": timestep,
            "added_cond_kwargs": {"time_ids": time_ids},
            "return_dict": False,
        }
        native_prediction = pipe.unet(
            encoder_hidden_states=native.prompt_embeds,
            added_cond_kwargs={**common["added_cond_kwargs"], "text_embeds": native.pooled},
            sample=common["sample"],
            timestep=common["timestep"],
            return_dict=False,
        )[0]
        student_prediction = pipe.unet(
            encoder_hidden_states=student.prompt_embeds,
            added_cond_kwargs={**common["added_cond_kwargs"], "text_embeds": student.pooled},
            sample=common["sample"],
            timestep=common["timestep"],
            return_dict=False,
        )[0]
        results.append(
            {
                "timestep": timestep_value,
                "mse": float(
                    F.mse_loss(student_prediction.float(), native_prediction.float()).cpu()
                ),
                "cosine_distance": cosine_distance(student_prediction, native_prediction),
            }
        )
    return {
        "per_timestep": results,
        "mean_mse": sum(float(value["mse"]) for value in results) / len(results),
        "mean_cosine_distance": sum(
            float(value["cosine_distance"]) for value in results
        )
        / len(results),
    }


def labeled_pair(
    native_image: Image.Image,
    bridge_image: Image.Image,
    prompt: str,
) -> Image.Image:
    native_image = native_image.convert("RGB")
    bridge_image = bridge_image.convert("RGB")
    width = native_image.width + bridge_image.width
    header = 64
    canvas = Image.new("RGB", (width, native_image.height + header), "white")
    canvas.paste(native_image, (0, header))
    canvas.paste(bridge_image, (native_image.width, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), prompt[:150], fill="black")
    draw.text((8, 34), "Native SSD1B dual-CLIP", fill="black")
    draw.text((native_image.width + 8, 34), "Mobile-OV SSD1B Image Bridge", fill="black")
    return canvas


def make_contact_sheet(rows: list[Image.Image], output_path: Path) -> None:
    width = max(image.width for image in rows)
    height = sum(image.height for image in rows)
    sheet = Image.new("RGB", (width, height), "white")
    offset = 0
    for image in rows:
        sheet.paste(image, (0, offset))
        offset += image.height
    sheet.save(output_path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an SSD1B Image Bridge checkpoint against native dual-CLIP conditioning."
    )
    parser.add_argument("--config", default="configs/mobile_ov_ssd1b_image_bridge.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="output/ssd1b_image_bridge_step100k_test")
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--prompt-file")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument(
        "--append-prompt-modifier",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("SSD1B image generation requires a CUDA allocation.")
    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    prompts = load_prompts(args)
    if not prompts:
        raise ValueError("At least one prompt is required.")

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint)
    cfg = load_config(args.config)
    _, model_path = resolve_neodragon(cfg)

    from neodragon.first_frame_gen import SSD1B_FirstFrameGeneratorPipeline
    from neodragon.utils.generation_utils import DEFAULT_PROMPT_MODIFIER

    conditioned_prompts = [
        prompt + DEFAULT_PROMPT_MODIFIER if args.append_prompt_modifier else prompt
        for prompt in prompts
    ]

    start = time.perf_counter()
    bridge, checkpoint_metadata = load_bridge(
        cfg,
        checkpoint_path,
        device=device,
        dtype=dtype,
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        student_conditions = [
            SSD1BImageCondition(*(value.detach().cpu() for value in bridge([prompt])))
            for prompt in conditioned_prompts
        ]
    bridge_seconds = time.perf_counter() - start
    del bridge
    torch.cuda.empty_cache()

    start = time.perf_counter()
    pipe = SSD1B_FirstFrameGeneratorPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipeline_load_seconds = time.perf_counter() - start

    rows: list[Image.Image] = []
    prompt_metrics: list[dict[str, object]] = []
    for index, (prompt, conditioned_prompt, student_cpu) in enumerate(
        zip(prompts, conditioned_prompts, student_conditions)
    ):
        student = SSD1BImageCondition(
            *(value.to(device=device, dtype=dtype) for value in student_cpu)
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            native_prompt, _, native_pooled, _ = pipe.encode_prompt(
                prompt=conditioned_prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            native = split_native_condition(native_prompt, native_pooled)
            parity = unet_parity_metrics(
                pipe,
                student,
                native,
                seed=args.seed + index,
                device=device,
                dtype=dtype,
            )

            native_start = time.perf_counter()
            native_image = pipe(
                prompt_embeds=native.prompt_embeds,
                pooled_prompt_embeds=native.pooled,
                generator=torch.Generator(device=device).manual_seed(args.seed + index),
            ).images[0]
            native_seconds = time.perf_counter() - native_start

            bridge_start = time.perf_counter()
            bridge_image = pipe(
                prompt_embeds=student.prompt_embeds,
                pooled_prompt_embeds=student.pooled,
                generator=torch.Generator(device=device).manual_seed(args.seed + index),
            ).images[0]
            bridge_generation_seconds = time.perf_counter() - bridge_start

        stem = f"{index + 1:02d}_{safe_stem(prompt)}"
        native_path = image_dir / f"{stem}_native.png"
        bridge_path = image_dir / f"{stem}_bridge.png"
        pair_path = image_dir / f"{stem}_pair.jpg"
        native_image.save(native_path)
        bridge_image.save(bridge_path)
        pair = labeled_pair(native_image, bridge_image, prompt)
        pair.save(pair_path, quality=92)
        rows.append(pair)

        metrics = condition_metrics(student, native)
        prompt_metrics.append(
            {
                "index": index + 1,
                "prompt": prompt,
                "conditioned_prompt": conditioned_prompt,
                "seed": args.seed + index,
                "condition": metrics,
                "unet_parity": parity,
                "native_generation_seconds": native_seconds,
                "bridge_generation_seconds": bridge_generation_seconds,
                "native_image": str(native_path),
                "bridge_image": str(bridge_path),
                "pair_image": str(pair_path),
            }
        )
        print(
            f"[{index + 1}/{len(prompts)}] {prompt} "
            f"unet_mse={parity['mean_mse']:.6f} "
            f"native={native_seconds:.2f}s bridge={bridge_generation_seconds:.2f}s",
            flush=True,
        )

    contact_sheet_path = output_dir / "native_vs_bridge_contact_sheet.jpg"
    make_contact_sheet(rows, contact_sheet_path)
    summary = {
        "status": "ok",
        "config": args.config,
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": checkpoint_metadata,
        "device": str(device),
        "dtype": str(dtype),
        "num_prompts": len(prompts),
        "append_prompt_modifier": args.append_prompt_modifier,
        "bridge_batch_seconds": bridge_seconds,
        "pipeline_load_seconds": pipeline_load_seconds,
        "mean_unet_mse": sum(
            float(item["unet_parity"]["mean_mse"]) for item in prompt_metrics
        )
        / len(prompt_metrics),
        "mean_unet_cosine_distance": sum(
            float(item["unet_parity"]["mean_cosine_distance"]) for item in prompt_metrics
        )
        / len(prompt_metrics),
        "contact_sheet": str(contact_sheet_path),
        "prompts": prompt_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
