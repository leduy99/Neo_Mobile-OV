#!/usr/bin/env python
"""Render a controlled V8/V9/V10 DreamLite bridge image comparison."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.bridge import MobileOVDreamLiteImageBridge
from new_mobile_ov.config import load_config
from new_mobile_ov.generation.backends import DreamLiteMobileBackend


def dtype_from_name(value: str) -> torch.dtype:
    normalized = value.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype={value!r}")


def parse_variant(value: str) -> tuple[str, Path]:
    name, separator, checkpoint = value.partition("=")
    if not separator or not name or not checkpoint:
        raise argparse.ArgumentTypeError(
            "--variant must have the form NAME=CHECKPOINT"
        )
    return name, Path(checkpoint)


def load_prompts(path: Path, max_prompts: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    prompts = [
        {"name": row["name"].strip(), "prompt": row["prompt"].strip()}
        for row in rows
        if row.get("name", "").strip() and row.get("prompt", "").strip()
    ]
    if max_prompts > 0:
        prompts = prompts[:max_prompts]
    if not prompts:
        raise ValueError(f"No valid prompts found in {path}")
    return prompts


def load_bridge(config, checkpoint: Path, device: torch.device, dtype: torch.dtype):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    bridge = MobileOVDreamLiteImageBridge(
        config.bridge,
        config.dreamlite_bridge,
        device=device,
        dtype=dtype,
    ).eval()
    bridge.load_trainable_state_dict(payload.get("bridge", payload))
    return bridge, int(payload.get("step", -1))


def write_sheet(
    output_dir: Path,
    variants: list[tuple[str, Path]],
    prompts: list[dict[str, str]],
) -> Path:
    thumb_width, thumb_height = 384, 240
    label_width, header_height, row_height = 230, 60, 274
    sheet = Image.new(
        "RGB",
        (label_width + thumb_width * len(variants), header_height + row_height * len(prompts)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text(
        (12, 12),
        "Controlled DreamLite comparison: same prompt, noise, size, and schedule per row",
        fill="black",
        font=font,
    )
    for index, (name, _) in enumerate(variants):
        x = label_width + index * thumb_width + 12
        draw.text((x, 38), name, fill="black", font=font)

    for row_index, prompt in enumerate(prompts):
        y = header_height + row_index * row_height
        draw.text((12, y + 12), prompt["name"], fill="black", font=font)
        for variant_index, (variant, _) in enumerate(variants):
            path = output_dir / "images" / variant / f"{row_index:02d}_{prompt['name']}.png"
            if not path.is_file():
                raise FileNotFoundError(f"Missing rendered image: {path}")
            with Image.open(path) as source:
                image = source.convert("RGB").resize(
                    (thumb_width, thumb_height), Image.Resampling.LANCZOS
                )
            sheet.paste(image, (label_width + variant_index * thumb_width, y + 30))
    sheet_path = output_dir / "comparison_sheet.png"
    sheet.save(sheet_path)
    return sheet_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic, side-by-side DreamLite bridge comparison."
    )
    parser.add_argument("--config", default="configs/mobile_ov_dreamlite_compact_v10.yaml")
    parser.add_argument(
        "--prompts-csv",
        default="configs/prompts/dreamlite_condition_ablation_10.csv",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--variant",
        action="append",
        type=parse_variant,
        required=True,
        metavar="NAME=CHECKPOINT",
    )
    parser.add_argument("--max-prompts", type=int, default=10)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("DreamLite comparison requires CUDA.")
    variants: list[tuple[str, Path]] = args.variant
    names = [name for name, _ in variants]
    if len(set(names)) != len(names):
        raise ValueError(f"Variant names must be unique: {names}")
    for name, checkpoint in variants:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing {name} checkpoint: {checkpoint}")

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(Path(args.prompts_csv), args.max_prompts)
    config = load_config(args.config)
    dtype = dtype_from_name(args.dtype)
    device = torch.device("cuda")
    backend = DreamLiteMobileBackend(config.dreamlite, device=device, dtype=dtype, load_vae=True)
    started = time.perf_counter()
    manifest: dict[str, object] = {
        "protocol": {
            "seed_policy": "seed + prompt_index; identical across variants",
            "base_seed": args.seed,
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "dtype": args.dtype,
            "config": args.config,
        },
        "prompts": prompts,
        "variants": {},
    }

    for variant, checkpoint in variants:
        print(f"Loading {variant}: {checkpoint}", flush=True)
        bridge, step = load_bridge(config, checkpoint, device=device, dtype=dtype)
        variant_dir = output_dir / "images" / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        records = []
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=dtype, enabled=dtype != torch.float32
        ):
            for prompt_index, prompt in enumerate(prompts):
                seed = args.seed + prompt_index
                image = backend.generate_images(
                    bridge([prompt["prompt"]], mode="generate"),
                    height=args.height,
                    width=args.width,
                    num_steps=args.steps,
                    seed=seed,
                )[0].convert("RGB")
                path = variant_dir / f"{prompt_index:02d}_{prompt['name']}.png"
                image.save(path)
                records.append({"name": prompt["name"], "seed": seed, "path": str(path)})
                print(
                    f"{variant}: {prompt_index + 1}/{len(prompts)} {prompt['name']}",
                    flush=True,
                )
        manifest["variants"][variant] = {
            "checkpoint": str(checkpoint),
            "step": step,
            "images": records,
        }
        del bridge
        gc.collect()
        torch.cuda.empty_cache()

    sheet_path = write_sheet(output_dir, variants, prompts)
    manifest["seconds"] = time.perf_counter() - started
    manifest["comparison_sheet"] = str(sheet_path)
    manifest_path = output_dir / "comparison_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved sheet: {sheet_path}", flush=True)
    print(f"Saved manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
