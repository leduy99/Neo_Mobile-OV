#!/usr/bin/env python
# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.bridge import MobileOVDreamLiteImageBridge
from new_mobile_ov.config import load_config
from new_mobile_ov.generation.backends import DreamLiteMobileBackend


def dtype_from_name(value: str) -> torch.dtype:
    if value.lower() in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value.lower() in {"fp16", "float16"}:
        return torch.float16
    if value.lower() in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype={value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or edit with the Mobile-OV DreamLite bridge.")
    parser.add_argument("--config", default="configs/mobile_ov_dreamlite.yaml")
    parser.add_argument("--bridge-checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--source-image", default=None)
    parser.add_argument("--output", default="output/dreamlite_bridge_inference.png")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("DreamLite inference requires CUDA.")
    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    config = load_config(args.config)
    bridge = MobileOVDreamLiteImageBridge(
        config.bridge,
        config.dreamlite_bridge,
        device=device,
        dtype=dtype,
    )
    payload = torch.load(args.bridge_checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("bridge", payload)
    bridge.load_trainable_state_dict(state)
    bridge.eval()
    backend = DreamLiteMobileBackend(
        config.dreamlite,
        device=device,
        dtype=dtype,
        load_vae=True,
    )
    source_images = None
    mode = "generate"
    if args.source_image:
        source_images = [Image.open(args.source_image).convert("RGB")]
        mode = "edit"
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        condition = bridge(
            [args.prompt],
            mode=mode,
            images=source_images,
        )
        images = backend.generate_images(
            condition,
            source_images=source_images,
            height=args.height,
            width=args.width,
            num_steps=args.steps,
            seed=args.seed,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output)
    print(
        f"Saved {mode} output={output} size={images[0].size} "
        f"condition={tuple(condition.prompt_embeds.shape)}"
    )


if __name__ == "__main__":
    main()
