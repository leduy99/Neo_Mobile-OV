#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from diffusers.utils import export_to_video

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.bridge import MobileOVNeodragonTextBridge, MobileOVTextBridge
from new_mobile_ov.checkpoints import ensure_neodragon_assets
from new_mobile_ov.config import load_config
from new_mobile_ov.generation import build_generation_backend


def dtype_from_name(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def safe_stem(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9._ -]+", "_", text).strip().replace(" ", "_")
    return (text[:max_len] or "prompt").strip("_")


def load_neodragon_prompt_modifier(cfg) -> str:
    repo_path, _, _ = ensure_neodragon_assets(
        repo_path=cfg.backend.extra.get("repo_path"),
        cache_dir=cfg.backend.extra.get("cache_dir"),
        model_id=cfg.backend.extra.get("model_id", "karnewar/Neodragon"),
        repo_url=cfg.backend.extra.get("repo_url"),
    )
    repo_path = Path(repo_path).expanduser().resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    from neodragon.utils.generation_utils import DEFAULT_PROMPT_MODIFIER

    return DEFAULT_PROMPT_MODIFIER


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--prompt", default="A red fox walking through gentle snowfall.")
    parser.add_argument("--output-dir", default="output/mobile_ov_neodragon_smoke")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--skip-bridge", action="store_true")
    parser.add_argument("--condition-source", choices=["native", "bridge"], default="native")
    parser.add_argument("--bridge-ckpt", default=None, help="Optional Neodragon-shaped bridge checkpoint.")
    parser.add_argument("--dit-ckpt", default=None, help="Optional checkpoint supplying a separate `dit` state.")
    parser.add_argument(
        "--load-checkpoint-dit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically load `dit` from --bridge-ckpt when present.",
    )
    parser.add_argument(
        "--bridge-append-prompt-modifier",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Append Neodragon DEFAULT_PROMPT_MODIFIER before bridge encoding. Defaults to true for mobile_ov_neodragon.",
    )
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("FSDP_USE_ORIG_PARAMS", "true")

    cfg = load_config(args.config)
    if args.dtype:
        cfg.backend.dtype = args.dtype
        cfg.train.dtype = args.dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(cfg.backend.dtype)
    if device.type == "cpu":
        dtype = torch.float32

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, object] = {
        "config": args.config,
        "prompt": args.prompt,
        "device": str(device),
        "dtype": str(dtype),
        "backend": cfg.backend.name,
        "seed": args.seed,
    }

    bridge_outputs = None
    dit_state = None
    if not args.skip_bridge:
        t0 = time.time()
        bridge_prompt = args.prompt
        append_modifier = (
            cfg.backend.name == "mobile_ov_neodragon"
            if args.bridge_append_prompt_modifier is None
            else args.bridge_append_prompt_modifier
        )
        if append_modifier and cfg.backend.name == "mobile_ov_neodragon":
            bridge_prompt = bridge_prompt + load_neodragon_prompt_modifier(cfg)
        if cfg.backend.name == "mobile_ov_neodragon":
            bridge = MobileOVNeodragonTextBridge(cfg.bridge, device=device, dtype=dtype).eval()
        else:
            bridge = MobileOVTextBridge(cfg.bridge, device=device, dtype=dtype).eval()
        if args.bridge_ckpt:
            ckpt = torch.load(args.bridge_ckpt, map_location="cpu", weights_only=False)
            state = ckpt.get("bridge", ckpt.get("student_state", ckpt))
            missing, unexpected = bridge.load_state_dict(state, strict=False)
            checkpoint_dit = ckpt.get("dit")
            if args.load_checkpoint_dit and not args.dit_ckpt:
                dit_state = checkpoint_dit
            metrics["bridge_ckpt"] = args.bridge_ckpt
            metrics["checkpoint_step"] = ckpt.get("step")
            metrics["checkpoint_has_dit"] = checkpoint_dit is not None
            metrics["bridge_ckpt_missing"] = len(missing)
            metrics["bridge_ckpt_unexpected"] = len(unexpected)
            if missing or unexpected:
                raise RuntimeError(
                    "Bridge checkpoint does not exactly match the inference architecture: "
                    f"missing={missing[:10]} unexpected={unexpected[:10]}"
                )
            del checkpoint_dit, state, ckpt
        with torch.no_grad():
            prompt_embeds, prompt_mask, pooled = bridge.encode([bridge_prompt])
        bridge_outputs = (prompt_embeds, prompt_mask, pooled)
        metrics.update(
            {
                "bridge_seconds": time.time() - t0,
                "bridge_prompt": bridge_prompt,
                "bridge_append_prompt_modifier": append_modifier,
                "bridge_prompt_embeds_shape": list(prompt_embeds.shape),
                "bridge_prompt_mask_shape": list(prompt_mask.shape),
                "bridge_pooled_shape": list(pooled.shape),
                "bridge_pooled_norm": float(pooled.float().norm(dim=-1).mean().cpu()),
            }
        )
        del bridge
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.dit_ckpt:
        dit_ckpt = torch.load(args.dit_ckpt, map_location="cpu", weights_only=False)
        dit_state = dit_ckpt.get("dit", dit_ckpt)
        metrics["dit_ckpt"] = args.dit_ckpt
        metrics["dit_checkpoint_step"] = dit_ckpt.get("step") if isinstance(dit_ckpt, dict) else None
        del dit_ckpt

    t0 = time.time()
    backend = build_generation_backend(cfg.backend, device=device)
    metrics["backend_load_seconds"] = time.time() - t0
    if dit_state is not None:
        dit = getattr(getattr(backend, "pipeline", None), "dit", None)
        if dit is None:
            raise RuntimeError("Checkpoint contains DiT weights, but the backend has no pipeline.dit module.")
        missing, unexpected = dit.load_state_dict(dit_state, strict=False)
        metrics["dit_ckpt_loaded"] = True
        metrics["dit_ckpt_missing"] = len(missing)
        metrics["dit_ckpt_unexpected"] = len(unexpected)
        if missing or unexpected:
            raise RuntimeError(
                "DiT checkpoint does not exactly match the inference architecture: "
                f"missing={missing[:10]} unexpected={unexpected[:10]}"
            )
        del dit_state
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        metrics["dit_ckpt_loaded"] = False

    height = int(args.height or cfg.data.height)
    width = int(args.width or cfg.data.width)
    num_frames = int(args.num_frames or cfg.data.frame_num)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    t0 = time.time()
    if args.condition_source == "bridge":
        if bridge_outputs is None:
            raise RuntimeError("--condition-source bridge requires bridge encoding; remove --skip-bridge.")
        prompt_embeds, prompt_mask, pooled = bridge_outputs
        frames = backend.generate_video_from_bridge_condition(
            args.prompt,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            pooled_prompt_embeds=pooled,
            height=height,
            width=width,
            num_frames=num_frames,
            profile=args.profile,
        )
    else:
        frames = backend.generate_video_from_prompt(
            args.prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            profile=args.profile,
        )
    metrics["condition_source"] = args.condition_source
    metrics["generation_seconds"] = time.time() - t0
    metrics["num_output_frames"] = len(frames)
    metrics["height"] = height
    metrics["width"] = width
    metrics["num_frames"] = num_frames

    video_path = out_dir / f"{safe_stem(args.prompt)}.mp4"
    export_to_video(frames, video_path, fps=args.fps)
    metrics["video_path"] = str(video_path)
    metrics["video_bytes"] = video_path.stat().st_size
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
