#!/usr/bin/env python
# ruff: noqa: E402
"""Run a trained Pyramidal-DMD student over six controlled one-step video units.

This evaluator intentionally keeps the first latent unit from the synthetic
monolithic sample fixed.  It therefore measures the DMD DiT only, without
mixing in a separate first-frame generator such as SSD1B or DreamLite.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.checkpoints import ensure_neodragon_assets
from new_mobile_ov.config import load_config
from new_mobile_ov.training.neodragon_pyramidal_dmd import DMDCondition, predict_flow
from new_mobile_ov.training.neodragon_rollout import (
    downsample_noise_2x,
    prepare_past_conditions,
    upsample_pyramidal_latent,
)


def dtype_from_name(name: str) -> torch.dtype:
    return torch.bfloat16 if str(name).lower() in {"bf16", "bfloat16"} else torch.float32


def load_models(cfg, checkpoint: Path, device: torch.device, dtype: torch.dtype):
    repo_path, _, local_model_path = ensure_neodragon_assets(
        repo_path=cfg.backend.extra.get("repo_path"),
        cache_dir=cfg.backend.extra.get("cache_dir"),
        model_id=cfg.backend.extra.get("model_id", "karnewar/Neodragon"),
        repo_url=cfg.backend.extra.get("repo_url"),
    )
    repo_path = Path(repo_path).expanduser().resolve()
    sys.path.insert(0, str(repo_path))
    from neodragon.context_adapter import ContextAdapter
    from neodragon.pyramid_mmdit import PyramidMMDiT
    from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler
    from neodragon.text_encoder_bundle import TextEncoderBundle

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schedule") != "hybrid_1-1-1_video_units_only":
        raise ValueError(f"Not a Pyramidal-DMD student checkpoint: {checkpoint}")
    adapter_id = str(payload["context_adapter_id"])
    dit_id = str(payload["teacher_dit_id"])
    text = TextEncoderBundle.from_pretrained(local_model_path, torch_dtype=dtype).to(device).eval()
    adapter = ContextAdapter.from_pretrained(
        f"{local_model_path}/{adapter_id}", torch_dtype=dtype
    ).to(device).eval()
    student = PyramidMMDiT.from_pretrained(
        f"{local_model_path}/{dit_id}", torch_dtype=dtype
    ).to(device).eval()
    student.load_state_dict(payload["student"], strict=True)
    for module in (text, adapter, student):
        module.requires_grad_(False)
    return text, adapter, student, PyramidFlowMatchEulerDiscreteScheduler(), payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--dtype", default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    cfg = load_config(args.config)
    manifest = Path(args.manifest).resolve()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not 0 <= args.index < len(rows):
        raise IndexError(f"index={args.index} outside manifest with {len(rows)} rows")
    row = rows[args.index]
    latent_path = Path(row["latent_path"])
    if not latent_path.is_absolute():
        latent_path = manifest.parent / latent_path
    source = torch.load(latent_path, map_location="cpu", weights_only=False)
    target = (source["latent"] if isinstance(source, dict) else source).unsqueeze(0).to(device, dtype=dtype)
    if target.shape[2] != 7:
        raise ValueError(f"Expected seven latent units, got {tuple(target.shape)}")
    text, adapter, student, scheduler, payload = load_models(
        cfg, Path(args.checkpoint), device, dtype
    )
    prompt = str(row["condition_prompt"])
    with torch.no_grad():
        tokens, mask, pooled = text([prompt], device)
        condition = DMDCondition(tokens=adapter(tokens), mask=mask, pooled=pooled)
        generator = torch.Generator(device=device).manual_seed(args.seed + args.index)
        full_noise = torch.randn(target.shape, device=device, dtype=dtype, generator=generator)
        low_noise = downsample_noise_2x(full_noise, 2)
        generated = [target[:, :, :1]]
        for unit in range(6):
            histories = prepare_past_conditions(generated, num_stages=3)
            current = low_noise[:, :, unit + 1 : unit + 2]
            for stage in range(3):
                if stage > 0:
                    current = upsample_pyramidal_latent(
                        current,
                        orig_sigma=1.0 - scheduler.orig_start_sigmas[stage],
                        gamma=scheduler.config.gamma,
                        generator=generator,
                    )
                timestep = scheduler.get_stage_timesteps(1, stage, device=device)[0]
                velocity = predict_flow(
                    dit=student,
                    current=current,
                    history=tuple(histories[stage]),
                    condition=condition,
                    timestep=timestep,
                )
                current = current - velocity
            generated.append(current)
        prediction = torch.cat(generated, dim=2)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"latent": prediction.cpu(), "prompt": prompt}, output_dir / "student_rollout_latent.pt")
    metrics = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(payload["step"]),
        "sample_index": args.index,
        "student_video_mse": float(torch.nn.functional.mse_loss(prediction[:, :, 1:].float(), target[:, :, 1:].float()).cpu()),
        "anchor_mse": float(torch.nn.functional.mse_loss(prediction[:, :, :1].float(), target[:, :, :1].float()).cpu()),
        "shape": list(prediction.shape),
        "schedule": "six units x three stages x one conditional DiT call",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
