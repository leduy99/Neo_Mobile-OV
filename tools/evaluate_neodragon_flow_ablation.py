#!/usr/bin/env python
"""Generate controlled native-text rollouts for short NeoDragon flow ablations."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.checkpoints import ensure_neodragon_assets
from new_mobile_ov.config import load_config
from tools import audit_neodragon_dmd_motion as motion
from tools.audit_neodragon_monolithic_bridge_dmd import tensor_similarity
from tools.evaluate_neodragon_joint_monolithic import (
    DEFAULT_PROMPTS,
    decoded_pair_metrics,
    encode_native_conditions,
    make_contact_sheet,
    prompt_noise,
    run_cfg_rollout,
    save_system,
)
from tools.train_neodragon_dit_bridge import dtype_from_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--dtype", default="bf16")
    return parser.parse_args()


def require_slurm_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This evaluator requires a CUDA allocation.")
    if not (os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_STEP_ID")):
        raise RuntimeError("Run this evaluator through srun or sbatch.")


def load_dit(
    *,
    local_model_path: str,
    dit_id: str,
    checkpoint: Path | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    from neodragon.pyramid_mmdit import PyramidMMDiT

    model = PyramidMMDiT.from_pretrained(
        f"{local_model_path}/{dit_id}",
        torch_dtype=dtype,
    )
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
        model.load_state_dict(payload["dit"], strict=True)
        del payload
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main() -> None:
    args = parse_args()
    require_slurm_cuda()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = args.prompt or list(DEFAULT_PROMPTS)
    checkpoints = {
        path.parent.name: path
        for path in sorted(run_dir.glob("*/neodragon_ablation_dit.pt"))
    }
    if not checkpoints:
        raise FileNotFoundError(f"No ablation checkpoints below {run_dir}")

    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    cfg = load_config(args.config)
    repo_path, _, local_model_path = ensure_neodragon_assets(
        repo_path=cfg.backend.extra.get("repo_path"),
        cache_dir=cfg.backend.extra.get("cache_dir"),
        model_id=cfg.backend.extra.get("model_id", "karnewar/Neodragon"),
        repo_url=cfg.backend.extra.get("repo_url"),
    )
    repo_path = Path(repo_path).resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    from neodragon import MULTISTEP_CONTEXT_ADAPTER_ID, MULTISTEP_DIT_ID, VAE_ID
    from neodragon.asymmetric_causal_video_vae import AsymmetricCausalVideoVAE
    from neodragon.context_adapter import ContextAdapter
    from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler
    from neodragon.text_encoder_bundle import TextEncoderBundle
    from neodragon.utils.generation_utils import DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT_MODIFIER

    text = TextEncoderBundle.from_pretrained(local_model_path, torch_dtype=dtype).to(device).eval()
    adapter = ContextAdapter.from_pretrained(
        f"{local_model_path}/{MULTISTEP_CONTEXT_ADAPTER_ID}",
        torch_dtype=dtype,
    ).to(device).eval()
    positives, negative = encode_native_conditions(
        text=text,
        adapter=adapter,
        prompts=prompts,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        modifier=DEFAULT_PROMPT_MODIFIER,
        device=device,
    )
    del text, adapter
    torch.cuda.empty_cache()

    vae = AsymmetricCausalVideoVAE.from_pretrained(
        f"{local_model_path}/{VAE_ID}",
        torch_dtype=dtype,
    ).to(device).eval()
    scheduler = PyramidFlowMatchEulerDiscreteScheduler()
    channels = 16
    systems: list[tuple[str, Path | None]] = [("released_multistep", None), *checkpoints.items()]
    report: dict[str, Any] = {"prompts": {}, "systems": [name for name, _ in systems]}
    baseline_latents: dict[int, torch.Tensor] = {}
    baseline_frames: dict[int, Any] = {}
    contact_frames: dict[int, dict[str, Any]] = {index: {} for index in range(len(prompts))}

    for system_index, (name, checkpoint) in enumerate(systems):
        print(f"Controlled rollout: {name}", flush=True)
        dit = load_dit(
            local_model_path=local_model_path,
            dit_id=MULTISTEP_DIT_ID,
            checkpoint=checkpoint,
            device=device,
            dtype=dtype,
        )
        for prompt_index, (prompt, positive) in enumerate(zip(prompts, positives, strict=True)):
            full_noise = prompt_noise(
                args=args,
                index=prompt_index,
                channels=channels,
                device=device,
                dtype=dtype,
            )
            rollout, seconds = run_cfg_rollout(
                dit=dit,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                full_noise=full_noise,
                transition_seed=args.seed + prompt_index * 1009 + 77,
                device=device,
                dtype=dtype,
            )
            prompt_dir = output_dir / f"prompt_{prompt_index:02d}"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            metrics, latents, frames = save_system(
                name=name,
                rollout=rollout,
                seconds=seconds,
                vae=vae,
                prompt_dir=prompt_dir,
                fps=args.fps,
            )
            contact_frames[prompt_index][name] = frames
            if system_index == 0:
                baseline_latents[prompt_index] = latents
                baseline_frames[prompt_index] = frames
            else:
                metrics["vs_released_latent"] = tensor_similarity(
                    baseline_latents[prompt_index], latents
                )
                metrics["vs_released_decoded"] = decoded_pair_metrics(
                    baseline_frames[prompt_index], frames
                )
            prompt_report = report["prompts"].setdefault(
                str(prompt_index),
                {"prompt": prompt, "systems": {}},
            )
            prompt_report["systems"][name] = metrics
        del dit
        gc.collect()
        torch.cuda.empty_cache()

    for prompt_index, frames in contact_frames.items():
        make_contact_sheet(output_dir / f"prompt_{prompt_index:02d}_contact.jpg", frames)
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
