#!/usr/bin/env python
"""Infer with the Stage-1 training condition contract, without an external anchor."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from PIL import Image

from tools.train_mobileov_stage1 import FORMAT, load_stack, autocast
from new_mobile_ov.training.stage1_alignment import frozen_signatures, JsonlRecords, prepare_sample, TASKS
from new_mobile_ov.training.stage1_alignment_data import load_sources
from new_mobile_ov.generation.neodragon_compat import install_neodragon_generation_patches
from tools.data_prepare.download_alignment_images import sha256_file


def pad_condition(condition, length):
    tokens, mask, pooled = condition
    extra = length - tokens.shape[1]
    if extra < 0:
        raise ValueError("CFG padding must not crop condition tokens")
    return F.pad(tokens, (0, 0, 0, extra)), F.pad(mask, (0, extra)), pooled


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--prompt", default="A red panda walking through a bamboo forest.")
    parser.add_argument("--reconstruct-image", type=Path)
    parser.add_argument("--validation-task", choices=TASKS)
    parser.add_argument("--data-root", type=Path, default=Path("download_data/data/univideo_stage1"))
    parser.add_argument("--validation-index", type=int, default=0)
    parser.add_argument("--frames", type=int, choices=(1, 49), default=49)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--first-steps", type=int, default=20)
    parser.add_argument("--video-steps", type=int, default=10)
    parser.add_argument("--guidance", type=float, default=7)
    parser.add_argument("--video-guidance", type=float, default=5)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run GPU inference inside srun/sbatch")
    if args.height % 64 or args.width % 64 or min(args.height, args.width, args.first_steps, args.video_steps) < 1:
        raise ValueError("Invalid shape/steps")
    if args.reconstruct_image and args.frames != 1:
        raise ValueError("Stage-1 reconstruction is an image task; use --frames 1")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError("Not a Stage-1 connector checkpoint")
    contract = payload["contract"]
    validation_sample = None
    if args.validation_task:
        if args.reconstruct_image:
            raise ValueError("Use either a validation sample or an external reconstruction image")
        if sha256_file(args.data_root / "stage1_summary.json") != contract["data_summary_sha256"]:
            raise ValueError("Validation release differs from training")
        records = JsonlRecords(args.data_root / "validation" / f"{args.validation_task}.jsonl")
        if not 0 <= args.validation_index < len(records):
            raise ValueError("Invalid validation index")
        validation_sample = prepare_sample(records[args.validation_index], load_sources(args.data_root),
                                            contract["short_side"], contract["long_side"])
        args.prompt = validation_sample["prompt"]
        args.frames = 49 if args.validation_task == "t2v" else 1
        args.height, args.width = validation_sample["video"].shape[-2:]
    if sha256_file(Path(args.config)) != contract["config_sha256"]:
        raise ValueError("Inference configuration differs from training")
    args.processor, args.max_tokens = contract["processor"], contract["max_tokens"]
    device = torch.device("cuda", 0)
    torch.manual_seed(args.seed)
    cfg, encoder, connector, dit, vae, scheduler = load_stack(args, device)
    if encoder.processor_sha256 != contract.get("processor_sha256"):
        raise ValueError("Processor differs from training")
    if frozen_signatures(dict(smolvlm2=encoder, dit=dit, vae=vae)) != contract["frozen_weights_sha256"]:
        raise ValueError("Frozen generator/VLM/VAE weights differ from training")
    if sha256_file(Path(cfg.bridge.smolvlm2_ckpt_path)) != contract["smolvlm2_sha256"]:
        raise ValueError("SmolVLM2 weights differ from training")
    connector.load_state_dict(payload["connector"], strict=True)
    connector.eval()
    del payload
    image = validation_sample["image"] if validation_sample is not None else None
    if args.reconstruct_image:
        with Image.open(args.reconstruct_image) as source:
            image = source.convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS)
    with autocast(device):
        positive = connector(*encoder("" if image is not None else args.prompt, image))
        negative = connector(*encoder("", drop_condition=True))
        length = max(positive[0].shape[1], negative[0].shape[1])
        positive, negative = pad_condition(positive, length), pad_condition(negative, length)
        condition = [torch.cat([neg, pos]) for neg, pos in zip(negative, positive)]
        from neodragon.utils.generation_utils import (
            _prepare_latent_noise, _downsample_noise_2x, _prepare_past_condition_latents,
            _generate_one_unit, _decode_latent,
        )
        install_neodragon_generation_patches(device=device)
        units = 1 + (args.frames - 1) // 8
        latent = _prepare_latent_noise(1, dit.config.in_channels, units,
                                       args.height // 8, args.width // 8, torch.bfloat16, device)
        noise = _downsample_noise_2x(latent, 2)
        generated = []
        unit_statistics = []
        for unit in range(units):
            history = _prepare_past_condition_latents(generated, 3, True)
            generated.append(_generate_one_unit(
                scheduler, dit, 3, noise[:, :, unit:unit + 1], history, *condition,
                num_inference_steps=[args.first_steps if unit == 0 else args.video_steps] * 3,
                device=device, dtype=torch.bfloat16, guidance_scale=args.guidance,
                video_guidance_scale=args.video_guidance))
            if not bool(torch.isfinite(generated[-1]).all()):
                raise RuntimeError(f"Non-finite generated latent at unit {unit}")
            unit_statistics.append(dict(unit=unit, mean=float(generated[-1].float().mean()),
                                         std=float(generated[-1].float().std())))
        frames = _decode_latent(vae, torch.cat(generated, dim=2))
    if len(frames) != args.frames:
        raise RuntimeError("Decoded frame count mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if image is not None:
        image.save(args.output_dir / "reconstruction_input.png")
    if args.frames == 1:
        frames[0].save(args.output_dir / "image.png")
    else:
        from diffusers.utils import export_to_video
        export_to_video(frames, str(args.output_dir / "video.mp4"), fps=24)
    (args.output_dir / "inference.json").write_text(json.dumps(dict(
        checkpoint=str(args.checkpoint), seed=args.seed, prompt=args.prompt if image is None else "",
        task="image_reconstruction" if image is not None else ("t2i" if args.frames == 1 else "t2v"),
        external_anchor=False, validation_index=args.validation_index if args.validation_task else None,
        first_steps=args.first_steps, video_steps=args.video_steps,
        guidance=args.guidance, video_guidance=args.video_guidance, frames=len(frames),
        unit_statistics=unit_statistics), indent=2))


if __name__ == "__main__":
    main()
