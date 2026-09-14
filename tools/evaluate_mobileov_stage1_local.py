#!/usr/bin/env python
"""Small paired Stage-1/native inference audit, not a benchmark or held-out evaluation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import textwrap
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch

from tools.infer_mobileov_stage1 import (
    FORMAT, load_stack, autocast, pad_condition, generate_frames, verify_inference_stack,
)
from tools.train_neodragon_dit_bridge import resolve_neodragon_stack, scale_vae_latents
from new_mobile_ov.checkpoints import ensure_neodragon_assets
from new_mobile_ov.training.stage1_alignment import JsonlRecords, prepare_sample, verify_release
from new_mobile_ov.training.stage1_alignment_data import load_sources
from tools.data_prepare.download_alignment_images import sha256_file, atomic_json

PROMPTS = [
    ("red_panda", "A red panda walks along a mossy branch in a bamboo forest. The camera follows its movement.", True),
    ("toy_car", "A yellow toy car drives across a wooden table and turns left. A bookshelf stands in the background.", True),
    ("waterfall", "A waterfall cascades down a rocky cliff into a turquoise pool surrounded by lush green trees.", True),
    ("mugs", "A red ceramic mug sits to the left of a blue ceramic mug on a white kitchen counter.", False),
    ("dogs", "Two golden retriever dogs sit side by side on green grass in a park, with trees in the background.", False),
]


def sheet(rows, path, cell=(320, 200)):
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    width = max(len(images) for _, images in rows) * cell[0]
    out = Image.new("RGB", (width, len(rows) * (cell[1] + 64)), "#f0f3f5")
    draw = ImageDraw.Draw(out)
    for row, (label, images) in enumerate(rows):
        y = row * (cell[1] + 64)
        draw.multiline_text((8, y + 4), textwrap.fill(label, width=max(30, width // 9)),
                            font=font, fill="#142b39", spacing=2)
        for column, image in enumerate(images):
            tile = ImageOps.contain(image, cell, Image.Resampling.LANCZOS)
            out.paste(tile, (column * cell[0] + (cell[0] - tile.width) // 2, y + 64 + (cell[1] - tile.height) // 2))
    out.save(path)


def frame_metrics(frames):
    pixels = np.stack([np.asarray(frame).astype(np.float32) / 255 for frame in frames])
    result = dict(pixel_mean=float(pixels.mean()), pixel_std=float(pixels.std()))
    if len(frames) > 1:
        result["frame_delta_mae_0to1"] = float(np.abs(np.diff(pixels, axis=0)).mean())
        grays = [cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2GRAY) for frame in frames]
        flow = [cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 15, 3, 5, 1.2, 0)
                for a, b in zip(grays, grays[1:])]
        result["farneback_mean_pixels_per_frame"] = float(np.mean([np.linalg.norm(f, axis=-1).mean() for f in flow]))
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--reference-data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU evaluation must run through SLURM")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "audit.json").exists():
        raise FileExistsError("Use a fresh audit output directory")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError("Wrong checkpoint format")
    contract = payload["contract"]
    if sha256_file(Path(args.config)) != contract["config_sha256"]:
        raise ValueError("Config differs from training")
    args.processor, args.max_tokens = contract["processor"], contract["max_tokens"]
    device = torch.device("cuda", 0)
    cfg, encoder, connector, dit, vae, scheduler = load_stack(args, device)
    weight_audit = verify_inference_stack(cfg, encoder, dit, vae, contract)
    connector.load_state_dict(payload["connector"], strict=True)
    connector.eval()
    report = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=sha256_file(args.checkpoint),
                  step=payload["step"], weight_audit=weight_audit, seed=args.seed,
                  conditioning="same literal prompt and empty negative; no native prompt modifier",
                  external_anchor=False, first_steps_per_stage=20, video_steps_per_stage=10,
                  main_cfg_first=7, main_cfg_video=5,
                  scope="small qualitative audit; not VBench; reference image overlap with training unknown",
                  metric_warning="Pixel change and optical flow are not quality scores; noise can increase both",
                  runs=[])
    del payload
    print("Checkpoint and all frozen tensor hashes verified. Loading native reference encoder.", flush=True)
    _, _, native_path = ensure_neodragon_assets(
        repo_path=cfg.backend.extra.get("repo_path"), cache_dir=cfg.backend.extra.get("cache_dir"),
        model_id=cfg.backend.extra.get("model_id", "karnewar/Neodragon"), repo_url=cfg.backend.extra.get("repo_url"))
    from neodragon.text_encoder_bundle import TextEncoderBundle
    from neodragon.context_adapter import ContextAdapter
    from neodragon.utils.generation_utils import _decode_latent
    from diffusers.utils import export_to_video
    text_encoder = TextEncoderBundle.from_pretrained(native_path, torch_dtype=torch.bfloat16).to(device).eval().requires_grad_(False)
    adapter = ContextAdapter.from_pretrained(
        f"{native_path}/{resolve_neodragon_stack('multistep')['context_adapter_id']}",
        torch_dtype=torch.bfloat16).to(device).eval().requires_grad_(False)

    def condition(prompt, *, native=False, image=None):
        with autocast(device):
            if native:
                tokens, mask, pooled = text_encoder(["", prompt], device)
                return (adapter(tokens), mask, pooled)
            positive = connector(*encoder("" if image is not None else prompt, image))
            negative = connector(*encoder("", drop_condition=True))
            length = max(positive[0].shape[1], negative[0].shape[1])
            return tuple(torch.cat([a, b]) for a, b in zip(
                pad_condition(negative, length), pad_condition(positive, length)))

    def render(name, prompt, cond, *, count=1, size=(512, 320), cfg_first=7., cfg_video=5.):
        print(f"Generating {name}: frames={count} cfg={cfg_first}/{cfg_video}", flush=True)
        begin = time.monotonic()
        frames, stats = generate_frames(
            dit, vae, scheduler, cond, device=device, seed=args.seed, width=size[0], height=size[1],
            num_frames=count, guidance=cfg_first, video_guidance=cfg_video)
        directory = out / name
        directory.mkdir(parents=True, exist_ok=True)
        frames[0].save(directory / "first_frame.png")
        if count > 1:
            export_to_video(frames, str(directory / "video.mp4"), fps=24)
            sheet([(name + " | frames 0,12,24,36,48", [frames[i] for i in (0, 12, 24, 36, 48)])],
                  directory / "frames.png")
        entry = dict(name=name, prompt=prompt, frames=count, width=size[0], height=size[1],
                     cfg_first=cfg_first, cfg_video=cfg_video, condition_shape=list(cond[0].shape),
                     positive_tokens=int(cond[1][1].sum()), seconds=time.monotonic()-begin,
                     unit_statistics=stats, **frame_metrics(frames))
        report["runs"].append(entry)
        atomic_json(directory / "metrics.json", entry)
        atomic_json(out / "audit_partial.json", report)
        return frames

    image_rows, video_rows = [], []
    for name, prompt, make_video in PROMPTS:
        student, native = condition(prompt), condition(prompt, native=True)
        outputs = {}
        for label, cond in (("stage1", student), ("native", native)):
            outputs[label] = render(f"t2i/{name}/{label}", prompt, cond)[0]
            if make_video:
                frames = render(f"t2v/{name}/{label}", prompt, cond, count=49)
                video_rows.append((f"{name} / {label} | frames 0,12,24,36,48", [frames[i] for i in (0,12,24,36,48)]))
        image_rows.append((f"{name}: native LEFT / stage1 RIGHT | {prompt}", [outputs["native"], outputs["stage1"]]))
        sheet(image_rows, out / "t2i_comparison.png")
        if video_rows:
            sheet(video_rows, out / "t2v_comparison.png")

    verify_release(args.reference_data_root)
    records = JsonlRecords(args.reference_data_root / "validation/image_reconstruction.jsonl")
    sources = load_sources(args.reference_data_root)
    report["reference_data_sha256"] = sha256_file(args.reference_data_root / "stage1_summary.json")
    recon_rows = []
    for index in (20, 21, 22):
        record = records[index]
        sample = prepare_sample(record, sources, contract["short_side"], contract["long_side"])
        target = sample["image"]
        frames = render(f"reconstruction/{index}", "", condition("", image=target), size=target.size)
        with autocast(device):
            latent = vae.encode(sample["video"].unsqueeze(0).to(device, torch.bfloat16), temporal_chunk=False).latent_dist.mode()
            vae_image = _decode_latent(vae, scale_vae_latents(latent).to(torch.bfloat16))[0]
        target.save(out / "reconstruction" / str(index) / "input.png")
        vae_image.save(out / "reconstruction" / str(index) / "vae_roundtrip.png")
        report["runs"][-1]["reference_sample_id"] = record["sample_id"]
        for label, im in (("generated", frames[0]), ("vae_roundtrip", vae_image)):
            mse = float(np.mean((np.asarray(target).astype(float) / 255 - np.asarray(im).astype(float) / 255) ** 2))
            report["runs"][-1][label + "_psnr_db"] = float(-10 * np.log10(max(mse, 1e-12)))
        recon_rows.append((f"Reference {index}: INPUT / VAE roundtrip / STAGE1 | overlap with training unknown", [target, vae_image, frames[0]]))
        sheet(recon_rows, out / "reconstruction_comparison.png")

    # A small guidance check distinguishes a CFG sensitivity issue from every output being bad.
    prompt = PROMPTS[0][1]
    cond = condition(prompt)
    cfg_images = [render(f"cfg_check/{value}", prompt, cond, cfg_first=value)[0] for value in (1.,3.,7.)]
    sheet([("Same prompt and seed: Stage1 CFG 1 / 3 / 7", cfg_images)], out / "cfg_comparison.png")
    report["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    report["status"] = "complete"
    atomic_json(out / "audit.json", report)
    print(f"Completed audit: {out / 'audit.json'}", flush=True)


if __name__ == "__main__":
    main()
