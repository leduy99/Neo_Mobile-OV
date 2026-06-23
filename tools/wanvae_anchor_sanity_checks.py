#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "sana"))

from diffusion.model.wan.vae import WanVAE
from tools.wanvae_anchor_similarity import (
    dtype_from_name,
    iter_video_paths,
    latent_pair_metrics,
    read_video_frames,
    summarize,
)


def metric_fieldnames(prefixes: list[str]) -> list[str]:
    base = ["out_idx", "manifest_row", "video_path", "status", "z_full_shape", "z_one_shape", "z_frame_shape", "error"]
    suffixes = [
        "cos",
        "l1",
        "rmse",
        "rel",
        "token_cos_mean",
        "token_cos_median",
        "token_cos_p10",
        "token_cos_p90",
        "token_cos_min",
    ]
    return base + [f"{prefix}_{suffix}" for prefix in prefixes for suffix in suffixes]


def compact_summary(rows: list[dict[str, object]], prefixes: list[str]) -> dict[str, object]:
    out = summarize(rows)
    # Keep a small top-level table for quick reading in logs.
    table = {}
    for prefix in prefixes:
        table[prefix] = {}
        for suffix in ["cos", "rel", "token_cos_mean", "token_cos_p10"]:
            key = f"{prefix}_{suffix}"
            value = out.get(key)
            if isinstance(value, dict):
                table[prefix][suffix] = {
                    "mean": value["mean"],
                    "median": value["median"],
                    "p10": value["p10"],
                    "p90": value["p90"],
                    "min": value["min"],
                    "max": value["max"],
                }
    out["quick_table"] = table
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/data/openvid_1m/manifests/by_part/part_0111.csv")
    parser.add_argument("--vae-path", default="/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/omni_ckpts/sana_video_2b_480p/vae/Wan2.1_VAE.pth")
    parser.add_argument("--output-dir", default="output/wanvae_anchor_sanity_100")
    parser.add_argument("--num-videos", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--frame-num", type=int, default=81)
    parser.add_argument("--frame-idx", type=int, default=10)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--sampling-rate", type=int, default=3)
    parser.add_argument("--dtype", default="bf16")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    if device.type == "cpu":
        dtype = torch.float32

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefixes = ["same_instance_one_full0", "independent_one_full0", "frameidx_full0", "frameidx_one0"]
    fieldnames = metric_fieldnames(prefixes)
    csv_path = output_dir / "per_video_sanity_metrics.csv"

    print(f"Device: {device}, dtype={dtype}")
    print(f"Frame index sanity check: frame_idx={args.frame_idx}")
    print(f"Output: {output_dir}")
    print("Loading two independent WanVAE instances...")
    vae_full = WanVAE(vae_pth=args.vae_path, dtype=dtype, device=device)
    vae_one = WanVAE(vae_pth=args.vae_path, dtype=dtype, device=device)

    video_items = list(iter_video_paths(Path(args.manifest), limit=args.num_videos, offset=args.offset))
    if len(video_items) < args.num_videos:
        print(f"Warning: requested {args.num_videos}, found {len(video_items)} existing videos.")

    rows: list[dict[str, object]] = []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

    t_start = time.time()
    for out_idx, (row_idx, video_path) in enumerate(tqdm(video_items, desc="WanVAE sanity")):
        row: dict[str, object] = {
            "out_idx": out_idx,
            "manifest_row": row_idx,
            "video_path": video_path,
            "status": "ok",
        }
        try:
            frames_tchw = read_video_frames(Path(video_path), args.frame_num, (args.height, args.width), args.sampling_rate)
            x_full = frames_tchw.transpose(0, 1).to(device)
            x_first = x_full[:, :1]
            frame_idx = min(max(args.frame_idx, 0), x_full.shape[1] - 1)
            x_frame = x_full[:, frame_idx : frame_idx + 1]

            with torch.no_grad():
                z_full = vae_full.encode([x_full])[0].to(device)
                z_one_same = vae_full.encode([x_first])[0].to(device)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                z_one_ind = vae_one.encode([x_first])[0].to(device)
                z_frame_ind = vae_one.encode([x_frame])[0].to(device)

            row["z_full_shape"] = "x".join(map(str, z_full.shape))
            row["z_one_shape"] = "x".join(map(str, z_one_ind.shape))
            row["z_frame_shape"] = "x".join(map(str, z_frame_ind.shape))

            z_full0 = z_full[:, 0]
            z_one_same0 = z_one_same[:, 0]
            z_one_ind0 = z_one_ind[:, 0]
            z_frame0 = z_frame_ind[:, 0]

            row.update(latent_pair_metrics(z_one_same0, z_full0, "same_instance_one_full0"))
            row.update(latent_pair_metrics(z_one_ind0, z_full0, "independent_one_full0"))
            row.update(latent_pair_metrics(z_frame0, z_full0, "frameidx_full0"))
            row.update(latent_pair_metrics(z_frame0, z_one_ind0, "frameidx_one0"))
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = repr(exc)

        rows.append(row)
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writerow(row)

    summary = compact_summary(rows, prefixes)
    summary.update(
        {
            "elapsed_seconds": time.time() - t_start,
            "frame_num": args.frame_num,
            "frame_idx": args.frame_idx,
            "height": args.height,
            "width": args.width,
            "sampling_rate": args.sampling_rate,
            "manifest": str(args.manifest),
            "vae_path": str(args.vae_path),
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["quick_table"], indent=2))
    print(json.dumps({k: summary[k] for k in ["num_rows", "num_ok", "num_failed", "elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
