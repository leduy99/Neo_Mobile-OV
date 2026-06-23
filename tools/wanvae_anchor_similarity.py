#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "sana"))

from diffusion.model.wan.vae import WanVAE


def dtype_from_name(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def iter_video_paths(manifest: Path, limit: int, offset: int = 0) -> Iterable[tuple[int, str]]:
    """Yield existing video paths from a Mobile-OV/OpenVid manifest."""

    yielded = 0
    seen = 0
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader):
            path = row.get("video_path") or row.get("media_path") or row.get("path")
            if not path:
                continue
            if not os.path.isfile(path):
                continue
            if seen < offset:
                seen += 1
                continue
            yield row_idx, path
            yielded += 1
            if yielded >= limit:
                return


def center_crop_resize_frames(frames: list[np.ndarray], target_size: tuple[int, int]) -> torch.Tensor:
    h, w = frames[0].shape[:2]
    target_h, target_w = target_size
    ratio = float(target_w) / float(target_h)
    if w < h * ratio:
        crop_size = (int(float(w) / ratio), w)
    else:
        crop_size = (h, int(float(h) * ratio))
    transform = transforms.Compose(
        [
            transforms.CenterCrop(crop_size),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return torch.stack([transform(Image.fromarray(frame)) for frame in frames], dim=0)


def read_video_frames(path: Path, frame_num: int, target_size: tuple[int, int], sampling_rate: int) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rate = max(1, int(sampling_rate))
    while total < frame_num * rate and rate > 1:
        rate -= 1

    frames: list[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % rate == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(frames) >= frame_num:
                break
        idx += 1
    cap.release()
    if len(frames) != frame_num:
        raise RuntimeError(f"Need {frame_num} frames, got {len(frames)} from {path}")
    return center_crop_resize_frames(frames, target_size)


def latent_pair_metrics(a: torch.Tensor, b: torch.Tensor, prefix: str) -> dict[str, float]:
    """Compare two latent slices shaped [C,H,W]."""

    a = a.float()
    b = b.float()
    diff = a - b
    flat_a = a.flatten()
    flat_b = b.flatten()

    token_a = a.permute(1, 2, 0).reshape(-1, a.shape[0])
    token_b = b.permute(1, 2, 0).reshape(-1, b.shape[0])
    token_sims = F.cosine_similarity(token_a, token_b, dim=-1)

    return {
        f"{prefix}_cos": F.cosine_similarity(flat_a, flat_b, dim=0).item(),
        f"{prefix}_l1": diff.abs().mean().item(),
        f"{prefix}_rmse": torch.sqrt((diff * diff).mean()).item(),
        f"{prefix}_rel": (torch.norm(diff) / (torch.norm(b) + 1e-8)).item(),
        f"{prefix}_token_cos_mean": token_sims.mean().item(),
        f"{prefix}_token_cos_median": token_sims.median().item(),
        f"{prefix}_token_cos_p10": token_sims.quantile(0.10).item(),
        f"{prefix}_token_cos_p90": token_sims.quantile(0.90).item(),
        f"{prefix}_token_cos_min": token_sims.min().item(),
    }


def decode_first_slice(vae: WanVAE, z0: torch.Tensor) -> torch.Tensor:
    decoded = vae.decode([z0[:, None].float().to(vae.device)])[0]
    if decoded.dim() == 4:
        return decoded[:, 0]
    return decoded


def save_image(tensor_chw: torch.Tensor, path: Path) -> None:
    arr = ((tensor_chw.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).byte()
    arr = arr.permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    numeric: dict[str, list[float]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                numeric.setdefault(key, []).append(float(value))

    summary: dict[str, object] = {
        "num_rows": len(rows),
        "num_ok": sum(1 for row in rows if row.get("status") == "ok"),
        "num_failed": sum(1 for row in rows if row.get("status") != "ok"),
    }
    for key, values in sorted(numeric.items()):
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "p10": float(np.quantile(arr, 0.10)),
            "p90": float(np.quantile(arr, 0.90)),
            "max": float(arr.max()),
        }
    return summary


def metric_fieldnames() -> list[str]:
    base = ["out_idx", "manifest_row", "video_path", "status", "z_full_shape", "z_one_shape", "error"]
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
    prefixes = ["one_full0", "full0_full1", "full0_full2", "decode_one_full0"]
    return base + [f"{prefix}_{suffix}" for prefix in prefixes for suffix in suffixes]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/data/openvid_1m/manifests/by_part/part_0111.csv")
    parser.add_argument("--vae-path", default="/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/omni_ckpts/sana_video_2b_480p/vae/Wan2.1_VAE.pth")
    parser.add_argument("--output-dir", default="output/wanvae_anchor_similarity_100")
    parser.add_argument("--num-videos", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--frame-num", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--sampling-rate", type=int, default=3)
    parser.add_argument("--decode-count", type=int, default=8)
    parser.add_argument("--dtype", default="bf16")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    if device.type == "cpu":
        dtype = torch.float32

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "decoded_compare"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}, dtype={dtype}")
    print(f"Manifest: {args.manifest}")
    print(f"Output: {output_dir}")
    print("Loading WanVAE...")
    vae = WanVAE(vae_pth=args.vae_path, dtype=dtype, device=device)

    video_items = list(iter_video_paths(Path(args.manifest), limit=args.num_videos, offset=args.offset))
    if len(video_items) < args.num_videos:
        print(f"Warning: requested {args.num_videos}, found {len(video_items)} existing videos.")

    rows: list[dict[str, object]] = []
    csv_path = output_dir / "per_video_metrics.csv"
    fieldnames = metric_fieldnames()
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

    t_start = time.time()
    for out_idx, (row_idx, video_path) in enumerate(tqdm(video_items, desc="WanVAE similarity")):
        row: dict[str, object] = {
            "out_idx": out_idx,
            "manifest_row": row_idx,
            "video_path": video_path,
            "status": "ok",
        }
        try:
            frames_tchw = read_video_frames(Path(video_path), args.frame_num, (args.height, args.width), args.sampling_rate)
            x_full = frames_tchw.transpose(0, 1).to(device)  # [C,T,H,W]
            x_first = x_full[:, :1]
            with torch.no_grad():
                z_full = vae.encode([x_full])[0].to(device)
                z_one = vae.encode([x_first])[0].to(device)

            row["z_full_shape"] = "x".join(map(str, z_full.shape))
            row["z_one_shape"] = "x".join(map(str, z_one.shape))

            z_full0 = z_full[:, 0]
            z_one0 = z_one[:, 0]
            row.update(latent_pair_metrics(z_one0, z_full0, "one_full0"))

            if z_full.shape[1] > 1:
                row.update(latent_pair_metrics(z_full[:, 0], z_full[:, 1], "full0_full1"))
            if z_full.shape[1] > 2:
                row.update(latent_pair_metrics(z_full[:, 0], z_full[:, 2], "full0_full2"))

            if out_idx < args.decode_count:
                with torch.no_grad():
                    dec_one = decode_first_slice(vae, z_one0)
                    dec_full = decode_first_slice(vae, z_full0)
                row.update(latent_pair_metrics(dec_one, dec_full, "decode_one_full0"))
                save_image(dec_one, image_dir / f"{out_idx:04d}_one.png")
                save_image(dec_full, image_dir / f"{out_idx:04d}_full0.png")
                save_image((dec_one - dec_full).abs() * 2.0 - 1.0, image_dir / f"{out_idx:04d}_absdiff_vis.png")

        except Exception as exc:
            row["status"] = "failed"
            row["error"] = repr(exc)
        rows.append(row)

        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writerow(row)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = summarize(rows)
    summary.update(
        {
            "elapsed_seconds": time.time() - t_start,
            "frame_num": args.frame_num,
            "height": args.height,
            "width": args.width,
            "sampling_rate": args.sampling_rate,
            "decode_count": args.decode_count,
            "manifest": str(args.manifest),
            "vae_path": str(args.vae_path),
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
