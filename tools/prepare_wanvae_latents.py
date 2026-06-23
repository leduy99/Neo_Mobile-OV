#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import glob
import pickle
import sys
from pathlib import Path

import cv2
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


def add_sana_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sana_root = repo_root / "third_party" / "sana"
    if str(sana_root) not in sys.path:
        sys.path.insert(0, str(sana_root))


def center_crop_resize(frames: list, target_size: tuple[int, int]) -> torch.Tensor:
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


def read_video_frames(path: Path, frame_num: int, target_size: tuple[int, int], sampling_rate: int) -> torch.Tensor | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rate = max(1, int(sampling_rate))
    while total < frame_num * rate and rate > 1:
        rate -= 1
    frames = []
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
        return None
    return center_crop_resize(frames, target_size)


def build_prompt_lookup(csv_path: str | None) -> dict[str, str]:
    if not csv_path:
        return {}
    df = pd.read_csv(csv_path)
    prompt_col = "caption" if "caption" in df.columns else "prompt"
    lookup = {}
    for _, row in df.iterrows():
        video = str(row.get("video_path") or row.get("video") or row.get("media_path") or "").strip()
        prompt = str(row.get(prompt_col) or "").strip()
        if video and prompt:
            lookup[Path(video).name] = prompt
    return lookup


def videos_from_manifest(manifest_path: str, max_videos: int, offset: int = 0) -> list[tuple[Path, str]]:
    df = pd.read_csv(manifest_path)
    rows: list[tuple[Path, str]] = []
    seen = 0
    for _, row in df.iterrows():
        path = str(row.get("video_path") or row.get("media_path") or row.get("path") or "").strip()
        if not path or not Path(path).is_file():
            continue
        if seen < offset:
            seen += 1
            continue
        prompt = str(row.get("prompt") or row.get("caption") or Path(path).stem.replace("_", " ")).strip()
        rows.append((Path(path), prompt))
        if len(rows) >= max_videos:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-glob", default=None, help="Glob for raw mp4 files.")
    parser.add_argument("--manifest", default=None, help="Optional Mobile-OV/OpenVid manifest with video_path/caption columns.")
    parser.add_argument("--prompt-csv", default=None, help="Optional CSV with video/caption columns.")
    parser.add_argument("--vae-path", required=True, help="Path to Wan2.1_VAE.pth")
    parser.add_argument("--output-dir", default="data/lmw_smoke")
    parser.add_argument("--max-videos", type=int, default=16)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--frame-num", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--sampling-rate", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    add_sana_to_path()
    from diffusion.model.wan.vae import WanVAE

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    latent_dir = out_dir / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    prompt_lookup = build_prompt_lookup(args.prompt_csv)
    if args.manifest:
        video_items = videos_from_manifest(args.manifest, max_videos=args.max_videos, offset=args.offset)
    else:
        if not args.video_glob:
            raise ValueError("Either --manifest or --video-glob is required.")
        prompt_lookup = build_prompt_lookup(args.prompt_csv)
        videos = [Path(p) for p in sorted(glob.glob(args.video_glob, recursive=True))[: args.max_videos]]
        video_items = [(video_path, prompt_lookup.get(video_path.name, video_path.stem.replace("_", " "))) for video_path in videos]
    if not video_items:
        raise FileNotFoundError(f"No videos found from manifest={args.manifest} glob={args.video_glob}")

    vae = WanVAE(vae_pth=args.vae_path, device=device)
    rows = []
    for i, (video_path, prompt) in enumerate(tqdm(video_items, desc="WanVAE encode")):
        frames = read_video_frames(video_path, args.frame_num, (args.height, args.width), args.sampling_rate)
        if frames is None:
            continue
        with torch.no_grad():
            latent = vae.encode([frames.transpose(0, 1).to(device)])[0].cpu()
        out_path = latent_dir / f"sample_{i:06d}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "latent_feature": latent,
                    "prompt": prompt,
                    "video_path": str(video_path),
                    "frame_num": int(args.frame_num),
                    "target_size": [int(args.height), int(args.width)],
                },
                f,
            )
        rows.append({"latent_path": str(out_path.relative_to(out_dir)), "prompt": prompt, "video_path": str(video_path)})
    manifest = out_dir / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["latent_path", "prompt", "video_path"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {manifest} rows={len(rows)}")


if __name__ == "__main__":
    main()
