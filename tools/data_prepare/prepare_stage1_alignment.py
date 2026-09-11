#!/usr/bin/env python
"""Prepare public image/video sources and manifests for all three alignment tasks."""
from __future__ import annotations

import argparse
from pathlib import Path

from tools.data_prepare import build_stage1_alignment_manifests as manifests
from tools.data_prepare import download_alignment_images as images
from tools.data_prepare import download_alignment_videos as videos


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=Path("download_data/data/univideo_alignment_images"))
    parser.add_argument("--video-root", type=Path, default=Path("download_data/data/univideo_alignment_videos"))
    parser.add_argument("--output-dir", type=Path, default=Path("download_data/data/univideo_stage1"))
    parser.add_argument("--image-shards", type=int, default=100)
    parser.add_argument("--video-shards", type=int, default=100)
    parser.add_argument("--image-revision", default="main")
    parser.add_argument("--video-revision", default="main")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--max-image-gib", type=float, default=80)
    parser.add_argument("--max-video-gib", type=float, default=450)
    parser.add_argument("--disk-margin-gib", type=float, default=20)
    parser.add_argument("--min-video-seconds", type=float, default=2)
    parser.add_argument("--max-video-seconds", type=float, default=12)
    parser.add_argument("--min-video-side", type=int, default=256)
    parser.add_argument("--min-video-frames", type=int, default=49)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--exclude-prompts", type=Path, action="append", default=[])
    parser.add_argument("--reader-check-samples", type=int, default=8)
    parser.add_argument("--skip-download", action="store_true", help="Build tasks from already verified source releases")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.min_video_frames < 49:
        raise ValueError("Stage-1 manifests require at least 49 distinct video frames")
    if len({path.expanduser().resolve() for path in (args.image_root, args.video_root, args.output_dir)}) != 3:
        raise ValueError("Image, video, and task-manifest roots must be different")
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / ".stage1_data_complete").unlink(missing_ok=True)
    common = ["--seed", str(args.seed), "--workers", str(args.workers), "--retries", str(args.retries),
              "--disk-margin-gib", str(args.disk_margin_gib)]
    if args.dry_run:
        common.append("--dry-run")
    if not args.skip_download or args.dry_run:
        print("Source 1/2: BLIP3o image-caption pairs for T2I and image reconstruction", flush=True)
        images.run(images.parse_args(common + ["--output-dir", str(args.image_root),
                   "--num-shards", str(args.image_shards), "--revision", args.image_revision,
                   "--max-download-gib", str(args.max_image_gib)]))
        print("Source 2/2: Vchitect video-caption pairs for T2V, not OpenVid", flush=True)
        videos.run(videos.parse_args(common + ["--output-dir", str(args.video_root),
                   "--num-shards", str(args.video_shards), "--revision", args.video_revision,
                   "--max-download-gib", str(args.max_video_gib),
                   "--min-seconds", str(args.min_video_seconds), "--max-seconds", str(args.max_video_seconds),
                   "--min-side", str(args.min_video_side), "--min-frames", str(args.min_video_frames)]))
    if not args.dry_run:
        print("Building disjoint train/validation manifests for all three tasks; media are not copied.", flush=True)
        manifests.run(args)


if __name__ == "__main__":
    main()
