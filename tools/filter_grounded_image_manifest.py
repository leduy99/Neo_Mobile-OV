#!/usr/bin/env python
"""Create a verified image-caption manifest for DreamLite grounded supervision."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.training.grounded_manifest import filter_grounded_manifest


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter a caption manifest to rows with verified readable images."
    )
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--image-columns", default="image_path,media_path,video_path")
    parser.add_argument(
        "--image-path-roots",
        default="",
        help="Comma-separated roots used to resolve relocated absolute image paths.",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument(
        "--skip-image-verify",
        action="store_true",
        help="Accept existing files without PIL integrity verification.",
    )
    args = parser.parse_args()
    filter_grounded_manifest(
        source_manifest=args.source_manifest,
        output_manifest=args.output_manifest,
        source_name=args.source_name,
        image_columns=parse_csv(args.image_columns),
        image_path_roots=parse_csv(args.image_path_roots),
        workers=args.workers,
        verify_images=not args.skip_image_verify,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
