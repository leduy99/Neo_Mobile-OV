#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.training.image_bridge_data_v1 import (  # noqa: E402
    ImageBridgeBuildConfig,
    ImageBridgeSource,
    build_image_bridge_data_v1,
)


def split_paths(value: str) -> list[Path]:
    return [Path(item.strip()).expanduser() for item in value.split(";") if item.strip()]


def split_names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def source_group(
    *,
    paths_value: str,
    names_value: str,
    role: str,
) -> list[ImageBridgeSource]:
    paths = split_paths(paths_value)
    names = split_names(names_value)
    if not paths:
        return []
    if not names:
        names = [path.stem for path in paths]
    if len(paths) != len(names):
        raise ValueError(f"{role} paths and names must have equal length")
    return [
        ImageBridgeSource(name=name, role=role, path=path)
        for path, name in zip(paths, names)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the immutable ImageBridge-Data-v1 catalog and mutually exclusive "
            "broad, compositional, and grounded-candidate views."
        )
    )
    parser.add_argument("--broad-manifests", required=True)
    parser.add_argument("--broad-source-names", default="")
    parser.add_argument("--compositional-manifests", default="")
    parser.add_argument("--compositional-source-names", default="")
    parser.add_argument("--grounded-manifests", default="")
    parser.add_argument("--grounded-source-names", default="")
    parser.add_argument("--output-dir", default="data/image_bridge_v1")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--validation-fraction", type=float, default=0.005)
    parser.add_argument("--hard-validation-per-capability", type=int, default=100)
    parser.add_argument("--max-broad-records", type=int, default=3_000_000)
    parser.add_argument("--max-compositional-records", type=int, default=200_000)
    parser.add_argument("--max-grounded-records", type=int, default=200_000)
    parser.add_argument("--skip-source-hashes", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = [
        *source_group(
            paths_value=args.broad_manifests,
            names_value=args.broad_source_names,
            role="broad",
        ),
        *source_group(
            paths_value=args.compositional_manifests,
            names_value=args.compositional_source_names,
            role="compositional",
        ),
        *source_group(
            paths_value=args.grounded_manifests,
            names_value=args.grounded_source_names,
            role="grounded",
        ),
    ]
    names = [source.name for source in sources]
    if len(set(names)) != len(names):
        raise ValueError("Source names must be unique")
    config = ImageBridgeBuildConfig(
        output_dir=Path(args.output_dir),
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        hard_validation_per_capability=args.hard_validation_per_capability,
        max_broad_records=args.max_broad_records,
        max_compositional_records=args.max_compositional_records,
        max_grounded_records=args.max_grounded_records,
        source_hashes=not args.skip_source_hashes,
    )
    build_image_bridge_data_v1(
        sources,
        config,
        force=args.force,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
