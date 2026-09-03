#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.training.mobileov_data_release import (  # noqa: E402
    ReleaseBuildConfig,
    ReleaseSource,
    build_mobileov_data_release,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze heterogeneous Mobile-OV data into one auditable release."
    )
    parser.add_argument("--config", default="configs/mobileov_data_v1.yaml")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-path-verification", action="store_true")
    parser.add_argument("--skip-source-hashes", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    release = payload.get("release", {})
    output_dir = Path(args.output_dir or release["output_dir"])
    sources = [
        ReleaseSource(
            name=str(value["name"]),
            pool=str(value["pool"]),
            path=Path(str(value["path"])).expanduser(),
            max_records=int(value.get("max_records", -1)),
            fixed_split=str(value.get("fixed_split", "")),
            require_artifact=bool(value.get("require_artifact", True)),
        )
        for value in payload.get("sources", [])
        if bool(value.get("enabled", True))
    ]
    summary = build_mobileov_data_release(
        sources,
        ReleaseBuildConfig(
            output_dir=output_dir,
            seed=int(release.get("seed", 20260904)),
            validation_fraction=float(release.get("validation_fraction", 0.002)),
            leakage_threshold=float(release.get("leakage_threshold", 0.85)),
            verify_paths=not args.skip_path_verification,
            source_hashes=not args.skip_source_hashes,
            force=args.force,
        ),
        benchmark_prompt_paths=[
            Path(str(path)).expanduser() for path in payload.get("benchmark_prompts", [])
        ],
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
