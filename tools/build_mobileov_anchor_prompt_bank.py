#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.training.mobileov_prompt_bank import (  # noqa: E402
    PromptSource,
    build_anchor_prompt_bank,
)


def read_benchmark_prompts(paths: list[Path]) -> list[str]:
    prompts: list[str] = []
    for path in paths:
        if path.suffix.lower() in {".csv", ".tsv"}:
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter=delimiter):
                    prompt = str(
                        row.get("prompt_en") or row.get("prompt") or row.get("caption") or ""
                    ).strip()
                    if prompt:
                        prompts.append(prompt)
        else:
            prompts.extend(
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the benchmark-clean prompt bank for matched anchor/teacher generation."
    )
    parser.add_argument("--config", default="configs/mobileov_data_v1.yaml")
    parser.add_argument("--target-records", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    value = payload["anchor_prompt_bank"]
    sources = [
        PromptSource(
            name=str(source["name"]),
            path=Path(str(source["path"])).expanduser(),
            weight=float(source["weight"]),
            hard_fraction=float(source.get("hard_fraction", 0.5)),
            require_any_capabilities=tuple(source.get("require_any_capabilities", [])),
            text_columns=tuple(
                source.get(
                    "text_columns",
                    [
                        "caption_long",
                        "caption_medium",
                        "prompt",
                        "caption",
                        "text",
                        "caption_short",
                    ],
                )
            ),
        )
        for source in value["sources"]
        if bool(source.get("enabled", True))
    ]
    benchmark_paths = [Path(str(path)) for path in payload.get("benchmark_prompts", [])]
    summary = build_anchor_prompt_bank(
        sources,
        output_path=Path(str(value["output"])),
        target_records=(
            args.target_records if args.target_records > 0 else int(value["target_records"])
        ),
        seed=int(value.get("seed", payload.get("release", {}).get("seed", 20260904))),
        benchmark_prompts=read_benchmark_prompts(benchmark_paths),
        leakage_threshold=float(
            value.get("leakage_threshold", payload.get("release", {}).get("leakage_threshold", 0.85))
        ),
        min_words=int(value.get("min_words", 4)),
        max_words=int(value.get("max_words", 80)),
        force=args.force,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
