#!/usr/bin/env python
"""Freeze a benchmark-clean, self-contained V12 training-data release."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.training.image_bridge_grounding_cascade import (  # noqa: E402
    LeakageIndex,
    file_sha256,
    read_prompts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/image_bridge_v1")
    parser.add_argument("--output-root", default="data/image_bridge_v1/releases/v12")
    parser.add_argument("--leakage-prompts", action="append", required=True)
    parser.add_argument("--leakage-threshold", type=float, default=0.85)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress-every", type=int, default=250_000)
    return parser.parse_args()


def filter_manifest(
    source: Path,
    destination: Path,
    *,
    leakage: LeakageIndex,
    progress_every: int,
) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    filtered = 0
    examples: list[dict[str, str]] = []
    with source.open("r", encoding="utf-8", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        if not reader.fieldnames or not {"record_id", "caption"}.issubset(reader.fieldnames):
            raise ValueError(f"Invalid training manifest schema: {source}")
        with destination.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            for processed, row in enumerate(reader, start=1):
                caption = str(row.get("caption", "")).strip()
                if leakage.matches(caption):
                    filtered += 1
                    if len(examples) < 50:
                        examples.append(
                            {
                                "record_id": str(row.get("record_id", "")),
                                "caption": caption,
                            }
                        )
                else:
                    writer.writerow(row)
                    kept += 1
                if progress_every > 0 and processed % progress_every == 0:
                    print(
                        f"source={source.name} processed={processed} "
                        f"kept={kept} leakage_filtered={filtered}",
                        flush=True,
                    )
    return {
        "source": str(source),
        "source_sha256": file_sha256(source),
        "output": str(destination),
        "output_sha256": file_sha256(destination),
        "kept_rows": kept,
        "filtered_rows": filtered,
        "filtered_examples": examples,
    }


def existing_release_is_current(
    output_root: Path,
    *,
    leakage_prompt_paths: list[str],
    leakage_threshold: float,
) -> bool:
    summary_path = output_root / "stats/release_summary.json"
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if float(summary.get("leakage_threshold", -1.0)) != float(leakage_threshold):
        return False
    expected_benchmarks = {
        str(Path(path).expanduser()): file_sha256(Path(path).expanduser())
        for path in leakage_prompt_paths
        if Path(path).expanduser().is_file()
    }
    recorded_benchmarks = {
        str(value.get("path", "")): str(value.get("sha256", ""))
        for value in summary.get("benchmark_sources", [])
    }
    if expected_benchmarks != recorded_benchmarks:
        return False
    for value in summary.get("manifests", {}).values():
        source = Path(str(value.get("source", "")))
        output = Path(str(value.get("output", "")))
        if not source.is_file() or not output.is_file():
            return False
        if file_sha256(source) != value.get("source_sha256"):
            return False
        if file_sha256(output) != value.get("output_sha256"):
            return False
    return True


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    prompts = read_prompts(args.leakage_prompts)
    if not prompts:
        raise ValueError("No benchmark prompts were loaded; refusing to freeze V12 release.")
    if output_root.exists() and not args.force:
        if existing_release_is_current(
            output_root,
            leakage_prompt_paths=args.leakage_prompts,
            leakage_threshold=args.leakage_threshold,
        ):
            summary = json.loads(
                (output_root / "stats/release_summary.json").read_text(encoding="utf-8")
            )
            print(json.dumps({"reused_existing_release": True, **summary}, indent=2))
            return
        raise FileExistsError(
            f"Existing release is incomplete or stale: {output_root}. Set FORCE=1 to rebuild."
        )

    partial = output_root.with_name(output_root.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    if output_root.exists():
        shutil.rmtree(output_root)
    (partial / "manifests").mkdir(parents=True)
    (partial / "mixtures").mkdir()
    (partial / "stats").mkdir()

    leakage = LeakageIndex(prompts, threshold=args.leakage_threshold)
    sources = {
        "broad": data_root / "manifests/d1_broad_train.csv",
        "compositional": data_root / "manifests/d2_compositional_train.csv",
        "grounded": data_root / "manifests/d2_grounded_high_precision_50k.csv",
    }
    for source in sources.values():
        if not source.is_file():
            raise FileNotFoundError(source)

    manifests = {
        name: filter_manifest(
            source,
            partial / "manifests" / source.name,
            leakage=leakage,
            progress_every=args.progress_every,
        )
        for name, source in sources.items()
    }
    # The temporary tree is atomically renamed below, so persist only final
    # release paths in the reusable release contract.
    for name, source in sources.items():
        manifests[name]["output"] = str(output_root / "manifests" / source.name)
    for name in ("validation.csv", "hard_validation.csv"):
        source = data_root / "manifests" / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, partial / "manifests" / name)

    final_manifest_paths = {
        name: output_root / "manifests" / source.name for name, source in sources.items()
    }
    mixture = {
        "name": "v12_release_70_20_10_benchmark_clean",
        "overall_sampling": {"broad": 0.70, "compositional": 0.20, "grounded": 0.10},
        "trainer_contract": {
            "generation_prompt_manifests": [
                str(final_manifest_paths["broad"]),
                str(final_manifest_paths["compositional"]),
                str(final_manifest_paths["grounded"]),
            ],
            "generation_source_names": ["broad", "compositional", "grounded_cascade"],
            "generation_source_weights": [7.0 / 9.0, 2.0 / 9.0, 0.0],
            "grounded_source_names": ["grounded_cascade"],
            "grounded_batch_probability": 0.10,
            "semantic_prompt_probability": 0.0,
        },
        "benchmark_filter": {
            "prompt_sources": args.leakage_prompts,
            "jaccard_threshold": args.leakage_threshold,
        },
    }
    mixture_partial = partial / "mixtures/v12_release_70_20_10.json"
    mixture_partial.write_text(json.dumps(mixture, indent=2) + "\n", encoding="utf-8")
    summary = {
        "dataset": "ImageBridge-Data-v1",
        "release": "v12",
        "release_contract": "benchmark-clean derived manifests; source captions unchanged",
        "leakage_prompt_count": len(prompts),
        "leakage_threshold": args.leakage_threshold,
        "benchmark_sources": [
            {
                "path": str(Path(path).expanduser()),
                "sha256": file_sha256(Path(path).expanduser()),
            }
            for path in args.leakage_prompts
            if Path(path).expanduser().is_file()
        ],
        "manifests": manifests,
        "mixture": {
            "path": str(output_root / "mixtures/v12_release_70_20_10.json"),
            "sha256": file_sha256(mixture_partial),
        },
    }
    (partial / "stats/release_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    partial.replace(output_root)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
