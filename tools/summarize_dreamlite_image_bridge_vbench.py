#!/usr/bin/env python
"""Summarize paired native-Qwen versus V8 image-bridge VBench scores."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def describe(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def flatten(score: dict) -> dict[str, float]:
    values = {
        name: float(value)
        for name, value in score.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    for group in ("raw", "scaled"):
        for name, value in score.get(group, {}).items():
            values[f"{group}.{name}"] = float(value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    prepared = json.loads((root / "prepared.json").read_text(encoding="utf-8"))
    if prepared.get("status") != "ok":
        raise RuntimeError(f"Incomplete prepared ablation: {root / 'prepared.json'}")
    seeds = [int(value) for value in prepared["seeds"]]
    variants = [str(value) for value in prepared["variants"]]
    if variants != ["native_qwen", "v8_imageonly"]:
        raise RuntimeError(f"Unexpected variants in prepared ablation: {variants}")

    by_seed: dict[str, dict[str, dict]] = {variant: {} for variant in variants}
    for variant in variants:
        for seed in seeds:
            score_path = root / variant / f"seed_{seed}" / "scores" / "vbench" / "summary.json"
            if not score_path.is_file():
                raise FileNotFoundError(f"Missing completed VBench score: {score_path}")
            by_seed[variant][str(seed)] = json.loads(score_path.read_text(encoding="utf-8"))

    aggregate: dict[str, dict[str, dict[str, float | int]]] = {}
    for variant, values in by_seed.items():
        flat = [flatten(score) for score in values.values()]
        keys = sorted(set.intersection(*(set(row) for row in flat)))
        aggregate[variant] = {key: describe([row[key] for row in flat]) for key in keys}

    paired: dict[str, list[float]] = {}
    by_seed_delta: dict[str, dict[str, float]] = {}
    for seed in seeds:
        native = flatten(by_seed["native_qwen"][str(seed)])
        student = flatten(by_seed["v8_imageonly"][str(seed)])
        delta = {key: student[key] - native[key] for key in sorted(set(native) & set(student))}
        by_seed_delta[str(seed)] = delta
        for key, value in delta.items():
            paired.setdefault(key, []).append(value)
    paired_aggregate = {key: describe(values) for key, values in sorted(paired.items())}

    output = {
        "status": "ok",
        "protocol": prepared["protocol"],
        "prompt_count": int(prepared["prompt_count"]),
        "seeds": seeds,
        "variants": variants,
        "vbench_subset": prepared["vbench_subset"],
        "image_condition_alignment": {
            "by_vbench_dimension": prepared["condition_metrics_by_vbench_dimension"],
            "highest_content_condition_mismatch_prompts": prepared[
                "highest_content_condition_mismatch_prompts"
            ],
            "contact_sheets": prepared["contact_sheets"],
        },
        "vbench_by_seed": by_seed,
        "vbench_aggregate": aggregate,
        "paired_v8_image_bridge_minus_native_qwen_by_seed": by_seed_delta,
        "paired_v8_image_bridge_minus_native_qwen": paired_aggregate,
        "decision_metrics": {
            "quality_score": paired_aggregate.get("quality_score"),
            "semantic_score": paired_aggregate.get("semantic_score"),
            "total_score": paired_aggregate.get("total_score"),
            "scaled.object_class": paired_aggregate.get("scaled.object_class"),
            "scaled.multiple_objects": paired_aggregate.get("scaled.multiple_objects"),
            "scaled.color": paired_aggregate.get("scaled.color"),
            "scaled.spatial_relationship": paired_aggregate.get("scaled.spatial_relationship"),
            "scaled.scene": paired_aggregate.get("scaled.scene"),
        },
    }
    destination = args.output or root / "image_bridge_final_ablation_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
