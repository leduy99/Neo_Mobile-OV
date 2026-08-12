#!/usr/bin/env python
"""Aggregate repeated, fixed-subset VBench runs into mean and seed variation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated list.")
    return values


def describe(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--variants", required=True, help="Comma-separated variant directory names.")
    parser.add_argument("--seeds", required=True, help="Comma-separated seed values.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    variants = parse_csv(args.variants)
    seeds = parse_csv(args.seeds)
    records: dict[str, dict[str, dict]] = {}
    for variant in variants:
        records[variant] = {}
        for seed in seeds:
            summary_path = args.root / variant / f"seed_{seed}" / "scores" / "vbench" / "summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(f"Missing completed VBench summary: {summary_path}")
            records[variant][seed] = json.loads(summary_path.read_text(encoding="utf-8"))

    aggregate: dict[str, dict] = {}
    for variant, by_seed in records.items():
        metric_values: dict[str, list[float]] = {}
        for payload in by_seed.values():
            for key in ("quality_score", "semantic_score", "total_score"):
                metric_values.setdefault(key, []).append(float(payload[key]))
            for key, value in payload.get("scaled", {}).items():
                metric_values.setdefault(f"scaled.{key}", []).append(float(value))
        aggregate[variant] = {
            "by_seed": by_seed,
            "aggregate": {key: describe(values) for key, values in sorted(metric_values.items())},
        }

    output = {
        "protocol": (
            "Comparative result on one fixed stratified VBench subset. Values are means over "
            "independently seeded generations and are not a canonical full-VBench result."
        ),
        "root": str(args.root.resolve()),
        "variants": variants,
        "seeds": seeds,
        "results": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
