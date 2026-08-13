#!/usr/bin/env python
"""Summarize factorial bridge effects from VBench scores without regenerating media."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

VIDEO_CELLS = (
    "image_native_qwen__text_native_neodragon",
    "image_native_qwen__text_exp1_64k",
    "image_v8_imageonly__text_native_neodragon",
    "image_v8_imageonly__text_exp1_64k",
)
ANCHOR_CELLS = ("anchor_native_qwen", "anchor_v8_imageonly")
ANCHOR_DIMENSIONS = ("object_class", "multiple_objects", "color", "spatial_relationship", "scene")
# Keep the VBench standard normalization local so this reporting-only tool does
# not need to import VBench or CUDA dependencies just to read finished JSON files.
NORMALIZATION = {
    "object_class": (0.0, 1.0),
    "multiple_objects": (0.0, 1.0),
    "color": (0.0, 1.0),
    "spatial_relationship": (0.0, 1.0),
    "scene": (0.0, 0.8222),
}


def describe(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def score_path(root: Path, seed: int, cell: str) -> Path:
    return root / "scores" / f"seed_{seed}" / cell / "vbench"


def anchor_score(root: Path, seed: int, cell: str) -> dict[str, float]:
    directory = score_path(root, seed, cell)
    raw: dict[str, float] = {}
    for dimension in ANCHOR_DIMENSIONS:
        payload = json.loads((directory / f"{dimension}_eval_results.json").read_text(encoding="utf-8"))
        raw[dimension] = float(payload[dimension][0])
    scaled = {
        dimension: (value - NORMALIZATION[dimension][0])
        / (NORMALIZATION[dimension][1] - NORMALIZATION[dimension][0])
        for dimension, value in raw.items()
    }
    return {"raw": raw, "scaled": scaled, "semantic_proxy": statistics.mean(scaled.values())}


def video_score(root: Path, seed: int, cell: str) -> dict:
    path = score_path(root, seed, cell) / "summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "quality_score": float(payload["quality_score"]),
        "semantic_score": float(payload["semantic_score"]),
        "total_score": float(payload["total_score"]),
        "scaled": {name: float(value) for name, value in payload["scaled"].items()},
        "raw": {name: float(value) for name, value in payload["raw"].items()},
    }


def flattened(score: dict) -> dict[str, float]:
    result = {
        name: float(value)
        for name, value in score.items()
        if isinstance(value, (int, float))
    }
    for group in ("raw", "scaled"):
        for name, value in score.get(group, {}).items():
            result[f"{group}.{name}"] = float(value)
    return result


def delta(left: dict, right: dict) -> dict[str, float]:
    lhs, rhs = flattened(left), flattened(right)
    keys = sorted(set(lhs) & set(rhs))
    return {key: lhs[key] - rhs[key] for key in keys}


def average_deltas(*effects: dict[str, float]) -> dict[str, float]:
    keys = sorted(set.intersection(*(set(effect) for effect in effects)))
    return {key: statistics.mean(effect[key] for effect in effects) for key in keys}


def aggregate_by_seed(by_seed: dict[int, dict[str, float]]) -> dict[str, dict[str, float | int]]:
    keys = sorted(set.intersection(*(set(effect) for effect in by_seed.values())))
    return {key: describe([effect[key] for effect in by_seed.values()]) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    prepared = json.loads((root / "prepared.json").read_text(encoding="utf-8"))
    if prepared.get("status") != "ok":
        raise RuntimeError(f"Prepared ablation is incomplete: {root / 'prepared.json'}")
    seeds = [int(value) for value in prepared["seeds"]]

    results: dict[str, dict[int, dict]] = {cell: {} for cell in (*ANCHOR_CELLS, *VIDEO_CELLS)}
    for seed in seeds:
        for cell in ANCHOR_CELLS:
            results[cell][seed] = anchor_score(root, seed, cell)
        for cell in VIDEO_CELLS:
            results[cell][seed] = video_score(root, seed, cell)

    effects: dict[str, dict[int, dict[str, float]]] = {
        "anchor_v8_minus_native": {},
        "image_effect_at_native_text": {},
        "image_effect_at_exp1_text": {},
        "video_effect_at_native_anchor": {},
        "video_effect_at_v8_anchor": {},
        "image_video_interaction": {},
    }
    for seed in seeds:
        native_native = results["image_native_qwen__text_native_neodragon"][seed]
        native_exp1 = results["image_native_qwen__text_exp1_64k"][seed]
        v8_native = results["image_v8_imageonly__text_native_neodragon"][seed]
        v8_exp1 = results["image_v8_imageonly__text_exp1_64k"][seed]
        effects["anchor_v8_minus_native"][seed] = delta(
            results["anchor_v8_imageonly"][seed], results["anchor_native_qwen"][seed]
        )
        effects["image_effect_at_native_text"][seed] = delta(v8_native, native_native)
        effects["image_effect_at_exp1_text"][seed] = delta(v8_exp1, native_exp1)
        effects["video_effect_at_native_anchor"][seed] = delta(native_exp1, native_native)
        effects["video_effect_at_v8_anchor"][seed] = delta(v8_exp1, v8_native)
        effects["image_video_interaction"][seed] = delta(
            effects["image_effect_at_exp1_text"][seed],
            effects["image_effect_at_native_text"][seed],
        )

    effects["mean_image_effect"] = {
        seed: average_deltas(
            effects["image_effect_at_native_text"][seed], effects["image_effect_at_exp1_text"][seed]
        )
        for seed in seeds
    }
    effects["mean_video_effect"] = {
        seed: average_deltas(
            effects["video_effect_at_native_anchor"][seed], effects["video_effect_at_v8_anchor"][seed]
        )
        for seed in seeds
    }

    aggregate_results = {
        cell: {
            metric: describe([flattened(value)[metric] for value in by_seed.values()])
            for metric in sorted(set.intersection(*(set(flattened(value)) for value in by_seed.values())))
        }
        for cell, by_seed in results.items()
    }
    aggregate_effects = {name: aggregate_by_seed(by_seed) for name, by_seed in effects.items()}
    output = {
        "status": "ok",
        "protocol": {
            "unit": "seed-level paired factorial effect on the same 20 VBench prompts",
            "seeds": seeds,
            "prompt_count": int(prepared["prompt_count"]),
            "video_cells": list(VIDEO_CELLS),
            "anchor_cells": list(ANCHOR_CELLS),
            "important_limit": (
                "VBench provides aggregate scores per cell. This report has two independent seed pairs, "
                "not per-prompt bootstrap confidence intervals; use contact sheets to validate borderline effects."
            ),
        },
        "by_seed": {cell: {str(seed): score for seed, score in values.items()} for cell, values in results.items()},
        "aggregate_cells": aggregate_results,
        "paired_effects_by_seed": {
            name: {str(seed): values for seed, values in by_seed.items()} for name, by_seed in effects.items()
        },
        "paired_effects_aggregate": aggregate_effects,
        "decision_metrics": {
            "anchor_image_effect.semantic_proxy": aggregate_effects["anchor_v8_minus_native"].get("semantic_proxy"),
            "video_image_effect.semantic_score": aggregate_effects["mean_image_effect"].get("semantic_score"),
            "video_bridge_effect.semantic_score": aggregate_effects["mean_video_effect"].get("semantic_score"),
            "image_video_interaction.semantic_score": aggregate_effects["image_video_interaction"].get("semantic_score"),
        },
        "contact_sheets": str(root / "contact_sheets"),
    }
    destination = args.output or root / "decision_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
