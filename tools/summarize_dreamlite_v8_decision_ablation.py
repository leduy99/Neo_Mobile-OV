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
DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "temporal_flickering",
    "object_class",
    "multiple_objects",
    "color",
    "spatial_relationship",
    "scene",
    "temporal_style",
    "overall_consistency",
    "human_action",
    "appearance_style",
)
QUALITY = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
    "dynamic_degree",
)
SEMANTIC = (
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
    "overall_consistency",
)
NORMALIZATION = {
    "subject_consistency": (0.1462, 1.0),
    "background_consistency": (0.2615, 1.0),
    "temporal_flickering": (0.6293, 1.0),
    "motion_smoothness": (0.706, 0.9975),
    "dynamic_degree": (0.0, 1.0),
    "aesthetic_quality": (0.0, 1.0),
    "imaging_quality": (0.0, 1.0),
    "object_class": (0.0, 1.0),
    "multiple_objects": (0.0, 1.0),
    "human_action": (0.0, 1.0),
    "color": (0.0, 1.0),
    "spatial_relationship": (0.0, 1.0),
    "scene": (0.0, 0.8222),
    "appearance_style": (0.0009, 0.2855),
    "temporal_style": (0.0, 0.364),
    "overall_consistency": (0.0, 0.364),
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


def partial_score(scaled: dict[str, float], dimensions: tuple[str, ...]) -> float | None:
    present = [dimension for dimension in dimensions if dimension in scaled]
    if not present:
        return None
    weights = {dimension: 0.5 if dimension == "dynamic_degree" else 1.0 for dimension in present}
    return sum(scaled[dimension] for dimension in present) / sum(weights.values())


def video_score(root: Path, seed: int, cell: str, active_dimensions: list[str]) -> dict:
    directory = score_path(root, seed, cell)
    raw: dict[str, float] = {}
    for dimension in active_dimensions:
        payload = json.loads((directory / f"{dimension}_eval_results.json").read_text(encoding="utf-8"))
        raw[dimension] = float(payload[dimension][0])
    scaled = {
        dimension: (value - NORMALIZATION[dimension][0])
        / (NORMALIZATION[dimension][1] - NORMALIZATION[dimension][0])
        * (0.5 if dimension == "dynamic_degree" else 1.0)
        for dimension, value in raw.items()
    }
    quality_score = partial_score(scaled, QUALITY)
    semantic_score = partial_score(scaled, SEMANTIC)
    return {
        "evaluated_dimensions": active_dimensions,
        "quality_score": quality_score,
        "semantic_score": semantic_score,
        "total_score": (
            (4.0 * quality_score + semantic_score) / 5.0
            if quality_score is not None and semantic_score is not None
            else None
        ),
        "scaled": scaled,
        "raw": raw,
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


def condition_metrics_by_seed(prepared: dict, seeds: list[int]) -> dict[int, dict[str, float]]:
    source_by_seed = {int(source["seed"]): source for source in prepared.get("sources", [])}
    metrics: dict[int, dict[str, float]] = {}
    for seed in seeds:
        source = source_by_seed.get(seed)
        if source is None:
            raise RuntimeError(f"Prepared metadata does not identify the source for seed={seed}")
        payload = source.get("mean_condition_metrics")
        if not isinstance(payload, dict) or not payload:
            raise RuntimeError(f"Source seed={seed} has no mean_condition_metrics")
        metrics[seed] = {name: float(value) for name, value in payload.items() if isinstance(value, (int, float))}
    return metrics


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
    active_dimensions = [str(value) for value in prepared.get("active_dimensions", [])]
    if not active_dimensions:
        raise RuntimeError("Prepared ablation has no active VBench dimensions")
    if any(dimension not in DIMENSIONS for dimension in active_dimensions):
        raise RuntimeError(f"Unsupported VBench dimension in metadata: {active_dimensions}")

    results: dict[str, dict[int, dict]] = {cell: {} for cell in VIDEO_CELLS}
    for seed in seeds:
        for cell in VIDEO_CELLS:
            results[cell][seed] = video_score(root, seed, cell, active_dimensions)
    condition_metrics = condition_metrics_by_seed(prepared, seeds)

    effects: dict[str, dict[int, dict[str, float]]] = {
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
    aggregate_condition_metrics = aggregate_by_seed(condition_metrics)
    output = {
        "status": "ok",
        "protocol": {
            "unit": "seed-level paired factorial effect on the same 20 prompts and matched VBench dimensions",
            "seeds": seeds,
            "prompt_count": int(prepared["prompt_count"]),
            "video_cells": list(VIDEO_CELLS),
            "evaluated_vbench_dimensions": active_dimensions,
            "important_limit": (
                "The selected 20 prompts do not cover the entire VBench suite. Scores are computed only over "
                "matched dimensions represented by those prompts and are not comparable to full VBench totals. "
                "This report has two independent seed pairs, not per-prompt bootstrap confidence intervals; "
                "use contact sheets to validate borderline effects. "
                "Image-bridge alignment is reported from the original direct condition-distance measurements."
            ),
        },
        "image_condition_alignment_by_seed": {
            str(seed): values for seed, values in condition_metrics.items()
        },
        "image_condition_alignment_aggregate": aggregate_condition_metrics,
        "by_seed": {cell: {str(seed): score for seed, score in values.items()} for cell, values in results.items()},
        "aggregate_cells": aggregate_results,
        "paired_effects_by_seed": {
            name: {str(seed): values for seed, values in by_seed.items()} for name, by_seed in effects.items()
        },
        "paired_effects_aggregate": aggregate_effects,
        "decision_metrics": {
            "image_bridge_alignment.token_cosine_distance": aggregate_condition_metrics.get("token_cosine_distance"),
            "image_bridge_alignment.content_token_cosine_distance": aggregate_condition_metrics.get(
                "content_token_cosine_distance"
            ),
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
