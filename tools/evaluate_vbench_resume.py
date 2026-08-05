#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from vbench import VBench


DIMENSIONS = [
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
]

QUALITY = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
    "dynamic_degree",
]
SEMANTIC = [
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
    "overall_consistency",
]
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


def valid_result(path: Path, dimension: str) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return dimension in value
    except Exception:
        return False


def tabulate(score_dir: Path) -> dict:
    raw = {}
    for dimension in DIMENSIONS:
        path = score_dir / f"{dimension}_eval_results.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw[dimension] = float(payload[dimension][0])
    weights = {dimension: 1.0 for dimension in DIMENSIONS}
    weights["dynamic_degree"] = 0.5
    scaled = {}
    for dimension, value in raw.items():
        minimum, maximum = NORMALIZATION[dimension]
        scaled[dimension] = (value - minimum) / (maximum - minimum) * weights[dimension]
    quality = sum(scaled[name] for name in QUALITY) / sum(weights[name] for name in QUALITY)
    semantic = sum(scaled[name] for name in SEMANTIC) / sum(weights[name] for name in SEMANTIC)
    result = {
        "raw": raw,
        "scaled": scaled,
        "quality_score": quality,
        "semantic_score": semantic,
        "total_score": (4.0 * quality + semantic) / 5.0,
    }
    (score_dir / "all_results.json").write_text(json.dumps(raw, indent=2) + "\n")
    (score_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume VBench one dimension at a time.")
    parser.add_argument("--videos", required=True)
    parser.add_argument("--full-info", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(DIMENSIONS))
    args = parser.parse_args()

    output = Path(args.output).resolve()
    score_dir = output / "vbench"
    score_dir.mkdir(parents=True, exist_ok=True)
    evaluator = VBench(torch.device("cuda"), str(Path(args.full_info).resolve()), str(score_dir))
    for dimension in DIMENSIONS[args.start : args.end]:
        result_path = score_dir / f"{dimension}_eval_results.json"
        if valid_result(result_path, dimension):
            print(f"Skipping completed dimension: {dimension}", flush=True)
            continue
        print(f"Evaluating dimension: {dimension}", flush=True)
        evaluator.evaluate(
            videos_path=str(Path(args.videos).resolve()),
            name=dimension,
            local=True,
            read_frame=False,
            dimension_list=[dimension],
            mode="vbench_standard",
            imaging_quality_preprocessing_mode="longer",
        )
    requested = DIMENSIONS[args.start : args.end]
    missing_requested = [
        dimension
        for dimension in requested
        if not valid_result(score_dir / f"{dimension}_eval_results.json", dimension)
    ]
    if missing_requested:
        raise SystemExit(f"Incomplete requested VBench dimensions: {missing_requested}")
    if args.start != 0 or args.end < len(DIMENSIONS):
        print(f"Completed partial VBench slice: {requested}", flush=True)
        return

    missing = [
        dimension
        for dimension in DIMENSIONS
        if not valid_result(score_dir / f"{dimension}_eval_results.json", dimension)
    ]
    if missing:
        raise SystemExit(f"Incomplete full VBench run: {missing}")
    summary = tabulate(score_dir)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
