#!/usr/bin/env python
"""Attribute the Joint-v2 VBench gap to image conditioning and QuickSR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_CELLS = (
    "bridge_no_qsr",
    "bridge_qsr",
    "native_no_qsr",
    "native_qsr",
)

# NeoDragon Table 8 reports percentages. Keep only directly comparable fields.
NEODRAGON_E2E = {
    "quality_score": 0.8368,
    "semantic_score": 0.7336,
    "total_score": 0.8161,
    "raw.temporal_flickering": 0.9927,
    "raw.aesthetic_quality": 0.6071,
    "raw.imaging_quality": 0.5978,
    "raw.object_class": 0.9237,
    "raw.scene": 0.5656,
    "raw.overall_consistency": 0.2809,
}


def flatten_metrics(payload: dict) -> dict[str, float]:
    metrics = {
        name: float(payload[name])
        for name in ("quality_score", "semantic_score", "total_score")
    }
    for group in ("raw", "scaled"):
        values = payload.get(group, {})
        for name, value in values.items():
            metrics[f"{group}.{name}"] = float(value)
    return metrics


def subtract(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {
        name: left[name] - right[name]
        for name in sorted(left.keys() & right.keys())
    }


def selected(values: dict[str, float]) -> dict[str, float]:
    names = tuple(NEODRAGON_E2E)
    return {name: values[name] for name in names if name in values}


def parse_cell(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Cells must use LABEL=/path/to/summary.json")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("Cells must use LABEL=/path/to/summary.json")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", action="append", type=parse_cell, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = dict(args.cell)
    missing = [name for name in REQUIRED_CELLS if name not in paths]
    if missing:
        raise ValueError(f"Missing required cells: {missing}")

    cells: dict[str, dict[str, float]] = {}
    for label in REQUIRED_CELLS:
        path = paths[label]
        payload = json.loads(path.read_text(encoding="utf-8"))
        cells[label] = flatten_metrics(payload)

    image_effect_no_qsr = subtract(cells["native_no_qsr"], cells["bridge_no_qsr"])
    image_effect_qsr = subtract(cells["native_qsr"], cells["bridge_qsr"])
    qsr_effect_bridge = subtract(cells["bridge_qsr"], cells["bridge_no_qsr"])
    qsr_effect_native = subtract(cells["native_qsr"], cells["native_no_qsr"])
    interaction = subtract(image_effect_qsr, image_effect_no_qsr)
    paper_gap = {
        name: cells["native_qsr"][name] - target
        for name, target in NEODRAGON_E2E.items()
        if name in cells["native_qsr"]
    }

    bridge_qsr_total = cells["bridge_qsr"]["total_score"]
    native_qsr_total = cells["native_qsr"]["total_score"]
    if bridge_qsr_total >= NEODRAGON_E2E["total_score"]:
        decision = "bridge_pipeline_reaches_directional_1x_target"
    elif native_qsr_total >= NEODRAGON_E2E["total_score"]:
        decision = "image_bridge_is_the_remaining_primary_bottleneck"
    else:
        decision = "image_bridge_and_non_image_pipeline_both_need_improvement"

    report = {
        "protocol": {
            "prompts": 944,
            "samples_per_prompt": 1,
            "paired_seed_and_video_checkpoint": True,
            "warning": (
                "This is a controlled 1-video-per-prompt attribution study. "
                "NeoDragon Table 8 is the published target, not a protocol-matched 1x control."
            ),
        },
        "paper_target": NEODRAGON_E2E,
        "cells": {name: selected(values) for name, values in cells.items()},
        "effects": {
            "native_qwen_minus_bridge_no_qsr": selected(image_effect_no_qsr),
            "native_qwen_minus_bridge_with_qsr": selected(image_effect_qsr),
            "qsr_on_bridge": selected(qsr_effect_bridge),
            "qsr_on_native_qwen": selected(qsr_effect_native),
            "image_qsr_interaction": selected(interaction),
        },
        "native_qwen_qsr_minus_neodragon_e2e": paper_gap,
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
