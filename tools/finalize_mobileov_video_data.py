#!/usr/bin/env python
"""Freeze a high-precision OpenVid subset from SigLIP2 and motion scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from new_mobile_ov.training.mobileov_data_release import file_sha256


METRICS = (
    "siglip_logit",
    "motion_frame_diff_mean",
    "motion_optical_flow_mean",
    "transition_max_diff",
)


def finite_float(value: object) -> float | None:
    try:
        result = float(str(value).strip())
        return result if math.isfinite(result) else None
    except Exception:
        return None


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--alignment-quantile", type=float, default=0.25)
    parser.add_argument("--motion-low-quantile", type=float, default=0.10)
    parser.add_argument("--transition-high-quantile", type=float, default=0.99)
    parser.add_argument("--absolute-min-frame-diff", type=float, default=0.005)
    parser.add_argument("--absolute-min-optical-flow", type=float, default=0.05)
    parser.add_argument("--absolute-max-transition-diff", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.scores)
    output = Path(args.output)
    summary_path = Path(args.summary)
    values = {metric: [] for metric in METRICS}
    valid_rows = 0
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing score schema: {source}")
        fieldnames = list(reader.fieldnames)
        for row in reader:
            parsed = {metric: finite_float(row.get(metric)) for metric in METRICS}
            if row.get("video_score_status") != "ok" or any(
                value is None for value in parsed.values()
            ):
                continue
            valid_rows += 1
            for metric, value in parsed.items():
                values[metric].append(float(value))
    if valid_rows == 0:
        raise RuntimeError(f"No valid video scores in {source}")
    distributions = {metric: quantiles(metric_values) for metric, metric_values in values.items()}
    thresholds = {
        "siglip_logit_min": float(np.quantile(values["siglip_logit"], args.alignment_quantile)),
        "motion_frame_diff_min": max(
            float(np.quantile(values["motion_frame_diff_mean"], args.motion_low_quantile)),
            args.absolute_min_frame_diff,
        ),
        "motion_optical_flow_min": max(
            float(np.quantile(values["motion_optical_flow_mean"], args.motion_low_quantile)),
            args.absolute_min_optical_flow,
        ),
        "transition_max_diff_max": min(
            float(np.quantile(values["transition_max_diff"], args.transition_high_quantile)),
            args.absolute_max_transition_diff,
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    accepted = rejected = errors = 0
    rejection_counts = {
        "alignment": 0,
        "frame_motion": 0,
        "optical_flow": 0,
        "transition": 0,
    }
    output_fields = list(
        dict.fromkeys(
            [*fieldnames, "verification_status", "verification_source", "alignment_score"]
        )
    )
    with source.open("r", encoding="utf-8", newline="") as source_handle, temporary.open(
        "w", encoding="utf-8", newline=""
    ) as output_handle:
        reader = csv.DictReader(source_handle)
        writer = csv.DictWriter(output_handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        for row in reader:
            parsed = {metric: finite_float(row.get(metric)) for metric in METRICS}
            if row.get("video_score_status") != "ok" or any(
                value is None for value in parsed.values()
            ):
                errors += 1
                continue
            failures = []
            if parsed["siglip_logit"] < thresholds["siglip_logit_min"]:
                failures.append("alignment")
            if parsed["motion_frame_diff_mean"] < thresholds["motion_frame_diff_min"]:
                failures.append("frame_motion")
            if parsed["motion_optical_flow_mean"] < thresholds["motion_optical_flow_min"]:
                failures.append("optical_flow")
            if parsed["transition_max_diff"] > thresholds["transition_max_diff_max"]:
                failures.append("transition")
            if failures:
                rejected += 1
                for reason in failures:
                    rejection_counts[reason] += 1
                continue
            row["verification_status"] = "video_cascade_pass"
            row["verification_source"] = "siglip2_multiframe_and_motion"
            row["alignment_score"] = row["siglip_logit"]
            writer.writerow(row)
            accepted += 1
    temporary.replace(output)
    summary = {
        "source": str(source),
        "source_sha256": file_sha256(source),
        "output": str(output),
        "output_sha256": file_sha256(output),
        "valid_scored_rows": valid_rows,
        "accepted_rows": accepted,
        "rejected_rows": rejected,
        "error_rows": errors,
        "threshold_policy": (
            "quantile gates with absolute non-static and hard-transition safety bounds"
        ),
        "thresholds": thresholds,
        "score_distributions": distributions,
        "rejection_counts": rejection_counts,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
