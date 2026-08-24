#!/usr/bin/env python
"""Report the lowest-scoring VBench prompts with their generated video paths."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any


DEFAULT_DIMENSIONS = (
    "object_class",
    "multiple_objects",
    "spatial_relationship",
    "scene",
    "human_action",
    "overall_consistency",
    "dynamic_degree",
    "temporal_style",
    "appearance_style",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dimension", action="append", default=[])
    parser.add_argument(
        "--failure-dir",
        type=Path,
        help="Optional directory populated with symlinks to the reported videos.",
    )
    return parser.parse_args()


def scalar_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, list):
        values = [score for item in value if (score := scalar_score(item)) is not None]
        return sum(values) / len(values) if values else None
    return None


def prompt_from_video(path: str) -> str:
    return re.sub(r"-\d+$", "", Path(path).stem)


def metadata_index(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_path: dict[str, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return by_path, by_name
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload:
        metadata = {
            "prompt": row.get("prompt_en", ""),
            "auxiliary_info": row.get("auxiliary_info", {}),
        }
        for video_path in row.get("video_list", []):
            by_path[str(Path(video_path).expanduser().resolve())] = metadata
            by_name[Path(video_path).name] = metadata
    return by_path, by_name


def load_dimension(score_dir: Path, dimension: str, top_k: int) -> dict[str, Any] | None:
    result_path = score_dir / f"{dimension}_eval_results.json"
    if not result_path.is_file():
        print(f"Skipping {dimension}: missing {result_path}")
        return None
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    result = payload.get(dimension)
    if not isinstance(result, list) or len(result) != 2 or not isinstance(result[1], list):
        raise ValueError(f"Unexpected VBench result schema in {result_path}")

    by_path, by_name = metadata_index(score_dir / f"{dimension}_full_info.json")
    rows: list[dict[str, Any]] = []
    for video_result in result[1]:
        if not isinstance(video_result, dict):
            continue
        video_path = str(video_result.get("video_path", ""))
        score = scalar_score(video_result.get("video_results"))
        if not video_path or score is None:
            continue
        metadata = by_path.get(str(Path(video_path).expanduser().resolve()))
        if metadata is None:
            metadata = by_name.get(Path(video_path).name, {})
        rows.append(
            {
                "score": score,
                "prompt": metadata.get("prompt") or prompt_from_video(video_path),
                "video_path": video_path,
                "auxiliary_info": metadata.get("auxiliary_info", {}),
            }
        )
    rows.sort(key=lambda row: (row["score"], row["prompt"], row["video_path"]))
    aggregate = scalar_score(result[0])
    return {
        "aggregate_score": aggregate,
        "evaluated_videos": len(rows),
        "zero_score_videos": sum(row["score"] == 0.0 for row in rows),
        "bottom_prompts": rows[:top_k],
    }


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# VBench Lowest-Scoring Prompts",
        "",
        f"Score directory: `{report['score_dir']}`",
        "",
        (
            "Lower scores are worse for every listed VBench dimension. These are diagnostic "
            "examples, not a new aggregate metric."
        ),
    ]
    for dimension, payload in report["dimensions"].items():
        aggregate = payload["aggregate_score"]
        aggregate_text = "n/a" if aggregate is None else f"{aggregate:.6f}"
        lines.extend(
            [
                "",
                f"## {dimension}",
                "",
                (
                    f"Aggregate: `{aggregate_text}`; evaluated videos: "
                    f"`{payload['evaluated_videos']}`; zero-score videos: "
                    f"`{payload['zero_score_videos']}`."
                ),
                "",
                "| Rank | Score | Prompt | Video |",
                "|---:|---:|---|---|",
            ]
        )
        for rank, row in enumerate(payload["bottom_prompts"], start=1):
            lines.append(
                f"| {rank} | {row['score']:.6f} | "
                f"{markdown_escape(row['prompt'])} | `{markdown_escape(row['video_path'])}` |"
            )
    return "\n".join(lines) + "\n"


def link_failures(report: dict[str, Any], failure_dir: Path) -> None:
    failure_dir.mkdir(parents=True, exist_ok=True)
    for dimension, payload in report["dimensions"].items():
        dimension_dir = failure_dir / dimension
        dimension_dir.mkdir(parents=True, exist_ok=True)
        for rank, row in enumerate(payload["bottom_prompts"], start=1):
            source = Path(row["video_path"]).expanduser().resolve()
            if not source.is_file():
                continue
            destination = dimension_dir / f"{rank:02d}_{row['score']:.4f}_{source.name}"
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            os.symlink(source, destination)


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    score_dir = args.score_dir.expanduser().resolve()
    dimensions = tuple(args.dimension) if args.dimension else DEFAULT_DIMENSIONS
    report: dict[str, Any] = {
        "score_dir": str(score_dir),
        "top_k": args.top_k,
        "dimensions": {},
    }
    for dimension in dimensions:
        payload = load_dimension(score_dir, dimension, args.top_k)
        if payload is not None:
            report["dimensions"][dimension] = payload
    if not report["dimensions"]:
        raise FileNotFoundError(f"No requested VBench result files found under {score_dir}")

    output_prefix = args.output_prefix.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    markdown_path = output_prefix.with_suffix(".md")
    markdown = render_markdown(report)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    if args.failure_dir:
        link_failures(report, args.failure_dir.expanduser().resolve())
    print(markdown)
    print(f"Saved low-prompt JSON: {json_path}")
    print(f"Saved low-prompt Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
