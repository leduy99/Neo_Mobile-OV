#!/usr/bin/env python
"""Aggregate per-prompt reports written by the staged NeoDragon DMD audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def average(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in rows[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reports = []
    for path in sorted(Path(args.root).glob("*/motion_audit.json")):
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    if not reports:
        raise FileNotFoundError(f"No per-prompt motion_audit.json files under {args.root}")

    systems = sorted({name for report in reports for name in report["decoded"]["systems"]})
    summary = {
        "prompt_count": len(reports),
        "teacher_target_mismatch": average(
            [report["teacher"]["target_mismatch"]["endpoint"]["mean"] for report in reports]
        ),
        "oracle_local_dmd": average(
            [report["student"]["oracle_local_dmd"]["mean"] for report in reports]
        ),
        "decoded_motion": {
            name: average([report["decoded"]["systems"][name] for report in reports])
            for name in systems
        },
        "latent_motion": {
            name: average([report["student"]["systems"][name]["latent_motion"] for report in reports])
            for name in reports[0]["student"]["systems"]
        },
        "ssd": average([report["ssd"]["latent_motion"] for report in reports if report["ssd"]]),
        "released_hybrid": average(
            [report["hybrid"]["latent_motion"] for report in reports if report["hybrid"]]
        ),
        "reports": [str(path.resolve()) for path in sorted(Path(args.root).glob("*/motion_audit.json"))],
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
