#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


AUDIT_FIELDS = (
    "record_id",
    "caption",
    "image_path",
    "qwen_accepted",
    "qwen_confidence",
    "qwen_reason",
    "human_supported",
    "human_comment",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic, balanced human-audit sheet for Qwen labels."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--score-filled-sheet",
        action="store_true",
        help="Score an existing output CSV after human_supported has been filled.",
    )
    parser.add_argument("--minimum-agreement", type=float, default=0.95)
    return parser.parse_args()


def deterministic_key(record_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest()


def score_sheet(
    path: Path, minimum_agreement: float, samples_per_class: int
) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    completed = [row for row in rows if row.get("human_supported", "").strip()]
    invalid = [
        row
        for row in completed
        if row["human_supported"].strip().lower() not in {"true", "false", "1", "0"}
    ]
    if invalid:
        raise ValueError("human_supported must contain only true/false or 1/0")
    matches = 0
    for row in completed:
        human = row["human_supported"].strip().lower() in {"true", "1"}
        qwen = row["qwen_accepted"].strip().lower() == "true"
        matches += int(human == qwen)
    agreement = matches / max(len(completed), 1)
    class_counts = {
        value: sum(row["qwen_accepted"].strip().lower() == value for row in rows)
        for value in ("true", "false")
    }
    coverage_passed = all(
        count >= samples_per_class for count in class_counts.values()
    )
    summary = {
        "audit_sheet": str(path),
        "rows": len(rows),
        "completed_rows": len(completed),
        "agreement": agreement,
        "class_counts": class_counts,
        "required_samples_per_class": samples_per_class,
        "minimum_agreement": minimum_agreement,
        "passed": (
            coverage_passed
            and len(completed) == len(rows)
            and agreement >= minimum_agreement
        ),
    }
    summary_path = path.with_suffix(path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    output = Path(args.output_csv)
    if args.score_filled_sheet:
        if not output.is_file():
            raise FileNotFoundError(f"Missing audit sheet: {output}")
        score_sheet(output, args.minimum_agreement, args.samples_per_class)
        return

    input_path = Path(args.input_jsonl)
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing Qwen result JSONL: {input_path}")
    groups: dict[bool, list[dict[str, object]]] = {True: [], False: []}
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if "error" in record:
                continue
            groups[bool(record.get("accepted"))].append(record)
    selected: list[dict[str, object]] = []
    for accepted, records in groups.items():
        records.sort(
            key=lambda record: deterministic_key(str(record["record_id"]), args.seed)
        )
        selected.extend(records[: args.samples_per_class])
    selected.sort(key=lambda record: deterministic_key(str(record["record_id"]), args.seed))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        for record in selected:
            annotation = record.get("annotation") or {}
            writer.writerow(
                {
                    "record_id": record["record_id"],
                    "caption": record.get("caption", ""),
                    "image_path": record.get("image_path", ""),
                    "qwen_accepted": str(bool(record.get("accepted"))).lower(),
                    "qwen_confidence": annotation.get("confidence", ""),
                    "qwen_reason": annotation.get("reason", ""),
                    "human_supported": "",
                    "human_comment": "",
                }
            )
    print(
        json.dumps(
            {
                "output_csv": str(output),
                "accepted_rows": min(len(groups[True]), args.samples_per_class),
                "rejected_rows": min(len(groups[False]), args.samples_per_class),
                "seed": args.seed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
