#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from new_mobile_ov.training.image_bridge_grounding_cascade import (  # noqa: E402
    FINAL_BUCKET_WEIGHTS,
    QWEN_BUCKET_WEIGHTS,
    LeakageIndex,
    alignment_score,
    balanced_select,
    capability_bucket,
    file_sha256,
    qwen_decisions,
    read_manifest,
    read_prompts,
    selection_counts,
    stable_row_key,
    write_manifest,
)


CASCADE_FIELDS = (
    "capability_bucket",
    "grounding_tier",
    "verification_source",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select Qwen audits and finalize the ImageBridge grounding cascade."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select-qwen")
    select.add_argument("--scored-manifest", required=True)
    select.add_argument("--output-manifest", required=True)
    select.add_argument("--qwen-jsonl", default="")
    select.add_argument("--target", type=int, default=12_000)
    select.add_argument("--leakage-prompts", action="append", default=[])
    select.add_argument("--leakage-threshold", type=float, default=0.85)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--scored-manifest", required=True)
    finalize.add_argument("--qwen-jsonl", required=True)
    finalize.add_argument("--qwen-accepted-manifest", default="")
    finalize.add_argument("--candidate-output", required=True)
    finalize.add_argument("--high-precision-output", required=True)
    finalize.add_argument("--summary-output", required=True)
    finalize.add_argument("--mixture-output", required=True)
    finalize.add_argument("--candidate-target", type=int, default=100_000)
    finalize.add_argument("--high-precision-target", type=int, default=50_000)
    finalize.add_argument("--min-caption-words", type=int, default=3)
    finalize.add_argument("--max-caption-words", type=int, default=80)
    finalize.add_argument("--leakage-prompts", action="append", default=[])
    finalize.add_argument("--leakage-threshold", type=float, default=0.85)
    finalize.add_argument("--allow-shortfall", action="store_true")
    return parser.parse_args()


def leakage_index(paths: list[str], threshold: float) -> LeakageIndex | None:
    prompts = read_prompts(paths)
    return LeakageIndex(prompts, threshold=threshold) if prompts else None


def valid_scored_rows(
    rows: list[dict[str, str]],
    *,
    leakage: LeakageIndex | None,
) -> tuple[list[dict[str, str]], Counter[str]]:
    stats: Counter[str] = Counter()
    valid: list[dict[str, str]] = []
    for row in rows:
        if str(row.get("siglip_status", "")) != "ok":
            stats["siglip_error"] += 1
            continue
        caption = str(row.get("caption", "")).strip()
        if leakage is not None and leakage.matches(caption):
            stats["benchmark_leakage"] += 1
            continue
        valid.append(row)
    return valid, stats


def select_qwen(args: argparse.Namespace) -> None:
    fields, rows = read_manifest(args.scored_manifest)
    leakage = leakage_index(args.leakage_prompts, args.leakage_threshold)
    valid, filtered = valid_scored_rows(rows, leakage=leakage)
    decisions = qwen_decisions(args.qwen_jsonl)
    errors = [row for row in valid if decisions.get(str(row.get("record_id"))) == "error"]
    pending = [
        row
        for row in valid
        if decisions.get(str(row.get("record_id"))) not in {"accepted", "rejected", "error"}
    ]
    errors.sort(key=lambda row: (-alignment_score(row), stable_row_key(row)))
    retry_rows = errors[: args.target]
    retry_ids = {str(row.get("record_id")) for row in retry_rows}
    remaining = max(0, args.target - len(retry_rows))
    selected = retry_rows + balanced_select(
        pending,
        target=remaining,
        weights=QWEN_BUCKET_WEIGHTS,
    )
    for row in selected:
        row["capability_bucket"] = capability_bucket(row)
        row["qwen_selection_reason"] = (
            "retry_previous_error"
            if str(row.get("record_id")) in retry_ids
            else "balanced_hard_adjudication"
        )
    write_manifest(
        args.output_manifest,
        selected,
        preferred_fields=[*fields, "capability_bucket", "qwen_selection_reason"],
    )
    summary = {
        "command": "select-qwen",
        "scored_manifest": args.scored_manifest,
        "qwen_jsonl": args.qwen_jsonl or None,
        "requested": args.target,
        "selected": len(selected),
        "retries": len(retry_rows),
        "filtered": dict(filtered),
        "selection": selection_counts(selected),
    }
    summary_path = Path(args.output_manifest).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def accepted_metadata(path: str) -> dict[str, dict[str, str]]:
    if not path or not Path(path).is_file():
        return {}
    _, rows = read_manifest(path)
    return {
        str(row.get("record_id", "")): row
        for row in rows
        if str(row.get("record_id", ""))
    }


def finalize(args: argparse.Namespace) -> None:
    fields, rows = read_manifest(args.scored_manifest)
    leakage = leakage_index(args.leakage_prompts, args.leakage_threshold)
    valid, filtered = valid_scored_rows(rows, leakage=leakage)
    decisions = qwen_decisions(args.qwen_jsonl)
    qwen_rows = accepted_metadata(args.qwen_accepted_manifest)
    eligible: list[dict[str, object]] = []
    state_counts: Counter[str] = Counter()
    for raw_row in valid:
        row: dict[str, object] = dict(raw_row)
        record_id = str(row.get("record_id", ""))
        caption_words = len(str(row.get("caption", "")).split())
        if not args.min_caption_words <= caption_words <= args.max_caption_words:
            filtered["caption_length"] += 1
            continue
        state = decisions.get(record_id, "not_audited")
        state_counts[state] += 1
        if state == "rejected":
            filtered["qwen_rejected"] += 1
            continue
        row["capability_bucket"] = capability_bucket(row)
        if state == "accepted":
            row.update(qwen_rows.get(record_id, {}))
            row["grounding_status"] = "qwen36_verified"
            row["grounding_tier"] = "qwen_verified"
            row["verification_source"] = "qwen36"
        elif state == "error":
            row["grounding_tier"] = "candidate_only"
            row["verification_source"] = "qwen_error"
        else:
            row["grounding_tier"] = "siglip_ranked"
            row["verification_source"] = "siglip2"
        eligible.append(row)

    accepted_ids = {record_id for record_id, state in decisions.items() if state == "accepted"}
    candidate = balanced_select(
        eligible,
        target=args.candidate_target,
        weights=FINAL_BUCKET_WEIGHTS,
        preferred_ids=accepted_ids,
    )
    high_eligible = [
        row
        for row in candidate
        if str(row.get("verification_source")) != "qwen_error"
    ]
    high_precision = balanced_select(
        high_eligible,
        target=args.high_precision_target,
        weights=FINAL_BUCKET_WEIGHTS,
        preferred_ids=accepted_ids,
    )
    for row in high_precision:
        if str(row.get("verification_source")) == "siglip2":
            row["grounding_status"] = "siglip2_high_confidence"
            row["grounding_tier"] = "high_precision"

    shortfalls = {
        "candidate": args.candidate_target - len(candidate),
        "high_precision": args.high_precision_target - len(high_precision),
    }
    if not args.allow_shortfall and any(value > 0 for value in shortfalls.values()):
        raise RuntimeError(
            f"Cascade target shortfall: {shortfalls}. Use --allow-shortfall only for smoke tests."
        )
    candidate_path = Path(args.candidate_output)
    high_path = Path(args.high_precision_output)
    output_fields = [*fields, *CASCADE_FIELDS]
    write_manifest(candidate_path, candidate, preferred_fields=output_fields)
    write_manifest(high_path, high_precision, preferred_fields=output_fields)

    mixture = {
        "name": "v12_grounding_cascade_70_20_10",
        "overall_sampling": {"broad": 0.70, "compositional": 0.20, "grounded": 0.10},
        "trainer_contract": {
            "generation_prompt_manifests": [
                "manifests/d1_broad_train.csv",
                "manifests/d2_compositional_train.csv",
                f"manifests/{high_path.name}",
            ],
            "generation_source_names": ["broad", "compositional", "grounded_cascade"],
            "generation_source_weights": [7.0 / 9.0, 2.0 / 9.0, 0.0],
            "grounded_source_names": ["grounded_cascade"],
            "grounded_batch_probability": 0.10,
            "semantic_prompt_probability": 0.0,
        },
        "grounded_gate": {
            "candidate_manifest": candidate_path.name,
            "training_manifest": high_path.name,
            "qwen_rejections_are_excluded": True,
            "qwen_errors_are_excluded_from_training": True,
            "siglip_is_used_for_ranking_not_as_a_factual_label": True,
        },
    }
    mixture_path = Path(args.mixture_output)
    mixture_path.parent.mkdir(parents=True, exist_ok=True)
    mixture_path.write_text(json.dumps(mixture, indent=2) + "\n", encoding="utf-8")

    summary = {
        "dataset": "ImageBridge-Data-v1-grounding-cascade",
        "inputs": {
            "scored_manifest": args.scored_manifest,
            "qwen_jsonl": args.qwen_jsonl,
            "qwen_accepted_manifest": args.qwen_accepted_manifest or None,
        },
        "targets": {
            "candidate": args.candidate_target,
            "high_precision": args.high_precision_target,
        },
        "shortfalls": shortfalls,
        "filtered": dict(sorted(filtered.items())),
        "qwen_states": dict(sorted(state_counts.items())),
        "eligible_after_filters": len(eligible),
        "candidate": selection_counts(candidate),
        "high_precision": selection_counts(high_precision),
        "outputs": {
            "candidate": {
                "path": str(candidate_path),
                "sha256": file_sha256(candidate_path),
            },
            "high_precision": {
                "path": str(high_path),
                "sha256": file_sha256(high_path),
            },
            "mixture": {
                "path": str(mixture_path),
                "sha256": file_sha256(mixture_path),
            },
        },
    }
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.command == "select-qwen":
        select_qwen(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
