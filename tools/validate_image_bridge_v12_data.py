#!/usr/bin/env python
"""Validate the frozen ImageBridge-Data-v1 contract before V12 training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.training.image_bridge_grounding_cascade import (  # noqa: E402
    LeakageIndex,
    read_prompts,
)


REQUIRED_BASE_FIELDS = {"record_id", "caption"}
REQUIRED_GROUNDED_FIELDS = {
    "record_id",
    "caption",
    "image_path",
    "capabilities",
    "grounding_status",
    "verification_source",
    "siglip_status",
}
ALLOWED_GROUNDED_SOURCES = {"qwen36", "siglip2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/image_bridge_v1")
    parser.add_argument("--mixture", default="mixtures/v12_grounding_cascade_70_20_10.json")
    parser.add_argument("--cascade-summary", default="stats/grounding_cascade_summary.json")
    parser.add_argument("--output", default="stats/v12_data_preflight.json")
    parser.add_argument("--broad-manifest", default="manifests/d1_broad_train.csv")
    parser.add_argument(
        "--compositional-manifest", default="manifests/d2_compositional_train.csv"
    )
    parser.add_argument(
        "--grounded-manifest", default="manifests/d2_grounded_high_precision_50k.csv"
    )
    parser.add_argument("--validation-manifest", default="manifests/validation.csv")
    parser.add_argument("--hard-validation-manifest", default="manifests/hard_validation.csv")
    parser.add_argument("--candidate-manifest", default="manifests/d2_grounded_candidate_100k.csv")
    parser.add_argument("--skip-cascade-hash-check", action="store_true")
    parser.add_argument("--rescore-manifest", default="")
    parser.add_argument("--audit-sheet", default="")
    parser.add_argument("--leakage-prompts", action="append", default=[])
    parser.add_argument("--leakage-threshold", type=float, default=0.85)
    parser.add_argument("--expected-grounded-rows", type=int, default=50_000)
    parser.add_argument("--decode-samples", type=int, default=2_048)
    parser.add_argument("--decode-workers", type=int, default=16)
    parser.add_argument(
        "--allow-leakage",
        action="store_true",
        help="Report benchmark leakage without failing. Intended only for diagnosis.",
    )
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path
    try:
        path.relative_to(root)
    except ValueError:
        return root / path
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(record_id: str) -> str:
    return hashlib.sha256(record_id.encode("utf-8")).hexdigest()


def iter_manifest(path: Path) -> tuple[list[str], Iterable[dict[str, str]]]:
    handle = path.open("r", encoding="utf-8", newline="")
    reader = csv.DictReader(handle)
    if not reader.fieldnames:
        handle.close()
        raise ValueError(f"Manifest has no header: {path}")

    def rows() -> Iterable[dict[str, str]]:
        try:
            yield from reader
        finally:
            handle.close()

    return list(reader.fieldnames), rows()


def inspect_manifest(
    path: Path,
    *,
    required_fields: set[str],
    leakage: LeakageIndex | None,
    keep_grounded_rows: bool = False,
) -> tuple[dict[str, object], set[str], list[dict[str, str]]]:
    fields, rows = iter_manifest(path)
    missing_fields = sorted(required_fields - set(fields))
    if missing_fields:
        raise ValueError(f"{path} is missing required fields: {missing_fields}")
    ids: set[str] = set()
    duplicate_ids = 0
    empty_ids = 0
    empty_captions = 0
    leakage_count = 0
    leakage_examples: list[dict[str, str]] = []
    grounded_rows: list[dict[str, str]] = []
    capabilities: Counter[str] = Counter()
    verification_sources: Counter[str] = Counter()
    siglip_statuses: Counter[str] = Counter()
    grounding_statuses: Counter[str] = Counter()
    invalid_grounded_source = 0
    invalid_siglip_status = 0
    missing_image_path = 0
    row_count = 0
    for row in rows:
        row_count += 1
        record_id = str(row.get("record_id", "")).strip()
        caption = str(row.get("caption", "")).strip()
        if not record_id:
            empty_ids += 1
        elif record_id in ids:
            duplicate_ids += 1
        else:
            ids.add(record_id)
        if not caption:
            empty_captions += 1
        if leakage is not None and leakage.matches(caption):
            leakage_count += 1
            if len(leakage_examples) < 20:
                leakage_examples.append({"record_id": record_id, "caption": caption})
        if keep_grounded_rows:
            grounded_rows.append(row)
            source = str(row.get("verification_source", "")).strip()
            status = str(row.get("grounding_status", "")).strip()
            verification_sources[source] += 1
            siglip_status = str(row.get("siglip_status", "")).strip()
            siglip_statuses[siglip_status] += 1
            grounding_statuses[status] += 1
            capabilities.update(
                item for item in str(row.get("capabilities", "")).split(";") if item
            )
            invalid_grounded_source += int(source not in ALLOWED_GROUNDED_SOURCES)
            # A factual Qwen decision supersedes the ranking model. Only rows
            # admitted solely by SigLIP2 require a valid SigLIP scoring status.
            invalid_siglip_status += int(source == "siglip2" and siglip_status != "ok")
            missing_image_path += int(not str(row.get("image_path", "")).strip())
    summary = {
        "path": str(path),
        "rows": row_count,
        "unique_ids": len(ids),
        "duplicate_ids": duplicate_ids,
        "empty_ids": empty_ids,
        "empty_captions": empty_captions,
        "leakage_count": leakage_count,
        "leakage_examples": leakage_examples,
    }
    if keep_grounded_rows:
        summary.update(
            {
                "verification_sources": dict(sorted(verification_sources.items())),
                "siglip_statuses": dict(sorted(siglip_statuses.items())),
                "grounding_statuses": dict(sorted(grounding_statuses.items())),
                "capabilities": dict(sorted(capabilities.items())),
                "invalid_grounded_source": invalid_grounded_source,
                "invalid_siglip_status": invalid_siglip_status,
                "missing_image_path": missing_image_path,
            }
        )
    return summary, ids, grounded_rows


def verify_image(path_text: str) -> tuple[str, str]:
    path = Path(path_text).expanduser()
    if not path.is_file():
        return "missing", str(path)
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, ValueError) as error:
        return "unreadable", f"{path}: {type(error).__name__}: {error}"
    return "ok", str(path)


def image_preflight(
    rows: list[dict[str, str]], *, sample_count: int, workers: int
) -> dict[str, object]:
    ordered = sorted(rows, key=lambda row: stable_key(str(row.get("record_id", ""))))
    selected = ordered if sample_count < 0 else ordered[:sample_count]
    paths = [str(row.get("image_path", "")).strip() for row in selected]
    counts: Counter[str] = Counter()
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for status, detail in executor.map(verify_image, paths):
            counts[status] += 1
            if status != "ok" and len(failures) < 20:
                failures.append(detail)
    return {
        "requested": sample_count,
        "checked": len(selected),
        "status_counts": dict(sorted(counts.items())),
        "failure_examples": failures,
    }


def float_or_nan(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def compare_rescore(
    grounded_rows: list[dict[str, str]], path: Path | None
) -> dict[str, object]:
    if path is None or not path.is_file():
        return {"enabled": False}
    original = {
        str(row.get("record_id", "")): float_or_nan(row.get("siglip_logit"))
        for row in grounded_rows
    }
    _, rescored = iter_manifest(path)
    pairs: list[tuple[float, float]] = []
    status_counts: Counter[str] = Counter()
    missing_original = 0
    for row in rescored:
        status_counts[str(row.get("siglip_status", ""))] += 1
        record_id = str(row.get("record_id", ""))
        before = original.get(record_id, float("nan"))
        after = float_or_nan(row.get("siglip_logit"))
        if not math.isfinite(before):
            missing_original += 1
        elif math.isfinite(after):
            pairs.append((before, after))
    correlation = float("nan")
    mean_abs_delta = float("nan")
    if pairs:
        before_mean = sum(item[0] for item in pairs) / len(pairs)
        after_mean = sum(item[1] for item in pairs) / len(pairs)
        covariance = sum(
            (before - before_mean) * (after - after_mean) for before, after in pairs
        )
        before_var = sum((before - before_mean) ** 2 for before, _ in pairs)
        after_var = sum((after - after_mean) ** 2 for _, after in pairs)
        if before_var > 0 and after_var > 0:
            correlation = covariance / math.sqrt(before_var * after_var)
        mean_abs_delta = sum(abs(before - after) for before, after in pairs) / len(pairs)
    return {
        "enabled": True,
        "path": str(path),
        "rows_compared": len(pairs),
        "missing_original_score": missing_original,
        "status_counts": dict(sorted(status_counts.items())),
        "pearson_logit": correlation,
        "mean_absolute_logit_delta": mean_abs_delta,
    }


def audit_status(path: Path | None) -> dict[str, object]:
    if path is None or not path.is_file():
        return {"available": False}
    fields, rows = iter_manifest(path)
    completed = 0
    total = 0
    for row in rows:
        total += 1
        completed += int(bool(str(row.get("human_supported", "")).strip()))
    summary_path = path.with_suffix(path.suffix + ".summary.json")
    scored_summary = None
    if summary_path.is_file():
        scored_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "available": True,
        "path": str(path),
        "fields": fields,
        "rows": total,
        "completed_rows": completed,
        "scored_summary": scored_summary,
    }


def expected_hash(summary: dict[str, object], key: str) -> str | None:
    outputs = summary.get("outputs", {})
    if not isinstance(outputs, dict):
        return None
    value = outputs.get(key, {})
    return str(value.get("sha256")) if isinstance(value, dict) and value.get("sha256") else None


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser()
    mixture_path = resolve(data_root, args.mixture)
    cascade_summary_path = resolve(data_root, args.cascade_summary)
    output_path = resolve(data_root, args.output)
    required_files = {
        "broad": resolve(data_root, args.broad_manifest),
        "compositional": resolve(data_root, args.compositional_manifest),
        "grounded": resolve(data_root, args.grounded_manifest),
        "validation": resolve(data_root, args.validation_manifest),
        "hard_validation": resolve(data_root, args.hard_validation_manifest),
        "candidate": resolve(data_root, args.candidate_manifest),
        "mixture": mixture_path,
        "cascade_summary": cascade_summary_path,
    }
    missing = {name: str(path) for name, path in required_files.items() if not path.is_file()}
    if missing:
        raise FileNotFoundError(f"Missing V12 data files: {missing}")

    prompts = read_prompts(args.leakage_prompts)
    leakage = LeakageIndex(prompts, threshold=args.leakage_threshold) if prompts else None
    manifests: dict[str, dict[str, object]] = {}
    id_sets: dict[str, set[str]] = {}
    grounded_rows: list[dict[str, str]] = []
    for name in ("broad", "compositional", "grounded", "validation", "hard_validation"):
        summary, ids, kept = inspect_manifest(
            required_files[name],
            required_fields=(REQUIRED_GROUNDED_FIELDS if name == "grounded" else REQUIRED_BASE_FIELDS),
            leakage=leakage,
            keep_grounded_rows=name == "grounded",
        )
        manifests[name] = summary
        id_sets[name] = ids
        if kept:
            grounded_rows = kept

    train_names = ("broad", "compositional", "grounded")
    overlaps: dict[str, int] = {}
    for index, left in enumerate(train_names):
        for right in train_names[index + 1 :]:
            overlaps[f"{left}__{right}"] = len(id_sets[left] & id_sets[right])
    train_ids = set().union(*(id_sets[name] for name in train_names))
    overlaps["train__validation"] = len(train_ids & id_sets["validation"])
    overlaps["train__hard_validation"] = len(train_ids & id_sets["hard_validation"])
    overlaps["validation__hard_validation"] = len(
        id_sets["validation"] & id_sets["hard_validation"]
    )

    mixture = json.loads(mixture_path.read_text(encoding="utf-8"))
    trainer = mixture.get("trainer_contract", {})
    probabilities = mixture.get("overall_sampling", {})
    mixture_ok = (
        probabilities == {"broad": 0.7, "compositional": 0.2, "grounded": 0.1}
        and trainer.get("generation_source_names")
        == ["broad", "compositional", "grounded_cascade"]
        and trainer.get("grounded_source_names") == ["grounded_cascade"]
        and math.isclose(float(trainer.get("grounded_batch_probability", -1)), 0.1)
    )

    cascade_summary = json.loads(cascade_summary_path.read_text(encoding="utf-8"))
    hash_checks = {}
    for key, path in (
        ("candidate", required_files["candidate"]),
        ("high_precision", required_files["grounded"]),
        ("mixture", mixture_path),
    ):
        expected = expected_hash(cascade_summary, key)
        actual = file_sha256(path)
        hash_checks[key] = {
            "expected": expected,
            "actual": actual,
            "matched": args.skip_cascade_hash_check or expected is None or expected == actual,
            "skipped": args.skip_cascade_hash_check,
        }

    image_check = image_preflight(
        grounded_rows,
        sample_count=args.decode_samples,
        workers=args.decode_workers,
    )
    rescore_path = Path(args.rescore_manifest).expanduser() if args.rescore_manifest else None
    qwen_audit_path = Path(args.audit_sheet).expanduser() if args.audit_sheet else None
    rescore = compare_rescore(grounded_rows, rescore_path)
    audit = audit_status(qwen_audit_path)

    machine_failures: list[str] = []
    for name, summary in manifests.items():
        for field in ("duplicate_ids", "empty_ids", "empty_captions"):
            if int(summary[field]) > 0:
                machine_failures.append(f"{name}.{field}={summary[field]}")
        if int(summary["leakage_count"]) > 0 and not args.allow_leakage:
            machine_failures.append(f"{name}.leakage_count={summary['leakage_count']}")
    grounded = manifests["grounded"]
    if args.expected_grounded_rows >= 0 and int(grounded["rows"]) != args.expected_grounded_rows:
        machine_failures.append(
            f"grounded.rows={grounded['rows']} expected={args.expected_grounded_rows}"
        )
    for field in ("invalid_grounded_source", "invalid_siglip_status", "missing_image_path"):
        if int(grounded[field]) > 0:
            machine_failures.append(f"grounded.{field}={grounded[field]}")
    for name, count in overlaps.items():
        if count:
            machine_failures.append(f"overlap.{name}={count}")
    if not mixture_ok:
        machine_failures.append("mixture_contract_mismatch")
    for name, check in hash_checks.items():
        if not check["matched"]:
            machine_failures.append(f"hash_mismatch.{name}")
    if image_check["status_counts"].get("missing", 0):
        machine_failures.append("decoded_sample_has_missing_images")
    if image_check["status_counts"].get("unreadable", 0):
        machine_failures.append("decoded_sample_has_unreadable_images")
    if rescore.get("enabled"):
        if rescore.get("status_counts", {}).get("error", 0):
            machine_failures.append("siglip_rescore_has_errors")
        correlation = float(rescore.get("pearson_logit", float("nan")))
        if int(rescore.get("rows_compared", 0)) >= 32 and (
            not math.isfinite(correlation) or correlation < 0.98
        ):
            machine_failures.append(f"siglip_rescore_correlation={correlation}")

    report = {
        "dataset": "ImageBridge-Data-v1",
        "target_experiment": "DreamLite-Image-Bridge-V12",
        "machine_preflight_passed": not machine_failures,
        "machine_failures": machine_failures,
        "human_audit_required": True,
        "human_audit": audit,
        "manifests": manifests,
        "overlaps": overlaps,
        "mixture_contract": {"passed": mixture_ok, "payload": mixture},
        "hash_checks": hash_checks,
        "image_decode_check": image_check,
        "siglip_rescore": rescore,
        "leakage_prompts": args.leakage_prompts,
        "leakage_threshold": args.leakage_threshold,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if machine_failures:
        raise RuntimeError(f"V12 data preflight failed: {machine_failures}")


if __name__ == "__main__":
    main()
