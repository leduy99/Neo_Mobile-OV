#!/usr/bin/env python
"""Machine-check a frozen MobileOV-Data-v1 release before any training run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import sqlite3
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

from new_mobile_ov.training.mobileov_data_release import CANONICAL_FIELDS, file_sha256


def sample_key(record_id: str) -> int:
    return int(hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:16], 16)


def add_sample(heap: list[tuple[int, dict[str, str]]], row: dict[str, str], limit: int) -> None:
    key = sample_key(row["record_id"])
    item = (-key, row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def validate_image(path: Path) -> str:
    try:
        with Image.open(path) as image:
            image.verify()
        return "ok"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def validate_latent(path: Path, *, require_pair_contract: bool) -> str:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        latent = payload.get("latent", payload) if isinstance(payload, dict) else payload
        if not isinstance(latent, torch.Tensor) or latent.ndim != 4:
            return f"invalid latent shape/type: {type(latent).__name__}"
        if latent.shape[1] != 7:
            return f"expected seven temporal units, got {tuple(latent.shape)}"
        if require_pair_contract and payload.get("pair_contract") != "same_dreamlite_anchor_native_teacher":
            return "missing exact anchor/teacher pair contract"
        if require_pair_contract:
            anchor_path = path.parent.parent / str(payload.get("anchor_image_path", ""))
            if not anchor_path.is_file():
                return f"missing paired anchor: {anchor_path}"
            expected = str(payload.get("anchor_sha256", ""))
            if not expected or file_sha256(anchor_path) != expected:
                return f"anchor hash mismatch: {anchor_path}"
        return "ok"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", default="data/mobileov_data_v1/releases/v1")
    parser.add_argument("--output", default="")
    parser.add_argument("--sample-checks-per-pool", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.release_root)
    summary_path = root / "stats/release_summary.json"
    contract_path = root / "release.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if contract.get("release_summary_sha256") != file_sha256(summary_path):
        failures.append("release summary hash mismatch")
    for source in summary.get("sources", []):
        path = Path(str(source.get("path", "")))
        expected_hash = source.get("sha256")
        if not path.is_file():
            failures.append(f"release source disappeared: {path}")
        elif expected_hash and file_sha256(path) != expected_hash:
            failures.append(f"release source hash mismatch: {path}")

    database = root / ".preflight.sqlite3"
    database.unlink(missing_ok=True)
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE records (record_id TEXT PRIMARY KEY)")
    connection.execute(
        "CREATE TABLE families (family_id TEXT PRIMARY KEY, split TEXT NOT NULL)"
    )
    pool_counts: Counter[str] = Counter()
    samples: dict[str, list[tuple[int, dict[str, str]]]] = {}
    try:
        for name, expected in summary.get("manifests", {}).items():
            path = root / "manifests" / f"{name}.csv"
            if not path.is_file():
                failures.append(f"missing manifest: {path}")
                continue
            if file_sha256(path) != expected.get("sha256"):
                failures.append(f"manifest hash mismatch: {path}")
                continue
            rows = 0
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                missing_fields = set(CANONICAL_FIELDS) - set(reader.fieldnames or ())
                if missing_fields:
                    failures.append(f"{path}: missing fields {sorted(missing_fields)}")
                    continue
                for row in reader:
                    rows += 1
                    pool = row["pool"]
                    pool_counts[pool] += 1
                    try:
                        connection.execute("INSERT INTO records VALUES (?)", (row["record_id"],))
                    except sqlite3.IntegrityError:
                        failures.append(f"duplicate record_id: {row['record_id']}")
                    prior = connection.execute(
                        "SELECT split FROM families WHERE family_id = ?", (row["family_id"],)
                    ).fetchone()
                    if prior and prior[0] != row["split"]:
                        failures.append(
                            f"family split leakage: {row['family_id']} in {prior[0]} and {row['split']}"
                        )
                    elif not prior:
                        connection.execute(
                            "INSERT INTO families VALUES (?, ?)",
                            (row["family_id"], row["split"]),
                        )
                    heap = samples.setdefault(pool, [])
                    add_sample(heap, row, args.sample_checks_per_pool)
            if rows != int(expected.get("rows", -1)):
                failures.append(f"{path}: rows={rows}, expected={expected.get('rows')}")
            connection.commit()
    finally:
        connection.close()
        database.unlink(missing_ok=True)

    required_pools = {
        "image_broad",
        "image_compositional",
        "image_grounded",
        "video_real",
        "video_teacher_t2v",
        "video_anchor_teacher",
    }
    for pool in required_pools:
        if pool_counts[pool] == 0:
            failures.append(f"required pool is empty: {pool}")

    checks: dict[str, Counter[str]] = {}
    examples: list[dict[str, str]] = []
    for pool, heap in samples.items():
        pool_checks: Counter[str] = Counter()
        checks[pool] = pool_checks
        for _, row in heap:
            if pool == "image_grounded":
                status = validate_image(Path(row["image_path"]))
            elif pool in {"video_real", "video_teacher_t2v", "video_anchor_teacher"}:
                status = validate_latent(
                    Path(row["latent_path"]),
                    require_pair_contract=pool == "video_anchor_teacher",
                )
            else:
                status = "ok"
            pool_checks["ok" if status == "ok" else "error"] += 1
            if status != "ok" and len(examples) < 50:
                examples.append(
                    {"pool": pool, "record_id": row["record_id"], "error": status}
                )
    for pool, values in checks.items():
        if values["error"]:
            failures.append(f"{pool}: sampled artifact errors={values['error']}")

    report = {
        "dataset": summary.get("dataset"),
        "release_root": str(root),
        "passed": not failures,
        "failures": failures,
        "pool_counts": dict(pool_counts),
        "sample_checks": {pool: dict(value) for pool, value in checks.items()},
        "failure_examples": examples,
        "benchmark_prompt_count": summary.get("benchmark_prompt_count"),
        "leakage_threshold": summary.get("leakage_threshold"),
    }
    output = Path(args.output) if args.output else root / "stats/preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if failures:
        raise RuntimeError(f"MobileOV-Data-v1 preflight failed: {failures[:10]}")


if __name__ == "__main__":
    main()
