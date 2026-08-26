from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from new_mobile_ov.training.image_bridge_data_v1 import normalized_caption_key


FINAL_BUCKET_WEIGHTS = {
    "broad": 0.60,
    "attribute_binding": 0.05,
    "spatial_relation": 0.08,
    "count": 0.05,
    "multiple_objects": 0.06,
    "action": 0.05,
    "scene": 0.05,
    "color_attribute": 0.03,
    "style": 0.03,
}

QWEN_BUCKET_WEIGHTS = {
    "broad": 0.10,
    "attribute_binding": 0.12,
    "spatial_relation": 0.16,
    "count": 0.12,
    "multiple_objects": 0.12,
    "action": 0.12,
    "scene": 0.12,
    "color_attribute": 0.07,
    "style": 0.07,
}

BUCKET_PRIORITY = (
    "attribute_binding",
    "spatial_relation",
    "count",
    "multiple_objects",
    "action",
    "scene",
    "color_attribute",
    "style",
)


def parse_capabilities(value: object) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return {item for item in str(value or "").split(";") if item}


def capability_bucket(row: Mapping[str, object]) -> str:
    capabilities = parse_capabilities(row.get("capabilities"))
    for capability in BUCKET_PRIORITY:
        if capability == "color_attribute":
            if capabilities & {"color", "attribute"}:
                return capability
        elif capability in capabilities:
            return capability
    return "broad"


def alignment_score(row: Mapping[str, object]) -> float:
    try:
        value = float(row.get("siglip_logit") or row.get("siglip_score", "nan"))
    except (TypeError, ValueError):
        return float("-inf")
    return value if math.isfinite(value) else float("-inf")


def stable_row_key(row: Mapping[str, object]) -> str:
    record_id = str(row.get("record_id", "")).strip()
    return hashlib.sha256(record_id.encode("utf-8")).hexdigest()


def qwen_decisions(path: str | Path | None) -> dict[str, str]:
    """Return the latest valid state for each Qwen-audited record."""
    if not path:
        return {}
    source = Path(path)
    if not source.is_file():
        return {}
    decisions: dict[str, str] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_id = str(record.get("record_id", "")).strip()
            if not record_id:
                continue
            if record.get("error"):
                decisions[record_id] = "error"
            elif bool(record.get("accepted")):
                decisions[record_id] = "accepted"
            else:
                decisions[record_id] = "rejected"
    return decisions


def read_manifest(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {source}")
        return list(reader.fieldnames), list(reader)


def write_manifest(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
    *,
    preferred_fields: Sequence[str] = (),
) -> list[str]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(preferred_fields))
    extras = sorted({str(key) for row in rows for key in row if key not in fields})
    fields.extend(extras)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return fields


def _tokens(caption: str) -> tuple[str, ...]:
    return tuple(token for token in normalized_caption_key(caption).split() if token)


class LeakageIndex:
    """Small inverted index for exact and high-overlap benchmark prompts."""

    def __init__(self, prompts: Iterable[str], *, threshold: float = 0.85) -> None:
        self.threshold = float(threshold)
        self.exact: set[str] = set()
        self.prompt_tokens: list[frozenset[str]] = []
        self.trigram_index: dict[tuple[str, str, str], set[int]] = defaultdict(set)
        for prompt in prompts:
            normalized = normalized_caption_key(prompt)
            if not normalized or normalized in self.exact:
                continue
            self.exact.add(normalized)
            tokens = _tokens(prompt)
            index = len(self.prompt_tokens)
            self.prompt_tokens.append(frozenset(tokens))
            for offset in range(max(0, len(tokens) - 2)):
                self.trigram_index[tokens[offset : offset + 3]].add(index)

    def matches(self, caption: str) -> bool:
        normalized = normalized_caption_key(caption)
        if normalized in self.exact:
            return True
        tokens = _tokens(caption)
        if len(tokens) < 3:
            return False
        candidate_indices: set[int] = set()
        for offset in range(len(tokens) - 2):
            candidate_indices.update(self.trigram_index.get(tokens[offset : offset + 3], ()))
        token_set = frozenset(tokens)
        for index in candidate_indices:
            prompt = self.prompt_tokens[index]
            union = len(token_set | prompt)
            if union and len(token_set & prompt) / union >= self.threshold:
                return True
        return False


def read_prompts(paths: Sequence[str | Path]) -> list[str]:
    prompts: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = payload if isinstance(payload, list) else payload.get("records", [])
            for row in values:
                if isinstance(row, str):
                    prompts.append(row)
                elif isinstance(row, Mapping):
                    prompts.append(str(row.get("prompt_en") or row.get("prompt") or ""))
        elif path.suffix.lower() in {".csv", ".tsv"}:
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter=delimiter):
                    prompts.append(str(row.get("prompt_en") or row.get("prompt") or ""))
        else:
            prompts.extend(path.read_text(encoding="utf-8").splitlines())
    return [prompt.strip() for prompt in prompts if prompt.strip()]


def _quota_counts(target: int, weights: Mapping[str, float]) -> dict[str, int]:
    if target < 0:
        raise ValueError("target must be non-negative")
    total_weight = sum(weights.values())
    if not math.isclose(total_weight, 1.0, abs_tol=1e-6):
        raise ValueError(f"bucket weights must sum to one, got {total_weight}")
    raw = {bucket: target * weight for bucket, weight in weights.items()}
    quotas = {bucket: int(value) for bucket, value in raw.items()}
    remainder = target - sum(quotas.values())
    order = sorted(weights, key=lambda bucket: (raw[bucket] % 1, bucket), reverse=True)
    for bucket in order[:remainder]:
        quotas[bucket] += 1
    return quotas


def balanced_select(
    rows: Sequence[Mapping[str, object]],
    *,
    target: int,
    weights: Mapping[str, float] = FINAL_BUCKET_WEIGHTS,
    preferred_ids: set[str] | None = None,
) -> list[dict[str, object]]:
    """Select a deterministic score-ranked set while preserving capability quotas."""
    preferred_ids = preferred_ids or set()
    deduplicated: dict[str, dict[str, object]] = {}
    used_images: set[str] = set()
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            str(row.get("record_id", "")) not in preferred_ids,
            -alignment_score(row),
            stable_row_key(row),
        ),
    )
    for row in ordered:
        record_id = str(row.get("record_id", "")).strip()
        image_path = str(row.get("image_path", "")).strip()
        if not record_id or record_id in deduplicated:
            continue
        if image_path and image_path in used_images:
            continue
        row["capability_bucket"] = capability_bucket(row)
        deduplicated[record_id] = row
        if image_path:
            used_images.add(image_path)

    quotas = _quota_counts(target, weights)
    by_bucket: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in deduplicated.values():
        by_bucket[str(row["capability_bucket"])].append(row)
    for bucket_rows in by_bucket.values():
        bucket_rows.sort(
            key=lambda row: (
                str(row.get("record_id", "")) not in preferred_ids,
                -alignment_score(row),
                stable_row_key(row),
            )
        )

    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    for bucket, quota in quotas.items():
        for row in by_bucket.get(bucket, ())[:quota]:
            selected.append(row)
            selected_ids.add(str(row["record_id"]))
    if len(selected) < target:
        overflow = sorted(
            (
                row
                for row in deduplicated.values()
                if str(row["record_id"]) not in selected_ids
            ),
            key=lambda row: (
                str(row.get("record_id", "")) not in preferred_ids,
                -alignment_score(row),
                stable_row_key(row),
            ),
        )
        selected.extend(overflow[: target - len(selected)])
    return selected[:target]


def selection_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    buckets = Counter(capability_bucket(row) for row in rows)
    capabilities: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    for row in rows:
        capabilities.update(parse_capabilities(row.get("capabilities")))
        sources[str(row.get("verification_source", "unspecified"))] += 1
    scores = sorted(
        score for row in rows if math.isfinite(score := alignment_score(row))
    )
    score_summary: dict[str, float | int | None] = {"count": len(scores)}
    for name, fraction in (("min", 0.0), ("p10", 0.1), ("median", 0.5), ("p90", 0.9), ("max", 1.0)):
        score_summary[name] = (
            scores[min(int(fraction * max(len(scores) - 1, 0)), len(scores) - 1)]
            if scores
            else None
        )
    return {
        "count": len(rows),
        "buckets": dict(sorted(buckets.items())),
        "capabilities": dict(sorted(capabilities.items())),
        "verification_sources": dict(sorted(sources.items())),
        "siglip_ranking_values": score_summary,
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
