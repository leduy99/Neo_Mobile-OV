from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from new_mobile_ov.training.image_bridge_data_v1 import (
    compositional_score,
    infer_capabilities,
    iter_manifest_rows,
    normalize_caption,
    normalized_caption_key,
)
from new_mobile_ov.training.image_bridge_grounding_cascade import LeakageIndex
from new_mobile_ov.training.mobileov_data_release import file_sha256, stable_hash


PROMPT_FIELDS = (
    "prompt_id",
    "prompt",
    "caption_short",
    "caption_medium",
    "caption_long",
    "capabilities",
    "prompt_bucket",
    "source_name",
    "source_key",
    "source_manifest",
    "source_row",
    "selection_key",
)


@dataclass(frozen=True)
class PromptSource:
    name: str
    path: Path
    weight: float
    hard_fraction: float = 0.5
    require_any_capabilities: tuple[str, ...] = ()
    text_columns: tuple[str, ...] = (
        "caption_long",
        "caption_medium",
        "prompt",
        "caption",
        "text",
        "caption_short",
    )

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("Prompt source weights must be non-negative")
        if not 0 <= self.hard_fraction <= 1:
            raise ValueError("hard_fraction must be in [0, 1]")


def _first(row: Mapping[str, object], columns: Sequence[str]) -> str:
    for column in columns:
        value = normalize_caption(row.get(column))
        if value:
            return value
    return ""


def _source_key(row: Mapping[str, object], row_number: int) -> str:
    return _first(
        row,
        ("source_key", "source_id", "record_id", "sample_id", "index", "id"),
    ) or str(row_number)


def _bucket(capabilities: Sequence[str]) -> str:
    values = set(capabilities)
    for name in (
        "attribute_binding",
        "spatial_relation",
        "count",
        "multiple_objects",
        "action",
        "scene",
        "style",
    ):
        if name in values:
            return name
    return "general"


def _quota_counts(target: int, sources: Sequence[PromptSource]) -> dict[str, int]:
    total_weight = sum(source.weight for source in sources)
    if total_weight <= 0:
        raise ValueError("At least one prompt source must have positive weight")
    raw = {source.name: target * source.weight / total_weight for source in sources}
    quotas = {name: int(value) for name, value in raw.items()}
    remainder = target - sum(quotas.values())
    order = sorted(raw, key=lambda name: (raw[name] - quotas[name], name), reverse=True)
    for name in order[:remainder]:
        quotas[name] += 1
    return quotas


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        """
        CREATE TABLE candidates (
            source_name TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            selection_key TEXT NOT NULL,
            prompt TEXT NOT NULL,
            caption_short TEXT NOT NULL,
            caption_medium TEXT NOT NULL,
            caption_long TEXT NOT NULL,
            capabilities TEXT NOT NULL,
            prompt_bucket TEXT NOT NULL,
            source_key TEXT NOT NULL,
            source_manifest TEXT NOT NULL,
            source_row INTEGER NOT NULL,
            hard INTEGER NOT NULL,
            PRIMARY KEY (source_name, prompt_hash)
        )
        """
    )
    connection.execute(
        "CREATE INDEX candidates_select_idx ON candidates(source_name, hard, selection_key)"
    )
    connection.execute(
        "CREATE INDEX candidates_source_key_idx ON candidates(source_name, selection_key)"
    )
    connection.execute("CREATE INDEX candidates_key_idx ON candidates(selection_key)")
    return connection


def _select_rows(
    connection: sqlite3.Connection,
    *,
    source: PromptSource,
    target: int,
    already_selected: set[str],
) -> list[tuple[str, ...]]:
    fields = (
        "prompt_hash",
        "selection_key",
        "prompt",
        "caption_short",
        "caption_medium",
        "caption_long",
        "capabilities",
        "prompt_bucket",
        "source_name",
        "source_key",
        "source_manifest",
        "source_row",
    )
    selected: list[tuple[str, ...]] = []
    hard_target = round(target * source.hard_fraction)

    def add(where: str, limit: int) -> None:
        if limit <= 0:
            return
        query = (
            f"SELECT {', '.join(fields)} FROM candidates "
            f"WHERE source_name = ? AND {where} ORDER BY selection_key"
        )
        for row in connection.execute(query, (source.name,)):
            prompt_hash = str(row[0])
            if prompt_hash in already_selected:
                continue
            selected.append(tuple(str(value) for value in row))
            already_selected.add(prompt_hash)
            if len(selected) >= limit:
                return

    add("hard = 1", hard_target)
    add("1 = 1", target)
    return selected[:target]


def build_anchor_prompt_bank(
    sources: Sequence[PromptSource],
    *,
    output_path: Path,
    target_records: int,
    seed: int,
    benchmark_prompts: Sequence[str] = (),
    leakage_threshold: float = 0.85,
    min_words: int = 4,
    max_words: int = 80,
    force: bool = False,
    progress_every: int = 100_000,
) -> dict[str, object]:
    if target_records <= 0:
        raise ValueError("target_records must be positive")
    if not sources:
        raise ValueError("At least one prompt source is required")
    if output_path.exists() and not force:
        raise FileExistsError(f"Prompt bank already exists: {output_path}")
    for source in sources:
        if not source.path.is_file():
            raise FileNotFoundError(source.path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output_path.parent / f".{output_path.stem}.work"
    work_dir.mkdir(parents=True, exist_ok=True)
    database = work_dir / "candidates.sqlite3"
    database.unlink(missing_ok=True)
    connection = _connect(database)
    leakage = LeakageIndex(benchmark_prompts, threshold=leakage_threshold)
    stats: dict[str, Counter[str]] = {source.name: Counter() for source in sources}
    try:
        for source in sources:
            required = set(source.require_any_capabilities)
            for row_number, row in enumerate(iter_manifest_rows(source.path), start=1):
                stats[source.name]["rows"] += 1
                prompt = _first(row, source.text_columns)
                if not prompt:
                    stats[source.name]["empty"] += 1
                    continue
                word_count = len(prompt.split())
                if word_count < min_words or word_count > max_words:
                    stats[source.name]["length_filtered"] += 1
                    continue
                if str(row.get("caption_status", "")).strip().lower() == "failed":
                    stats[source.name]["caption_failed"] += 1
                    continue
                if leakage.matches(prompt):
                    stats[source.name]["benchmark_leakage"] += 1
                    continue
                capabilities = tuple(infer_capabilities(prompt))
                if required and not required.intersection(capabilities):
                    stats[source.name]["capability_filtered"] += 1
                    continue
                prompt_hash = stable_hash(normalized_caption_key(prompt))
                selection_key = stable_hash(f"{seed}:{source.name}:{prompt_hash}")
                caption_short = _first(row, ("caption_short",)) or prompt
                caption_medium = _first(row, ("caption_medium", "caption")) or prompt
                caption_long = _first(row, ("caption_long", "caption", "prompt")) or prompt
                hard = int(compositional_score(capabilities) >= 3)
                try:
                    connection.execute(
                        "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            source.name,
                            prompt_hash,
                            selection_key,
                            prompt,
                            caption_short,
                            caption_medium,
                            caption_long,
                            ";".join(capabilities),
                            _bucket(capabilities),
                            _source_key(row, row_number),
                            str(source.path),
                            row_number,
                            hard,
                        ),
                    )
                    stats[source.name]["eligible"] += 1
                except sqlite3.IntegrityError:
                    stats[source.name]["duplicate"] += 1
                if row_number % 10_000 == 0:
                    connection.commit()
                if progress_every > 0 and row_number % progress_every == 0:
                    print(
                        f"source={source.name} rows={row_number} "
                        f"eligible={stats[source.name]['eligible']} "
                        f"leakage={stats[source.name]['benchmark_leakage']}",
                        flush=True,
                    )
            connection.commit()

        quotas = _quota_counts(target_records, sources)
        selected_hashes: set[str] = set()
        selected: list[tuple[str, ...]] = []
        for source in sources:
            selected.extend(
                _select_rows(
                    connection,
                    source=source,
                    target=quotas[source.name],
                    already_selected=selected_hashes,
                )
            )

        if len(selected) < target_records:
            query = (
                "SELECT prompt_hash, selection_key, prompt, caption_short, caption_medium, "
                "caption_long, capabilities, prompt_bucket, source_name, source_key, "
                "source_manifest, source_row FROM candidates ORDER BY selection_key"
            )
            for row in connection.execute(query):
                prompt_hash = str(row[0])
                if prompt_hash in selected_hashes:
                    continue
                selected.append(tuple(str(value) for value in row))
                selected_hashes.add(prompt_hash)
                if len(selected) >= target_records:
                    break
        if len(selected) != target_records:
            raise RuntimeError(
                f"Only {len(selected)} unique eligible prompts were available for target={target_records}"
            )

        selected.sort(key=lambda row: row[1])
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        source_selected: Counter[str] = Counter()
        bucket_selected: Counter[str] = Counter()
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PROMPT_FIELDS)
            writer.writeheader()
            for prompt_hash, selection_key, prompt, short, medium, long, caps, bucket, source_name, source_key, source_manifest, source_row in selected:
                writer.writerow(
                    {
                        "prompt_id": prompt_hash[:24],
                        "prompt": prompt,
                        "caption_short": short,
                        "caption_medium": medium,
                        "caption_long": long,
                        "capabilities": caps,
                        "prompt_bucket": bucket,
                        "source_name": source_name,
                        "source_key": source_key,
                        "source_manifest": source_manifest,
                        "source_row": source_row,
                        "selection_key": selection_key,
                    }
                )
                source_selected[source_name] += 1
                bucket_selected[bucket] += 1
        temporary.replace(output_path)
    finally:
        connection.close()
        database.unlink(missing_ok=True)
        try:
            work_dir.rmdir()
        except OSError:
            pass

    summary = {
        "name": "mobileov_anchor_teacher_prompt_bank_v1",
        "target_records": target_records,
        "seed": seed,
        "leakage_threshold": leakage_threshold,
        "source_quotas": quotas,
        "source_selected": dict(source_selected),
        "bucket_selected": dict(bucket_selected),
        "source_stats": {name: dict(value) for name, value in stats.items()},
        "sources": [
            {
                "name": source.name,
                "path": str(source.path),
                "sha256": file_sha256(source.path),
                "weight": source.weight,
                "hard_fraction": source.hard_fraction,
                "require_any_capabilities": list(source.require_any_capabilities),
                "text_columns": list(source.text_columns),
            }
            for source in sources
        ],
        "output": str(output_path),
        "output_sha256": file_sha256(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
