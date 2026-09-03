from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from new_mobile_ov.training.image_bridge_data_v1 import (
    infer_capabilities,
    iter_manifest_rows,
    normalize_caption,
    normalized_caption_key,
)
from new_mobile_ov.training.image_bridge_grounding_cascade import LeakageIndex


DATASET_NAME = "MobileOV-Data-v1"
VALID_POOLS = {
    "image_broad",
    "image_compositional",
    "image_grounded",
    "video_real",
    "video_teacher_t2v",
    "video_anchor_teacher",
}
POOL_TASKS = {
    "image_broad": "image_condition_distill",
    "image_compositional": "image_condition_distill",
    "image_grounded": "image_condition_distill",
    "video_real": "video_flow_matching",
    "video_teacher_t2v": "video_teacher_distill",
    "video_anchor_teacher": "video_anchor_conditioned_teacher_distill",
}
POOL_MODALITIES = {
    "image_broad": "text",
    "image_compositional": "text",
    "image_grounded": "image_text",
    "video_real": "video_text",
    "video_teacher_t2v": "latent_text",
    "video_anchor_teacher": "image_latent_text",
}

CAPTION_COLUMNS = (
    "prompt",
    "caption_long",
    "caption_medium",
    "caption",
    "text",
    "caption_short",
)
SOURCE_KEY_COLUMNS = (
    "source_key",
    "source_id",
    "record_id",
    "sample_id",
    "index",
    "id",
    "recaption_row_id",
)
CANONICAL_FIELDS = (
    "record_id",
    "split",
    "pool",
    "task",
    "modality",
    "prompt",
    "caption_short",
    "caption_medium",
    "caption_long",
    "capabilities",
    "source_name",
    "source_key",
    "source_manifest",
    "source_row",
    "image_path",
    "video_path",
    "anchor_image_path",
    "latent_path",
    "condition_prompt",
    "width",
    "height",
    "num_frames",
    "fps",
    "seed",
    "verification_status",
    "verification_source",
    "alignment_score",
    "family_id",
)


@dataclass(frozen=True)
class ReleaseSource:
    name: str
    pool: str
    path: Path
    max_records: int = -1
    fixed_split: str = ""
    require_artifact: bool = True

    def __post_init__(self) -> None:
        if self.pool not in VALID_POOLS:
            raise ValueError(f"Unsupported pool {self.pool!r}; expected one of {sorted(VALID_POOLS)}")
        if self.fixed_split not in {"", "train", "validation"}:
            raise ValueError("fixed_split must be empty, 'train', or 'validation'")


@dataclass(frozen=True)
class ReleaseBuildConfig:
    output_dir: Path
    seed: int = 20260904
    validation_fraction: float = 0.002
    leakage_threshold: float = 0.85
    verify_paths: bool = True
    source_hashes: bool = True
    force: bool = False


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def first_value(row: Mapping[str, object], columns: Sequence[str]) -> str:
    for column in columns:
        value = normalize_caption(row.get(column))
        if value:
            return value
    return ""


def prompt_from_row(row: Mapping[str, object]) -> str:
    return first_value(row, CAPTION_COLUMNS)


def source_key_from_row(row: Mapping[str, object], row_number: int) -> str:
    return first_value(row, SOURCE_KEY_COLUMNS) or str(row_number)


def resolve_artifact(value: str, source_manifest: Path) -> tuple[str, bool]:
    value = str(value or "").strip()
    if not value:
        return "", False
    raw = Path(os.path.expandvars(os.path.expanduser(value)))
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, source_manifest.parent / raw]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve()), True
    return str(raw), False


def artifact_values(
    row: Mapping[str, object], source_manifest: Path
) -> tuple[dict[str, str], dict[str, bool]]:
    aliases = {
        "image_path": ("image_path", "image", "jpg", "png"),
        "video_path": ("video_path", "media_path", "video", "path", "mp4"),
        "anchor_image_path": (
            "anchor_image_path",
            "anchor_path",
            "first_frame_path",
            "first_frame",
        ),
        "latent_path": (
            "latent_path",
            "teacher_trajectory_path",
            "trajectory_path",
            "latents",
        ),
    }
    values: dict[str, str] = {}
    exists: dict[str, bool] = {}
    for field, columns in aliases.items():
        values[field], exists[field] = resolve_artifact(
            first_value(row, columns), source_manifest
        )
    return values, exists


def required_artifacts(pool: str) -> tuple[tuple[str, ...], ...]:
    if pool == "image_grounded":
        return (("image_path",),)
    if pool == "video_real":
        return (("latent_path", "video_path"),)
    if pool == "video_teacher_t2v":
        return (("latent_path",),)
    if pool == "video_anchor_teacher":
        return (("latent_path",), ("anchor_image_path",))
    return ()


def artifact_requirement_met(
    requirements: Iterable[tuple[str, ...]], exists: Mapping[str, bool]
) -> bool:
    return all(any(exists.get(field, False) for field in alternatives) for alternatives in requirements)


def split_for_prompt(prompt: str, *, seed: int, validation_fraction: float) -> tuple[str, str]:
    family_id = stable_hash(normalized_caption_key(prompt))[:24]
    value = int(stable_hash(f"{seed}:{family_id}")[:12], 16) / float(16**12)
    split = "validation" if value < validation_fraction else "train"
    return split, family_id


def canonicalize_record(
    source: ReleaseSource,
    row: Mapping[str, object],
    row_number: int,
    *,
    seed: int,
    validation_fraction: float,
    verify_paths: bool = True,
) -> tuple[dict[str, str] | None, str, dict[str, bool]]:
    prompt = prompt_from_row(row)
    if not prompt:
        return None, "empty_prompt", {}
    artifacts, exists = artifact_values(row, source.path)
    if verify_paths and source.require_artifact and not artifact_requirement_met(
        required_artifacts(source.pool), exists
    ):
        return None, "missing_artifact", exists

    split, family_id = split_for_prompt(
        prompt, seed=seed, validation_fraction=validation_fraction
    )
    if source.fixed_split:
        split = source.fixed_split
    source_key = source_key_from_row(row, row_number)
    primary_path = next(
        (
            artifacts[field]
            for field in ("latent_path", "video_path", "image_path", "anchor_image_path")
            if artifacts[field]
        ),
        "",
    )
    duplicate_key = stable_hash(
        "|".join((source.pool, normalized_caption_key(prompt), primary_path))
    )
    record_id = stable_hash(f"{source.name}|{source_key}|{duplicate_key}")[:24]
    supplied_capabilities = first_value(row, ("capabilities",))
    capabilities = supplied_capabilities or ";".join(infer_capabilities(prompt))
    caption_short = first_value(row, ("caption_short",)) or prompt
    caption_medium = first_value(row, ("caption_medium", "caption")) or prompt
    caption_long = first_value(row, ("caption_long", "caption", "prompt")) or prompt
    verification_status = first_value(
        row,
        (
            "verification_status",
            "grounding_status",
            "caption_status",
            "qwen_status",
            "siglip_status",
        ),
    )
    verification_source = first_value(
        row, ("verification_source", "grounding_source", "caption_source")
    )
    alignment_score = first_value(
        row, ("alignment_score", "siglip_logit", "siglip_score")
    )
    record = {
        "record_id": record_id,
        "split": split,
        "pool": source.pool,
        "task": POOL_TASKS[source.pool],
        "modality": POOL_MODALITIES[source.pool],
        "prompt": prompt,
        "caption_short": caption_short,
        "caption_medium": caption_medium,
        "caption_long": caption_long,
        "capabilities": capabilities,
        "source_name": source.name,
        "source_key": source_key,
        "source_manifest": str(source.path),
        "source_row": str(row_number),
        **artifacts,
        "condition_prompt": first_value(row, ("condition_prompt",)),
        "width": first_value(row, ("width", "source_width")),
        "height": first_value(row, ("height", "source_height")),
        "num_frames": first_value(row, ("num_frames", "clip_num_frames")),
        "fps": first_value(row, ("fps", "clip_fps", "target_fps")),
        "seed": first_value(row, ("seed",)),
        "verification_status": verification_status,
        "verification_source": verification_source,
        "alignment_score": alignment_score,
        "family_id": family_id,
    }
    return record, duplicate_key, exists


def _connect_seen(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("CREATE TABLE seen (duplicate_key TEXT PRIMARY KEY)")
    return connection


def _read_benchmark_prompts(paths: Sequence[Path]) -> list[str]:
    prompts: list[str] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing benchmark prompt source: {path}")
        if path.suffix.lower() in {".csv", ".tsv"}:
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter=delimiter):
                    prompt = first_value(row, ("prompt_en", "prompt", "caption", "text"))
                    if prompt:
                        prompts.append(prompt)
        else:
            prompts.extend(
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    return prompts


def _source_contract(source: ReleaseSource, *, include_hash: bool) -> dict[str, object]:
    stat = source.path.stat()
    return {
        **asdict(source),
        "path": str(source.path),
        "resolved_path": str(source.path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": file_sha256(source.path) if include_hash else None,
    }


def build_mobileov_data_release(
    sources: Sequence[ReleaseSource],
    config: ReleaseBuildConfig,
    *,
    benchmark_prompt_paths: Sequence[Path] = (),
    progress_every: int = 100_000,
) -> dict[str, object]:
    if not sources:
        raise ValueError("At least one release source is required")
    if len({source.name for source in sources}) != len(sources):
        raise ValueError("Release source names must be unique")
    if not 0 <= config.validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    for source in sources:
        if not source.path.is_file():
            raise FileNotFoundError(f"Missing release source: {source.path}")

    output_dir = config.output_dir.expanduser()
    partial = output_dir.with_name(f"{output_dir.name}.partial")
    if output_dir.exists():
        if not config.force:
            raise FileExistsError(f"Release already exists: {output_dir}; use --force to rebuild")
        shutil.rmtree(output_dir)
    if partial.exists():
        shutil.rmtree(partial)
    manifests_dir = partial / "manifests"
    stats_dir = partial / "stats"
    manifests_dir.mkdir(parents=True)
    stats_dir.mkdir()

    benchmark_prompts = _read_benchmark_prompts(benchmark_prompt_paths)
    leakage = LeakageIndex(benchmark_prompts, threshold=config.leakage_threshold)
    handles: dict[tuple[str, str], object] = {}
    writers: dict[tuple[str, str], csv.DictWriter] = {}
    for pool in sorted(VALID_POOLS):
        for split in ("train", "validation"):
            path = manifests_dir / f"{pool}_{split}.csv"
            handle = path.open("w", encoding="utf-8", newline="")
            writer = csv.DictWriter(handle, fieldnames=CANONICAL_FIELDS)
            writer.writeheader()
            handles[(pool, split)] = handle
            writers[(pool, split)] = writer

    counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    capability_counts: dict[str, Counter[str]] = defaultdict(Counter)
    seen = _connect_seen(partial / "seen.sqlite3")
    try:
        for source in sources:
            accepted_for_source = 0
            for row_number, row in enumerate(iter_manifest_rows(source.path), start=1):
                source_counts[source.name]["rows"] += 1
                record, duplicate_key, _ = canonicalize_record(
                    source,
                    row,
                    row_number,
                    seed=config.seed,
                    validation_fraction=config.validation_fraction,
                    verify_paths=config.verify_paths,
                )
                if record is None:
                    source_counts[source.name][duplicate_key] += 1
                    continue
                if leakage.matches(record["prompt"]):
                    source_counts[source.name]["benchmark_leakage"] += 1
                    continue
                try:
                    seen.execute("INSERT INTO seen VALUES (?)", (duplicate_key,))
                except sqlite3.IntegrityError:
                    source_counts[source.name]["duplicate"] += 1
                    continue
                writers[(source.pool, record["split"])].writerow(record)
                accepted_for_source += 1
                source_counts[source.name]["accepted"] += 1
                counts[f"{source.pool}_{record['split']}"] += 1
                for capability in record["capabilities"].split(";"):
                    if capability:
                        capability_counts[source.pool][capability] += 1
                if accepted_for_source % 10_000 == 0:
                    seen.commit()
                if progress_every > 0 and row_number % progress_every == 0:
                    print(
                        f"source={source.name} rows={row_number} accepted={accepted_for_source} "
                        f"missing={source_counts[source.name]['missing_artifact']} "
                        f"duplicates={source_counts[source.name]['duplicate']} "
                        f"leakage={source_counts[source.name]['benchmark_leakage']}",
                        flush=True,
                    )
                if source.max_records > 0 and accepted_for_source >= source.max_records:
                    break
            seen.commit()
    finally:
        seen.close()
        for handle in handles.values():
            handle.close()

    manifest_contract: dict[str, dict[str, object]] = {}
    for path in sorted(manifests_dir.glob("*.csv")):
        key = path.stem
        manifest_contract[key] = {
            "path": str(output_dir / "manifests" / path.name),
            "rows": int(counts.get(key, 0)),
            "sha256": file_sha256(path),
        }
    source_contract = [
        _source_contract(source, include_hash=config.source_hashes) for source in sources
    ]
    benchmark_contract = [
        {"path": str(path), "sha256": file_sha256(path)}
        for path in benchmark_prompt_paths
    ]
    summary = {
        "dataset": DATASET_NAME,
        "release_contract": (
            "immutable benchmark-clean pool manifests; sampling recipes remain separate"
        ),
        "seed": config.seed,
        "validation_fraction": config.validation_fraction,
        "leakage_threshold": config.leakage_threshold,
        "benchmark_prompt_count": len(benchmark_prompts),
        "sources": source_contract,
        "source_counts": {name: dict(value) for name, value in source_counts.items()},
        "pool_counts": dict(counts),
        "capability_counts": {
            pool: dict(values) for pool, values in capability_counts.items()
        },
        "manifests": manifest_contract,
        "benchmark_sources": benchmark_contract,
    }
    summary_path = stats_dir / "release_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    contract = {
        "dataset": DATASET_NAME,
        "schema_version": 1,
        "canonical_fields": list(CANONICAL_FIELDS),
        "pools": {
            pool: {"task": POOL_TASKS[pool], "modality": POOL_MODALITIES[pool]}
            for pool in sorted(VALID_POOLS)
        },
        "release_summary": str(output_dir / "stats/release_summary.json"),
        "release_summary_sha256": file_sha256(summary_path),
    }
    (partial / "release.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    (partial / "seen.sqlite3").unlink(missing_ok=True)
    partial.replace(output_dir)
    return summary
