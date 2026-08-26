from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sqlite3
import tarfile
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence


CAPTION_COLUMNS = (
    "caption",
    "prompt",
    "text",
    "caption_medium",
    "caption_long",
    "caption_short",
)
CAPTION_VARIANT_COLUMNS = ("caption_short", "caption_medium", "caption_long")
IMAGE_COLUMNS = ("image_path", "media_path", "path", "file_path", "video_path")
SOURCE_KEY_COLUMNS = ("record_id", "id", "sample_id", "key", "source_key", "filename")

COLOR_WORDS = (
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
    "gray",
    "grey",
    "silver",
    "gold",
    "golden",
    "turquoise",
    "cyan",
    "magenta",
)
COUNT_WORDS = (
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "single",
    "pair",
    "couple",
    "dozen",
)
SPATIAL_PHRASES = (
    "to the left of",
    "to the right of",
    "left of",
    "right of",
    "in front of",
    "behind",
    "above",
    "below",
    "under",
    "over",
    "next to",
    "beside",
    "between",
    "inside",
    "outside",
    "on top of",
    "near",
)
ACTION_WORDS = (
    "walking",
    "running",
    "jumping",
    "dancing",
    "cooking",
    "reading",
    "writing",
    "driving",
    "riding",
    "flying",
    "swimming",
    "playing",
    "painting",
    "holding",
    "carrying",
    "talking",
    "working",
    "standing",
    "sitting",
    "lying",
    "eating",
    "drinking",
)
SCENE_WORDS = (
    "kitchen",
    "bedroom",
    "bathroom",
    "office",
    "classroom",
    "library",
    "laboratory",
    "hospital",
    "restaurant",
    "cafe",
    "airport",
    "station",
    "street",
    "city",
    "village",
    "forest",
    "beach",
    "mountain",
    "desert",
    "ocean",
    "underwater",
    "stadium",
    "museum",
    "market",
    "store",
    "workshop",
    "farm",
    "garden",
    "park",
)
STYLE_WORDS = (
    "photograph",
    "cinematic",
    "documentary",
    "watercolor",
    "illustration",
    "painting",
    "anime",
    "sketch",
    "poster",
    "pixel art",
    "3d render",
    "macro",
    "wide angle",
    "black and white",
)
ATTRIBUTE_WORDS = (
    "wooden",
    "metal",
    "metallic",
    "glass",
    "ceramic",
    "plastic",
    "paper",
    "striped",
    "spotted",
    "shiny",
    "transparent",
    "furry",
    "small",
    "large",
    "tall",
    "short",
)

_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_DIGIT_COUNT_RE = re.compile(r"\b(?:[2-9]|10)\b")
_QUOTED_TEXT_RE = re.compile(r"[\"'][A-Za-z0-9][A-Za-z0-9 _-]{1,30}[\"']")
_READS_TEXT_RE = re.compile(r"\b(?:reads?|reading|says?|saying|written|text)\b", re.I)


@dataclass(frozen=True)
class ImageBridgeSource:
    name: str
    role: str
    path: Path

    def __post_init__(self) -> None:
        if self.role not in {"broad", "compositional", "grounded"}:
            raise ValueError(f"Unsupported source role: {self.role}")


@dataclass(frozen=True)
class ImageBridgeBuildConfig:
    output_dir: Path
    seed: int = 20260825
    validation_fraction: float = 0.005
    hard_validation_per_capability: int = 100
    max_broad_records: int = 3_000_000
    max_compositional_records: int = 200_000
    max_grounded_records: int = 200_000
    source_hashes: bool = True


def normalize_caption(value: object) -> str:
    if value is None:
        return ""
    return _SPACE_RE.sub(" ", str(value).strip())


def normalized_caption_key(caption: str) -> str:
    value = unicodedata.normalize("NFKC", normalize_caption(caption)).casefold()
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    return value.rstrip(".!?")


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _contains_term(text: str, tokens: set[str], term: str) -> bool:
    if " " not in term:
        return term in tokens
    return f" {term} " in text


def infer_capabilities(caption: str) -> tuple[str, ...]:
    text = f" {normalized_caption_key(caption)} "
    tokens = set(_WORD_RE.findall(text))
    capabilities: set[str] = set()
    colors = {word for word in COLOR_WORDS if word in tokens}
    if colors:
        capabilities.add("color")
    if any(word in tokens for word in COUNT_WORDS) or _DIGIT_COUNT_RE.search(text):
        capabilities.add("count")
    if any(_contains_term(text, tokens, phrase) for phrase in SPATIAL_PHRASES):
        capabilities.add("spatial_relation")
    if any(word in tokens for word in ACTION_WORDS):
        capabilities.add("action")
    if any(word in tokens for word in SCENE_WORDS):
        capabilities.add("scene")
    if any(_contains_term(text, tokens, phrase) for phrase in STYLE_WORDS):
        capabilities.add("style")
    if any(word in tokens for word in ATTRIBUTE_WORDS):
        capabilities.add("attribute")
    if _QUOTED_TEXT_RE.search(caption) or _READS_TEXT_RE.search(caption):
        capabilities.add("rendered_text")
    multi_signal = (
        " and " in text
        or " both " in text
        or " pair of " in text
        or " group of " in text
        or " several " in text
        or " multiple " in text
        or "spatial_relation" in capabilities
    )
    if multi_signal:
        capabilities.add("multiple_objects")
    if len(colors) >= 2 and "multiple_objects" in capabilities:
        capabilities.add("attribute_binding")
    return tuple(sorted(capabilities))


def compositional_score(capabilities: Sequence[str]) -> int:
    values = set(capabilities)
    weights = {
        "attribute_binding": 4,
        "spatial_relation": 3,
        "count": 3,
        "multiple_objects": 2,
        "color": 1,
        "attribute": 1,
        "action": 1,
        "scene": 1,
        "rendered_text": 2,
    }
    return sum(weights.get(value, 0) for value in values)


def is_compositional(capabilities: Sequence[str]) -> bool:
    values = set(capabilities)
    return bool(
        values & {"attribute_binding", "spatial_relation", "count"}
        or "multiple_objects" in values
        and bool(values & {"color", "attribute", "action"})
    )


def _manifest_delimiter(path: Path) -> str:
    return "\t" if path.suffix.lower() == ".tsv" else ","


def _iter_tar_rows(path: Path) -> Iterator[dict[str, str]]:
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            suffix = Path(member.name).suffix.lower()
            if suffix not in {".txt", ".caption", ".json"}:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            raw = handle.read().decode("utf-8", errors="replace").strip()
            if not raw:
                continue
            caption = raw
            if suffix == ".json":
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                caption = next(
                    (
                        normalize_caption(payload.get(column))
                        for column in CAPTION_COLUMNS
                        if normalize_caption(payload.get(column))
                    ),
                    "",
                )
            if caption:
                yield {"caption": caption, "source_key": member.name}


def iter_manifest_rows(path: str | Path) -> Iterator[dict[str, str]]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Missing source manifest: {source}")
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {source}:{line_number}: {exc}"
                    ) from exc
                if isinstance(row, Mapping):
                    yield {str(key): str(value) for key, value in row.items() if value is not None}
        return
    if suffix in {".tar", ".tgz", ".gz"} or source.name.endswith(".tar.gz"):
        yield from _iter_tar_rows(source)
        return
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=_manifest_delimiter(source))
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {source}")
        for row in reader:
            yield {str(key): value or "" for key, value in row.items() if key is not None}


def _caption_from_row(row: Mapping[str, object]) -> str:
    for column in CAPTION_COLUMNS:
        caption = normalize_caption(row.get(column))
        if caption:
            return caption
    return ""


def _value_from_columns(row: Mapping[str, object], columns: Sequence[str]) -> str:
    for column in columns:
        value = normalize_caption(row.get(column))
        if value:
            return value
    return ""


def _source_key(row: Mapping[str, object], row_number: int) -> str:
    return _value_from_columns(row, SOURCE_KEY_COLUMNS) or str(row_number)


def _source_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _connect_catalog(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        """
        CREATE TABLE records (
            caption_hash TEXT PRIMARY KEY,
            record_id TEXT NOT NULL,
            selection_key TEXT NOT NULL,
            caption TEXT NOT NULL,
            caption_short TEXT NOT NULL,
            caption_medium TEXT NOT NULL,
            caption_long TEXT NOT NULL,
            source_name TEXT NOT NULL,
            source_names TEXT NOT NULL,
            source_role TEXT NOT NULL,
            source_key TEXT NOT NULL,
            source_path TEXT NOT NULL,
            image_path TEXT NOT NULL,
            grounding_status TEXT NOT NULL,
            capabilities TEXT NOT NULL,
            capability_score INTEGER NOT NULL,
            split TEXT NOT NULL,
            selected_pool TEXT NOT NULL DEFAULT ''
        )
        """
    )
    connection.execute("CREATE INDEX records_split_idx ON records(split)")
    connection.execute("CREATE INDEX records_role_idx ON records(source_role)")
    connection.execute("CREATE INDEX records_score_idx ON records(capability_score)")
    return connection


def _role_priority(role: str) -> int:
    return {"broad": 0, "compositional": 1, "grounded": 2}[role]


def _selection_key(caption_hash: str, seed: int) -> str:
    return stable_hash(f"{seed}:{caption_hash}")


def _split_for_hash(caption_hash: str, validation_fraction: float, seed: int) -> str:
    value = int(_selection_key(caption_hash, seed)[:12], 16) / float(16**12)
    return "validation" if value < validation_fraction else "train"


def _merge_source_names(existing: str, new_name: str) -> str:
    values = [value for value in existing.split(";") if value]
    if new_name not in values:
        values.append(new_name)
    return ";".join(values)


def _insert_row(
    connection: sqlite3.Connection,
    *,
    source: ImageBridgeSource,
    row: Mapping[str, object],
    row_number: int,
    validation_fraction: float,
    seed: int,
) -> str:
    caption = _caption_from_row(row)
    if not caption:
        return "empty"
    caption_key = normalized_caption_key(caption)
    if len(caption_key) < 3:
        return "empty"
    caption_hash = stable_hash(caption_key)
    capabilities = infer_capabilities(caption)
    capabilities_text = ";".join(capabilities)
    score = compositional_score(capabilities)
    image_path = _value_from_columns(row, IMAGE_COLUMNS)
    if source.role == "grounded" and not image_path:
        return "missing_image"
    variants = {
        column: normalize_caption(row.get(column)) for column in CAPTION_VARIANT_COLUMNS
    }
    source_key = _source_key(row, row_number)
    inserted = connection.execute(
        """
        INSERT OR IGNORE INTO records (
            caption_hash, record_id, selection_key, caption, caption_short, caption_medium,
            caption_long, source_name, source_names, source_role, source_key,
            source_path, image_path, grounding_status, capabilities,
            capability_score, split
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            caption_hash,
            caption_hash,
            _selection_key(caption_hash, seed),
            caption,
            variants["caption_short"],
            variants["caption_medium"],
            variants["caption_long"],
            source.name,
            source.name,
            source.role,
            source_key,
            str(source.path.resolve()),
            image_path,
            "path_verified" if source.role == "grounded" else "caption_only",
            capabilities_text,
            score,
            _split_for_hash(caption_hash, validation_fraction, seed),
        ),
    )
    if inserted.rowcount:
        return "inserted"

    existing = connection.execute(
        """
        SELECT caption, source_name, source_names, source_role, source_key,
               source_path, image_path, grounding_status, caption_short,
               caption_medium, caption_long, capabilities, capability_score
        FROM records WHERE caption_hash = ?
        """,
        (caption_hash,),
    ).fetchone()
    assert existing is not None
    (
        existing_caption,
        existing_name,
        existing_names,
        existing_role,
        existing_key,
        existing_path,
        existing_image,
        existing_grounding,
        existing_short,
        existing_medium,
        existing_long,
        existing_capabilities,
        existing_score,
    ) = existing
    prefer_new = _role_priority(source.role) > _role_priority(existing_role)
    connection.execute(
        """
        UPDATE records SET
            caption = ?, source_name = ?, source_names = ?, source_role = ?, source_key = ?,
            source_path = ?, image_path = ?, grounding_status = ?,
            caption_short = ?, caption_medium = ?, caption_long = ?,
            capabilities = ?, capability_score = ?
        WHERE caption_hash = ?
        """,
        (
            caption if prefer_new else existing_caption,
            source.name if prefer_new else existing_name,
            _merge_source_names(existing_names, source.name),
            source.role if prefer_new else existing_role,
            source_key if prefer_new else existing_key,
            str(source.path.resolve()) if prefer_new else existing_path,
            image_path or existing_image,
            "path_verified" if image_path and source.role == "grounded" else existing_grounding,
            variants["caption_short"] or existing_short,
            variants["caption_medium"] or existing_medium,
            variants["caption_long"] or existing_long,
            capabilities_text if prefer_new else existing_capabilities,
            score if prefer_new else existing_score,
            caption_hash,
        ),
    )
    return "duplicate"


def _assign_hard_validation(
    connection: sqlite3.Connection,
    *,
    per_capability: int,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    if per_capability <= 0:
        return counts
    capabilities = (
        "attribute_binding",
        "spatial_relation",
        "count",
        "multiple_objects",
        "color",
        "action",
        "scene",
    )
    for capability in capabilities:
        rows = connection.execute(
            """
            SELECT caption_hash FROM records
            WHERE split = 'train' AND (';' || capabilities || ';') LIKE ?
            ORDER BY selection_key LIMIT ?
            """,
            (f"%;{capability};%", per_capability),
        ).fetchall()
        connection.executemany(
            "UPDATE records SET split = 'hard_validation' WHERE caption_hash = ?",
            rows,
        )
        counts[capability] = len(rows)
    return counts


def _assign_pool(
    connection: sqlite3.Connection,
    *,
    pool: str,
    where: str,
    limit: int,
    order_by: str,
) -> int:
    limit_clause = "" if limit < 0 else f"LIMIT {int(limit)}"
    rows = connection.execute(
        f"""
        SELECT caption_hash FROM records
        WHERE split = 'train' AND selected_pool = '' AND ({where})
        ORDER BY {order_by} {limit_clause}
        """
    ).fetchall()
    connection.executemany(
        "UPDATE records SET selected_pool = ? WHERE caption_hash = ?",
        ((pool, row[0]) for row in rows),
    )
    return len(rows)


EXPORT_FIELDS = (
    "record_id",
    "caption",
    "caption_short",
    "caption_medium",
    "caption_long",
    "image_path",
    "source_name",
    "source_names",
    "source_role",
    "source_key",
    "capabilities",
    "capability_score",
    "grounding_status",
    "split",
)


def _export_query(
    connection: sqlite3.Connection,
    *,
    output_path: Path,
    where: str,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    query = f"SELECT {', '.join(EXPORT_FIELDS)} FROM records WHERE {where} ORDER BY record_id"
    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(EXPORT_FIELDS)
        for row in connection.execute(query):
            writer.writerow(row)
            count += 1
    return count


def _source_registry(
    sources: Sequence[ImageBridgeSource], *, include_hashes: bool
) -> list[dict[str, object]]:
    registry = []
    for source in sources:
        stat = source.path.stat()
        registry.append(
            {
                "name": source.name,
                "role": source.role,
                "path": str(source.path),
                "resolved_path": str(source.path.resolve()),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _source_file_sha256(source.path) if include_hashes else None,
            }
        )
    return registry


def build_image_bridge_data_v1(
    sources: Sequence[ImageBridgeSource],
    config: ImageBridgeBuildConfig,
    *,
    force: bool = False,
    progress_every: int = 100_000,
) -> dict[str, object]:
    if not sources:
        raise ValueError("At least one source manifest is required.")
    source_names = [source.name for source in sources]
    if len(set(source_names)) != len(source_names):
        raise ValueError("Source names must be unique.")
    for source in sources:
        if not source.path.is_file():
            raise FileNotFoundError(f"Missing source manifest: {source.path}")
    if not 0 <= config.validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")

    output_dir = config.output_dir.expanduser()
    work_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"Output already exists: {output_dir}. Use force=True to replace it."
            )
        shutil.rmtree(output_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    (work_dir / "manifests").mkdir(parents=True)
    (work_dir / "mixtures").mkdir()
    (work_dir / "stats").mkdir()

    catalog_path = work_dir / "catalog.sqlite3"
    connection = _connect_catalog(catalog_path)
    source_stats: dict[str, Counter[str]] = {}
    try:
        for source in sources:
            stats: Counter[str] = Counter()
            source_stats[source.name] = stats
            for row_number, row in enumerate(iter_manifest_rows(source.path), start=1):
                status = _insert_row(
                    connection,
                    source=source,
                    row=row,
                    row_number=row_number,
                    validation_fraction=config.validation_fraction,
                    seed=config.seed,
                )
                stats[status] += 1
                stats["rows"] += 1
                if row_number % 10_000 == 0:
                    connection.commit()
                if progress_every > 0 and row_number % progress_every == 0:
                    print(
                        f"source={source.name} rows={row_number} "
                        f"inserted={stats['inserted']} duplicates={stats['duplicate']} "
                        f"empty={stats['empty']} missing_image={stats['missing_image']}",
                        flush=True,
                    )
            connection.commit()

        hard_counts = _assign_hard_validation(
            connection,
            per_capability=config.hard_validation_per_capability,
        )
        connection.commit()
        grounded_count = _assign_pool(
            connection,
            pool="grounded",
            where="image_path != '' AND grounding_status = 'path_verified'",
            limit=config.max_grounded_records,
            order_by="capability_score DESC, selection_key",
        )
        compositional_count = _assign_pool(
            connection,
            pool="compositional",
            where=(
                "source_role = 'compositional' OR capability_score >= 3 OR "
                "instr(';' || capabilities || ';', ';spatial_relation;') > 0 OR "
                "instr(';' || capabilities || ';', ';count;') > 0"
            ),
            limit=config.max_compositional_records,
            order_by="capability_score DESC, selection_key",
        )
        broad_count = _assign_pool(
            connection,
            pool="broad",
            where="1 = 1",
            limit=config.max_broad_records,
            order_by="selection_key",
        )
        connection.commit()

        export_counts = {
            "broad_train": _export_query(
                connection,
                output_path=work_dir / "manifests" / "d1_broad_train.csv",
                where="split = 'train' AND selected_pool = 'broad'",
            ),
            "compositional_train": _export_query(
                connection,
                output_path=work_dir / "manifests" / "d2_compositional_train.csv",
                where="split = 'train' AND selected_pool = 'compositional'",
            ),
            "grounded_candidates": _export_query(
                connection,
                output_path=work_dir / "manifests" / "d2_grounded_candidates.csv",
                where="split = 'train' AND selected_pool = 'grounded'",
            ),
            "validation": _export_query(
                connection,
                output_path=work_dir / "manifests" / "validation.csv",
                where="split = 'validation'",
            ),
            "hard_validation": _export_query(
                connection,
                output_path=work_dir / "manifests" / "hard_validation.csv",
                where="split = 'hard_validation'",
            ),
        }
        assert export_counts["broad_train"] == broad_count
        assert export_counts["compositional_train"] == compositional_count
        assert export_counts["grounded_candidates"] == grounded_count

        capability_counts: Counter[str] = Counter()
        for (capabilities,) in connection.execute(
            "SELECT capabilities FROM records WHERE selected_pool != ''"
        ):
            capability_counts.update(value for value in capabilities.split(";") if value)

        mixture = {
            "name": "v12_70_20_10",
            "overall_sampling": {"broad": 0.70, "compositional": 0.20, "grounded": 0.10},
            "trainer_contract": {
                "generation_prompt_manifests": [
                    "manifests/d1_broad_train.csv",
                    "manifests/d2_compositional_train.csv",
                    "manifests/d2_grounded_high_precision_50k.csv",
                ],
                "generation_source_names": ["broad", "compositional", "grounded_cascade"],
                "generation_source_weights": [7.0 / 9.0, 2.0 / 9.0, 0.0],
                "grounded_source_names": ["grounded_cascade"],
                "grounded_batch_probability": 0.10,
                "semantic_prompt_probability": 0.0,
            },
            "note": (
                "The grounded CSV is produced by the separate SigLIP2-ranking and "
                "Qwen3.6-adjudication cascade. Do not substitute the raw candidate CSV."
            ),
        }
        summary = {
            "dataset": "ImageBridge-Data-v1",
            "build_config": {**asdict(config), "output_dir": str(config.output_dir)},
            "source_stats": {
                name: dict(stats) for name, stats in source_stats.items()
            },
            "unique_records": connection.execute("SELECT COUNT(*) FROM records").fetchone()[0],
            "export_counts": export_counts,
            "hard_validation_by_capability": hard_counts,
            "selected_capabilities": dict(sorted(capability_counts.items())),
        }
        (work_dir / "source_registry.json").write_text(
            json.dumps(
                _source_registry(sources, include_hashes=config.source_hashes),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (work_dir / "mixtures" / "v12_70_20_10.json").write_text(
            json.dumps(mixture, indent=2) + "\n", encoding="utf-8"
        )
        (work_dir / "stats" / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        (work_dir / "DATASET_CARD.md").write_text(
            "# ImageBridge-Data-v1\n\n"
            "This immutable catalog preserves source captions and creates mutually exclusive "
            "broad, compositional, and grounded-candidate views. No prompt is freely generated. "
            "Grounded candidates must pass the separate SigLIP2-ranking and Qwen3.6-adjudication "
            "cascade before they are used by DreamLite V12.\n",
            encoding="utf-8",
        )
    except BaseException:
        connection.close()
        raise
    connection.close()
    work_dir.replace(output_dir)
    print(json.dumps(summary, indent=2), flush=True)
    return summary
