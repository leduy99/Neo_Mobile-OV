from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

from new_mobile_ov.training.prompt_curriculum import (
    clean_text,
    existing_manifest_image_path,
)


DEFAULT_CAPTION_COLUMNS = (
    "caption",
    "caption_short",
    "caption_medium",
    "caption_long",
    "prompt",
    "text",
)


@dataclass
class GroundedManifestSummary:
    source_manifest: str
    output_manifest: str
    source_name: str
    scanned_rows: int = 0
    accepted_rows: int = 0
    missing_image_rows: int = 0
    unreadable_image_rows: int = 0


def _batches(values: Iterable[dict[str, str]], size: int) -> Iterable[list[dict[str, str]]]:
    iterator = iter(values)
    while batch := list(islice(iterator, size)):
        yield batch


def _verify_image(path: str) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError, SyntaxError):
        return False


def filter_grounded_manifest(
    *,
    source_manifest: str | Path,
    output_manifest: str | Path,
    source_name: str,
    image_columns: Sequence[str],
    image_path_roots: Sequence[str | Path] = (),
    workers: int = 16,
    verify_images: bool = True,
    progress_every: int = 10_000,
) -> GroundedManifestSummary:
    """Write only caption rows whose referenced images resolve and can be opened."""
    source = Path(source_manifest).expanduser()
    output = Path(output_manifest).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Missing source manifest: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    delimiter = "\t" if source.suffix.lower() == ".tsv" else ","
    partial = output.with_suffix(output.suffix + ".partial")
    summary = GroundedManifestSummary(
        source_manifest=str(source),
        output_manifest=str(output),
        source_name=str(source_name),
    )

    with source.open("r", encoding="utf-8", newline="") as input_handle:
        reader = csv.DictReader(input_handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {source}")
        caption_columns = [
            column for column in DEFAULT_CAPTION_COLUMNS if column in reader.fieldnames
        ]
        if not caption_columns:
            raise ValueError(f"Manifest has no caption columns: {source}")
        active_image_columns = [
            column for column in image_columns if column in reader.fieldnames
        ]
        if not active_image_columns:
            raise ValueError(
                f"Manifest has no image columns among {list(image_columns)}: {source}"
            )
        fieldnames = [*caption_columns, "image_path", "source_name"]

        def inspect(row: dict[str, str]) -> tuple[dict[str, str] | None, str]:
            image_path = ""
            for column in active_image_columns:
                image_path = existing_manifest_image_path(
                    row.get(column, ""),
                    manifest_path=source,
                    image_path_roots=image_path_roots,
                )
                if image_path:
                    break
            if not image_path:
                return None, "missing"
            if verify_images and not _verify_image(image_path):
                return None, "unreadable"
            filtered = {column: clean_text(row.get(column, "")) for column in caption_columns}
            filtered["image_path"] = image_path
            filtered["source_name"] = str(source_name)
            return filtered, "accepted"

        try:
            with partial.open("w", encoding="utf-8", newline="") as output_handle:
                writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
                writer.writeheader()
                with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
                    for rows in _batches(reader, max(512, int(workers) * 64)):
                        for filtered, status in executor.map(inspect, rows):
                            summary.scanned_rows += 1
                            if status == "accepted":
                                writer.writerow(filtered)
                                summary.accepted_rows += 1
                            elif status == "missing":
                                summary.missing_image_rows += 1
                            else:
                                summary.unreadable_image_rows += 1
                            if (
                                progress_every > 0
                                and summary.scanned_rows % progress_every == 0
                            ):
                                print(
                                    "Filtered grounded rows="
                                    f"{summary.accepted_rows} scanned={summary.scanned_rows} "
                                    f"missing={summary.missing_image_rows} "
                                    f"unreadable={summary.unreadable_image_rows}",
                                    flush=True,
                                )
            partial.replace(output)
        except BaseException:
            if partial.exists():
                partial.unlink()
            raise

    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(asdict(summary), indent=2), flush=True)
    return summary
