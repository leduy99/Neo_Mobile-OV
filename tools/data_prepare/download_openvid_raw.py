#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.data_prepare.prepare_openvid_neodragon import _download_openvid_part, _load_split_index

PART_RE = re.compile(r"part[_-](\d+)")


def parse_part_spec(text: str) -> list[int]:
    parts: set[int] = set()
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid descending part range: {chunk}")
            parts.update(range(start, end + 1))
        else:
            parts.add(int(chunk))
    return sorted(parts)


def infer_parts_from_manifest(manifest: Path) -> list[int]:
    df = pd.read_csv(manifest)
    parts: set[int] = set()
    for col in ["part_remote", "part_user", "part", "openvid_part"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna().astype(int).tolist()
            parts.update(values)
    for col in ["video_path", "source_video_path", "path", "mp4", "media_path"]:
        if col not in df.columns:
            continue
        for value in df[col].dropna().astype(str):
            match = PART_RE.search(value)
            if match:
                parts.add(int(match.group(1)))
    if not parts:
        raise RuntimeError(
            f"Could not infer OpenVid part ids from {manifest}. "
            "Use --parts, for example --parts 0,1,35-40."
        )
    return sorted(parts)


def count_mp4s(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*.mp4"))


def extract_zip_if_needed(zip_path: Path, out_dir: Path, *, overwrite: bool) -> dict[str, object]:
    marker = out_dir / ".extract_complete"
    before = count_mp4s(out_dir)
    if marker.exists() and before > 0 and not overwrite:
        return {"status": "exists", "mp4_count": before}
    if overwrite and out_dir.exists():
        for path in out_dir.glob(".extract_complete"):
            path.unlink(missing_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("unzip"):
        subprocess.run(["unzip", "-q", "-n", str(zip_path), "-d", str(out_dir)], check=True)
    else:
        with zipfile.ZipFile(zip_path, "r") as handle:
            handle.extractall(out_dir)
    marker.touch()
    return {"status": "extracted", "mp4_count": count_mp4s(out_dir)}


def process_part(
    *,
    part: int,
    download_root: Path,
    split_index: dict[int, list[str]],
    extract: bool,
    overwrite_extract: bool,
    keep_zip: bool,
    dry_run: bool,
) -> dict[str, object]:
    zips_root = download_root / "zips"
    raw_parts_root = download_root / "raw" / "parts"
    zip_path = zips_root / f"OpenVid_part{part}.zip"
    out_dir = raw_parts_root / f"part_{part:04d}"
    marker = out_dir / ".extract_complete"
    item: dict[str, object] = {
        "part": int(part),
        "zip_path": str(zip_path),
        "extract_dir": str(out_dir),
        "split_suffixes": split_index.get(part, []),
    }
    if dry_run:
        item["status"] = "dry_run"
        item["existing_mp4_count"] = count_mp4s(out_dir)
        return item
    if extract and marker.exists() and count_mp4s(out_dir) > 0 and not overwrite_extract:
        item["status"] = "exists"
        item["mp4_count"] = count_mp4s(out_dir)
        return item

    zip_path = _download_openvid_part(part, zips_root, split_index)
    item["zip_path"] = str(zip_path)
    item["zip_size_bytes"] = int(zip_path.stat().st_size)
    if extract:
        item.update(extract_zip_if_needed(zip_path, out_dir, overwrite=overwrite_extract))
        if not keep_zip:
            zip_path.unlink(missing_ok=True)
            item["zip_removed"] = True
    else:
        item["status"] = "downloaded"
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/extract OpenVid raw video parts into this repo.")
    parser.add_argument("--download-root", default="download_data/data/openvid")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--parts", default="", help="Comma/range list, e.g. 0,1,35-40. Overrides manifest inference.")
    parser.add_argument("--max-parts", type=int, default=-1, help="Safety limit after sorting inferred/requested parts.")
    parser.add_argument("--start-offset", type=int, default=0, help="Skip this many sorted parts before --max-parts.")
    parser.add_argument("--extract", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite-extract", action="store_true")
    parser.add_argument("--keep-zip", action="store_true", help="Keep zip files after successful extraction.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel part download/extract workers.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    download_root = Path(args.download_root).expanduser()
    if args.parts.strip():
        parts = parse_part_spec(args.parts)
        source = "args.parts"
    else:
        manifest = Path(args.manifest).expanduser()
        if not manifest.exists():
            raise FileNotFoundError(f"Missing manifest for part inference: {manifest}")
        parts = infer_parts_from_manifest(manifest)
        source = str(manifest)
    if args.start_offset > 0:
        parts = parts[args.start_offset :]
    if args.max_parts > 0:
        parts = parts[: args.max_parts]
    if not parts:
        raise RuntimeError("No OpenVid parts selected.")

    split_index = _load_split_index(download_root / ".openvid_split_index.json")

    summary = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "download_root": str(download_root),
        "part_source": source,
        "parts": parts,
        "dry_run": bool(args.dry_run),
        "extract": bool(args.extract),
        "keep_zip": bool(args.keep_zip),
        "workers": int(args.workers),
        "items": [],
    }

    workers = max(1, int(args.workers))
    if workers == 1:
        for part in parts:
            item = process_part(
                part=part,
                download_root=download_root,
                split_index=split_index,
                extract=args.extract,
                overwrite_extract=args.overwrite_extract,
                keep_zip=args.keep_zip,
                dry_run=args.dry_run,
            )
            print(json.dumps(item, ensure_ascii=False), flush=True)
            summary["items"].append(item)
    else:
        ordered: list[dict[str, object] | None] = [None] * len(parts)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_part,
                    part=part,
                    download_root=download_root,
                    split_index=split_index,
                    extract=args.extract,
                    overwrite_extract=args.overwrite_extract,
                    keep_zip=args.keep_zip,
                    dry_run=args.dry_run,
                ): idx
                for idx, part in enumerate(parts)
            }
            for future in as_completed(futures):
                idx = futures[future]
                item = future.result()
                ordered[idx] = item
                print(json.dumps(item, ensure_ascii=False), flush=True)
        summary["items"] = [item for item in ordered if item is not None]

    download_root.mkdir(parents=True, exist_ok=True)
    summary_path = download_root / "openvid_raw_download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
