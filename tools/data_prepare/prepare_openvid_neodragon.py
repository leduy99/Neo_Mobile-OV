#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OPENVID_CSV_URL = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVid-1M.csv"
OPENVID_TREE_URL = "https://huggingface.co/api/datasets/nkp37/OpenVid-1M/tree/main?recursive=true"
OPENVID_ZIP_URL = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{}.zip"
OPENVID_SPLIT_URL = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{}_part{}"
SPLIT_RE = re.compile(r"OpenVid_part(\d+)_part([a-z]+)$")


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _download_file(url: str, out_path: Path, *, skip_existing: bool = True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        return
    if shutil.which("wget"):
        _run(["wget", "-c", "--progress=dot:giga", url, "-O", str(out_path)])
    else:
        _run(["curl", "-L", "-C", "-", url, "-o", str(out_path)])


def _try_download_file(url: str, out_path: Path, *, skip_existing: bool = True) -> bool:
    try:
        _download_file(url, out_path, skip_existing=skip_existing)
        return out_path.exists() and out_path.stat().st_size > 0
    except subprocess.CalledProcessError:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass
        return False


def _is_valid_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path, "r") as handle:
            handle.infolist()
        return True
    except zipfile.BadZipFile:
        return False


def _load_split_index(cache_path: Path) -> dict[int, list[str]]:
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return {int(k): list(v) for k, v in payload.get("split_suffixes", {}).items()}
        except Exception:
            pass
    try:
        with urllib.request.urlopen(OPENVID_TREE_URL, timeout=60) as response:
            entries = json.load(response)
    except Exception as exc:
        print(f"Warning: could not query OpenVid HF tree ({exc}); using aa/ab fallback.", flush=True)
        return {}
    split_suffixes: dict[int, list[str]] = {}
    for item in entries if isinstance(entries, list) else []:
        path = item.get("path", "")
        if not isinstance(path, str):
            continue
        match = SPLIT_RE.fullmatch(path)
        if match:
            part = int(match.group(1))
            suffix = match.group(2)
            split_suffixes.setdefault(part, [])
            if suffix not in split_suffixes[part]:
                split_suffixes[part].append(suffix)
    split_suffixes = {k: sorted(v) for k, v in split_suffixes.items()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"split_suffixes": split_suffixes}, indent=2), encoding="utf-8")
    return split_suffixes


def _download_openvid_part(part: int, zips_root: Path, split_index: dict[int, list[str]]) -> Path:
    zip_path = zips_root / f"OpenVid_part{part}.zip"
    if _is_valid_zip(zip_path):
        return zip_path

    # Keep partial zips from interrupted jobs; wget/curl can resume them.
    if _try_download_file(OPENVID_ZIP_URL.format(part), zip_path, skip_existing=False) and _is_valid_zip(zip_path):
        return zip_path
    try:
        zip_path.unlink()
    except FileNotFoundError:
        pass

    suffixes = split_index.get(part) or ["aa", "ab"]
    split_paths: list[Path] = []
    for suffix in suffixes:
        split_path = zips_root / f"OpenVid_part{part}_part{suffix}"
        if not _try_download_file(OPENVID_SPLIT_URL.format(part, suffix), split_path, skip_existing=False):
            for path in split_paths:
                path.unlink(missing_ok=True)
            split_path.unlink(missing_ok=True)
            raise RuntimeError(f"Could not download OpenVid split part={part} suffix={suffix}")
        split_paths.append(split_path)

    tmp_zip = zip_path.with_suffix(".zip.tmp")
    with tmp_zip.open("wb") as out_handle:
        for split_path in split_paths:
            with split_path.open("rb") as in_handle:
                shutil.copyfileobj(in_handle, out_handle, length=8 * 1024 * 1024)
    tmp_zip.replace(zip_path)
    for split_path in split_paths:
        split_path.unlink(missing_ok=True)
    if not _is_valid_zip(zip_path):
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(f"Merged split files did not create a valid zip for OpenVid part {part}")
    return zip_path


def maybe_download_openvid(download_root: Path, *, start_part: int, num_parts: int, extract: bool) -> tuple[Path, Path]:
    """Download OpenVid CSV and optional zip parts into a simple local layout."""
    csv_path = download_root / "OpenVid-1M.csv"
    _download_file(OPENVID_CSV_URL, csv_path)

    parts_root = download_root / "raw" / "parts"
    zips_root = download_root / "zips"
    if num_parts <= 0:
        return csv_path, parts_root

    split_index = _load_split_index(download_root / ".openvid_split_index.json")
    for part in range(start_part, start_part + num_parts):
        zip_path = _download_openvid_part(part, zips_root, split_index)
        if extract:
            out_dir = parts_root / f"part_{part:04d}"
            marker = out_dir / ".extract_complete"
            if marker.exists():
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as handle:
                handle.extractall(out_dir)
            marker.touch()
    return csv_path, parts_root


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def _parse_rate(text: str | None) -> float:
    if not text:
        return float("nan")
    if "/" in text:
        num, den = text.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f else float("nan")
    return float(text)


def ffprobe_video(path: Path) -> dict[str, Any] | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        payload = json.loads(subprocess.check_output(cmd, text=True))
        stream = payload["streams"][0]
    except Exception:
        return None
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    return {
        "source_width": int(stream.get("width") or 0),
        "source_height": int(stream.get("height") or 0),
        "source_fps": fps,
        "source_duration_sec": _safe_float(stream.get("duration")),
        "source_nb_frames": int(float(stream.get("nb_frames") or 0)),
    }


def _video_key(value: Any) -> str:
    path = Path(str(value))
    return path.stem if path.suffix else path.name


def index_videos(parts_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for video_path in sorted(parts_root.rglob("*.mp4")):
        out.setdefault(video_path.stem, video_path)
    return out


def build_source_rows(args: argparse.Namespace) -> pd.DataFrame:
    source_manifest = Path(args.source_manifest).expanduser() if args.source_manifest else None
    if source_manifest and source_manifest.exists():
        return pd.read_csv(source_manifest)

    download_root = Path(args.download_root).expanduser()
    csv_path = Path(args.openvid_csv).expanduser() if args.openvid_csv else None
    parts_root = Path(args.parts_root).expanduser() if args.parts_root else None
    if args.download_csv or args.download_parts > 0:
        csv_path, parts_root = maybe_download_openvid(
            download_root,
            start_part=args.start_part,
            num_parts=args.download_parts,
            extract=args.extract_zips,
        )

    if not csv_path or not csv_path.exists():
        raise FileNotFoundError("Need --source-manifest, --openvid-csv, or --download-csv.")
    if not parts_root or not parts_root.exists():
        raise FileNotFoundError("Need --parts-root with extracted OpenVid videos.")

    df = pd.read_csv(csv_path)
    if "video" not in df.columns or "caption" not in df.columns:
        raise RuntimeError(f"OpenVid CSV must contain video/caption columns: {csv_path}")
    video_index = index_videos(parts_root)
    rows: list[dict[str, Any]] = []
    for row_idx, row in df.iterrows():
        key = _video_key(row["video"])
        video_path = video_index.get(key)
        if video_path is None:
            continue
        rows.append(
            {
                "sample_idx": int(row_idx),
                "video": str(row["video"]),
                "caption": str(row["caption"]),
                "video_path": str(video_path),
            }
        )
    return pd.DataFrame(rows)


def _first_text(row: pd.Series, columns: list[str]) -> str:
    for col in columns:
        if col in row:
            value = row.get(col)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                text = str(value).strip()
                if text:
                    return text
    return ""


def _copy_or_link(src: Path, dst: Path, mode: str) -> Path:
    if mode == "none":
        return src
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src)
    else:
        raise ValueError(f"Unsupported copy mode: {mode}")
    return dst


def choose_clip_start(duration: float, clip_seconds: float, policy: str, rng: random.Random) -> float:
    if not math.isfinite(duration) or duration <= clip_seconds:
        return 0.0
    max_start = max(0.0, duration - clip_seconds)
    if policy == "first":
        return 0.0
    if policy == "center":
        return max_start * 0.5
    if policy == "random":
        return rng.uniform(0.0, max_start)
    raise ValueError(f"Unknown clip policy: {policy}")


def make_neodragon_manifest(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_source_rows(args)
    if args.max_input_rows > 0:
        df = df.head(args.max_input_rows)

    text_columns = [x.strip() for x in args.text_columns.split(",") if x.strip()]
    rng = random.Random(args.seed)
    clip_seconds = float(args.clip_seconds or (args.num_frames / args.target_fps))
    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {"missing_path": 0, "missing_caption": 0, "ffprobe": 0, "too_short": 0}
    durations: list[float] = []

    for row_idx, row in df.iterrows():
        raw_video = row.get("video_path") or row.get("media_path") or row.get("path") or row.get("mp4")
        if not raw_video:
            skipped["missing_path"] += 1
            continue
        src = Path(str(raw_video)).expanduser()
        if not src.exists():
            skipped["missing_path"] += 1
            continue
        prompt = _first_text(row, text_columns)
        if not prompt:
            skipped["missing_caption"] += 1
            continue
        meta = ffprobe_video(src)
        if not meta:
            skipped["ffprobe"] += 1
            continue
        duration = float(meta["source_duration_sec"])
        if duration < args.min_duration_sec and not args.allow_short:
            skipped["too_short"] += 1
            continue
        if duration < clip_seconds and not args.allow_short:
            skipped["too_short"] += 1
            continue

        dst = _copy_or_link(src, out_dir / "videos" / f"{len(rows):07d}_{src.name}", args.copy_mode)
        start = choose_clip_start(duration, clip_seconds, args.clip_policy, rng)
        end = min(duration, start + clip_seconds) if math.isfinite(duration) else start + clip_seconds
        item = {
            "sample_id": len(rows),
            "video_path": str(dst.resolve() if args.copy_mode != "symlink" else dst),
            "prompt": prompt,
            "caption": str(row.get("caption") or prompt),
            "clip_start_sec": round(start, 6),
            "clip_end_sec": round(end, 6),
            "clip_num_frames": int(args.num_frames),
            "clip_fps": float(args.target_fps),
            "clip_policy": args.clip_policy,
            "source_video_path": str(src),
            "source_row": int(row_idx),
            **meta,
        }
        for col in ["caption_short", "caption_medium", "caption_long"]:
            if col in row:
                item[col] = str(row.get(col) or "")
        rows.append(item)
        durations.append(duration)
        if args.max_samples > 0 and len(rows) >= args.max_samples:
            break

    if not rows:
        raise RuntimeError(f"No valid OpenVid samples prepared. skipped={skipped}")

    manifest = out_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    summary = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest),
        "output_dir": str(out_dir),
        "rows": len(rows),
        "input_rows_seen": int(len(df)),
        "skipped": skipped,
        "num_frames": int(args.num_frames),
        "target_fps": float(args.target_fps),
        "clip_seconds": clip_seconds,
        "clip_policy": args.clip_policy,
        "copy_mode": args.copy_mode,
        "duration_sec": {
            "min": min(durations),
            "mean": sum(durations) / len(durations),
            "max": max(durations),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 2s OpenVid clips for NeoDragon DiT training.")
    parser.add_argument("--source-manifest", default="")
    parser.add_argument("--openvid-csv", default="")
    parser.add_argument("--parts-root", default="")
    parser.add_argument("--download-root", default="download_data/data/openvid")
    parser.add_argument("--download-csv", action="store_true")
    parser.add_argument("--download-parts", type=int, default=0)
    parser.add_argument("--start-part", type=int, default=0)
    parser.add_argument("--extract-zips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="data/openvid_neodragon_2s")
    parser.add_argument("--max-input-rows", type=int, default=-1)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--target-fps", type=float, default=24.0)
    parser.add_argument("--clip-seconds", type=float, default=0.0)
    parser.add_argument("--clip-policy", choices=["first", "center", "random"], default="first")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-duration-sec", type=float, default=2.0)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--copy-mode", choices=["none", "copy", "hardlink", "symlink"], default="none")
    parser.add_argument(
        "--text-columns",
        default="caption_long,caption_medium,caption_short,prompt,caption,text",
        help="Prompt column priority for the primary prompt field.",
    )
    args = parser.parse_args()
    make_neodragon_manifest(args)


if __name__ == "__main__":
    main()
