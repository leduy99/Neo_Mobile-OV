#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def copy_video(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
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
        raise ValueError(f"Unknown copy mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        default="/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/data/openvid_1m/manifests/by_part/part_0111.csv",
    )
    parser.add_argument("--output-dir", default="data/neodragon_openvid100")
    parser.add_argument("--num-videos", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="copy")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_manifest = Path(args.source_manifest)
    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source_manifest)
    rows: list[dict[str, object]] = []
    skipped = 0
    for row_idx, row in df.iloc[max(args.offset, 0) :].iterrows():
        src_value = row.get("video_path") or row.get("media_path") or row.get("path")
        caption = str(row.get("caption") or row.get("prompt") or row.get("text") or "").strip()
        if not src_value or not caption:
            skipped += 1
            continue
        src = Path(str(src_value))
        if not src.exists():
            skipped += 1
            continue
        stem = f"{len(rows):04d}_{src.name}"
        dst = video_dir / stem
        if args.overwrite and dst.exists() and not dst.is_symlink():
            dst.unlink()
        copy_video(src, dst, args.mode)
        rows.append(
            {
                "sample_id": len(rows),
                "video_path": str(dst.resolve() if args.mode != "symlink" else dst),
                "prompt": caption,
                "caption": caption,
                "source_video_path": str(src),
                "source_manifest": str(source_manifest),
                "source_row": int(row_idx),
                "source_id": str(row.get("source_id") or src.name),
            }
        )
        if len(rows) >= args.num_videos:
            break

    if len(rows) < args.num_videos:
        raise RuntimeError(f"Only prepared {len(rows)} videos from {source_manifest}; skipped={skipped}")

    manifest = output_dir / "manifest.csv"
    prompt_file = output_dir / "prompts.txt"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    prompt_file.write_text("\n".join(str(r["prompt"]) for r in rows) + "\n", encoding="utf-8")
    summary = {
        "source_manifest": str(source_manifest),
        "output_dir": str(output_dir),
        "manifest": str(manifest),
        "prompt_file": str(prompt_file),
        "video_dir": str(video_dir),
        "num_videos": len(rows),
        "copy_mode": args.mode,
        "skipped": skipped,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
