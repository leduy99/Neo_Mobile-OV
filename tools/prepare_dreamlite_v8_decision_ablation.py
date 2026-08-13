#!/usr/bin/env python
"""Prepare a reusable factorial DreamLite/NeoDragon ablation for VBench scoring.

The source runs already contain four controlled cells per prompt and seed. This
tool only creates VBench-compatible symlinks and contact sheets; it never
regenerates an image or video.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from decord import VideoReader
from PIL import Image, ImageDraw


VIDEO_CELLS = (
    "image_native_qwen__text_native_neodragon",
    "image_native_qwen__text_exp1_64k",
    "image_v8_imageonly__text_native_neodragon",
    "image_v8_imageonly__text_exp1_64k",
)
DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "temporal_flickering",
    "object_class",
    "multiple_objects",
    "color",
    "spatial_relationship",
    "scene",
    "temporal_style",
    "overall_consistency",
    "human_action",
    "appearance_style",
)


def normalize_prompt(value: object) -> str:
    return " ".join(str(value).strip().split())


def valid_image(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def valid_video(path: Path, expected_frames: int) -> bool:
    if not path.is_file() or path.stat().st_size < 4096:
        return False
    try:
        return len(VideoReader(str(path), num_threads=1)) == expected_frames
    except Exception:
        return False


def link_asset(source: Path, destination: Path, *, expected_frames: int) -> None:
    if not valid_video(source, expected_frames):
        raise RuntimeError(f"Missing or invalid source video: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve() and valid_video(destination, expected_frames):
            return
        destination.unlink()
    elif destination.exists():
        if valid_video(destination, expected_frames):
            return
        destination.unlink()
    destination.symlink_to(source.resolve())
    if not valid_video(destination, expected_frames):
        raise RuntimeError(f"Invalid linked video: {destination}")


def load_source_run(path: Path) -> dict:
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing ablation summary: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        raise RuntimeError(f"Ablation run is incomplete: {summary_path}")
    seed = int(payload["seed"])
    prompts = payload.get("prompts", [])
    if not isinstance(prompts, list) or not prompts:
        raise RuntimeError(f"Ablation summary has no prompt records: {summary_path}")
    return {"root": path.resolve(), "summary": payload, "seed": seed, "prompts": prompts}


def ordered_vbench_subset(info_path: Path, prompts: list[str]) -> list[dict]:
    rows = json.loads(info_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"Expected VBench metadata list: {info_path}")
    requested = set(prompts)
    available = {normalize_prompt(row.get("prompt_en", "")) for row in rows if isinstance(row, dict)}
    missing = requested - available
    if missing:
        raise RuntimeError(f"VBench info is missing prompt: {sorted(missing)[0]!r}")
    return [row for row in rows if normalize_prompt(row.get("prompt_en", "")) in requested]


def dimensions_for(row: dict) -> list[str]:
    value = row.get("dimension", [])
    if isinstance(value, str):
        return [value]
    return value if isinstance(value, list) else []


def write_dimension_subsets(rows: list[dict], directory: Path) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    active: list[str] = []
    for dimension in DIMENSIONS:
        matching = [row for row in rows if dimension in dimensions_for(row)]
        if not matching:
            continue
        (directory / f"{dimension}.json").write_text(
            json.dumps(matching, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        active.append(dimension)
    if not active:
        raise RuntimeError("The selected VBench prompts do not cover any VBench dimension.")
    return active


def thumbnail(image: Image.Image, width: int, height: int) -> Image.Image:
    result = image.convert("RGB")
    result.thumbnail((width, height))
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(result, ((width - result.width) // 2, (height - result.height) // 2))
    return canvas


def video_strip(path: Path, *, width: int, height: int) -> Image.Image:
    reader = VideoReader(str(path), num_threads=1)
    indices = (0, max(0, len(reader) // 2), max(0, len(reader) - 1))
    frames = [thumbnail(Image.fromarray(reader[index].asnumpy()), width, height) for index in indices]
    strip = Image.new("RGB", (width * len(frames), height), "white")
    for index, frame in enumerate(frames):
        strip.paste(frame, (index * width, 0))
    return strip


def create_contact_sheet(
    prompt_record: dict,
    output: Path,
    *,
    expected_frames: int,
) -> None:
    anchors = prompt_record["anchors"]
    videos = prompt_record["videos"]
    for path in anchors.values():
        if not valid_image(Path(path)):
            raise RuntimeError(f"Cannot create contact sheet from invalid anchor: {path}")
    frame_width, frame_height = 144, 90
    card_width = frame_width * 3
    row_height = frame_height + 26
    header_height = 62
    canvas = Image.new("RGB", (card_width * 2, header_height + row_height * 3), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), prompt_record["prompt"], fill="black")
    draw.text((8, 30), "Each video card shows first / middle / last frame.", fill="black")

    anchor_images = [
        thumbnail(Image.open(anchors["native_qwen"]), card_width // 2, frame_height),
        thumbnail(Image.open(anchors["v8_imageonly"]), card_width // 2, frame_height),
    ]
    draw.text((4, header_height), "native Qwen anchor", fill="black")
    draw.text((card_width // 2 + 4, header_height), "V8 image-only anchor", fill="black")
    canvas.paste(anchor_images[0], (0, header_height + 18))
    canvas.paste(anchor_images[1], (card_width // 2, header_height + 18))

    labels = (
        ("native Qwen + native NeoDragon", "image_native_qwen__text_native_neodragon"),
        ("native Qwen + Exp1-64k", "image_native_qwen__text_exp1_64k"),
        ("V8 + native NeoDragon", "image_v8_imageonly__text_native_neodragon"),
        ("V8 + Exp1-64k", "image_v8_imageonly__text_exp1_64k"),
    )
    for index, (label, key) in enumerate(labels):
        path = Path(videos[key])
        if not valid_video(path, expected_frames):
            raise RuntimeError(f"Cannot create contact sheet from invalid video: {path}")
        column = index % 2
        row = 1 + index // 2
        x = column * card_width
        y = header_height + row * row_height
        draw.text((x + 4, y), label, fill="black")
        canvas.paste(video_strip(path, width=frame_width, height=frame_height), (x, y + 18))

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def vbench_video_name(prompt: str) -> str:
    return f"{prompt}-0.mp4"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-roots", required=True, nargs="+", type=Path)
    parser.add_argument("--vbench-info", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    sources = [load_source_run(path) for path in args.run_roots]
    seeds = [source["seed"] for source in sources]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Each source run must use a different seed: {seeds}")

    canonical_prompts = [normalize_prompt(row["prompt"]) for row in sources[0]["prompts"]]
    if not canonical_prompts:
        raise RuntimeError("First source has no prompts.")
    for source in sources[1:]:
        candidate = [normalize_prompt(row["prompt"]) for row in source["prompts"]]
        if candidate != canonical_prompts:
            raise RuntimeError("All source ablations must use the same prompt order.")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    subset = ordered_vbench_subset(args.vbench_info, canonical_prompts)
    subset_path = output / "vbench_control20_info.json"
    subset_path.write_text(json.dumps(subset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    active_dimensions = write_dimension_subsets(subset, output / "vbench_dimension_info")

    prepared_sources: list[dict] = []
    for source in sources:
        seed_root = output / "inputs" / f"seed_{source['seed']}"
        prompt_records = source["prompts"]
        for record in prompt_records:
            prompt = normalize_prompt(record["prompt"])
            for cell in VIDEO_CELLS:
                link_asset(
                    Path(record["videos"][cell]),
                    seed_root / "branches" / cell / "videos" / vbench_video_name(prompt),
                    expected_frames=args.num_frames,
                )
            stem = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
            create_contact_sheet(
                record,
                output / "contact_sheets" / f"seed_{source['seed']}" / f"{record['index']:02d}_{stem}.jpg",
                expected_frames=args.num_frames,
            )
        prepared_sources.append(
            {
                "seed": source["seed"],
                "source_root": str(source["root"]),
                "source_summary": str(source["root"] / "summary.json"),
                "prompt_count": len(prompt_records),
                "mean_condition_metrics": source["summary"].get("mean_condition_metrics", {}),
            }
        )

    metadata = {
        "status": "ok",
        "protocol": (
            "Evaluation-only 2x2 factorial ablation. Existing videos are linked exactly as generated; "
            "the original per-seed image-condition distances are retained for bridge-alignment reporting."
        ),
        "vbench_info": str(args.vbench_info.resolve()),
        "subset_info": str(subset_path),
        "dimension_info_dir": str(output / "vbench_dimension_info"),
        "active_dimensions": active_dimensions,
        "prompt_count": len(canonical_prompts),
        "seeds": seeds,
        "video_cells": list(VIDEO_CELLS),
        "sources": prepared_sources,
    }
    (output / "prepared.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
