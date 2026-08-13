#!/usr/bin/env python
"""Prepare the final native-Qwen versus V8 image-bridge VBench comparison.

The source ablations generate exactly two cells per prompt: native Qwen3-VL
DreamLite conditioning and the V8 Mobile-OV image bridge.  Both cells use the
released NeoDragon text stack and Hybrid DiT.  This tool only validates and
links those videos into the filenames required by VBench; it never regenerates
an image or video.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from decord import VideoReader
from PIL import Image, ImageDraw


VARIANTS = ("native_qwen", "v8_imageonly")
NATIVE_TEXT = "native_neodragon"


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


def link_video(source: Path, destination: Path, expected_frames: int) -> None:
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


def load_source(root: Path) -> dict[str, object]:
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing ablation summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "ok":
        raise RuntimeError(f"Ablation did not complete: {summary_path}")
    protocol = summary.get("protocol", {})
    if protocol.get("image_conditions") != list(VARIANTS):
        raise RuntimeError(f"Source does not contain the required image cells: {summary_path}")
    if protocol.get("video_conditions") != [NATIVE_TEXT]:
        raise RuntimeError(
            "This final ablation requires native NeoDragon text conditioning only: "
            f"{summary_path}"
        )
    prompts = summary.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise RuntimeError(f"Source has no prompt records: {summary_path}")
    return {
        "root": root.resolve(),
        "summary": summary,
        "seed": int(summary["seed"]),
        "prompts": prompts,
    }


def load_vbench_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"Expected VBench metadata list: {path}")
    return [row for row in rows if isinstance(row, dict)]


def prompt_dimensions(rows: list[dict]) -> dict[str, list[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        prompt = normalize_prompt(row.get("prompt_en", ""))
        dimensions = row.get("dimension", [])
        if isinstance(dimensions, str):
            dimensions = [dimensions]
        if prompt and isinstance(dimensions, list):
            result[prompt].update(str(value) for value in dimensions)
    return {prompt: sorted(values) for prompt, values in result.items()}


def float_means(rows: list[dict[str, object]]) -> dict[str, float]:
    names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (float, int)) and not isinstance(value, bool)
        }
    )
    return {
        name: sum(float(row[name]) for row in rows if name in row) / len(rows)
        for name in names
        if any(name in row for row in rows)
    }


def thumbnail(image: Image.Image, width: int, height: int) -> Image.Image:
    value = image.convert("RGB")
    value.thumbnail((width, height))
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(value, ((width - value.width) // 2, (height - value.height) // 2))
    return canvas


def video_strip(path: Path, width: int, height: int) -> Image.Image:
    reader = VideoReader(str(path), num_threads=1)
    indices = (0, max(0, len(reader) // 2), max(0, len(reader) - 1))
    strip = Image.new("RGB", (width * len(indices), height), "white")
    for index, frame_index in enumerate(indices):
        frame = Image.fromarray(reader[frame_index].asnumpy())
        strip.paste(thumbnail(frame, width, height), (index * width, 0))
    return strip


def create_contact_sheet(record: dict, output: Path, expected_frames: int) -> None:
    anchors = record["anchors"]
    videos = record["videos"]
    native_anchor = Path(anchors["native_qwen"])
    student_anchor = Path(anchors["v8_imageonly"])
    native_video = Path(videos[f"image_native_qwen__text_{NATIVE_TEXT}"])
    student_video = Path(videos[f"image_v8_imageonly__text_{NATIVE_TEXT}"])
    for path in (native_anchor, student_anchor):
        if not valid_image(path):
            raise RuntimeError(f"Invalid source anchor for contact sheet: {path}")
    for path in (native_video, student_video):
        if not valid_video(path, expected_frames):
            raise RuntimeError(f"Invalid source video for contact sheet: {path}")

    frame_width, frame_height = 128, 80
    video_width = frame_width * 3
    header_height = 60
    row_height = 122
    canvas = Image.new("RGB", (video_width * 2, header_height + row_height * 2), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), record["prompt"], fill="black")
    draw.text((6, 25), "Top: DreamLite anchor. Bottom: first / middle / last video frames.", fill="black")
    for column, (label, anchor, video) in enumerate(
        (
            ("native Qwen3-VL", native_anchor, native_video),
            ("V8 image bridge", student_anchor, student_video),
        )
    ):
        x = column * video_width
        draw.text((x + 4, header_height), label, fill="black")
        with Image.open(anchor) as image:
            canvas.paste(thumbnail(image, video_width, frame_height), (x, header_height + 18))
        draw.text((x + 4, header_height + row_height), label + " video", fill="black")
        canvas.paste(video_strip(video, frame_width, frame_height), (x, header_height + row_height + 18))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-roots", required=True, nargs="+", type=Path)
    parser.add_argument("--vbench-info", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--contact-sheet-count", type=int, default=12)
    args = parser.parse_args()

    if args.contact_sheet_count < 0:
        raise ValueError("--contact-sheet-count must be non-negative")
    sources = [load_source(path) for path in args.run_roots]
    seeds = [int(source["seed"]) for source in sources]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Each source run must use a distinct seed, received {seeds}")

    records_by_seed = {int(source["seed"]): source["prompts"] for source in sources}
    canonical = [normalize_prompt(record["prompt"]) for record in records_by_seed[seeds[0]]]
    if not canonical or len(canonical) != len(set(canonical)):
        raise RuntimeError("Source prompts must be unique and non-empty.")
    for seed in seeds[1:]:
        candidate = [normalize_prompt(record["prompt"]) for record in records_by_seed[seed]]
        if candidate != canonical:
            raise RuntimeError("All source runs must use the same ordered prompt list.")

    vbench_rows = load_vbench_rows(args.vbench_info)
    dimensions = prompt_dimensions(vbench_rows)
    missing = [prompt for prompt in canonical if prompt not in dimensions]
    if missing:
        raise RuntimeError(f"Prompt absent from VBench metadata: {missing[0]!r}")
    canonical_set = set(canonical)
    subset = [row for row in vbench_rows if normalize_prompt(row.get("prompt_en", "")) in canonical_set]

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    subset_path = output / f"vbench_stratified_{len(canonical)}_info.json"
    subset_path.write_text(json.dumps(subset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    sources_payload: list[dict[str, object]] = []
    for source in sources:
        seed = int(source["seed"])
        records = source["prompts"]
        for variant in VARIANTS:
            cell = f"image_{variant}__text_{NATIVE_TEXT}"
            for record in records:
                prompt = normalize_prompt(record["prompt"])
                source_video = Path(record["videos"][cell])
                destination = output / variant / f"seed_{seed}" / "videos" / f"{prompt}-0.mp4"
                link_video(source_video, destination, args.num_frames)
        metric_rows = [record["condition_metrics_v8_vs_native"] for record in records]
        sources_payload.append(
            {
                "seed": seed,
                "source_root": str(source["root"]),
                "source_summary": str(Path(source["root"]) / "summary.json"),
                "mean_condition_metrics": float_means(metric_rows),
            }
        )

    first_records = records_by_seed[seeds[0]]
    ranked = sorted(
        first_records,
        key=lambda row: float(row["condition_metrics_v8_vs_native"].get("content_token_cosine_distance", 0.0)),
        reverse=True,
    )
    contact_sheets: list[str] = []
    for index, record in enumerate(ranked[: args.contact_sheet_count], start=1):
        destination = output / "contact_sheets" / f"{index:02d}.jpg"
        create_contact_sheet(record, destination, args.num_frames)
        contact_sheets.append(str(destination))

    by_dimension: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in first_records:
        prompt = normalize_prompt(record["prompt"])
        for dimension in dimensions[prompt]:
            by_dimension[dimension].append(record["condition_metrics_v8_vs_native"])
    condition_by_dimension = {
        dimension: {"prompt_count": len(values), **float_means(values)}
        for dimension, values in sorted(by_dimension.items())
    }
    prepared = {
        "status": "ok",
        "protocol": {
            "comparison": "Native Qwen3-VL DreamLite condition versus V8 image bridge only.",
            "fixed_components": [
                "raw VBench prompt",
                "DreamLite generator and VAE",
                "DreamLite seed",
                "released native NeoDragon ContextAdapter",
                "released NeoDragon Hybrid DiT",
                "NeoDragon seed",
                "image and video render settings",
            ],
            "important_limit": (
                "This is a stratified 100-prompt comparative subset with three paired seeds, not the "
                "canonical 944-prompt VBench result."
            ),
        },
        "prompt_count": len(canonical),
        "seeds": seeds,
        "variants": list(VARIANTS),
        "vbench_source": str(args.vbench_info.resolve()),
        "vbench_subset": str(subset_path),
        "sources": sources_payload,
        "condition_metrics_by_vbench_dimension": condition_by_dimension,
        "highest_content_condition_mismatch_prompts": [
            {
                "prompt": record["prompt"],
                "metrics": record["condition_metrics_v8_vs_native"],
            }
            for record in ranked[: args.contact_sheet_count]
        ],
        "contact_sheets": contact_sheets,
    }
    destination = output / "prepared.json"
    destination.write_text(json.dumps(prepared, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(prepared, indent=2), flush=True)


if __name__ == "__main__":
    main()
