#!/usr/bin/env python
"""Export a bounded before/after visual audit, including rejected data, without models."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import html
import json
from pathlib import Path
import sys

import av
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.training.exp2_data import decode_selected, subset_frames  # noqa: E402
from new_mobile_ov.training.stage1_alignment_data import load_sources, read_alignment_sample, read_media_bytes  # noqa: E402
from tools.data_prepare.prepare_mobileov_exp2 import completed_results, load_view  # noqa: E402


def select_audit(release, per_group):
    contract = json.loads((release / "exp2_contract.json").read_text())
    groups = {}
    for phase in ("quality", "caption"):
        for value in completed_results(release, phase, contract[f"{phase}_workers"]):
            if phase == "quality" and value["status"] == "accepted":
                continue  # Its final caption verdict is included by the other stream.
            group = value["record"]["task"], value["status"]
            score = int(hashlib.sha256(value["record"]["sample_id"].encode()).hexdigest(), 16)
            heap = groups.setdefault(group, [])
            item = (-score, value["ordinal"], value)
            if len(heap) < per_group:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    return [v for group in sorted(groups) for _, _, v in sorted(groups[group], reverse=True)]


def source_frames(value, sources):
    record = value["record"]
    spec = value.get("transform")
    if not spec or spec["temporal"] is None:
        return subset_frames(read_alignment_sample(record, sources, num_video_frames=3)["target_frames"], 3)
    target = record["target"]
    width, height = target["width"], target["height"]
    identity = dict(source_size=[width, height], resized_size=[width, height],
                    crop_box=[0, 0, width, height], output_size=[width, height])
    return subset_frames(decode_selected(read_media_bytes(sources, target), spec["temporal"], identity), 3)


def render(release, output, per_group=16):
    if per_group < 1:
        raise ValueError("Audit group size must be positive")
    output.mkdir(parents=True, exist_ok=True)
    sources, cards = load_sources(release), []
    values = select_audit(release, per_group)
    for index, value in enumerate(values):
        before = source_frames(value, sources)
        exact = load_view(value, sources) if "transform" in value else []
        after = subset_frames(exact, 3)
        columns = max(len(before), len(after), 1)
        canvas = Image.new("RGB", (320 * columns, 230 * (2 if after else 1)), "#f5f1e8")
        draw = ImageDraw.Draw(canvas)
        for row, (label, frames) in enumerate((("Source view", before), ("Exact training crop/window", after))):
            if not frames:
                continue
            for column, image in enumerate(frames):
                tile = ImageOps.contain(image, (310, 200))
                canvas.paste(tile, (column * 320 + (320 - tile.width) // 2, row * 230 + 25 + (200 - tile.height) // 2))
            draw.text((8, row * 230 + 5), label, fill="#29251e")
        filename = f"sample_{index:04d}.jpg"
        canvas.save(output / filename, quality=90)
        player = ""
        if len(exact) == 49:
            video_name = f"sample_{index:04d}.mp4"
            with av.open(str(output / video_name), "w") as container:
                stream = container.add_stream("mpeg4", rate=24)
                stream.width, stream.height = exact[0].size
                stream.pix_fmt = "yuv420p"
                for image in exact:
                    for packet in stream.encode(av.VideoFrame.from_image(image)):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            player = f"<p>Exact 49-frame training view played at 24 fps:</p><video controls preload='none' src='{video_name}'></video>"
        annotation = value.get("annotation", {})
        caption = annotation.get("caption_long", "No accepted replacement caption")
        temporal = value.get("transform", {}).get("temporal")
        details = dict(status=value["status"], reasons=value.get("reasons", []),
                       caption_supported=annotation.get("caption_supported"),
                       motion_observation=annotation.get("motion_observation"),
                       confidence=annotation.get("confidence"),
                       window=[temporal["start_seconds"], temporal["end_seconds"]] if temporal else None,
                       quality=value.get("quality"))
        cards.append(f"<article><h2>{html.escape(value['record']['sample_id'])}</h2>"
                     f"<img src='{filename}'>{player}<p><b>Original:</b> {html.escape(value['record']['prompt'])}</p>"
                     f"<p><b>Replacement:</b> {html.escape(caption)}</p>"
                     f"<pre>{html.escape(json.dumps(details, indent=2))}</pre></article>")
    page = ("<!doctype html><meta charset='utf-8'><title>Exp-2 Data Audit</title>"
            "<style>body{max-width:1100px;margin:40px auto;font:17px Georgia;color:#29251e;background:#fff}"
            "article{padding:24px;margin:24px 0;border:1px solid #c9b789;border-radius:8px}"
            "img,video{max-width:100%}pre{white-space:pre-wrap;font-size:13px}h2{overflow-wrap:anywhere;font-size:18px}</style>"
            "<h1>Exp-2 Data Audit</h1><p>Accepted and rejected images/videos. Source and cropped views use the same "
            "selected timestamps when a window exists. Inspect caption validity, crop omissions, action speed, "
            "and false rejections. Automated model judgments are not ground truth.</p>" + "".join(cards))
    (output / "index.html").write_text(page)
    print(f"Rendered {len(cards)} samples: {output / 'index.html'}", flush=True)
    return len(cards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--per-group", type=int, default=16)
    args = parser.parse_args()
    render(args.release, args.output_dir or args.release / "audit", args.per_group)
