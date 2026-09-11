"""CPU readers for Stage-1 raw media manifests (no VAE or trainer assumptions)."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath

import av
from PIL import Image


def read_media_bytes(sources: dict[str, Path], reference: dict) -> bytes:
    root = sources[reference["source"]].resolve()
    relative = PurePosixPath(reference["shard"])
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
        raise ValueError("Unsafe media reference")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Media reference escapes source root")
    offset, size = reference["offset"], reference["size"]
    if not isinstance(offset, int) or not isinstance(size, int) or offset < 0 or not 0 < size <= 256 * 1024**2:
        raise ValueError("Invalid media offset/size")
    with path.open("rb") as stream:
        stream.seek(offset)
        payload = stream.read(size)
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise RuntimeError("Media checksum mismatch")
    return payload


def load_sources(manifest_root: Path) -> dict[str, Path]:
    config = json.loads((manifest_root / "sources.json").read_text())
    return {key: (manifest_root / value["root"]).resolve() for key, value in config.items()}


def read_alignment_sample(record: dict, sources: dict[str, Path], *, num_video_frames: int = 49) -> dict:
    """Return PIL targets and VLM inputs; images are not clean DiT conditioning latents."""
    task, target = record["task"], record["target"]
    payload = read_media_bytes(sources, target)
    if task in {"t2i", "image_reconstruction"}:
        if target["kind"] != "image":
            raise ValueError("Image task requires an image target")
        with Image.open(io.BytesIO(payload)) as image:
            image = image.convert("RGB")
        if image.size != (target["width"], target["height"]):
            raise ValueError("Image shape differs from manifest")
        return dict(task=task, prompt=record["prompt"], target_frames=[image],
                    conditioning_images=[image.copy()] if task == "image_reconstruction" else [],
                    conditioning_route="mllm_only", frame_indices=[0])
    if task != "t2v" or target["kind"] != "video":
        raise ValueError(f"Unsupported Stage-1 task: {task}")
    count = target["frame_count"]
    if not 2 <= num_video_frames <= count:
        raise ValueError("T2V must sample distinct frames; do not repeat a still image")
    indices = [(index * (count - 1)) // (num_video_frames - 1) for index in range(num_video_frames)]
    selected, frames, times, decoded = set(indices), [], [], 0
    with av.open(io.BytesIO(payload)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for index, frame in enumerate(container.decode(stream)):
            if index in selected:
                frames.append(frame.to_image().convert("RGB"))
                times.append(float(frame.time) if frame.time is not None else index / target["fps"])
            decoded += 1
    if decoded != count or len(frames) != num_video_frames:
        raise ValueError("Decoded video length differs from manifest")
    if any(frame.size != (target["width"], target["height"]) for frame in frames):
        raise ValueError("Video shape differs from manifest")
    return dict(task=task, prompt=record["prompt"], target_frames=frames, conditioning_images=[],
                conditioning_route="mllm_only", frame_indices=indices, frame_times=times,
                source_duration_seconds=target["duration_seconds"],
                effective_sample_fps=(num_video_frames - 1) / (times[-1] - times[0]))
