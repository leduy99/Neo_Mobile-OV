"""Versioned, deterministic Exp-2 views of existing raw media. No model imports."""
from __future__ import annotations

import bisect
import hashlib
import io
import json
import math

import av
import cv2
import numpy as np
from PIL import Image

FORMAT = "mobileov_exp2_data_v1"


class RejectedSample(ValueError):
    """A data-policy rejection, not a worker/runtime failure."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def fraction(key, seed, purpose):
    return int(hashlib.sha256(f"{seed}:{purpose}:{key}".encode()).hexdigest()[:16], 16) / 2**64


def validate_policy(policy):
    if policy["format"] != FORMAT or policy["video_frames"] != 49 or policy["video_fps"] != 24:
        raise ValueError("Exp-2 requires the versioned 49-frame/24-fps policy")
    if policy.get("caption_schema") != "visual_only_single_pass_v2":
        raise ValueError("Exp-2 requires visual-only, single-pass captioning")
    for key in ("min_source_fps", "max_timestamp_error_seconds", "scene_adaptive_threshold",
                "scene_min_content_val", "blur_laplacian_min"):
        if not math.isfinite(policy[key]) or policy[key] <= 0:
            raise ValueError(f"Invalid {key}")
    for key in ("short_caption_fraction", "min_unique_frame_fraction", "bad_frame_fraction",
                "caption_min_confidence"):
        if not 0 <= policy[key] <= 1:
            raise ValueError(f"Invalid {key}")
    if (policy["scene_guard_seconds"] < 0 or policy["quality_frames"] < 2
            or not 2 <= policy["caption_frames"] <= 49):
        raise ValueError("Invalid sampling policy")
    if policy["buckets"] != [[512, 320], [320, 512], [384, 384]]:
        raise ValueError("Exp-2 buckets must match the current generator")
    for key in ("min_musiq", "min_aesthetic", "max_motion_pixels", "siglip_min_logit"):
        value = policy[key]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError(f"Invalid threshold {key}")
    for key in ("qwen_model", "qwen_revision", "siglip_model", "siglip_revision"):
        if not isinstance(policy[key], str) or not policy[key].strip():
            raise ValueError(f"Invalid model identifier {key}")
    return policy


def spatial_spec(width, height, policy, key, split):
    if min(width, height) < 1:
        raise RejectedSample("invalid_dimensions")
    tw, th = min(policy["buckets"], key=lambda s: abs(math.log(width / height / (s[0] / s[1]))))
    scale = max(tw / width, th / height)
    rw, rh = max(tw, math.ceil(width * scale)), max(th, math.ceil(height * scale))
    # A fixed random training crop is caption-checked once, then identical at every epoch.
    fx = fraction(key, policy["seed"], "crop_x") if split == "train" else 0.5
    fy = fraction(key, policy["seed"], "crop_y") if split == "train" else 0.5
    x, y = round((rw - tw) * fx), round((rh - th) * fy)
    return dict(source_size=[width, height], resized_size=[rw, rh],
                crop_box=[x, y, x + tw, y + th], output_size=[tw, th],
                retained_area_fraction=(tw * th) / (rw * rh),
                crop_policy="fixed_seeded_train_crop" if split == "train" else "center_crop")


def apply_spatial(image, spec):
    orientation = spec.get("exif_orientation", 1)
    if "raw_source_size" in spec:
        if list(image.size) != spec["raw_source_size"]:
            raise ValueError("Source geometry changed after data preparation")
        methods = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
                   4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
                   6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE, 8: Image.Transpose.ROTATE_90}
        if orientation in methods:
            image = image.transpose(methods[orientation])
    if list(image.size) != spec["source_size"]:
        raise ValueError("Source geometry changed after data preparation")
    rw, rh = spec["resized_size"]
    x, y, x2, y2 = spec["crop_box"]
    if (min(rw, rh) < 1 or not (0 <= x < x2 <= rw and 0 <= y < y2 <= rh)
            or [x2 - x, y2 - y] != spec["output_size"]):
        raise ValueError("Invalid Exp-2 spatial transform")
    result = image.resize((rw, rh), Image.Resampling.LANCZOS).crop((x, y, x2, y2))
    # The VLM processor must not apply the original orientation a second time.
    result.info.pop("exif", None)
    result.getexif().clear()
    return result


def choose_window(times, cuts, policy, key, split):
    if len(times) < 2 or any(not math.isfinite(t) for t in times):
        raise RejectedSample("invalid_timestamps")
    intervals = np.diff(times)
    if np.any(intervals <= 0):
        raise RejectedSample("non_monotonic_timestamps")
    actual_fps = 1 / float(np.median(intervals))
    if actual_fps < policy["min_source_fps"] - 1e-6:
        raise RejectedSample("source_fps_below_policy")
    span = (policy["video_frames"] - 1) / policy["video_fps"]
    boundaries = [0, *sorted(set(int(i) for i in cuts if 0 < i < len(times))), len(times)]
    options = []
    for begin, end in zip(boundaries, boundaries[1:]):
        left = times[begin] + (policy["scene_guard_seconds"] if begin else 0)
        right = times[end - 1] - (policy["scene_guard_seconds"] if end < len(times) else 0) - span
        if right >= left - 1e-6:
            options.append((begin, end, left, max(left, right)))
    if not options:
        raise RejectedSample("no_two_second_single_shot_window")
    f = fraction(key, policy["seed"], "window") if split == "train" else 0.5
    # Weight by available duration, rather than favoring short shots equally.
    weights = [max(right - left, 1 / policy["video_fps"]) for _, _, left, right in options]
    cursor = f * sum(weights)
    for option, weight in zip(options, weights):
        if cursor <= weight:
            break
        cursor -= weight
    begin, end, left, right = option
    start = left + (right - left) * min(cursor / weight, 1)
    starts = [t for t in times[begin:end] if left - 1e-6 <= t <= right + 1e-6]
    if not starts:
        raise RejectedSample("no_frame_aligned_single_shot_window")
    # Anchor the sampling grid to a real source timestamp. A half-frame offset plus
    # floating-point nearest-neighbor ties can otherwise duplicate 24-fps frames.
    start = min(starts, key=lambda t: abs(t - start))
    desired = [start + i / policy["video_fps"] for i in range(policy["video_frames"])]
    indices = []
    for t in desired:
        pos = bisect.bisect_left(times, t, begin, end)
        choices = [i for i in (pos - 1, pos) if begin <= i < end]
        indices.append(min(choices, key=lambda i: abs(times[i] - t)))
    error = max(abs(times[i] - t) for i, t in zip(indices, desired))
    if error > policy["max_timestamp_error_seconds"]:
        raise RejectedSample("timestamp_resampling_error")
    if len(set(indices)) / len(indices) < policy["min_unique_frame_fraction"]:
        raise RejectedSample("too_many_repeated_frames")
    return dict(start_seconds=start, end_seconds=start + span, fps=policy["video_fps"],
                frame_indices=indices, source_frame_times=[times[i] for i in indices],
                grid_times=desired, measured_source_fps=actual_fps, max_timestamp_error_seconds=error,
                unique_frames=len(set(indices)), source_frames=len(times), scene_bounds=[begin, end])


def video_probe(payload, policy):
    from scenedetect.detectors import AdaptiveDetector

    detector = AdaptiveDetector(adaptive_threshold=policy["scene_adaptive_threshold"],
                                min_scene_len=1, min_content_val=policy["scene_min_content_val"])
    times, cuts, size = [], [], None
    with av.open(io.BytesIO(payload), metadata_errors="replace") as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for index, frame in enumerate(container.decode(stream)):
            if frame.time is None:
                raise RejectedSample("missing_frame_timestamp")
            current_size = [frame.width, frame.height]
            if size is not None and size != current_size:
                raise RejectedSample("changing_frame_geometry")
            size = current_size
            times.append(float(frame.time))
            # Analyze every source frame, not just the sparse frames sent to the VLM.
            w, h = current_size
            thumb = frame.reformat(width=256, height=max(1, round(h * 256 / w))).to_ndarray(format="bgr24")
            cuts.extend(detector.process_frame(index, thumb))
    cuts.extend(detector.post_process(len(times)))
    return times, cuts, size


def decode_selected(payload, temporal, spatial):
    chosen = set(temporal["frame_indices"])
    images, observed_times = {}, {}
    with av.open(io.BytesIO(payload), metadata_errors="replace") as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for index, frame in enumerate(container.decode(stream)):
            if index in chosen:
                images[index] = apply_spatial(frame.to_image().convert("RGB"), spatial)
                observed_times[index] = float(frame.time) if frame.time is not None else None
            if index >= max(chosen):
                break
    if set(images) != chosen:
        raise ValueError("Prepared video frames no longer decode")
    for index, timestamp in zip(temporal["frame_indices"], temporal["source_frame_times"]):
        if observed_times[index] is None or abs(observed_times[index] - timestamp) > 1e-5:
            raise ValueError("Prepared video timestamps changed")
    return [images[i] for i in temporal["frame_indices"]]


def make_view(record, payload, policy, split):
    target = record["target"]
    if record["task"] == "t2i":
        with Image.open(io.BytesIO(payload)) as image:
            orientation = image.getexif().get(274, 1)
            if orientation not in range(1, 9):
                raise RejectedSample("invalid_exif_orientation")
            image = image.convert("RGB")
        size = list(image.size)
        temporal = None
    else:
        times, cuts, size = video_probe(payload, policy)
        if len(times) != target["frame_count"]:
            raise RejectedSample("source_frame_count_changed")
        temporal = choose_window(times, cuts, policy, record["sample_id"], split)
    if size != [target["width"], target["height"]]:
        raise RejectedSample("source_geometry_changed")
    oriented_size = size[::-1] if temporal is None and orientation in (5, 6, 7, 8) else size
    spatial = spatial_spec(*oriented_size, policy, record["sample_id"], split)
    if temporal is None:
        spatial.update(raw_source_size=size, exif_orientation=orientation)
    spec = dict(format=FORMAT, spatial=spatial, temporal=temporal)
    frames = [apply_spatial(image, spatial)] if temporal is None else decode_selected(payload, temporal, spatial)
    return spec, frames


def quality_stats(frames):
    values = {"laplacian": [], "black_fraction": [], "white_fraction": []}
    for image in frames:
        rgb = np.asarray(image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        values["laplacian"].append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
        values["black_fraction"].append(float((gray < 5).mean()))
        values["white_fraction"].append(float((gray > 250).mean()))
    return values


def cheap_rejections(stats, policy):
    reasons = []
    if np.mean(np.asarray(stats["laplacian"]) < policy["blur_laplacian_min"]) >= policy["bad_frame_fraction"]:
        reasons.append("severe_blur")
    if max(np.mean(np.asarray(stats[k]) > 0.98) for k in ("black_fraction", "white_fraction")) >= policy["bad_frame_fraction"]:
        reasons.append("blank_or_saturated")
    return reasons


def subset_frames(frames, count):
    return [frames[i] for i in np.linspace(0, len(frames) - 1, min(count, len(frames))).round().astype(int)]


def validate_caption(result, policy, *, video=None):
    for key in ("usable", "caption_supported"):
        if not isinstance(result.get(key), bool):
            raise ValueError(f"Missing boolean {key}")
    confidence = result.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid caption confidence")
    needs_caption = result["usable"] and result["caption_supported"]
    for key, low, high in (("caption_short", 2, 20), ("caption_long", 5, 120)):
        text = result.get(key)
        if not isinstance(text, str) or not (low if needs_caption else 0) <= len(text.split()) <= high:
            raise ValueError(f"Invalid {key}")
    if result.get("text_overlay") not in {"none", "minor", "obstructive"}:
        raise ValueError("Invalid text_overlay assessment")
    if not isinstance(result.get("reason"), str):
        raise ValueError("Missing explanation")
    motion = result.get("motion_observation")
    allowed = {"static", "object_motion", "camera_motion", "both", "unclear", "not_applicable"}
    if motion not in allowed or (video is True and motion == "not_applicable") or (video is False and motion != "not_applicable"):
        raise ValueError("Invalid motion_observation")
    accepted = (needs_caption and confidence >= policy["caption_min_confidence"]
                and result["text_overlay"] != "obstructive")
    return dict(result, accepted=accepted)


def caption_instruction(*, video):
    medium = "chronologically ordered frames from ONE two-second video window" if video else "ONE cropped image"
    motion = ("Classify visible motion as static, object_motion, camera_motion, both, or unclear. "
              "Describe motion only when it is supported across the ordered frames. "
              "A static scene can still be useful; do not invent action or reject it solely for being static."
              if video else "Set motion_observation to not_applicable. Do not infer actions from a single image.")
    return f"""Inspect ONLY the supplied {medium} and write captions from visual evidence.
You are not being asked to select a different time window or reconstruct a story outside this view.
Preserve visible identity, count, colors, layout, scene, actions and style. Do not assume objects
outside the crop, earlier/later events, hidden facts, sounds, or causal stories.
{motion}
Short caption: 2-20 words. Long caption: 5-100 words in one or two concise factual sentences;
do not add details just to reach a length. No 'the video shows' boilerplate or numerical motion scores.
Assess whether the view is usable and your captions are visually supported in this SAME response.
Flag obstructive subtitles/watermarks/overlays; ordinary scene text or a small mark is not enough to reject.
Return one strict JSON object only, with exactly these fields:
- usable: boolean; false if the view is genuinely ambiguous or unusable, not merely static.
- caption_supported: boolean; whether your generated captions are supported by the supplied view.
- confidence: number from 0 to 1 for that judgment; do not default to high confidence.
- caption_short and caption_long: strings; may be empty when rejecting an unusable view.
- motion_observation: the classification requested above.
- text_overlay: none, minor, or obstructive.
- reason: a short explanation, at most 30 words.
Do not invent content to make an unusable view acceptable."""


def read_exp2_sample(record, sources, *, num_video_frames=49):
    from new_mobile_ov.training.stage1_alignment_data import read_media_bytes

    spec = record["exp2_transform"]
    if spec["format"] != FORMAT or record.get("exp2_view_sha256") != digest(dict(target=record["target"], transform=spec)):
        raise ValueError("Invalid or modified Exp-2 view")
    payload = read_media_bytes(sources, record["target"])
    temporal = spec["temporal"]
    if temporal is None:
        if record["task"] not in {"t2i", "image_reconstruction"}:
            raise ValueError("Image view with a video task")
        with Image.open(io.BytesIO(payload)) as image:
            frames = [apply_spatial(image.convert("RGB"), spec["spatial"])]
        indices, times = [0], [0.0]
    else:
        if (record["task"] != "t2v" or num_video_frames != 49 or len(temporal["frame_indices"]) != 49
                or temporal["fps"] != 24 or abs(temporal["grid_times"][-1] - temporal["grid_times"][0] - 2) > 1e-6):
            raise ValueError("Invalid fixed-FPS Exp-2 video view")
        frames = decode_selected(payload, temporal, spec["spatial"])
        indices, times = temporal["frame_indices"], temporal["grid_times"]
    return dict(task=record["task"], prompt=record["prompt"], target_frames=frames,
                conditioning_images=[frames[0].copy()] if record["task"] == "image_reconstruction" else [],
                conditioning_route="mllm_only", frame_indices=indices, frame_times=times,
                source_duration_seconds=record["target"].get("duration_seconds"),
                effective_sample_fps=24.0 if temporal else None, preprocessing_format=FORMAT)
