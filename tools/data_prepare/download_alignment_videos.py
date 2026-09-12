#!/usr/bin/env python
"""Download a pinned public Vchitect T2V subset; no OpenVid or recaptioning."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import tarfile

import av
from huggingface_hub import HfApi

from tools.data_prepare.download_alignment_images import (
    Shard, atomic_json, download_shard, output_lock, safe_member, sha256_file,
)

REPO_ID = "Vchitect/Vchitect_T2V_DataVerse"
INDEX_VERSION = 1


class RejectedVideo(ValueError):
    pass


def select_shards(shards: list[Shard], count: int, seed: int) -> list[Shard]:
    if not 1 <= count <= len(shards):
        raise ValueError(f"Requested {count} video shards; available={len(shards)}")
    groups = defaultdict(list)
    for item in shards:
        if not safe_member(item.path) or not item.sha256 or item.size <= 0:
            raise ValueError(f"Invalid video archive metadata: {item.path}")
        groups[str(PurePosixPath(item.path).parent)].append(item)
    quotas = {key: count * len(items) // len(shards) for key, items in groups.items()}
    order = sorted(groups, key=lambda key: (-(count * len(groups[key]) % len(shards)), key))
    for key in order[:count - sum(quotas.values())]:
        quotas[key] += 1
    selected = []
    for key in sorted(groups):
        ranked = sorted(groups[key], key=lambda item: hashlib.sha256(f"{seed}:{item.path}".encode()).digest())
        selected.extend(ranked[:quotas[key]])
    return sorted(selected, key=lambda item: item.path)


def resolve_plan(output: Path, *, revision: str, count: int, seed: int) -> dict:
    settings = dict(repo_id=REPO_ID, requested_revision=revision, num_shards=count, seed=seed)
    path = output / "download_plan.json"
    if path.exists():
        plan = json.loads(path.read_text())
        if plan.get("settings") != settings:
            raise ValueError("Video plan differs. Use a new output directory instead of mixing releases.")
        return plan
    api = HfApi()
    pinned = api.dataset_info(REPO_ID, revision=revision).sha
    if not pinned:
        raise RuntimeError("Missing dataset commit")
    candidates, annotation = [], None
    for item in api.list_repo_tree(REPO_ID, repo_type="dataset", revision=pinned, recursive=True):
        if not (item.path.endswith(".tar") or item.path == "annotation.json"):
            continue
        checksum = getattr(getattr(item, "lfs", None), "sha256", "")
        if len(checksum) != 64:
            raise ValueError(f"Missing LFS SHA256: {item.path}")
        spec = Shard(item.path, item.size, checksum)
        if item.path == "annotation.json":
            annotation = spec
        else:
            candidates.append(spec)
    if annotation is None:
        raise RuntimeError("Vchitect annotation.json is missing")
    chosen = select_shards(candidates, count, seed)
    return dict(settings=settings, revision=pinned, available_shards=len(candidates),
                selection="seeded proportional sampling by archive directory, not semantic balancing",
                selected_groups=dict(Counter(str(PurePosixPath(item.path).parent) for item in chosen)),
                annotation=asdict(annotation), shards=[asdict(item) for item in chosen],
                expected_bytes=annotation.size + sum(item.size for item in chosen))


def load_captions(path: Path) -> dict[str, str]:
    # The released ~1GB JSON fits the 64GB preparation job. Load it once, not per worker.
    with path.open(encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise ValueError("Expected Vchitect annotation.json to contain a list")
    captions = {}
    for row in rows:
        name, text = row.get("video"), row.get("text")
        if not isinstance(name, str) or not safe_member(name) or PurePosixPath(name).name != name:
            raise ValueError("Unexpected annotation video key")
        if name in captions:
            raise ValueError(f"Duplicate annotation key: {name}")
        if not isinstance(text, str) or not text.strip() or "\x00" in text:
            raise ValueError(f"Invalid caption: {name}")
        captions[name] = text.strip()
    if not captions:
        raise ValueError("Empty video annotations")
    return captions


def probe_video(payload: bytes, *, min_seconds: float, max_seconds: float,
                min_side: int, min_frames: int) -> dict:
    # Legacy MP4 tags may not be UTF-8. These unused tags are not our annotation captions;
    # tolerate their encoding only, while keeping frame decoding and validation strict.
    with av.open(io.BytesIO(payload), metadata_errors="replace") as container:
        if len(container.streams.video) != 1:
            raise RejectedVideo("not_one_video_stream")
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        width, height = stream.codec_context.width, stream.codec_context.height
        if min(width, height) < min_side:
            raise RejectedVideo("low_resolution")
        fps = float(stream.average_rate or 0)
        if not math.isfinite(fps) or fps <= 0:
            raise RejectedVideo("invalid_fps")
        declared_duration = float(stream.duration * stream.time_base) if stream.duration is not None else None
        if declared_duration is not None and declared_duration > max_seconds + 1:
            raise RejectedVideo("too_long")
        count, first_time, last_time = 0, None, None
        for frame in container.decode(stream):
            if frame.width != width or frame.height != height:
                raise RejectedVideo("changing_resolution")
            timestamp = float(frame.time) if frame.time is not None else count / fps
            if not math.isfinite(timestamp) or (last_time is not None and timestamp <= last_time):
                raise RejectedVideo("invalid_timestamps")
            first_time = timestamp if first_time is None else first_time
            last_time = timestamp
            count += 1
            if count > 4096 or timestamp - first_time > max_seconds + 1:
                raise RejectedVideo("too_long")
        if count < min_frames:
            raise RejectedVideo("too_few_frames")
        if stream.frames and count != stream.frames:
            raise RejectedVideo("frame_count_mismatch")
        duration = last_time - first_time + 1 / fps
        if not min_seconds <= duration <= max_seconds:
            raise RejectedVideo("duration_out_of_range")
        return dict(width=width, height=height, frame_count=count, fps=fps,
                    duration_seconds=duration, first_frame_time=first_time)


def index_shard(archive: Path, shard: Shard, output: Path, captions: dict[str, str],
                *, revision: str, annotation_sha256: str, policy: dict) -> dict:
    index_path = output / "indexes" / f"{shard.path}.jsonl"
    receipt_path = index_path.with_suffix(".json")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    contract = dict(index_version=INDEX_VERSION, archive_sha256=shard.sha256,
                    annotation_sha256=annotation_sha256, revision=revision, policy=policy)
    if receipt_path.exists() and index_path.exists():
        try:
            receipt = json.loads(receipt_path.read_text())
            if (all(receipt.get(key) == value for key, value in contract.items())
                    and receipt.get("index_sha256") == sha256_file(index_path)):
                return receipt
        except (ValueError, OSError):
            pass
    temporary = index_path.with_suffix(".jsonl.tmp")
    rejected, errors, examples, seen = Counter(), Counter(), [], set()
    valid = 0
    with tarfile.open(archive, "r:") as tar, temporary.open("w", encoding="utf-8") as target:
        for member in tar:
            if not member.isfile():
                continue
            if not safe_member(member.name):
                raise ValueError(f"Unsafe archive member: {member.name}")
            name = PurePosixPath(member.name).name
            if not name.endswith(".mp4"):
                continue
            if name in seen:
                raise ValueError(f"Duplicate video key: {name}")
            seen.add(name)
            if name not in captions:
                errors["missing_caption"] += 1
                continue
            if not 0 < member.size <= 256 * 1024**2:
                rejected["oversized_or_empty"] += 1
                continue
            with tar.extractfile(member) as source:
                payload = source.read()
            try:
                info = probe_video(payload, **policy)
            except RejectedVideo as error:
                rejected[str(error)] += 1
                continue
            except av.error.FFmpegError as error:
                errors["decode_error"] += 1
                if len(examples) < 5:
                    examples.append(dict(video=name, error=str(error)))
                continue
            record = dict(sample_id=hashlib.sha256(f"{REPO_ID}@{revision}:{name}".encode()).hexdigest(),
                          key=name, shard=f"shards/{shard.path}", video_member=member.name,
                          video_offset=member.offset_data, video_size=len(payload),
                          video_sha256=hashlib.sha256(payload).hexdigest(), caption=captions[name], **info)
            target.write(json.dumps(record, ensure_ascii=True) + "\n")
            valid += 1
    if not seen:
        raise RuntimeError(f"No MP4 members: {shard.path}")
    temporary.replace(index_path)
    receipt = dict(**contract, shard=shard.path, videos_seen=len(seen), valid_pairs=valid,
                   rejected=dict(rejected), errors=dict(errors), error_examples=examples,
                   video_decode_check="all_frames_of_accepted_videos",
                   index=str(index_path.relative_to(output)), index_sha256=sha256_file(index_path))
    atomic_json(receipt_path, receipt)
    return receipt


def run(args: argparse.Namespace) -> dict:
    if min(args.num_shards, args.workers, args.retries, args.min_side, args.min_frames) < 1:
        raise ValueError("Counts must be positive")
    if not 0 < args.min_seconds <= args.max_seconds or not 0 <= args.max_error_fraction < 1:
        raise ValueError("Invalid duration/error limits")
    if not math.isfinite(args.max_download_gib) or args.max_download_gib <= 0 or not math.isfinite(args.disk_margin_gib) or args.disk_margin_gib < 0:
        raise ValueError("Invalid storage limits")
    output = args.output_dir.expanduser().resolve()
    with output_lock(output):
        plan = resolve_plan(output, revision=args.revision, count=args.num_shards, seed=args.seed)
        if plan["expected_bytes"] > args.max_download_gib * 1024**3:
            raise RuntimeError("Video selection exceeds --max-download-gib")
        print(json.dumps({key: value for key, value in plan.items() if key != "shards"}, indent=2), flush=True)
        if args.dry_run:
            return plan
        (output / ".download_complete").unlink(missing_ok=True)
        atomic_json(output / "download_plan.json", plan)
        shards = [Shard(**item) for item in plan["shards"]]
        annotation = Shard(**plan["annotation"])
        files = [annotation, *shards]
        pending = sum(max(0, item.size - ((output / "shards" / item.path).stat().st_size
                      if (output / "shards" / item.path).is_file() else 0)) for item in files)
        reserve = sum(sorted((item.size for item in files), reverse=True)[:args.workers])
        if shutil.disk_usage(output).free < pending + reserve + args.disk_margin_gib * 1024**3:
            raise RuntimeError("Insufficient disk space for video subset plus replacement/index margin")
        metadata = download_shard(annotation, output=output, repo_id=REPO_ID,
                                  revision=plan["revision"], retries=args.retries)
        captions = load_captions(metadata)
        print(f"Loaded {len(captions)} native video captions; no caption generation.", flush=True)
        policy = dict(min_seconds=args.min_seconds, max_seconds=args.max_seconds,
                      min_side=args.min_side, min_frames=args.min_frames)

        def process(shard):
            archive = download_shard(shard, output=output, repo_id=REPO_ID,
                                     revision=plan["revision"], retries=args.retries)
            return index_shard(archive, shard, output, captions, revision=plan["revision"],
                               annotation_sha256=annotation.sha256, policy=policy)

        results, failures = {}, []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process, item): item.path for item in shards}
            for number, future in enumerate(as_completed(futures), 1):
                name = futures[future]
                try:
                    receipt = future.result()
                    results[name] = receipt
                    print(f"[{number}/{len(shards)}] {name}: accepted={receipt['valid_pairs']} "
                          f"filtered={receipt['rejected']} errors={receipt['errors']}", flush=True)
                except Exception as error:
                    failures.append(dict(shard=name, error=repr(error)))
                    print(f"FAILED {name}: {error}", flush=True)
        rejected, errors = Counter(), Counter()
        for result in results.values():
            rejected.update(result["rejected"])
            errors.update(result["errors"])
        seen = sum(item["videos_seen"] for item in results.values())
        valid = sum(item["valid_pairs"] for item in results.values())
        if not valid or sum(errors.values()) / max(1, seen) > args.max_error_fraction:
            failures.append(dict(error="No accepted videos or excessive caption/decode errors"))
        summary = dict(status="incomplete" if failures else "complete", repo_id=REPO_ID,
                       revision=plan["revision"], annotation_sha256=annotation.sha256,
                       archive_bytes=plan["expected_bytes"], shards_requested=len(shards),
                       shards_verified=len(results), videos_seen=seen, valid_video_caption_pairs=valid,
                       rejected=dict(rejected), errors=dict(errors), failures=failures, policy=policy,
                       video_decode_check="all_frames_of_accepted_videos", semantics_verified=False,
                       temporal_policy="whole_source_clip_with_original_caption; no temporal cropping",
                       purpose="public_T2V_alignment_substitute_not_original_UniVideo_data")
        if not failures:
            temporary = output / "samples.jsonl.tmp"
            with temporary.open("wb") as merged:
                for name in sorted(results):
                    with (output / results[name]["index"]).open("rb") as source:
                        shutil.copyfileobj(source, merged)
            temporary.replace(output / "samples.jsonl")
            summary["samples_sha256"] = sha256_file(output / "samples.jsonl")
        atomic_json(output / "download_summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        if failures:
            raise RuntimeError("Video preparation incomplete. Fix errors and resubmit unchanged to resume.")
        atomic_json(output / ".download_complete", summary)
        return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("download_data/data/univideo_alignment_videos"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--num-shards", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--max-download-gib", type=float, default=450)
    parser.add_argument("--disk-margin-gib", type=float, default=20)
    parser.add_argument("--min-seconds", type=float, default=2)
    parser.add_argument("--max-seconds", type=float, default=12)
    parser.add_argument("--min-side", type=int, default=256)
    parser.add_argument("--min-frames", type=int, default=49)
    parser.add_argument("--max-error-fraction", type=float, default=0.01)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
