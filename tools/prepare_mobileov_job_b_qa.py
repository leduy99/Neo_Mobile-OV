#!/usr/bin/env python
"""Build bounded QA replay from pinned Job-B sources; no new model-generated answers."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import av
from PIL import Image
import pyarrow.parquet as pq

from new_mobile_ov.training import job_b_qa as qa
from tools.download_mobileov_job_b_data import verify_file
from tools.data_prepare.download_alignment_images import atomic_json, output_lock, sha256_file


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode()).hexdigest()


def prefixes(messages):
    normalized = []
    for message in messages:
        role = message.get("role", {"human": "user", "gpt": "assistant"}.get(message.get("from")))
        text = message.get("content", message.get("value"))
        if role not in {"system", "user", "assistant"} or not isinstance(text, str) or not text.strip():
            raise ValueError("Invalid source conversation")
        text = text.replace("<image>", "").replace("<video>", "").strip()
        if not text:
            raise ValueError("Empty source message after removing media placeholders")
        normalized.append(dict(role=role, content=text))
        if role == "assistant" and len(normalized) >= 2:
            yield list(normalized)


def put_image(image, root, *, seconds=None):
    image = image.convert("RGB")
    image.thumbnail((512, 512))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    body = buffer.getvalue()
    sha = digest(body)
    path = root / "media" / sha[:2] / f"{sha}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or sha256_file(path) != sha:
        temporary = path.with_suffix(f".{threading.get_ident()}.tmp")
        temporary.write_bytes(body)
        temporary.replace(path)
    result = dict(path=str(path.relative_to(root)), sha256=sha)
    if seconds is not None:
        result["seconds"] = seconds
    return result


def video_frames(body, root):
    if not 0 < len(body) <= 256 * 1024**2:
        raise ValueError("Video member size exceeds bound")
    def opened():
        container = av.open(io.BytesIO(body), metadata_errors="replace")
        container.streams.video[0].codec_context.thread_count = 1
        return container
    with opened() as container:
        count = sum(1 for _ in container.decode(video=0))
    if count < 8:
        raise ValueError("Video needs eight distinct frames")
    positions = {i * (count - 1) // 7 for i in range(8)}
    images = []
    with opened() as container:
        rate = float(container.streams.video[0].average_rate or 0)
        for index, frame in enumerate(container.decode(video=0)):
            if index in positions:
                if frame.time is None and rate <= 0:
                    raise ValueError("Video lacks timestamps and frame rate")
                images.append(put_image(frame.to_image(), root,
                                        seconds=float(frame.time) if frame.time is not None else index / rate))
    if len(images) != 8:
        raise ValueError("Video decode changed between passes")
    return images


def build(args, processor):
    source = args.source_root.resolve()
    source_report = json.loads((source / "source_files_complete.json").read_text())
    if source_report.get("status") != "source_files_verified_QA_release_NOT_prepared":
        raise ValueError("The bounded source download has not completed")
    files = source_report["files"]
    for row in files:
        path = source / "sources" / row["source"] / row["filename"]
        if not path.resolve().is_relative_to(source):
            raise ValueError("Unsafe source path")
        verify_file(path, row)
    root = args.output_dir.resolve()
    if root == source or source.is_relative_to(root):
        raise ValueError("Keep the QA release separate from raw sources")
    processor_sha = digest(dict(vocabulary=processor.tokenizer.get_vocab(), chat_template=processor.chat_template,
                                image_processor=processor.image_processor.to_dict()))
    contract = dict(format=qa.FORMAT, source_plan_sha256=source_report["plan_sha256"],
                    processor=args.processor, processor_sha256=processor_sha, max_tokens=args.max_tokens,
                    seed=args.seed, validation_fraction=args.validation_fraction,
                    video_frames=8, image_long_side=512, answer_supervision="last_assistant_only")
    rejected = Counter()
    with output_lock(root):
        existing = root / "qa_contract.json"
        if existing.exists() and json.loads(existing.read_text()) != contract:
            raise ValueError("QA preparation contract changed; choose a new output directory")
        if (root / "qa_summary.json").exists():
            return qa.verify_release(root)
        atomic_json(existing, contract)
        handles, counts, seen, media_index = {}, Counter(), set(), {}
        for split in ("train", "validation"):
            (root / split).mkdir(exist_ok=True)
            for modality in qa.MODALITIES:
                handles[split, modality] = (root / split / f"{modality}.jsonl.tmp").open("w")
        local = threading.local()
        def validate(row):
            if not hasattr(local, "processor"):
                local.processor = copy.deepcopy(processor)
            try:
                qa.pack_answer(local.processor, row, qa.read_images(root, row), args.max_tokens)
                return row, None
            except ValueError as error:
                return row, str(error)
        pending = []
        def flush(pool):
            for row, error in pool.map(validate, pending):
                if error:
                    rejected[error] += 1
                    continue
                if row["sample_id"] in seen:
                    rejected["exact_duplicate_qa"] += 1
                    continue
                seen.add(row["sample_id"])
                value = int(digest(f"{args.seed}:{row['group_id']}".encode())[:16], 16) / 2**64
                split = "validation" if value < args.validation_fraction else "train"
                handles[split, row["modality"]].write(json.dumps(row) + "\n")
                counts[split, row["modality"]] += 1
                for reference in row["media"]:
                    media_index[reference["path"]] = dict(path=reference["path"], sha256=reference["sha256"])
            pending.clear()
        def submit(pool, modality, messages, media, group, provenance):
            try:
                conversations = list(prefixes(messages))
            except ValueError:
                rejected["invalid_source_conversation"] += 1
                return
            for prefix in conversations:
                if sum(len(m["content"]) for m in prefix) > 16000:
                    rejected["source_text_exceeds_character_bound"] += 1
                    continue
                sample = digest(dict(messages=prefix, media=media))
                pending.append(dict(sample_id=f"qa:{sample}", modality=modality, messages=prefix,
                                    media=media, group_id=group, source=provenance))
                if len(pending) >= args.workers * 4:
                    flush(pool)
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for entry in files:
                    path = source / "sources" / entry["source"] / entry["filename"]
                    if entry["kind"] != "train_parquet":
                        continue
                    for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
                        for record in batch.to_pylist():
                            if entry["source"] == "image_qa":
                                images = record["images"]
                                if len(images) != 1 or not images[0].get("bytes"):
                                    raise ValueError("Selected Cauldron files must contain one embedded image")
                                body = images[0]["bytes"]
                                with Image.open(io.BytesIO(body)) as image:
                                    media = [put_image(image, root)]
                                for turn in record["texts"]:
                                    submit(pool, "image", [dict(role="user", content=turn["user"]),
                                           dict(role="assistant", content=turn["assistant"])], media,
                                           digest(body), entry["filename"])
                            else:
                                messages = record["messages"]
                                submit(pool, "text", messages, [], digest(messages), entry["filename"])
                    print(f"Prepared parquet {entry['filename']}", flush=True)
                annotations = {}
                for entry in files:
                    if entry["kind"] == "qa_annotation":
                        path = source / "sources" / entry["source"] / entry["filename"]
                        for record in json.loads(path.read_text()):
                            annotations.setdefault(record["video"], []).append((record, entry["filename"]))
                found = set()
                for entry in files:
                    if entry["kind"] != "media_archive":
                        continue
                    path = source / "sources" / entry["source"] / entry["filename"]
                    work = []
                    def finish_videos():
                        for name, group, future in work:
                            try:
                                media = future.result()
                            except (ValueError, av.error.FFmpegError):
                                rejected["selected_video_decode_failed"] += 1
                                continue
                            for record, provenance in annotations[name]:
                                submit(pool, "video", record["conversations"], media, group, provenance)
                        work.clear()
                    with tarfile.open(path, "r|gz") as archive:
                        for member in archive:
                            parts = PurePosixPath(member.name).parts
                            if member.name.startswith("/") or ".." in parts or "\\" in member.name:
                                raise ValueError("Unsafe video archive member")
                            if not member.isfile():
                                continue
                            name = str(PurePosixPath(*parts[parts.index("academic_source"):])) if "academic_source" in parts else member.name
                            if name not in annotations:
                                continue
                            if name in found or not 0 < member.size <= 256 * 1024**2:
                                raise ValueError("Duplicate or oversized selected video archive member")
                            found.add(name)
                            body = archive.extractfile(member).read()
                            work.append((name, digest(body), pool.submit(video_frames, body, root)))
                            if len(work) >= args.workers:
                                finish_videos()
                        finish_videos()
                rejected["annotation_video_not_in_selected_archive"] = len(set(annotations) - found)
                flush(pool)
        finally:
            for handle in handles.values():
                handle.close()
        manifest_info = {}
        for split, modality in handles:
            if counts[split, modality] < 1:
                raise ValueError(f"No retained QA examples for {split}/{modality}")
            name = f"{split}/{modality}.jsonl"
            (root / f"{name}.tmp").replace(root / name)
            manifest_info[name] = dict(rows=counts[split, modality], sha256=sha256_file(root / name))
        atomic_json(root / "media_index.json", list(media_index.values()))
        report = dict(contract, status="mechanically_verified_QA_release", files=manifest_info,
                      media_index_sha256=sha256_file(root / "media_index.json"), retained_media_files=len(media_index),
                      new_answers_generated_by_this_pipeline=False, source_annotations_human_gold_verified=False,
                      original_source_splits_independently_verified=False, benchmark_decontamination_complete=False,
                      split_unit="original_media_sha256_or_full_text_conversation_sha256",
                      rejected=dict(rejected), source_train_policy="provider_train_parquets; source-provided video QA",
                      research_scope="academic_only; source licenses still apply; no quality/forgetting guarantee")
        atomic_json(root / "qa_summary.json", report)
        return report


def main(argv=None):
    import argparse
    import os
    from transformers import AutoProcessor
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--processor", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--validation-fraction", type=float, default=.02)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Prepare QA in a Slurm data job, not the login shell")
    if not 1 <= args.workers <= 16 or not 64 <= args.max_tokens <= 2048 or not 0 < args.validation_fraction <= .2:
        parser.error("Invalid QA preparation limits")
    print(json.dumps(build(args, AutoProcessor.from_pretrained(args.processor)), indent=2), flush=True)


if __name__ == "__main__":
    main()
