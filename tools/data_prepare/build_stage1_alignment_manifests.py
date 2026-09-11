#!/usr/bin/env python
"""Build all three raw-data tasks from verified image and video releases."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3

from new_mobile_ov.training.stage1_alignment_data import load_sources, read_alignment_sample
from tools.data_prepare.download_alignment_images import atomic_json, output_lock, sha256_file

TASKS = ("t2i", "t2v", "image_reconstruction")


def completed_source(root: Path, count_key: str) -> dict:
    marker = root / ".download_complete"
    if not marker.is_file():
        raise RuntimeError(f"Source is not complete: {root}")
    summary = json.loads((root / "download_summary.json").read_text())
    if (summary != json.loads(marker.read_text()) or summary.get("status") != "complete"
            or summary.get(count_key, 0) < 1):
        raise RuntimeError(f"Invalid completion report: {root}")
    if sha256_file(root / "samples.jsonl") != summary.get("samples_sha256"):
        raise RuntimeError(f"Source index checksum mismatch: {root}")
    return summary


def media_reference(row: dict, alias: str, kind: str) -> dict:
    reference = dict(source=alias, kind=kind, shard=row["shard"],
                     offset=row[f"{kind}_offset"], size=row[f"{kind}_size"],
                     sha256=row[f"{kind}_sha256"], width=row["width"], height=row["height"])
    if kind == "video":
        for key in ("frame_count", "fps", "duration_seconds"):
            reference[key] = row[key]
    return reference


def split_for(kind: str, checksum: str, seed: int, validation_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{kind}:{checksum}".encode()).hexdigest()
    return "validation" if int(digest, 16) < validation_fraction * 2**256 else "train"


def normalize_prompt(text: str) -> str:
    return " ".join(text.casefold().split())


def run(args) -> dict:
    if not 0 <= args.validation_fraction < 0.5:
        raise ValueError("Validation fraction must be in [0, 0.5)")
    if args.reader_check_samples < 1:
        raise ValueError("At least one reader check per nonempty task manifest is required")
    output = args.output_dir.expanduser().resolve()
    roots = dict(blip3o=args.image_root.expanduser().resolve(), vchitect=args.video_root.expanduser().resolve())
    if output in roots.values():
        raise ValueError("Task manifests must use a separate output directory")
    with output_lock(output):
        (output / ".stage1_data_complete").unlink(missing_ok=True)
        reports = dict(blip3o=completed_source(roots["blip3o"], "valid_image_caption_pairs"),
                       vchitect=completed_source(roots["vchitect"], "valid_video_caption_pairs"))
        exclusions, exclusion_sources = set(), []
        for path in args.exclude_prompts:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip() and not line.startswith("#"):
                    exclusions.add(normalize_prompt(line))
            exclusion_sources.append(dict(path=str(path), sha256=sha256_file(path)))
        sources = {alias: dict(root=os.path.relpath(root, output), repo_id=reports[alias]["repo_id"],
                              revision=reports[alias]["revision"], samples_sha256=reports[alias]["samples_sha256"])
                   for alias, root in roots.items()}
        counts, rejected, scanned = Counter(), Counter(), Counter()
        database = output / "dedup.sqlite.tmp"
        database.unlink(missing_ok=True)
        with closing(sqlite3.connect(database)) as seen, seen, ExitStack() as stack:
            seen.execute("CREATE TABLE media (kind TEXT, digest TEXT, PRIMARY KEY (kind, digest))")
            writers = {}
            for split in ("train", "validation"):
                (output / split).mkdir(exist_ok=True)
                for task in TASKS:
                    writers[split, task] = stack.enter_context((output / split / f"{task}.jsonl.tmp").open("w"))
            for alias, kind, count_key in (("blip3o", "image", "valid_image_caption_pairs"),
                                            ("vchitect", "video", "valid_video_caption_pairs")):
                with (roots[alias] / "samples.jsonl").open(encoding="utf-8") as stream:
                    for line in stream:
                        row = json.loads(line)
                        scanned[alias] += 1
                        caption = row["caption"]
                        if not isinstance(caption, str) or not caption.strip():
                            raise ValueError("Invalid caption in supposedly verified source")
                        if normalize_prompt(caption) in exclusions:
                            rejected[f"{alias}.excluded_prompt"] += 1
                            continue
                        reference = media_reference(row, alias, kind)
                        digest = reference["sha256"]
                        if len(digest) != 64:
                            raise ValueError("Missing media checksum")
                        inserted = seen.execute("INSERT OR IGNORE INTO media VALUES (?, ?)", (kind, digest)).rowcount
                        if not inserted:
                            rejected[f"{alias}.exact_duplicate"] += 1
                            continue
                        split = split_for(kind, digest, args.seed, args.validation_fraction)
                        tasks = ("t2i", "image_reconstruction") if kind == "image" else ("t2v",)
                        for task in tasks:
                            record = dict(sample_id=f"{task}:{row['sample_id']}", task=task, target=reference,
                                          prompt="" if task == "image_reconstruction" else caption,
                                          conditioning_route="mllm_only")
                            writers[split, task].write(json.dumps(record, ensure_ascii=True) + "\n")
                            counts[f"{split}.{task}"] += 1
                        if scanned[alias] % 10000 == 0:
                            seen.commit()
                if scanned[alias] != reports[alias][count_key]:
                    raise RuntimeError(f"Source count mismatch: {alias}")
            for task in TASKS:
                if not counts[f"train.{task}"]:
                    raise RuntimeError(f"No training data for {task}; refusing an incomplete Stage-1 release")
        artifacts = {}
        for split in ("train", "validation"):
            for task in TASKS:
                final = output / split / f"{task}.jsonl"
                final.with_suffix(".jsonl.tmp").replace(final)
                artifacts[f"{split}.{task}"] = dict(path=str(final.relative_to(output)), sha256=sha256_file(final))
        database.unlink(missing_ok=True)
        atomic_json(output / "sources.json", sources)
        reader_checks = {}
        reader_sources = load_sources(output)
        for name, artifact in artifacts.items():
            checked = 0
            with (output / artifact["path"]).open() as stream:
                for line in stream:
                    record = json.loads(line)
                    read_alignment_sample(record, reader_sources)
                    checked += 1
                    if checked == args.reader_check_samples:
                        break
            reader_checks[name] = checked
        summary = dict(status="raw_data_ready", tasks=list(TASKS), counts=dict(counts), rejected=dict(rejected),
                       source_counts=dict(scanned), sources=sources, manifests=artifacts,
                       sources_sha256=sha256_file(output / "sources.json"), seed=args.seed,
                       reader_checks=reader_checks, reader_video_frames=49,
                       validation_fraction=args.validation_fraction, split_unit="exact_media_sha256",
                       duplicate_check="exact_encoded_media_bytes_only; near_duplicates_not_checked",
                       exclusion_sources=exclusion_sources, exclusion_policy="exact_normalized_prompt_only",
                       semantics_verified=False, vae_encoded=False, trainer_integrated=False,
                       recipe="public_substitution_for_three_alignment_tasks; not_UniVideo_original_data_or_exact_recipe",
                       temporal_policy="uniform_frames_across_whole_clip; preserve_caption_and_record_duration",
                       reconstruction_policy="same_image_as_target_and_MLLM_input; no_clean_target_latent_as_DiT_condition")
        atomic_json(output / "stage1_summary.json", summary)
        atomic_json(output / ".stage1_data_complete", summary)
        print(json.dumps(summary, indent=2), flush=True)
        return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--exclude-prompts", type=Path, action="append", default=[])
    parser.add_argument("--reader-check-samples", type=int, default=8)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
