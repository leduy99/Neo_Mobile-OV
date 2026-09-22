#!/usr/bin/env python
"""Expand T2V without modifying old media, image tasks, or validation manifests."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
from contextlib import closing

from new_mobile_ov.training.stage1_alignment_data import load_sources, read_alignment_sample
from tools.data_prepare import download_alignment_videos as videos
from tools.data_prepare.build_stage1_alignment_manifests import (
    TASKS, completed_source, media_reference, normalize_prompt, split_for,
)
from tools.data_prepare.download_alignment_images import atomic_json, output_lock, sha256_file


def read_base_release(root: Path) -> dict:
    summary = json.loads((root / "stage1_summary.json").read_text())
    if (summary != json.loads((root / ".stage1_data_complete").read_text())
            or summary.get("status") != "raw_data_ready" or set(summary["tasks"]) != set(TASKS)):
        raise ValueError("Base release is incomplete or modified")
    if sha256_file(root / "sources.json") != summary["sources_sha256"]:
        raise ValueError("Base source references changed")
    for split in ("train", "validation"):
        for task in TASKS:
            name = f"{split}.{task}"
            artifact = summary["manifests"][name]
            if (artifact["path"] != f"{split}/{task}.jsonl"
                    or sha256_file(root / artifact["path"]) != artifact["sha256"]):
                raise ValueError(f"Base manifest changed: {name}")
            if summary["counts"].get(name, 0) < 1:
                raise ValueError(f"Base manifest must be nonempty: {name}")
    return summary


def load_exclusions(paths: list[Path]) -> tuple[set[str], list[dict]]:
    prompts, provenance = set(), []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                prompts.add(normalize_prompt(line))
        provenance.append(dict(path=str(path.resolve()), sha256=sha256_file(path)))
    return prompts, provenance


def build_expanded_release(base: Path, extra: Path, output: Path, *, min_train_videos: int,
                           reader_checks: int = 8, exclude_prompts: list[Path] = ()) -> dict:
    base, extra, output = (path.resolve() for path in (base, extra, output))
    if min_train_videos < 1 or reader_checks < 1:
        raise ValueError("Target count and reader checks must be positive")
    original = read_base_release(base)
    report = completed_source(extra, "valid_video_caption_pairs")
    base_sources = json.loads((base / "sources.json").read_text())
    if (report["repo_id"] != base_sources["vchitect"]["repo_id"]
            or report["revision"] != base_sources["vchitect"]["revision"]):
        raise ValueError("Expanded video source must match the base repository and revision")
    roots = load_sources(base)
    all_roots = [base, extra, *roots.values()]
    if any(output == path or output.is_relative_to(path) or path.is_relative_to(output)
           for path in all_roots):
        raise ValueError("Output must be separate from source and base release directories")
    prompts, exclusions = load_exclusions(exclude_prompts)
    contract = dict(base_summary_sha256=sha256_file(base / "stage1_summary.json"),
                    extra_summary_sha256=sha256_file(extra / "download_summary.json"),
                    min_train_videos=min_train_videos, reader_checks=reader_checks,
                    exclude_prompts=exclusions)
    with output_lock(output):
        contract_path = output / "expansion_contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("Expansion contract changed; use a new manifest output directory")
        atomic_json(contract_path, contract)
        (output / ".stage1_data_complete").unlink(missing_ok=True)
        sources = {alias: {**spec, "root": os.path.relpath(roots[alias], output)}
                   for alias, spec in base_sources.items()}
        alias = "vchitect_added"
        if alias in sources:
            raise ValueError("Base is already expanded; choose a distinct expansion implementation")
        sources[alias] = dict(root=os.path.relpath(extra, output), repo_id=report["repo_id"],
                              revision=report["revision"], samples_sha256=report["samples_sha256"])
        counts = Counter(original["counts"])
        rejected = Counter()
        database = output / "dedup.sqlite.tmp"
        database.unlink(missing_ok=True)
        with closing(sqlite3.connect(database)) as seen, seen:
            seen.execute("CREATE TABLE media (digest TEXT PRIMARY KEY, sample_id TEXT UNIQUE)")
            seen.execute("CREATE TABLE val_prompts (prompt TEXT PRIMARY KEY)")
            for split in ("train", "validation"):
                (output / split).mkdir(exist_ok=True)
                for task in TASKS:
                    relative = f"{split}/{task}.jsonl"
                    # Copy small manifests, never the media archives. Validation stays byte-identical.
                    shutil.copyfile(base / relative, output / f"{relative}.tmp")
                scanned = 0
                with (base / split / "t2v.jsonl").open() as stream:
                    for line in stream:
                        row = json.loads(line)
                        seen.execute("INSERT INTO media VALUES (?, ?)",
                                     (row["target"]["sha256"], row["sample_id"]))
                        if split == "validation":
                            seen.execute("INSERT OR IGNORE INTO val_prompts VALUES (?)",
                                         (normalize_prompt(row["prompt"]),))
                        scanned += 1
                if scanned != counts[f"{split}.t2v"]:
                    raise ValueError("Base video count mismatch")
            seen.commit()
            (output / "heldout").mkdir(exist_ok=True)
            added_checks, scanned, added, heldout_count = [], 0, 0, 0
            with (output / "train/t2v.jsonl.tmp").open("a") as train, \
                    (output / "heldout/t2v_added.jsonl.tmp").open("w") as heldout, \
                    (extra / "samples.jsonl").open() as stream:
                for line in stream:
                    row = json.loads(line)
                    scanned += 1
                    caption = row["caption"]
                    if not isinstance(caption, str) or not caption.strip():
                        raise ValueError("Invalid caption in verified video source")
                    prompt = normalize_prompt(caption)
                    if prompt in prompts:
                        rejected["excluded_prompt"] += 1
                        continue
                    if seen.execute("SELECT 1 FROM val_prompts WHERE prompt=?", (prompt,)).fetchone():
                        rejected["matches_old_validation_prompt"] += 1
                        continue
                    reference = media_reference(row, alias, "video")
                    sample_id = f"t2v:{row['sample_id']}"
                    inserted = seen.execute("INSERT OR IGNORE INTO media VALUES (?, ?)",
                                            (reference["sha256"], sample_id)).rowcount
                    if not inserted:
                        rejected["duplicate_media_or_id"] += 1
                        continue
                    record = dict(sample_id=sample_id, task="t2v", target=reference,
                                  prompt=caption, conditioning_route="mllm_only")
                    split = split_for("video", reference["sha256"], original["seed"],
                                      original["validation_fraction"])
                    writer = train if split == "train" else heldout
                    writer.write(json.dumps(record, ensure_ascii=True) + "\n")
                    if split == "train":
                        added += 1
                        if len(added_checks) < reader_checks:
                            added_checks.append(record)
                    else:
                        heldout_count += 1
                    if scanned % 10000 == 0:
                        seen.commit()
                        print(f"Expanded videos: scanned={scanned} added_train={added} "
                              f"heldout={heldout_count}", flush=True)
            if scanned != report["valid_video_caption_pairs"]:
                raise ValueError("Expanded source count mismatch")
        database.unlink(missing_ok=True)
        counts["train.t2v"] += added
        count_report = dict(status="insufficient_videos" if counts["train.t2v"] < min_train_videos else "building",
                            old_train_videos=original["counts"]["train.t2v"], added_train_videos=added,
                            total_train_videos=counts["train.t2v"], required_train_videos=min_train_videos,
                            new_heldout_videos=heldout_count, rejected=dict(rejected))
        atomic_json(output / "expansion_report.json", count_report)
        if counts["train.t2v"] < min_train_videos:
            raise RuntimeError(f"Only {counts['train.t2v']} training videos; need {min_train_videos}. "
                               "Downloads are retained; no ready marker was written.")
        artifacts = {}
        for split in ("train", "validation"):
            for task in TASKS:
                relative = f"{split}/{task}.jsonl"
                final = output / relative
                (output / f"{relative}.tmp").replace(final)
                checksum = sha256_file(final)
                if (split == "validation" or task != "t2v") and checksum != original["manifests"][f"{split}.{task}"]["sha256"]:
                    raise RuntimeError(f"Frozen manifest changed: {relative}")
                artifacts[f"{split}.{task}"] = dict(path=relative, sha256=checksum)
        heldout_path = output / "heldout/t2v_added.jsonl"
        heldout_path.with_suffix(".jsonl.tmp").replace(heldout_path)
        atomic_json(output / "sources.json", sources)
        reader_sources = load_sources(output)
        checked = {}
        for name, artifact in artifacts.items():
            n = 0
            with (output / artifact["path"]).open() as stream:
                for line in stream:
                    read_alignment_sample(json.loads(line), reader_sources)
                    n += 1
                    if n == reader_checks:
                        break
            checked[name] = n
        for record in added_checks:
            read_alignment_sample(record, reader_sources)
        summary = {**original, "counts": dict(counts), "sources": sources, "manifests": artifacts,
                   "source_counts": {**original["source_counts"], alias: scanned},
                   "sources_sha256": sha256_file(output / "sources.json"), "reader_checks": checked,
                   "vae_encoded": False, "semantics_verified": False,
                   "expansion": {**count_report, "status": "complete", "contract": contract,
                                 "added_reader_checks": len(added_checks),
                                 "validation_policy": "original_six_manifests_except_train_t2v_unchanged",
                                 "new_heldout": dict(path="heldout/t2v_added.jsonl",
                                                     sha256=sha256_file(heldout_path)),
                                 "duplicate_check": "exact_media_sha256_and_sample_id; not_perceptual",
                                 "extra_source": report}}
        atomic_json(output / "stage1_summary.json", summary)
        atomic_json(output / ".stage1_data_complete", summary)
        atomic_json(output / "expansion_report.json", {**count_report, "status": "complete"})
        print(json.dumps(summary["expansion"], indent=2), flush=True)
        return summary


def run(args) -> dict:
    base, old_video, extra, output = (path.expanduser().resolve() for path in
                                    (args.base_root, args.video_root, args.extra_video_root, args.output_dir))
    paths = (base, old_video, extra, output)
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
           for i, a in enumerate(paths) for b in paths[i + 1:]):
        raise ValueError("Base, old video, extra video and output roots must be separate")
    if (args.total_video_shards < 1 or args.min_train_videos < 1 or args.workers < 1
            or args.reader_check_samples < 1 or args.retries < 1):
        raise ValueError("Counts must be positive")
    if any(not math.isfinite(value) or value <= 0
           for value in (args.max_total_video_gib, args.disk_margin_gib)):
        raise ValueError("Storage budgets must be finite and positive")
    base_summary = read_base_release(base)
    base_roots = load_sources(base)
    if base_roots["vchitect"] != old_video:
        raise ValueError("Base manifests do not reference the requested original video directory")
    for root in base_roots.values():
        for new_root in (extra, output):
            if new_root == root or new_root.is_relative_to(root) or root.is_relative_to(new_root):
                raise ValueError("New roots must not overlap existing media sources")
    old_report = completed_source(old_video, "valid_video_caption_pairs")
    old_plan_path = old_video / "download_plan.json"
    old_plan = json.loads(old_plan_path.read_text())
    source = base_summary["sources"]["vchitect"]
    if (old_report["repo_id"] != videos.REPO_ID or old_plan["revision"] != old_report["revision"]
            or source["revision"] != old_report["revision"]
            or source["samples_sha256"] != old_report["samples_sha256"]
            or old_plan["annotation"]["sha256"] != old_report["annotation_sha256"]):
        raise ValueError("Original video plan and completed source disagree")
    old_shards = old_plan["shards"]
    if len({item["path"] for item in old_shards}) != len(old_shards):
        raise ValueError("Duplicate shard in original plan")
    # Check existence/size here, not another full 400GB hash pass over an immutable release.
    for item in old_shards:
        path = old_video / "shards" / item["path"]
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise ValueError(f"Original archive missing or wrong size: {path}")
    additional = args.total_video_shards - len(old_shards)
    if additional < 1:
        raise ValueError("Total shard count must exceed the original selection")
    seed = old_plan["settings"]["seed"]
    with output_lock(output / ".expansion_job"):
        plan = videos.resolve_plan(extra, revision=old_plan["revision"], count=additional,
                                   seed=seed, exclude_plan=old_plan_path)
        old_bytes = sum(item["size"] for item in old_shards) + old_plan["annotation"]["size"]
        total_bytes = old_bytes + plan["expected_bytes"]
        if total_bytes > args.max_total_video_gib * 1024**3:
            raise RuntimeError(f"Video archive budget exceeded: {total_bytes / 1024**3:.1f} GiB > "
                               f"{args.max_total_video_gib} GiB. No downloads started.")
        existing_bytes = sum(min(item["size"], path.stat().st_size) if path.is_file() else 0
                             for item in [plan["annotation"], *plan["shards"]]
                             for path in [extra / "shards" / item["path"]])
        estimates = dict(old_shards=len(old_shards), additional_shards=additional,
                         total_shards=args.total_video_shards, old_archive_bytes=old_bytes,
                         added_archive_bytes=plan["expected_bytes"], total_archive_bytes=total_bytes,
                         remaining_bytes_before_integrity_checks=plan["expected_bytes"] - existing_bytes,
                         estimated_train_videos=round(base_summary["counts"]["train.t2v"] *
                                                      args.total_video_shards / len(old_shards)),
                         estimate_is_not_a_guarantee=True, minimum_train_videos=args.min_train_videos,
                         old_revision=old_plan["revision"], download_workers=args.workers,
                         filesystem_free_bytes=shutil.disk_usage(output).free,
                         quota_note="Filesystem free space is not project quota; check your quota separately.")
        print(json.dumps(estimates, indent=2), flush=True)
        if args.dry_run:
            return estimates
        with output_lock(extra):
            if (extra / "download_plan.json").exists():
                current = videos.resolve_plan(extra, revision=old_plan["revision"], count=additional,
                                              seed=seed, exclude_plan=old_plan_path)
                if current != plan:
                    raise ValueError("Expansion plan changed during preflight; refusing to overwrite it")
            atomic_json(extra / "download_plan.json", plan)
            # Reuse the native annotation, but never link/copy hundreds of GB of old video archives.
            annotation = old_video / "shards" / plan["annotation"]["path"]
            destination = extra / "shards" / plan["annotation"]["path"]
            if not destination.exists():
                if sha256_file(annotation) != plan["annotation"]["sha256"]:
                    raise ValueError("Original video annotation checksum mismatch")
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".copy.tmp")
                shutil.copyfile(annotation, temporary)
                temporary.replace(destination)
        policy = old_report["policy"]
        download_args = videos.parse_args([
            "--output-dir", str(extra), "--revision", old_plan["revision"], "--num-shards", str(additional),
            "--exclude-plan", str(old_plan_path), "--seed", str(seed), "--workers", str(args.workers),
            "--retries", str(args.retries), "--max-download-gib", str(args.max_total_video_gib),
            "--disk-margin-gib", str(args.disk_margin_gib), "--min-seconds", str(policy["min_seconds"]),
            "--max-seconds", str(policy["max_seconds"]), "--min-side", str(policy["min_side"]),
            "--min-frames", str(policy["min_frames"]),
        ])
        videos.run(download_args)
        return build_expanded_release(base, extra, output, min_train_videos=args.min_train_videos,
                                      reader_checks=args.reader_check_samples, exclude_prompts=args.exclude_prompts)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=Path("download_data/data/univideo_stage1"))
    parser.add_argument("--video-root", type=Path, default=Path("download_data/data/univideo_alignment_videos"))
    parser.add_argument("--extra-video-root", type=Path, default=Path("download_data/data/univideo_alignment_videos_added650"))
    parser.add_argument("--output-dir", type=Path, default=Path("download_data/data/univideo_stage1_expanded650"))
    parser.add_argument("--total-video-shards", type=int, default=650, help="Includes the existing 100 shards")
    parser.add_argument("--min-train-videos", type=int, default=500000)
    parser.add_argument("--max-total-video-gib", type=float, default=2500, help="Old + new archives; not latent quota")
    parser.add_argument("--disk-margin-gib", type=float, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--reader-check-samples", type=int, default=8)
    parser.add_argument("--exclude-prompts", type=Path, action="append", default=[])
    parser.add_argument("--dry-run", action="store_true", help="Read remote metadata only; no media downloads")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
