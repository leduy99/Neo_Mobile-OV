#!/usr/bin/env python
"""Prepare a separate, resumable Exp-2 data release without copying raw media.

plan -> quality (VBench environment) -> caption (Qwen data environment) -> finalize.
Only the explicit finalize phase publishes training manifests.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.training.exp2_data import (  # noqa: E402
    FORMAT, RejectedSample, apply_spatial, cheap_rejections, decode_selected, digest,
    fraction, make_view, quality_stats, subset_frames, validate_policy,
)
from new_mobile_ov.training.stage1_alignment import JsonlRecords, verify_release  # noqa: E402
from new_mobile_ov.training.stage1_alignment_data import load_sources, read_alignment_sample, read_media_bytes  # noqa: E402
from tools.data_prepare.download_alignment_images import atomic_json, sha256_file  # noqa: E402


@contextmanager
def lock(path, *, blocking=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


class Results:
    def __init__(self, path, contract_sha):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (contract_sha TEXT)")
        row = self.db.execute("SELECT contract_sha FROM metadata").fetchone()
        if row is not None and row[0] != contract_sha:
            self.db.close()
            raise ValueError("Worker database belongs to a different data contract")
        if row is None:
            self.db.execute("INSERT INTO metadata VALUES (?)", (contract_sha,))
            self.db.commit()
        self.db.execute("CREATE TABLE IF NOT EXISTS results (id TEXT PRIMARY KEY, ordinal INTEGER UNIQUE, status TEXT, payload TEXT)")

    def done(self, sample_id):
        row = self.db.execute("SELECT status FROM results WHERE id=?", (sample_id,)).fetchone()
        return row is not None and row[0] in {"accepted", "rejected"}

    def write(self, value):
        self.db.execute("INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?)",
                        (value["record"]["sample_id"], value["ordinal"], value["status"], json.dumps(value, allow_nan=False)))
        # Each receipt is independently durable. A stopped worker reruns only unfinished samples.
        self.db.commit()

    def close(self):
        self.db.close()


def result_paths(output, phase, workers):
    return [output / "work" / phase / f"worker_{rank:03d}.sqlite" for rank in range(workers)]


def completed_results(output, phase, workers):
    for path in result_paths(output, phase, workers):
        marker = path.with_suffix(".complete.json")
        if not marker.is_file() or not path.is_file():
            raise RuntimeError(f"Incomplete {phase} worker: {path}")
        receipt = json.loads(marker.read_text())
        if (receipt["sha256"] != sha256_file(path)
                or receipt.get("contract_sha256") != sha256_file(output / "exp2_contract.json")):
            raise ValueError(f"Modified worker results: {path}")
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            for status, payload in db.execute("SELECT status, payload FROM results ORDER BY ordinal"):
                if status == "error":
                    raise RuntimeError(f"Unresolved runtime error in {path}; resubmit unchanged")
                yield json.loads(payload)


def parent_records(contract):
    ordinal = 0
    parent = Path(contract["parent_root"])
    limit = contract["limit_per_manifest"]
    for split in ("train", "validation"):
        for task in ("t2i", "t2v"):
            records = JsonlRecords(parent / split / f"{task}.jsonl")
            for index in range(min(len(records), limit) if limit else len(records)):
                yield ordinal, split, records[index]
                ordinal += 1


def build_plan(args):
    output, parent = args.output_dir.resolve(), args.input_root.resolve()
    if output == parent or output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError("Exp-2 output must be separate from the original release")
    policy = validate_policy(json.loads(args.policy.read_text()))
    summary = verify_release(parent)
    if summary.get("format") == FORMAT:
        raise ValueError("Use the original expanded raw release as the Exp-2 parent")
    sources = load_sources(parent)
    if any(output == root or output.is_relative_to(root) or root.is_relative_to(output) for root in sources.values()):
        raise ValueError("Exp-2 output must not contain, or be inside, source media roots")
    contract = dict(format=FORMAT, parent_root=str(parent),
                    parent_summary_sha256=sha256_file(parent / "stage1_summary.json"),
                    parent_sources_sha256=summary["sources_sha256"], policy=policy,
                    quality_workers=args.quality_workers, caption_workers=args.caption_workers,
                    limit_per_manifest=args.limit_per_manifest,
                    preview_only=bool(args.limit_per_manifest),
                    media_storage="reference_original_tar_offsets; no_duplicate_media",
                    caption_scope="visual_only_single_pass_on_surviving_crops; source_caption_saved_for_audit_only",
                    implementation_sha256={name: sha256_file(ROOT / name) for name in (
                        "new_mobile_ov/training/exp2_data.py", "tools/data_prepare/exp2_models.py",
                        "tools/data_prepare/prepare_mobileov_exp2.py")},
                    original_data_unchanged=True)
    if min(args.quality_workers, args.caption_workers) < 1 or args.limit_per_manifest < 0:
        raise ValueError("Invalid worker count/limit")
    output.mkdir(parents=True, exist_ok=True)
    with lock(output / ".plan.lock"):
        target = output / "exp2_contract.json"
        if target.is_file() and json.loads(target.read_text()) != contract:
            raise ValueError("Different policy/input/sharding in this output; choose a NEW directory")
        atomic_json(target, contract)
        rewritten = json.loads((parent / "sources.json").read_text())
        for alias, source in rewritten.items():
            source["root"] = os.path.relpath(sources[alias], output)
        atomic_json(output / "sources.json", rewritten)
    print(json.dumps(contract, indent=2), flush=True)
    return contract


def load_contract(args):
    contract = json.loads((args.output_dir / "exp2_contract.json").read_text())
    parent = Path(contract["parent_root"])
    if (sha256_file(parent / "stage1_summary.json") != contract["parent_summary_sha256"]
            or sha256_file(parent / "sources.json") != contract["parent_sources_sha256"]):
        raise ValueError("Parent release changed during preparation")
    validate_policy(contract["policy"])
    for name, expected in contract["implementation_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError("Preparation code changed mid-release; use a new output directory")
    if load_sources(args.output_dir) != load_sources(parent):
        raise ValueError("Exp-2 source roots changed")
    return contract


def load_view(value, sources):
    from PIL import Image

    payload = read_media_bytes(sources, value["record"]["target"])
    spec = value["transform"]
    if spec["temporal"] is not None:
        return decode_selected(payload, spec["temporal"], spec["spatial"])
    with Image.open(io.BytesIO(payload)) as image:
        return [apply_spatial(image.convert("RGB"), spec["spatial"])]


def quality_worker(args, contract, model=None):
    if model is None:
        from tools.data_prepare.exp2_models import QualityModels
        model = QualityModels(args.vbench_cache, args.vbench_repo)
    verify_models(args.output_dir, "quality", model.provenance)
    policy, sources = contract["policy"], load_sources(args.output_dir)
    for ordinal, split, record in parent_records(contract):
        if ordinal % contract["quality_workers"] != args.rank:
            continue
        yield ordinal, record, lambda r=record, s=split, o=ordinal: quality_one(r, s, o, sources, policy, model)


def quality_one(record, split, ordinal, sources, policy, model):
    value = dict(ordinal=ordinal, split=split, record=record)
    try:
        spec, frames = make_view(record, read_media_bytes(sources, record["target"]), policy, split)
        stats = quality_stats(subset_frames(frames, policy["quality_frames"]))
        reasons = cheap_rejections(stats, policy)
        value.update(transform=spec, cheap_quality=stats)
        if not reasons:
            scores = model.score(frames, policy)
            value["quality"] = scores
            for key, threshold, op in (("musiq_mean", policy["min_musiq"], "min"),
                                       ("aesthetic_mean", policy["min_aesthetic"], "min"),
                                       ("motion_mean_pixels", policy["max_motion_pixels"], "max")):
                actual = scores[key]
                if actual is not None and threshold is not None and (actual < threshold if op == "min" else actual > threshold):
                    reasons.append(f"{key}_outside_policy")
        value.update(status="rejected" if reasons else "accepted", reasons=reasons)
    except RejectedSample as exc:
        value.update(status="rejected", reasons=[str(exc)])
    return value


def caption_worker(args, contract, model=None):
    if model is None:
        from tools.data_prepare.exp2_models import CaptionModels
        model = CaptionModels(contract["policy"])
    verify_models(args.output_dir, "caption", model.provenance)
    sources = load_sources(args.output_dir)
    for value in completed_results(args.output_dir, "quality", contract["quality_workers"]):
        if value["status"] != "accepted" or value["ordinal"] % contract["caption_workers"] != args.rank:
            continue
        def run(v=value):
            frames = load_view(v, sources)
            annotation = model.caption(frames, contract["policy"])
            return dict(v, annotation=annotation, status="accepted" if annotation["accepted"] else "rejected",
                        reasons=[] if annotation["accepted"] else ["caption_or_overlay_rejected"])
        yield value["ordinal"], value["record"], run


def verify_models(output, phase, provenance):
    with lock(output / f".{phase}_models.lock", blocking=True):
        path = output / f"{phase}_models.json"
        if path.is_file() and json.loads(path.read_text()) != provenance:
            raise ValueError("Model weights/revisions changed; use a new release directory")
        atomic_json(path, provenance)


def run_worker(args, *, model=None):
    contract = load_contract(args)
    workers = contract[f"{args.phase}_workers"]
    if not 0 <= args.rank < workers:
        raise ValueError("Invalid worker rank")
    path = result_paths(args.output_dir, args.phase, workers)[args.rank]
    contract_sha = sha256_file(args.output_dir / "exp2_contract.json")
    with lock(path.with_suffix(".lock")):
        marker = path.with_suffix(".complete.json")
        if marker.is_file():
            receipt = json.loads(marker.read_text())
            if receipt["sha256"] != sha256_file(path) or receipt.get("contract_sha256") != contract_sha:
                raise ValueError("Completed worker results changed")
            print(f"Reusing completed {args.phase} worker {args.rank}", flush=True)
            return
        results, errors, processed = Results(path, contract_sha), 0, 0
        iterator = quality_worker(args, contract, model) if args.phase == "quality" else caption_worker(args, contract, model)
        started = time.monotonic()
        try:
            for ordinal, record, operation in iterator:
                if results.done(record["sample_id"]):
                    continue
                sample_started = time.monotonic()
                try:
                    value = operation()
                except Exception as exc:
                    # Infrastructure/model/JSON failures are NOT silently dropped as bad data.
                    value = dict(ordinal=ordinal, record=record, status="error", error=f"{type(exc).__name__}: {exc}")
                    errors += 1
                value["processing_seconds"] = time.monotonic() - sample_started
                results.write(value)
                processed += 1
                if processed % args.log_every == 0:
                    print(json.dumps(dict(phase=args.phase, rank=args.rank, processed=processed,
                                          errors=errors, last_sample=record["sample_id"])), flush=True)
                if errors >= args.max_errors:
                    raise RuntimeError("Too many runtime/model failures; fix and resubmit unchanged")
        finally:
            results.close()
        if errors:
            raise RuntimeError(f"{errors} unresolved errors; resubmit unchanged to retry")
        elapsed = time.monotonic() - started
        atomic_json(marker, dict(sha256=sha256_file(path), contract_sha256=contract_sha, phase=args.phase,
                                 rank=args.rank, processed_this_invocation=processed, elapsed_seconds=elapsed))
        print(json.dumps(dict(phase=args.phase, rank=args.rank, processed=processed, elapsed_seconds=elapsed,
                              status="complete")), flush=True)


def quantile_report(db):
    # Exact SQL order statistics without keeping all score dictionaries in RAM.
    report = {}
    for key in ("musiq_mean", "aesthetic_mean", "motion_mean_pixels", "siglip_logit_mean", "source_fps",
                "quality_seconds_per_sample", "caption_seconds_per_sample"):
        count = db.execute("SELECT COUNT(*) FROM metrics WHERE name=?", (key,)).fetchone()[0]
        if count:
            report[key] = dict(count=count)
            for name, q in (("p01", .01), ("p10", .1), ("median", .5), ("p90", .9), ("p99", .99)):
                report[key][name] = db.execute("SELECT value FROM metrics WHERE name=? ORDER BY value LIMIT 1 OFFSET ?",
                                               (key, round((count - 1) * q))).fetchone()[0]
    return report


def finalize(args):
    contract = load_contract(args)
    output, policy = args.output_dir, contract["policy"]
    with lock(output / ".finalize.lock"), ExitStack() as stack:
        if (output / ".stage1_data_complete").exists():
            existing = json.loads((output / "stage1_summary.json").read_text())
            if existing != json.loads((output / ".stage1_data_complete").read_text()):
                raise ValueError("Modified completed release")
            if not existing.get("preview_only"):
                verify_release(output)
            print(f"Release already complete: {output}", flush=True)
            return
        # Temporary analysis DB is disposable, unlike the resumable worker DBs.
        db_path = output / "work" / "finalize.sqlite"
        db_path.unlink(missing_ok=True)
        for suffix in ("-journal", "-wal", "-shm"):
            db_path.with_name(db_path.name + suffix).unlink(missing_ok=True)
        db = sqlite3.connect(db_path)
        stack.callback(db.close)
        db.execute("CREATE TABLE seen (kind TEXT, hash TEXT, split TEXT, PRIMARY KEY(kind,hash))")
        db.execute("CREATE TABLE metrics (name TEXT, value REAL)")
        writers = {}
        for split in ("train", "validation"):
            (output / split).mkdir(exist_ok=True)
            for task in ("t2i", "t2v", "image_reconstruction"):
                writers[split, task] = stack.enter_context((output / split / f"{task}.jsonl.tmp").open("w"))
        audit = stack.enter_context((output / "human_audit.csv").open("w"))
        fields = ["sample_id", "split", "task", "original_caption", "caption_short", "caption_long",
                  "caption_supported", "motion_observation", "confidence", "window_start", "retained_crop_area", "manual_verdict", "notes"]
        audit_writer = csv.DictWriter(audit, fieldnames=fields)
        audit_writer.writeheader()
        counts, rejected, input_counts, motion_counts, short_count = Counter(), Counter(), Counter(), Counter(), Counter()
        quality_count = caption_count = 0
        for value in completed_results(output, "quality", contract["quality_workers"]):
            quality_count += 1
            record = value["record"]
            input_counts[f"{value['split']}.{record['task']}"] += 1
            db.execute("INSERT INTO metrics VALUES (?, ?)", ("quality_seconds_per_sample", value["processing_seconds"]))
            if value["status"] == "rejected":
                rejected.update(value["reasons"])
            for name, score in value.get("quality", {}).items():
                if name in {"musiq_mean", "aesthetic_mean", "motion_mean_pixels"} and score is not None:
                    db.execute("INSERT INTO metrics VALUES (?, ?)", (name, score))
            temporal = value.get("transform", {}).get("temporal")
            if temporal:
                db.execute("INSERT INTO metrics VALUES (?, ?)", ("source_fps", temporal["measured_source_fps"]))
        expected = sum(1 for _ in parent_records(contract))
        if quality_count != expected:
            raise ValueError("Quality workers did not cover all planned samples")
        accepted_quality = 0
        for path in result_paths(output, "quality", contract["quality_workers"]):
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as source:
                accepted_quality += source.execute("SELECT COUNT(*) FROM results WHERE status='accepted'").fetchone()[0]
        for value in completed_results(output, "caption", contract["caption_workers"]):
            caption_count += 1
            db.execute("INSERT INTO metrics VALUES (?, ?)", ("caption_seconds_per_sample", value["processing_seconds"]))
            motion_counts[f"{value['record']['task']}.{value['status']}.{value['annotation']['motion_observation']}"] += 1
            if value["status"] == "rejected":
                rejected.update(value["reasons"])
                continue
            record, annotation, split = value["record"], value["annotation"], value["split"]
            kind, sha = record["target"]["kind"], record["target"]["sha256"]
            previous = db.execute("SELECT split FROM seen WHERE kind=? AND hash=?", (kind, sha)).fetchone()
            if previous:
                raise ValueError(f"Duplicate or cross-split source media: {sha}")
            db.execute("INSERT INTO seen VALUES (?, ?, ?)", (kind, sha, split))
            short = fraction(record["sample_id"], policy["seed"], "caption_length") < policy["short_caption_fraction"]
            key = f"{split}.{record['task']}"
            short_count[key] += int(short)
            db.execute("INSERT INTO metrics VALUES (?, ?)", ("siglip_logit_mean", annotation["siglip_logit_mean"]))
            new = dict(record, exp2_transform=value["transform"],
                       exp2_view_sha256=digest(dict(target=record["target"], transform=value["transform"])),
                       prompt=annotation["caption_short" if short else "caption_long"],
                       caption_short=annotation["caption_short"], caption_long=annotation["caption_long"],
                       original_caption=record["prompt"], caption_length="short" if short else "long",
                       caption_verification=dict(model=annotation["qwen_model"], revision=annotation["qwen_revision"],
                                                 caption_supported=annotation["caption_supported"],
                                                 motion_observation=annotation["motion_observation"],
                                                 source_caption_provided=False, generation_passes=1,
                                                 confidence=annotation["confidence"], view="exact_fixed_crop_and_window"))
            tasks = (record["task"], "image_reconstruction") if kind == "image" else (record["task"],)
            for task in tasks:
                item = dict(new, task=task, sample_id=record["sample_id"].replace("t2i:", "image_reconstruction:", 1)
                            if task == "image_reconstruction" else record["sample_id"],
                            prompt="" if task == "image_reconstruction" else new["prompt"])
                writers[split, task].write(json.dumps(item, allow_nan=False) + "\n")
                counts[f"{split}.{task}"] += 1
            # Sample both static and moving clips; inspect uncertain responses without another model pass.
            if fraction(record["sample_id"], policy["seed"], "human_audit") < .001 or annotation["motion_observation"] == "unclear" or annotation["confidence"] < .9:
                temporal = value["transform"]["temporal"]
                audit_writer.writerow(dict(sample_id=record["sample_id"], split=split, task=record["task"],
                                           original_caption=record["prompt"], caption_short=annotation["caption_short"],
                                           caption_long=annotation["caption_long"], confidence=annotation["confidence"],
                                           caption_supported=annotation["caption_supported"],
                                           motion_observation=annotation["motion_observation"],
                                           window_start=temporal["start_seconds"] if temporal else "",
                                           retained_crop_area=value["transform"]["spatial"]["retained_area_fraction"]))
        if caption_count != accepted_quality:
            raise ValueError("Caption workers did not cover all quality survivors")
        if any(counts[f"{s}.{t}"] < 1 for s in ("train", "validation") for t in ("t2i", "t2v", "image_reconstruction")):
            raise ValueError("An image/video task or validation split is empty after filtering")
        for writer in writers.values():
            writer.flush()
        db.execute("CREATE INDEX metric_value ON metrics(name,value)")
        db.commit()
        metrics = quantile_report(db)
        artifacts = {}
        for (split, task), writer in writers.items():
            path = output / split / f"{task}.jsonl"
            Path(writer.name).replace(path)
            artifacts[f"{split}.{task}"] = dict(path=str(path.relative_to(output)), sha256=sha256_file(path))
        # Verify actual training readers, including matching reconstruction inputs.
        sources, checked = load_sources(output), {}
        for name, artifact in artifacts.items():
            checked[name] = 0
            with (output / artifact["path"]).open() as stream:
                for line in stream:
                    read_alignment_sample(json.loads(line), sources)
                    checked[name] += 1
                    if checked[name] >= args.reader_checks:
                        break
        summary = dict(status="raw_data_ready", format=FORMAT, tasks=["t2i", "t2v", "image_reconstruction"],
                       counts=dict(counts), input_counts=dict(input_counts), rejected=dict(rejected),
                       motion_observation_counts=dict(motion_counts), short_caption_counts=dict(short_count),
                       manifests=artifacts, sources_sha256=sha256_file(output / "sources.json"),
                       parent_summary_sha256=contract["parent_summary_sha256"], exp2_contract_sha256=sha256_file(output / "exp2_contract.json"),
                       scoring_models={phase: json.loads((output / f"{phase}_models.json").read_text())
                                       for phase in ("quality", "caption")},
                       reader_checks=checked, reader_video_frames=49, vae_encoded=False, trainer_integrated=True,
                       preview_only=contract["preview_only"], semantics_verified=False,
                       semantic_check="Qwen3.6 visual_only_self_assessment; one_generation_pass; not_ground_truth; human_audit_required",
                       overlay_check="Qwen3.6 visual judgment; not dedicated OCR bounding-box coverage",
                       temporal_policy="fixed_seeded_single_shot_49_frames_at_24fps; 2_seconds_timestamp_span",
                       spatial_policy="aspect_preserving_resize_then_fixed_crop; same_view_for_caption_and_training",
                       reconstruction_policy="same_crop_as_t2i; image_only_through_MLLM; no_clean_DiT_target_condition",
                       split_policy="inherited_source_media_split; never_resplit_windows",
                       seed=policy["seed"],
                       metrics=metrics, policy=policy, quality_claim="none; inspect human_audit and score distributions before training")
        atomic_json(output / "stage1_summary.json", summary)
        atomic_json(output / ".stage1_data_complete", summary)
        print(json.dumps(summary, indent=2), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("plan", "quality", "caption", "finalize"), required=True)
    parser.add_argument("--input-root", type=Path, default=Path("download_data/data/univideo_stage1_expanded650"))
    parser.add_argument("--output-dir", type=Path, default=Path("download_data/data/mobileov_exp2_visual_only_v2"))
    parser.add_argument("--policy", type=Path, default=ROOT / "configs/mobileov_exp2_data.json")
    parser.add_argument("--quality-workers", type=int, default=8)
    parser.add_argument("--caption-workers", type=int, default=4)
    parser.add_argument("--limit-per-manifest", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-errors", type=int, default=5)
    parser.add_argument("--reader-checks", type=int, default=8)
    parser.add_argument("--vbench-cache", type=Path, default=Path(os.environ.get("VBENCH_CACHE_DIR", Path.home() / ".cache/vbench")))
    parser.add_argument("--vbench-repo", type=Path, default=ROOT / "checkpoints/vbench_repo")
    args = parser.parse_args(argv)
    if min(args.log_every, args.max_errors, args.reader_checks) < 1:
        parser.error("Logging/error/check counts must be positive")
    args.output_dir = args.output_dir.resolve()
    return args


def main():
    args = parse_args()
    if args.phase == "plan":
        build_plan(args)
    elif args.phase == "finalize":
        finalize(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
