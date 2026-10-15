#!/usr/bin/env python
"""Download the bounded Job-B understanding sources, not a trainer-ready QA release."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.data_prepare.download_alignment_images import atomic_json, output_lock

DEFAULT_PLAN = ROOT / "configs/data_sources/mobileov_job_b_understanding_sources.json"
FORMAT = "mobileov_job_b_understanding_sources_v1"
KINDS = {"train_parquet", "qa_annotation", "media_archive", "dataset_card"}


def selected_files(plan):
    if not isinstance(plan, dict) or plan.get("format") != FORMAT or not isinstance(plan.get("sources"), dict) or set(plan["sources"]) != {
            "image_qa", "text_instruction", "video_qa"}:
        raise ValueError("Require the explicit image/text/video Job-B source plan")
    rows = []
    for name, source in plan["sources"].items():
        if not isinstance(source, dict):
            raise ValueError("Invalid source selection")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", source.get("repo_id", "")):
            raise ValueError("Invalid dataset repository")
        if not re.fullmatch(r"[0-9a-f]{40}", source.get("revision", "")):
            raise ValueError("Pin sources to exact commits, not main or a moving tag")
        files = source.get("files", [])
        if not isinstance(files, list) or not files:
            raise ValueError("Empty source selection")
        seen = set()
        for entry in files:
            if not isinstance(entry, dict):
                raise ValueError("Invalid file selection")
            filename = entry.get("filename")
            if not isinstance(filename, str) or not filename or "\\" in filename or any(ord(char) < 32 for char in filename):
                raise ValueError("Invalid source path")
            path = PurePosixPath(filename)
            if not path.parts or path.is_absolute() or ".." in path.parts or str(path) != filename or filename in seen:
                raise ValueError("Unsafe or duplicate source path")
            seen.add(filename)
            if type(entry.get("size")) is not int or entry["size"] < 1 or entry.get("kind") not in KINDS:
                raise ValueError("Invalid source size/kind")
            algorithm = entry.get("hash_algorithm")
            length = 64 if algorithm == "sha256" else 40 if algorithm == "git_blob_sha1" else 0
            if not length or not re.fullmatch(rf"[0-9a-f]{{{length}}}", entry.get("hash", "")):
                raise ValueError("Every selected source needs a pinned content digest")
            if entry["kind"] == "train_parquet" and not re.search(r"(^|/)train-[^/]+\.parquet$", filename):
                raise ValueError("Only declared train parquet files are allowed")
            rows.append(dict(entry, source=name, repo_id=source["repo_id"], revision=source["revision"]))
        if not any(entry["kind"] != "dataset_card" for entry in files):
            raise ValueError("A dataset card alone is not a training source")
    return rows


def verify_file(path, row):
    if not path.is_file() or path.stat().st_size != row["size"]:
        raise ValueError(f"Missing or wrong-size source: {row['source']}/{row['filename']}")
    digest = hashlib.sha256() if row["hash_algorithm"] == "sha256" else hashlib.sha1()
    if row["hash_algorithm"] == "git_blob_sha1":
        digest.update(f"blob {row['size']}\0".encode("ascii"))
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    if digest.hexdigest() != row["hash"]:
        raise ValueError(f"Source checksum differs: {row['source']}/{row['filename']}")


def verify_metadata(rows, api):
    for source in sorted({row["source"] for row in rows}):
        selected = [row for row in rows if row["source"] == source]
        info = api.dataset_info(selected[0]["repo_id"], revision=selected[0]["revision"], files_metadata=True)
        if info.sha != selected[0]["revision"]:
            raise ValueError("Hub did not resolve the selected dataset commit")
        entries = {entry.rfilename: entry for entry in info.siblings}
        for row in selected:
            entry = entries.get(row["filename"])
            if entry is None or entry.size != row["size"]:
                raise ValueError(f"Pinned Hub source size/path differs: {row['filename']}")
            lfs = getattr(entry, "lfs", None)
            algorithm = "sha256" if lfs else "git_blob_sha1"
            digest = (lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256) if lfs else entry.blob_id
            if (algorithm, digest) != (row["hash_algorithm"], row["hash"]):
                raise ValueError(f"Pinned Hub source digest differs: {row['filename']}")


def run(args, *, api=None, fetch=None, sleep=time.sleep):
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    rows = selected_files(plan)
    size = sum(row["size"] for row in rows)
    if size > args.max_download_gib * 1024**3:
        raise ValueError("Selected sources exceed the download storage cap")
    contract = dict(format=FORMAT, plan=plan, files=rows, bytes=size,
                    plan_sha256=hashlib.sha256(plan_bytes).hexdigest())
    if args.dry_run:
        return dict(status="offline_plan_only", bytes=size, source_gib=size / 1024**3,
                    selected_files=len(rows), trainer_ready=False, plan_sha256=contract["plan_sha256"])
    from huggingface_hub import HfApi, hf_hub_download
    api, fetch = api or HfApi(), fetch or hf_hub_download
    verify_metadata(rows, api)
    output = args.output_dir.expanduser().resolve()
    with output_lock(output):
        contract_path = output / "source_contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("Existing source selection changed; use a new output directory")
        atomic_json(contract_path, contract)
        if args.metadata_only:
            return dict(status="Hub_metadata_verified_NO_source_download", bytes=size,
                        source_gib=size / 1024**3, selected_files=len(rows), trainer_ready=False)
        marker = output / "source_files_complete.json"
        marker.unlink(missing_ok=True)
        destinations = [output / "sources" / row["source"] / row["filename"] for row in rows]
        if any(not path.resolve().is_relative_to(output) for path in destinations):
            raise ValueError("Local source destination escapes the output directory")
        # Reserve room for concurrently replacing partial or corrupt cache files.
        missing_bytes = sum(row["size"] for row, path in zip(rows, destinations)
                            if not path.is_file() or path.stat().st_size != row["size"])
        replacement_bytes = sum(sorted((row["size"] for row in rows), reverse=True)[:args.workers])
        required = missing_bytes + replacement_bytes + args.disk_margin_gib * 1024**3
        if shutil.disk_usage(output).free < required:
            raise ValueError(f"Insufficient free space: need at least {required / 1024**3:.2f} GiB including temporary files")
        receipts = output / "receipts"
        receipts.mkdir(exist_ok=True)

        def one(row):
            path = output / "sources" / row["source"] / row["filename"]
            kwargs = dict(repo_id=row["repo_id"], repo_type="dataset", revision=row["revision"],
                          filename=row["filename"], local_dir=str(output / "sources" / row["source"]))
            try:
                verify_file(path, row)
                cached = True
            except ValueError:
                cached = False
                force = path.exists()
                for attempt in range(args.retries):
                    try:
                        returned = Path(fetch(**kwargs, force_download=force))
                        if returned.resolve() != path.resolve():
                            raise ValueError("Hub returned an unexpected local destination")
                        verify_file(path, row)
                        break
                    except Exception as error:
                        if attempt + 1 == args.retries:
                            raise RuntimeError(f"Failed source {row['source']}/{row['filename']}: {error}") from error
                        force = path.exists()
                        sleep(min(2 ** attempt, 8))
            receipt = dict(row, path=str(path), source_file_verified=True)
            key = hashlib.sha256(f"{row['source']}/{row['filename']}".encode()).hexdigest()
            atomic_json(receipts / f"{key}.json", receipt)
            print(f"Verified {'cache' if cached else 'download'} {row['source']}/{row['filename']} bytes={row['size']}", flush=True)
            return receipt

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            verified = list(pool.map(one, rows))
        report = dict(status="source_files_verified_QA_release_NOT_prepared", bytes=size,
                      source_gib=size / 1024**3, files=verified, selected_files=len(rows),
                      plan_sha256=contract["plan_sha256"], trainer_ready=False, vae_encoded=False,
                      semantics_verified=False, original_splits_verified=False, unique_qa_counts=None,
                      next="prepare QA manifests: join selected media, check source train provenance/licenses, decode, deduplicate, split by media and audit answers; do not auto-launch training")
        atomic_json(marker, report)
        return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-dir", type=Path, default=Path("download_data/data/mobileov_job_b_understanding_sources"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-download-gib", type=float, default=9)
    parser.add_argument("--disk-margin-gib", type=float, default=2)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate the offline selection; no network or writes")
    mode.add_argument("--metadata-only", action="store_true", help="Verify real pinned Hub metadata without source downloads")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16 or not 1 <= args.retries <= 5:
        parser.error("Invalid concurrency/retry limit")
    if (not math.isfinite(args.max_download_gib) or not 0 < args.max_download_gib <= 100
            or not math.isfinite(args.disk_margin_gib) or args.disk_margin_gib < 0):
        parser.error("Invalid download cap/disk margin")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not (args.dry_run or args.metadata_only or os.environ.get("SLURM_JOB_ID")):
        raise RuntimeError("Submit the Slurm data job; do not download bulk data on the login node")
    report = run(args)
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
