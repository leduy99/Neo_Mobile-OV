#!/usr/bin/env python
"""Download/index public image-caption data, not UniVideo's private corpus."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import time

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

REPO_ID = "BLIP3o/BLIP3o-Pretrain-Long-Caption"
INDEX_VERSION = 2
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class Shard:
    path: str
    size: int
    sha256: str


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and "\\" not in name


def shard_group(path: str) -> str:
    return re.sub(r"[_-]\d+\.tar$", "", path)


def select_shards(shards: list[Shard], count: int, seed: int) -> list[Shard]:
    if not 1 <= count <= len(shards):
        raise ValueError(f"Requested {count} shards, but available={len(shards)}")
    groups: dict[str, list[Shard]] = defaultdict(list)
    for shard in shards:
        if not safe_member(shard.path) or len(PurePosixPath(shard.path).parts) != 1:
            raise ValueError(f"Expected a safe, flat archive path: {shard.path}")
        if shard.size <= 0 or not re.fullmatch(r"[0-9a-f]{64}", shard.sha256):
            raise ValueError(f"Missing size or LFS SHA-256: {shard.path}")
        groups[shard_group(shard.path)].append(shard)
    total = len(shards)
    quotas = {key: count * len(value) // total for key, value in groups.items()}
    order = sorted(groups, key=lambda key: (-(count * len(groups[key]) % total), key))
    for key in order[:count - sum(quotas.values())]:
        quotas[key] += 1
    selected = []
    for key in sorted(groups):
        candidates = sorted(groups[key], key=lambda item: hashlib.sha256(
            f"{seed}:{item.path}".encode()).hexdigest())
        selected.extend(candidates[:quotas[key]])
    return sorted(selected, key=lambda item: item.path)


def resolve_plan(output: Path, *, repo_id: str, revision: str, count: int, seed: int) -> dict:
    settings = dict(repo_id=repo_id, requested_revision=revision, num_shards=count, seed=seed)
    plan_path = output / "download_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("settings") != settings:
            raise ValueError("Existing download plan differs. Use a new output directory; do not mix releases.")
        return plan
    api = HfApi()
    pinned = api.dataset_info(repo_id, revision=revision).sha
    if not pinned:
        raise RuntimeError("Hugging Face did not return a dataset revision")
    candidates = []
    for item in api.list_repo_tree(repo_id, repo_type="dataset", revision=pinned):
        if item.path.endswith(".tar"):
            lfs = getattr(item, "lfs", None)
            candidates.append(Shard(item.path, item.size, getattr(lfs, "sha256", "")))
    selected = select_shards(candidates, count, seed)
    return {
        "settings": settings,
        "revision": pinned,
        "selection": "seeded proportional sampling by archive filename family, not semantic balancing",
        "available_shards": len(candidates),
        "selected_groups": dict(Counter(shard_group(item.path) for item in selected)),
        "expected_bytes": sum(item.size for item in selected),
        "shards": [asdict(item) for item in selected],
    }


def download_shard(shard: Shard, *, output: Path, repo_id: str, revision: str, retries: int) -> Path:
    if not safe_member(shard.path):
        raise ValueError(f"Unsafe download path: {shard.path}")
    destination = output / "shards" / shard.path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == shard.size:
        if sha256_file(destination) == shard.sha256:
            return destination
        print(f"Replacing checksum-mismatched archive: {shard.path}", flush=True)
    force = destination.exists()
    last_error = None
    for attempt in range(retries):
        try:
            downloaded = Path(hf_hub_download(
                repo_id=repo_id, repo_type="dataset", revision=revision,
                filename=shard.path, local_dir=output / "shards", force_download=force,
            ))
            if downloaded.resolve() != destination.resolve():
                raise RuntimeError(f"Unexpected download location: {downloaded}")
            if downloaded.stat().st_size != shard.size or sha256_file(downloaded) != shard.sha256:
                force = True
                raise RuntimeError(f"Size/SHA-256 mismatch: {shard.path}")
            return downloaded
        except Exception as error:
            last_error = error
            # Hub can fail before returning a path (e.g. an unusable partial/HTTP 416).
            # Restart only this shard on retry; verified archives remain untouched.
            force = True
            if attempt + 1 < retries:
                print(f"Retry {attempt + 1}/{retries}: {shard.path}: {error}", flush=True)
                time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"Download failed after {retries} attempts: {shard.path}") from last_error


def index_shard(archive: Path, shard: Shard, output: Path, *, repo_id: str, revision: str) -> dict:
    index_dir = output / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / f"{shard.path}.jsonl"
    receipt_path = index_dir / f"{shard.path}.json"
    contract = dict(index_version=INDEX_VERSION, archive_sha256=shard.sha256,
                    repo_id=repo_id, revision=revision)
    if receipt_path.exists() and index_path.exists():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (all(receipt.get(key) == value for key, value in contract.items())
                    and sha256_file(index_path) == receipt.get("index_sha256")):
                return receipt
        except (OSError, ValueError):
            pass

    pairs: dict[str, dict[str, tarfile.TarInfo]] = defaultdict(dict)
    rejected: Counter = Counter()
    examples = []
    temporary = index_path.with_suffix(".jsonl.tmp")
    valid = 0
    # r: intentionally rejects compressed archives: the index stores byte offsets.
    with tarfile.open(archive, mode="r:") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if not safe_member(member.name):
                raise ValueError(f"Unsafe archive member: {member.name}")
            path = PurePosixPath(member.name)
            suffix = path.suffix.lower()
            kind = "image" if suffix in IMAGE_SUFFIXES else "caption" if suffix == ".txt" else None
            if kind is None:
                continue
            pair = pairs[str(path.with_suffix(""))]
            if kind in pair:
                raise ValueError(f"Ambiguous duplicate {kind} member: {member.name}")
            pair[kind] = member
        with temporary.open("w", encoding="utf-8") as target:
            for key, pair in pairs.items():
                if "image" not in pair or "caption" not in pair:
                    rejected["unpaired_member"] += 1
                    continue
                if pair["caption"].size > 1024 * 1024 or pair["image"].size > 64 * 1024 * 1024:
                    rejected["oversized_member"] += 1
                    continue
                # Storage failures must abort the shard, never count as bad samples.
                with tar.extractfile(pair["caption"]) as stream:
                    caption_bytes = stream.read()
                with tar.extractfile(pair["image"]) as stream:
                    image_bytes = stream.read()
                try:
                    caption = caption_bytes.decode("utf-8").strip()
                    if not caption or "\x00" in caption:
                        raise ValueError("empty_or_invalid_caption")
                    with Image.open(io.BytesIO(image_bytes)) as img:
                        img.load()
                        width, height = img.size
                    record = {
                        "sample_id": hashlib.sha256(f"{repo_id}@{revision}:{shard.path}:{key}".encode()).hexdigest(),
                        "key": key, "shard": f"shards/{shard.path}",
                        "image_member": pair["image"].name, "caption_member": pair["caption"].name,
                        "image_offset": pair["image"].offset_data, "image_size": len(image_bytes),
                        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                        "width": width, "height": height, "caption": caption,
                    }
                except (OSError, ValueError, UnicodeError, Image.DecompressionBombError) as error:
                    rejected["invalid_image_or_caption"] += 1
                    if len(examples) < 5:
                        examples.append({"key": key, "error": str(error)})
                    continue
                target.write(json.dumps(record, ensure_ascii=True) + "\n")
                valid += 1
    if not valid:
        raise RuntimeError(f"No readable image-caption pairs in {archive}; check archive schema")
    temporary.replace(index_path)
    receipt = {
        **contract, "shard": shard.path, "valid_pairs": valid,
        "rejected": dict(rejected), "rejection_examples": examples,
        "image_decode_check": "all_indexed_images",
        "index": str(index_path.relative_to(output)), "index_sha256": sha256_file(index_path),
    }
    atomic_json(receipt_path, receipt)
    return receipt


def load_indexed_image(output: Path, record: dict) -> Image.Image:
    relative = record["shard"]
    if not safe_member(relative):
        raise ValueError("Unsafe shard reference")
    archive = (output / relative).resolve()
    if not archive.is_relative_to(output.resolve()):
        raise ValueError("Shard reference escapes output directory")
    with archive.open("rb") as stream:
        stream.seek(record["image_offset"])
        data = stream.read(record["image_size"])
    if hashlib.sha256(data).hexdigest() != record["image_sha256"]:
        raise RuntimeError("Indexed image checksum mismatch")
    with Image.open(io.BytesIO(data)) as img:
        return img.convert("RGB")


@contextmanager
def output_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".download.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another downloader owns this output directory; do not submit duplicate jobs") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run(args: argparse.Namespace) -> dict:
    if args.num_shards < 1 or args.workers < 1 or args.retries < 1:
        raise ValueError("Shard count, workers, and retries must be positive")
    if args.max_download_gib <= 0 or args.disk_margin_gib < 0 or not 0 <= args.max_reject_fraction < 1:
        raise ValueError("Invalid storage/rejection limits")
    output = args.output_dir.expanduser().resolve()
    with output_lock(output):
        plan = resolve_plan(output, repo_id=args.repo_id, revision=args.revision,
                            count=args.num_shards, seed=args.seed)
        expected = plan["expected_bytes"]
        if expected > args.max_download_gib * 1024 ** 3:
            raise RuntimeError(f"Selected archives exceed --max-download-gib: {expected / 1024**3:.2f} GiB")
        print(json.dumps({key: value for key, value in plan.items() if key != "shards"}, indent=2), flush=True)
        print(f"Archives: {expected / 10**9:.2f} GB ({expected / 1024**3:.2f} GiB). "
              "File list and hashes are saved in download_plan.json during a real run.", flush=True)
        if args.dry_run:
            print("Dry run: metadata only; no archives downloaded and no completion marker written.", flush=True)
            return plan
        (output / ".download_complete").unlink(missing_ok=True)
        atomic_json(output / "download_plan.json", plan)
        shards = [Shard(**item) for item in plan["shards"]]
        # Include space to replace one corrupt archive per concurrent worker.
        pending = sum(max(0, item.size - ((output / "shards" / item.path).stat().st_size
                      if (output / "shards" / item.path).exists() else 0)) for item in shards)
        replacement = sum(sorted((item.size for item in shards), reverse=True)[:args.workers])
        required = pending + replacement + int(args.disk_margin_gib * 1024**3)
        free = shutil.disk_usage(output).free
        if free < required:
            raise RuntimeError(f"Insufficient filesystem space: free={free}, required={required}. Check quota too.")
        (output / "shards").mkdir(exist_ok=True)

        def process(shard: Shard) -> dict:
            archive = download_shard(shard, output=output, repo_id=args.repo_id,
                                     revision=plan["revision"], retries=args.retries)
            return index_shard(archive, shard, output, repo_id=args.repo_id, revision=plan["revision"])

        results, failures = {}, []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process, shard): shard.path for shard in shards}
            for number, future in enumerate(as_completed(futures), 1):
                name = futures[future]
                try:
                    result = future.result()
                    results[name] = result
                    print(f"[{number}/{len(shards)}] verified {name}: pairs={result['valid_pairs']} rejected={result['rejected']}", flush=True)
                except Exception as error:
                    failures.append({"shard": name, "error": repr(error)})
                    print(f"[{number}/{len(shards)}] FAILED {name}: {error}", flush=True)
        rejected = Counter()
        for result in results.values():
            rejected.update(result["rejected"])
        valid = sum(result["valid_pairs"] for result in results.values())
        reject_fraction = sum(rejected.values()) / max(1, valid + sum(rejected.values()))
        if reject_fraction > args.max_reject_fraction:
            failures.append({"error": f"Rejected fraction {reject_fraction:.4f} exceeds {args.max_reject_fraction}"})
        summary = {
            "status": "incomplete" if failures else "complete",
            "repo_id": args.repo_id, "revision": plan["revision"],
            "shards_requested": len(shards), "shards_verified": len(results),
            "archive_bytes": expected, "valid_image_caption_pairs": valid,
            "rejected": dict(rejected), "reject_fraction": reject_fraction, "failures": failures,
            "archive_check": "size_and_lfs_sha256", "image_decode_check": "all_indexed_images",
            "semantics_verified": False, "deduplicated": False,
            "purpose": "public_single_frame_alignment_candidate_not_original_UniVideo_data",
            "remaining_before_training": ["deduplicate_and_split", "benchmark_leakage_check", "NeoDragon_VAE_encoding"],
        }
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
            raise RuntimeError("Download/index incomplete. Fix the reported error and resubmit unchanged to resume.")
        atomic_json(output / ".download_complete", summary)
        return summary


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", type=Path, default=Path("download_data/data/univideo_alignment_images"))
    parser.add_argument("--num-shards", type=int, default=100, help="About 1M pairs; actual count is reported after indexing")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--max-download-gib", type=float, default=80)
    parser.add_argument("--disk-margin-gib", type=float, default=10)
    parser.add_argument("--max-reject-fraction", type=float, default=0.01)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
