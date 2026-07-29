#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from fnmatch import fnmatch
import json
import multiprocessing
from pathlib import Path
import random
import shutil
import time
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


@dataclass(frozen=True)
class RepoFileSpec:
    path: str
    size: int
    oid: str


@dataclass(frozen=True)
class DownloadTask:
    repo_id: str
    revision: str
    output_dir: str
    file: RepoFileSpec
    retries: int


@dataclass(frozen=True)
class DownloadResult:
    path: str
    size: int
    status: str
    attempts: int
    error: str = ""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _matches(path: str, includes: list[str], excludes: list[str]) -> bool:
    if includes and not any(fnmatch(path, pattern) for pattern in includes):
        return False
    return not any(fnmatch(path, pattern) for pattern in excludes)


def _resolve_revision(
    *,
    api: HfApi,
    repo_id: str,
    requested_revision: str,
    output_dir: Path,
    refresh: bool,
) -> str:
    pin_path = output_dir / ".source_revision.json"
    if pin_path.is_file() and not refresh:
        payload = json.loads(pin_path.read_text())
        if payload.get("repo_id") == repo_id and payload.get("resolved_revision"):
            revision = str(payload["resolved_revision"])
            print(f"Reusing pinned revision: {revision}", flush=True)
            return revision

    info = api.repo_info(repo_id=repo_id, repo_type="dataset", revision=requested_revision)
    if not info.sha:
        raise RuntimeError(f"Could not resolve revision {requested_revision!r} for {repo_id}")
    payload = {
        "repo_id": repo_id,
        "requested_revision": requested_revision,
        "resolved_revision": info.sha,
        "resolved_at_unix": time.time(),
    }
    _atomic_json(pin_path, payload)
    print(f"Pinned {repo_id}@{requested_revision} to {info.sha}", flush=True)
    return info.sha


def _list_files(
    api: HfApi,
    *,
    repo_id: str,
    revision: str,
    includes: list[str],
    excludes: list[str],
) -> list[RepoFileSpec]:
    files: list[RepoFileSpec] = []
    for item in api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        recursive=True,
        expand=False,
    ):
        size = getattr(item, "size", None)
        path = getattr(item, "path", None)
        if size is None or path is None or not _matches(path, includes, excludes):
            continue
        files.append(
            RepoFileSpec(
                path=str(path),
                size=int(size),
                oid=str(getattr(item, "blob_id", "") or getattr(item, "oid", "")),
            )
        )
    files.sort(key=lambda item: item.path)
    return files


def _is_complete(output_dir: Path, file: RepoFileSpec) -> bool:
    path = output_dir / file.path
    return path.is_file() and path.stat().st_size == file.size


def _download_one(task: DownloadTask) -> DownloadResult:
    output_dir = Path(task.output_dir)
    destination = output_dir / task.file.path
    if destination.is_file() and destination.stat().st_size == task.file.size:
        return DownloadResult(task.file.path, task.file.size, "existing", 0)
    if destination.exists():
        destination.unlink()

    last_error = ""
    for attempt in range(1, task.retries + 1):
        try:
            downloaded = Path(
                hf_hub_download(
                    repo_id=task.repo_id,
                    filename=task.file.path,
                    repo_type="dataset",
                    revision=task.revision,
                    local_dir=output_dir,
                    force_download=False,
                    etag_timeout=60,
                )
            )
            actual_size = downloaded.stat().st_size
            if actual_size != task.file.size:
                raise RuntimeError(
                    f"size mismatch: expected={task.file.size}, actual={actual_size}"
                )
            return DownloadResult(task.file.path, task.file.size, "downloaded", attempt)
        except Exception as error:  # Retrying network and signed-URL failures is intentional.
            last_error = f"{type(error).__name__}: {error}"
            if attempt < task.retries:
                delay = min(60.0, 2.0 ** min(attempt, 5)) + random.random()
                time.sleep(delay)
    return DownloadResult(
        task.file.path,
        task.file.size,
        "failed",
        task.retries,
        last_error,
    )


def _gib(value: int) -> float:
    return value / (1024**3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face dataset with resumable file-level process parallelism."
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--refresh-revision", action="store_true")
    parser.add_argument("--skip-disk-check", action="store_true")
    parser.add_argument("--disk-margin-gib", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-files",
        type=int,
        default=-1,
        help="Testing only: download at most this many files after filtering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.retries < 1:
        raise ValueError("--retries must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_marker = output_dir / ".download_complete"
    if not args.dry_run:
        complete_marker.unlink(missing_ok=True)

    api = HfApi()
    revision = _resolve_revision(
        api=api,
        repo_id=args.repo_id,
        requested_revision=args.revision,
        output_dir=output_dir,
        refresh=args.refresh_revision,
    )
    files = _list_files(
        api,
        repo_id=args.repo_id,
        revision=revision,
        includes=args.include,
        excludes=args.exclude,
    )
    if args.max_files >= 0:
        files = files[: args.max_files]
    if not files:
        raise RuntimeError("No repository files matched the requested filters")

    expected_bytes = sum(item.size for item in files)
    pending = [item for item in files if not _is_complete(output_dir, item)]
    pending_bytes = sum(item.size for item in pending)
    manifest = {
        "repo_id": args.repo_id,
        "revision": revision,
        "output_dir": str(output_dir),
        "include": args.include,
        "exclude": args.exclude,
        "files": [asdict(item) for item in files],
        "expected_file_count": len(files),
        "expected_bytes": expected_bytes,
    }
    _atomic_json(output_dir / ".download_manifest.json", manifest)

    print(
        f"Dataset={args.repo_id}@{revision} files={len(files)} "
        f"total={_gib(expected_bytes):.2f} GiB",
        flush=True,
    )
    print(
        f"Already complete={len(files) - len(pending)} "
        f"pending={len(pending)} pending_size={_gib(pending_bytes):.2f} GiB "
        f"workers={args.workers}",
        flush=True,
    )

    if args.dry_run:
        print("Dry run complete; no dataset files were downloaded.", flush=True)
        return

    if not args.skip_disk_check:
        free_bytes = shutil.disk_usage(output_dir).free
        required = pending_bytes + int(args.disk_margin_gib * 1024**3)
        print(
            f"Disk free={_gib(free_bytes):.2f} GiB "
            f"required_with_margin={_gib(required):.2f} GiB",
            flush=True,
        )
        if free_bytes < required:
            raise RuntimeError(
                "Insufficient filesystem free space for pending files. "
                "Free space or use --skip-disk-check only after checking project quota."
            )

    results: list[DownloadResult] = [
        DownloadResult(item.path, item.size, "existing", 0)
        for item in files
        if item not in pending
    ]
    if pending:
        context = multiprocessing.get_context("spawn")
        tasks = [
            DownloadTask(
                repo_id=args.repo_id,
                revision=revision,
                output_dir=str(output_dir),
                file=item,
                retries=args.retries,
            )
            for item in pending
        ]
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
            futures = {executor.submit(_download_one, task): task.file.path for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                detail = f" error={result.error}" if result.error else ""
                print(
                    f"[{index}/{len(tasks)}] {result.status}: {result.path} "
                    f"({_gib(result.size):.2f} GiB, attempts={result.attempts}){detail}",
                    flush=True,
                )

    failed = [result for result in results if result.status == "failed"]
    verified = [item for item in files if _is_complete(output_dir, item)]
    summary = {
        "repo_id": args.repo_id,
        "revision": revision,
        "output_dir": str(output_dir),
        "expected_file_count": len(files),
        "expected_bytes": expected_bytes,
        "verified_file_count": len(verified),
        "verified_bytes": sum(item.size for item in verified),
        "failed": [asdict(item) for item in failed],
        "status": "complete" if len(verified) == len(files) and not failed else "incomplete",
        "finished_at_unix": time.time(),
    }
    _atomic_json(output_dir / "download_summary.json", summary)

    if summary["status"] != "complete":
        raise RuntimeError(
            f"Dataset download incomplete: verified={len(verified)}/{len(files)}, "
            f"failed={len(failed)}. Resubmit the same job to resume."
        )

    complete_marker.write_text(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "revision": revision,
                "file_count": len(files),
                "bytes": expected_bytes,
            },
            sort_keys=True,
        )
        + "\n"
    )
    print(
        f"Completed {args.repo_id}: files={len(files)} size={_gib(expected_bytes):.2f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
