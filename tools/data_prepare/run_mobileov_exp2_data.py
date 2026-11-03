#!/usr/bin/env python
"""Run isolated GPU workers; reuse existing environments, caches and durable receipts."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/data_prepare/prepare_mobileov_exp2.py"


def gpu_groups(devices, size):
    if not devices or len(devices) % size:
        raise ValueError("GPU count must be divisible by the per-worker group size")
    return [devices[start:start + size] for start in range(0, len(devices), size)]


def run_group(commands, *, deadline=None):
    running, handles = [], []
    previous = {}
    interrupted = [False]
    def stop(signum, frame):
        interrupted[0] = True
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, stop)
        for command, env, log in commands:
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("a", buffering=1)
            handles.append(handle)
            handle.write(f"\nSTART {time.strftime('%Y-%m-%dT%H:%M:%S')} {command!r}\n")
            running.append(subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT))
        while running:
            if interrupted[0] or (deadline is not None and time.monotonic() >= deadline):
                return 75
            if any(p.poll() not in (None, 0) for p in running):
                raise RuntimeError("A data worker failed; see the per-worker logs. Resubmit unchanged after fixing it.")
            if all(p.poll() == 0 for p in running):
                return 0
            time.sleep(2)
    finally:
        for proc in running:
            if proc.poll() is None:
                proc.terminate()
        for proc in running:
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for handle in handles:
            handle.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "download_data/data/univideo_stage1_expanded650")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "download_data/data/mobileov_exp2_visual_only_v2")
    parser.add_argument("--policy", type=Path, default=ROOT / "configs/mobileov_exp2_data.json")
    parser.add_argument("--quality-python", required=True)
    parser.add_argument("--caption-python", required=True)
    parser.add_argument("--vbench-cache", required=True)
    parser.add_argument("--vbench-repo", default=str(ROOT / "checkpoints/vbench_repo"))
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--caption-gpus-per-worker", type=int, choices=(1, 2), default=2,
                        help="One GPU requires >=100 GiB VRAM; default two GPUs for Berzelius")
    parser.add_argument("--limit-per-manifest", type=int, default=0)
    parser.add_argument("--max-runtime-seconds", type=float, default=66 * 3600)
    parser.add_argument("--phase", choices=("all", "quality", "caption", "finalize"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    quality_groups, caption_groups = gpu_groups(devices, 1), gpu_groups(devices, args.caption_gpus_per_worker)
    output = args.output_dir.resolve()
    common = ["--output-dir", str(output)]
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{os.environ.get('PYTHONPATH', '')}",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1")
    phases = ("quality", "caption") if args.phase == "all" else (() if args.phase == "finalize" else (args.phase,))
    if args.dry_run:
        print(json.dumps(dict(output=str(output), quality_workers=len(quality_groups),
                              caption_workers=len(caption_groups), phases=phases,
                              original_media="read_only", qwen="visual_only_single_pass; source_caption_not_provided",
                              preview_only=bool(args.limit_per_manifest)), indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".launcher.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = [sys.executable, str(TOOL), "--phase", "plan", *common,
                "--input-root", str(args.input_root), "--policy", str(args.policy),
                "--quality-workers", str(len(quality_groups)), "--caption-workers", str(len(caption_groups)),
                "--limit-per-manifest", str(args.limit_per_manifest)]
        subprocess.run(plan, check=True, env=env, cwd=ROOT)
        # No implicit model download or in-place changes to training/evaluation environments.
        if "caption" in phases:
            subprocess.run([args.caption_python, "-c",
                            "from transformers import AutoModelForMultimodalLM, AutoProcessor, AutoConfig; "
                            "from tools.data_prepare.exp2_models import cached_snapshot_info; import json,pathlib; "
                            f"p=json.loads(pathlib.Path({str(args.policy)!r}).read_text()); "
                            "[AutoConfig.from_pretrained(p[k],revision=p[k.replace('_model','_revision')],local_files_only=True) "
                            "for k in ('qwen_model','siglip_model')]; print(cached_snapshot_info(p))"],
                           check=True, env=env)
        deadline = time.monotonic() + args.max_runtime_seconds if args.max_runtime_seconds else None
        for phase in phases:
            groups = quality_groups if phase == "quality" else caption_groups
            python = args.quality_python if phase == "quality" else args.caption_python
            commands = []
            for rank, group in enumerate(groups):
                command = [python, "-u", str(TOOL), "--phase", phase, *common, "--rank", str(rank),
                           "--vbench-cache", args.vbench_cache, "--vbench-repo", args.vbench_repo]
                commands.append((command, dict(env, CUDA_VISIBLE_DEVICES=",".join(group)),
                                 output / "logs" / f"{phase}_{rank:03d}.log"))
            print(f"Starting {phase}: {len(groups)} workers; logs={output / 'logs'}", flush=True)
            code = run_group(commands, deadline=deadline)
            if code:
                raise SystemExit(code)
        if args.phase in {"all", "finalize"}:
            subprocess.run([sys.executable, str(TOOL), "--phase", "finalize", *common], check=True, env=env, cwd=ROOT)
            subprocess.run([sys.executable, str(ROOT / "tools/data_prepare/render_mobileov_exp2_audit.py"),
                            "--release", str(output)], check=True, env=env, cwd=ROOT)
        print(f"Data phase complete. No VAE encoding or training launched. Output: {output}", flush=True)


if __name__ == "__main__":
    main()
