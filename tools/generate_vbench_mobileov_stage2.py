#!/usr/bin/env python
"""VBench generation for Stage 2: trained MCP + multistep DiT, no image anchor."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FORMAT = "mobileov_stage2_mcp_dit_v1"
HF_FILE = "mobileov_stage2_from_alignment15k/17519025/stage2_dit_connector_latest.pt"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def run_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".generation.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another generator is using {output}") from error
        yield


def load_prompts(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("VBench full-info must be a list")
    prompts = []
    seen = set()
    for row in rows:
        value = row.get("prompt_en")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Every VBench entry needs a nonempty prompt_en")
        prompt = " ".join(value.split())
        if any(char in prompt for char in ("/", "\\", "\0")):
            raise ValueError(f"Unsafe VBench filename: {prompt!r}")
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
    if not prompts:
        raise ValueError("Empty VBench prompt set")
    return prompts


def sample_items(prompts, samples, seed):
    for index, prompt in enumerate(prompts):
        for sample in range(samples):
            filename = f"{prompt}-{sample}.mp4"
            if len(filename.encode("utf-8")) > 255:
                raise ValueError(f"VBench filename exceeds filesystem limit: {filename}")
            yield dict(prompt=prompt, prompt_index=index, sample_index=sample,
                       seed=seed + index * samples + sample, filename=filename)


def check_payload(payload, expected_step):
    import torch

    if payload.get("format") != FORMAT:
        raise ValueError("Expected a Stage-2 DiT+connector checkpoint, not Stage 1 or legacy bridge")
    if payload.get("step") != expected_step:
        raise ValueError(f"Checkpoint step={payload.get('step')}; expected {expected_step}")
    contract = payload["contract"]
    for key in ("config_sha256", "processor", "processor_sha256", "max_tokens",
                "smolvlm2_sha256", "initial_weights_sha256"):
        if key not in contract:
            raise ValueError(f"Missing training contract field: {key}")
    if not payload.get("connector_spec"):
        raise ValueError("Missing connector architecture")
    for component in ("connector", "dit"):
        if not payload.get(component):
            raise ValueError(f"Missing trained {component} weights")
        for name, value in payload[component].items():
            if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Invalid/non-finite {component} tensor: {name}")


def load_checkpoint(args):
    import torch

    path = args.checkpoint
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            from huggingface_hub import hf_hub_download

            print(f"Downloading {args.hf_repo}/{args.hf_file} at {args.hf_revision}", flush=True)
            source = hf_hub_download(repo_id=args.hf_repo, filename=args.hf_file,
                                     revision=args.hf_revision, force_download=True)
            temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
            try:
                shutil.copyfile(source, temporary)
                candidate = torch.load(temporary, map_location="cpu", weights_only=True)
                check_payload(candidate, args.expected_step)
                del candidate
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        # Do not delete an unreadable user checkpoint or silently replace a wrong run.
        payload = torch.load(path, map_location="cpu", weights_only=True)
        check_payload(payload, args.expected_step)
        return payload, sha256_file(path)


def decode_check(path, *, frames, width, height, fps):
    import cv2

    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        actual_fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not math.isfinite(actual_fps) or abs(actual_fps - fps) > .05:
            return False
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (height, width):
                return False
            count += 1
    finally:
        capture.release()
    return count == frames


def cached_video(video, metadata, item, shape):
    try:
        record = json.loads(metadata.read_text(encoding="utf-8"))
        return (record.get("sample") == item and video.is_file()
                and record.get("video_sha256") == sha256_file(video)
                and decode_check(video, **shape))
    except (OSError, ValueError, KeyError):
        return False


def bind_run(output, contract, items):
    path = output / "run_contract.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != contract:
            changed = sorted(key for key in set(previous) | set(contract)
                             if previous.get(key) != contract.get(key))
            raise ValueError(f"Resume contract differs: {changed}. Use a new RUN_ROOT.")
    else:
        if any((output / "videos").glob("*.mp4")) or (output / "scores").exists():
            raise ValueError("Untracked videos/scores in output; use a fresh RUN_ROOT")
        atomic_json(path, contract)
    expected = {item["filename"] for item in items}
    extra = {path.name for path in (output / "videos").glob("*.mp4")} - expected
    if extra:
        raise ValueError(f"Unexpected videos would contaminate VBench: {sorted(extra)[:5]}")


def load_models(args, payload, device):
    from tools.infer_mobileov_stage1 import load_stack, verify_inference_stack

    contract = payload["contract"]
    if sha256_file(args.config) != contract["config_sha256"]:
        raise ValueError("Config differs from training; do not patch paths in the training YAML")
    args.processor, args.max_tokens = contract["processor"], contract["max_tokens"]
    cfg, encoder, connector, dit, vae, scheduler = load_stack(args, device)
    base_contract = dict(contract, frozen_weights_sha256=contract["initial_weights_sha256"])
    audit = verify_inference_stack(cfg, encoder, dit, vae, base_contract)
    if payload["connector_spec"] != connector.spec:
        raise ValueError("Connector architecture differs from checkpoint")
    dit.load_state_dict(payload["dit"], strict=True)
    connector.load_state_dict(payload["connector"], strict=True)
    for module in (encoder, connector, dit, vae):
        module.eval().requires_grad_(False)
    audit.update(trained_dit_loaded=True, trained_connector_loaded=True,
                 checkpoint_step=payload["step"], target_stack="multistep")
    return encoder, connector, dit, vae, scheduler, audit


def generate(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU generation must run inside srun/sbatch")
    import torch
    from diffusers.utils import export_to_video
    from tools.infer_mobileov_stage1 import autocast, generate_frames, pad_condition

    payload, checkpoint_sha = load_checkpoint(args)
    prompts = load_prompts(args.vbench_info)
    total_prompts = len(prompts)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    items = list(sample_items(prompts, args.samples_per_prompt, args.seed))
    shape = dict(frames=args.frames, height=args.height, width=args.width, fps=args.fps)
    implementation = {str(path): sha256_file(ROOT / path) for path in (
        "tools/generate_vbench_mobileov_stage2.py", "tools/infer_mobileov_stage1.py",
        "tools/train_mobileov_stage1.py", "new_mobile_ov/bridge/stage1_conditioner.py",
        "new_mobile_ov/generation/neodragon_compat.py", "tools/evaluate_vbench_resume.py")}
    contract = dict(format="mobileov_stage2_vbench_v1", checkpoint_sha256=checkpoint_sha,
                    step=payload["step"], training_contract=payload["contract"],
                    config_sha256=sha256_file(args.config), vbench_info_sha256=sha256_file(args.vbench_info),
                    prompt_count=len(prompts), full_prompt_count=total_prompts,
                    max_prompts=args.max_prompts, samples_per_prompt=args.samples_per_prompt,
                    seed=args.seed, seed_policy="base + prompt_index * samples_per_prompt + sample_index",
                    shape=shape, first_steps_per_stage=args.first_steps, video_steps_per_stage=args.video_steps,
                    guidance=args.guidance, video_guidance=args.video_guidance,
                    conditioning="literal VBench prompt; trained connector empty negative; no modifier",
                    external_anchor=False, native_conditioner=False, quicksr=False, target_stack="multistep",
                    dtype="bf16_compute_fp32_connector", implementation_sha256=implementation,
                    library_versions={name: version(name) for name in ("torch", "transformers", "diffusers")})
    bind_run(args.output_dir, contract, items)
    videos = args.output_dir / "videos"
    metadata_dir = args.output_dir / "video_metadata"
    videos.mkdir(exist_ok=True)
    metadata_dir.mkdir(exist_ok=True)
    pending = []
    for item in items:
        metadata = metadata_dir / f"{item['prompt_index']:04d}-{item['sample_index']}.json"
        if not cached_video(videos / item["filename"], metadata, item, shape):
            pending.append((item, metadata))
    if pending and any((args.output_dir / "scores").rglob("*_eval_results.json")):
        raise ValueError("Videos need repair but cached scores exist. Use a new RUN_ROOT to avoid stale scores.")
    print(f"Stage 2 step={payload['step']} prompts={len(prompts)} samples={args.samples_per_prompt} "
          f"videos={len(items)} pending={len(pending)}; full monolithic, no external anchor", flush=True)
    audit_path = args.output_dir / "weight_audit.json"
    if pending:
        device = torch.device("cuda", 0)
        torch.manual_seed(args.seed)
        encoder, connector, dit, vae, scheduler, weight_audit = load_models(args, payload, device)
        atomic_json(audit_path, weight_audit)
        del payload
        gc.collect()
        with torch.inference_mode():
            with autocast(device):
                negative = connector(*encoder("", drop_condition=True))
            for ordinal, (item, metadata) in enumerate(pending, 1):
                begin = time.monotonic()
                with autocast(device):
                    positive = connector(*encoder(item["prompt"]))
                    length = max(positive[0].shape[1], negative[0].shape[1])
                    condition = tuple(torch.cat([neg, pos]) for neg, pos in zip(
                        pad_condition(negative, length), pad_condition(positive, length)))
                frames, stats = generate_frames(
                    dit, vae, scheduler, condition, device=device, seed=item["seed"],
                    height=args.height, width=args.width, num_frames=args.frames,
                    first_steps=args.first_steps, video_steps=args.video_steps,
                    guidance=args.guidance, video_guidance=args.video_guidance)
                destination = videos / item["filename"]
                # Keep interrupted encodes outside the directory scanned by VBench.
                temporary = metadata_dir / f".incomplete-{os.getpid()}.mp4"
                try:
                    export_to_video(frames, str(temporary), fps=args.fps)
                    if not decode_check(temporary, **shape):
                        raise RuntimeError(f"Video decode verification failed: {destination}")
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
                atomic_json(metadata, dict(sample=item, unit_statistics=stats,
                                          video_sha256=sha256_file(destination),
                                          seconds=time.monotonic() - begin))
                print(f"[{ordinal}/{len(pending)}] seed={item['seed']} {item['filename']}", flush=True)
    elif not audit_path.is_file():
        raise ValueError("Cached generation is missing its weight verification audit")
    summary = dict(status="complete", scope="smoke_subset" if args.max_prompts else "full_prompt_set",
                   protocol=f"VBench prompts, {args.samples_per_prompt} independently seeded video(s)/prompt",
                   checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_sha,
                   step=args.expected_step, videos=len(items), prompt_count=len(prompts),
                   samples_per_prompt=args.samples_per_prompt, external_anchor=False,
                   scores_computed=False, generation_only=True,
                   weight_audit=json.loads(audit_path.read_text(encoding="utf-8")))
    atomic_json(args.output_dir / "generation_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/hf_mobile_ov") / HF_FILE)
    parser.add_argument("--hf-repo", default="Amshaker/Mobile-OV")
    parser.add_argument("--hf-file", default=HF_FILE)
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--expected-step", type=int, default=100000)
    parser.add_argument("--config", type=Path, default=Path("configs/mobile_ov_neodragon.yaml"))
    parser.add_argument("--vbench-info", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    # The pinned VBench evaluator recognizes the official sample indices 0..4.
    parser.add_argument("--samples-per-prompt", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--max-prompts", type=int, default=0, help="Smoke only; no full VBench scoring")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--frames", type=int, choices=(49,), default=49)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--first-steps", type=int, default=20)
    parser.add_argument("--video-steps", type=int, default=10)
    parser.add_argument("--guidance", type=float, default=7.)
    parser.add_argument("--video-guidance", type=float, default=5.)
    args = parser.parse_args(argv)
    if min(args.expected_step, args.samples_per_prompt, args.height, args.width,
           args.first_steps, args.video_steps, args.fps) <= 0 or args.max_prompts < 0 or args.seed < 0:
        parser.error("Steps, samples, size and fps must be positive; max-prompts/seed cannot be negative")
    if args.height % 64 or args.width % 64:
        parser.error("Pyramidal inference requires height/width divisible by 64")
    if any(not math.isfinite(scale) or scale < 1 for scale in (args.guidance, args.video_guidance)):
        parser.error("CFG scales must be finite and >= 1")
    return args


def main():
    args = parse_args()
    with run_lock(args.output_dir):
        generate(args)


if __name__ == "__main__":
    main()
