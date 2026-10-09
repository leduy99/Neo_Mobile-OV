#!/usr/bin/env python
"""Multi-GPU Stage-2 T2V VBench: generate once, export/evaluate raw and QuickSR."""
from __future__ import annotations

import argparse
import gc
import hashlib
from importlib.machinery import ModuleSpec
from importlib.metadata import version
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def install_project_tools_namespace():
    # Legacy Stage-2 helpers import tools.*; an installed regular package can shadow it.
    directory = str(ROOT / "tools")
    current = sys.modules.get("tools")
    if current is not None and directory in getattr(current, "__path__", []):
        return
    package = ModuleType("tools")
    package.__path__ = [directory]
    package.__package__ = "tools"
    package.__spec__ = ModuleSpec("tools", loader=None, is_package=True)
    for name in list(sys.modules):
        if name == "tools" or name.startswith("tools."):
            del sys.modules[name]
    sys.modules["tools"] = package


install_project_tools_namespace()
import benchmark_joint_v2_native as core  # noqa: E402
import benchmark_joint_v2_quicksr_ablation as dual  # noqa: E402
import generate_vbench_mobileov_stage2 as stage2  # noqa: E402
from vbench_joint_native_protocol import (  # noqa: E402
    DIMENSIONS, atomic_json, bind_contract, cached_video, decode_video, fingerprint,
    load_prompt_rows, output_lock, read_json, sample_items, sha256, shard_items,
)

SYSTEM = "stage2_media3epochs"
VARIANTS = ("quicksr", "no_quicksr")
DEFAULT_CHECKPOINT = ROOT / "output/mobileov_stage2_media3epochs/17641364/stage2_dit_connector_latest.pt"


def check_media_checkpoint(payload, *, allow_incomplete=False):
    plan = payload.get("contract", {}).get("media_epochs", {})
    if (payload.get("format") != stage2.FORMAT or plan.get("recipe") != "media_epochs_i2v_v1"
            or plan.get("epochs") != 3):
        raise ValueError("Expected the three-media-epoch Stage-2 DiT+connector checkpoint")
    target = plan.get("steps", 0)
    if not isinstance(target, int) or target <= 0 or target != 3 * plan.get("steps_per_epoch", 0):
        raise ValueError("Inconsistent three-epoch training budget")
    step = payload.get("step", 0)
    if not isinstance(step, int) or not 0 < step <= target:
        raise ValueError(f"Invalid checkpoint step={step}; planned final step={target}")
    if step != target and not allow_incomplete:
        raise ValueError(f"Checkpoint is NOT complete: step={step}, three-epoch target={target}. "
                         "Use the final checkpoint in output/, not the old HF/cache 130K file.")
    tasks = {"t2i", "t2v", "i2v_reference", "i2v_anchor", "image_reconstruction"}
    if not tasks.issubset(payload["contract"].get("tasks", [])):
        raise ValueError("Checkpoint training tasks do not match the media-epoch recipe")
    stage2.check_payload(payload, step)
    return target


def read_checkpoint(path, *, allow_incomplete=False, expected_sha=None):
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"Missing local Stage-2 checkpoint: {path}. No old HF fallback is used.")
    before = path.stat()
    digest = sha256(path)
    if expected_sha and digest != expected_sha:
        raise ValueError("Checkpoint SHA256 changed; do not mix weights or training runs")
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    target = check_media_checkpoint(payload, allow_incomplete=allow_incomplete)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Checkpoint changed during preflight; wait until the final save is complete")
    return payload, digest, target


def items_for(contract):
    for item in sample_items(contract["prompts"], contract["samples_per_prompt"], contract["seed"], "extendable"):
        yield {key: value for key, value in item.items() if key not in ("anchor_seed", "video_seed")} | {"seed": item["video_seed"]}


def prepare(args):
    import torch
    from new_mobile_ov.generation.quicksrnet import load_quicksrnet

    torch.set_num_threads(4)
    payload, digest, target = read_checkpoint(args.checkpoint, allow_incomplete=args.allow_incomplete_checkpoint,
                                            expected_sha=args.expected_sha256)
    training = payload["contract"]
    if sha256(args.config) != training["config_sha256"]:
        raise ValueError("Inference config differs from training; do not rewrite YAML paths")
    prompts, rows = load_prompt_rows(args.vbench_info, args.max_prompts)
    if not args.max_prompts and len(prompts) != 944:
        raise ValueError(f"Expected 944 official unique prompts, got {len(prompts)}")
    model, checkpoint = load_quicksrnet(variant="medium", scale=2, cache_dir=str(args.quicksr_cache),
                                      device=torch.device("cpu"), dtype=torch.float32)
    del model
    files = ("tools/benchmark_mobileov_stage2_media3epochs_dual.py", "tools/generate_vbench_mobileov_stage2.py",
             "tools/infer_mobileov_stage1.py", "tools/train_mobileov_stage1.py",
             "new_mobile_ov/bridge/stage1_conditioner.py", "new_mobile_ov/generation/neodragon_compat.py",
             "tools/apply_quicksrnet_video.py", "tools/evaluate_vbench_resume.py",
             "tools/vbench_joint_native_protocol.py")
    contract = dict(
        format="mobileov_stage2_media3epochs_vbench_dual_v1", system=SYSTEM,
        checkpoint=dict(path=str(args.checkpoint), sha256=digest, step=payload["step"], final_step=target),
        training_complete=payload["step"] == target, training_contract=training, connector_spec=payload["connector_spec"],
        config=str(args.config), config_sha256=sha256(args.config), prompts=prompts,
        full_info_sha256=sha256(args.vbench_info), samples_per_prompt=args.samples_per_prompt,
        max_prompts=args.max_prompts, seed=args.seed, seed_mapping="extendable",
        seed_policy="base + global_prompt_index + sample_index * 1000003; reset inside generate_frames",
        exports={"no_quicksr": dict(width=512, height=320, frames=49, fps=24),
                 "quicksr": dict(width=1024, height=640, frames=49, fps=24)},
        quicksr=dict(variant="medium", scale=2, dtype="fp16", batch_frames=4,
                     path=str(Path(checkpoint).resolve()), sha256=sha256(checkpoint)),
        first_steps_per_stage=20, video_steps_per_stage=10, guidance=7., video_guidance=5.,
        external_anchor=False, native_conditioner=False, conditioning="literal prompt; trained empty negative; no modifier",
        dtype="bf16_compute_fp32_connector", same_generated_frames=True,
        implementation_sha256={name: sha256(ROOT / name) for name in files},
        library_versions={name: version(name) for name in ("torch", "transformers", "diffusers")},
        vbench_source_commit=subprocess.check_output(["git", "-C", str(args.vbench_repo), "rev-parse", "HEAD"], text=True).strip(),
        scope="Stage-2 T2V generates every frame, including frame zero. No DreamLite/SSD1B or observed image. "
              "Raw 512x320 vs QuickSR 1024x640 combines resolution and SR effects, not fixed-resolution SR. "
              "Not a Hybrid/anchor-only ablation or exact paper replication.",
    )
    bind_contract(args.output_dir, contract)
    atomic_json(args.output_dir / "vbench_info.json", rows)
    atomic_json(args.output_dir / "sample_plan.json", {"contract_sha256": fingerprint(contract), "samples": list(items_for(contract))})
    atomic_json(args.output_dir / "checkpoint_audit.json", {
        "checkpoint": contract["checkpoint"], "training_complete": contract["training_complete"],
        "connector_spec": payload["connector_spec"], "optimizer_present": "optimizer" in payload,
        "scope": "Preflight checks payload/finite tensors; frozen-stack tensor hashes are checked on each GPU before inference.",
    })
    del payload
    gc.collect()
    print(f"Stage 2 step={contract['checkpoint']['step']}/{target}; {len(prompts)} prompts x {args.samples_per_prompt} samples. "
          "Full monolithic T2V; raw/QuickSR from identical frames.", flush=True)
    return contract


def cached_pair(output, contract, item):
    records = []
    for variant in VARIANTS:
        folder = output / variant
        if not cached_video(folder / "videos" / item["filename"], folder / "video_metadata" / f"{item['key']}.json",
                            item, SYSTEM, fingerprint(contract), contract["exports"][variant]):
            return False
        records.append(read_json(folder / "video_metadata" / f"{item['key']}.json"))
    return (len(records[0].get("base_frames_sha256", "")) == 64
            and records[0]["base_frames_sha256"] == records[1].get("base_frames_sha256"))


def export_pair(frames, *, output, contract, item, model, device, statistics):
    import numpy as np
    import torch
    from PIL import Image
    from diffusers.utils import export_to_video
    from apply_quicksrnet_video import upscale_frames

    array = np.stack([np.asarray(frame.convert("RGB")) for frame in frames])
    raw_shape = contract["exports"]["no_quicksr"]
    if array.shape != (raw_shape["frames"], raw_shape["height"], raw_shape["width"], 3):
        raise ValueError(f"Wrong generated frame shape: {array.shape}")
    base_sha = hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest()
    with torch.inference_mode(), torch.autocast(device.type, enabled=False):
        sr, _ = upscale_frames(array, model=model, device=device, dtype=torch.float16, batch_frames=4, reset_peak_memory=False)
    shape = contract["exports"]["quicksr"]
    if sr is None or sr.shape != (shape["frames"], shape["height"], shape["width"], 3):
        raise ValueError("QuickSR output shape differs from export contract")
    records = []
    for variant, pixels in (("no_quicksr", array), ("quicksr", sr)):
        folder = output / variant
        destination = folder / "videos" / item["filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.parent / f".{item['key']}.tmp.{os.getpid()}.mp4"
        try:
            export_to_video([Image.fromarray(frame) for frame in pixels], str(temporary), fps=24)
            if not decode_video(temporary, contract["exports"][variant]):
                raise ValueError(f"Export/decode failed: {temporary}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        records.append((folder / "video_metadata" / f"{item['key']}.json", dict(
            sample=item, system=SYSTEM, shape=contract["exports"][variant], contract_sha256=fingerprint(contract),
            base_frames_sha256=base_sha, video_sha256=sha256(destination), unit_statistics=statistics,
            external_anchor=False, first_frame_generated=True, postprocessing_variant=variant)))
    for path, record in records:
        atomic_json(path, record)


def generation_worker(args, *, device=None):
    import torch
    from new_mobile_ov.generation.quicksrnet import load_quicksrnet
    from tools.infer_mobileov_stage1 import autocast, generate_frames, pad_condition

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Stage-2 GPU inference must run inside SLURM")
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("Each generation worker requires an allocated GPU")
        device = torch.device("cuda:0")
    torch.set_num_threads(4)
    contract = read_json(args.output_dir / "run_contract.json")
    if contract.get("format") != "mobileov_stage2_media3epochs_vbench_dual_v1":
        raise ValueError("Wrong Stage-2 benchmark contract")
    for name, expected in contract["implementation_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Code changed after preflight: {name}; keep code fixed while a benchmark runs")
    items = shard_items(list(items_for(contract)), args.rank, args.workers)
    pending = [item for item in items if not cached_pair(args.output_dir, contract, item)]
    audit_path = args.output_dir / "weight_audits" / f"rank_{args.rank}.json"
    print(f"Stage 2 rank={args.rank}: assigned={len(items)} pending pairs={len(pending)}", flush=True)
    started = time.monotonic()
    if pending:
        payload, _, _ = read_checkpoint(Path(contract["checkpoint"]["path"]),
                                        allow_incomplete=bool(contract["max_prompts"]), expected_sha=contract["checkpoint"]["sha256"])
        stack_args = SimpleNamespace(config=Path(contract["config"]))
        torch.manual_seed(contract["seed"])
        encoder, connector, dit, vae, scheduler, audit = stage2.load_models(stack_args, payload, device)
        atomic_json(audit_path, {**audit, "contract_sha256": fingerprint(contract)})
        del payload
        gc.collect()
        model, _ = load_quicksrnet(variant="medium", scale=2, checkpoint=contract["quicksr"]["path"],
                                  device=device, dtype=torch.float16)
        with torch.inference_mode():
            with autocast(device):
                negative = connector(*encoder("", drop_condition=True))
            for number, item in enumerate(pending, 1):
                with autocast(device):
                    positive = connector(*encoder(item["prompt"]))
                    length = max(positive[0].shape[1], negative[0].shape[1])
                    condition = tuple(torch.cat([neg, pos]) for neg, pos in zip(
                        pad_condition(negative, length), pad_condition(positive, length)))
                frames, stats = generate_frames(dit, vae, scheduler, condition, device=device, seed=item["seed"],
                    height=320, width=512, num_frames=49, first_steps=20, video_steps=10, guidance=7., video_guidance=5.)
                export_pair(frames, output=args.output_dir, contract=contract, item=item,
                            model=model, device=device, statistics=stats)
                if number == 1 or number % 20 == 0:
                    print(f"Stage 2 rank={args.rank}: pairs={number}/{len(pending)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    elif not audit_path.is_file() or read_json(audit_path).get("contract_sha256") != fingerprint(contract):
        # An empty shard requires no weights; a populated cached shard must retain its audit.
        if items:
            raise ValueError(f"Missing verified frozen-stack audit for cached shard: {audit_path}")
    atomic_json(args.output_dir / "worker_receipts" / f"rank_{args.rank}.json", dict(
        rank=args.rank, status="complete", contract_sha256=fingerprint(contract), assigned=len(items),
        generated_pairs=len(pending), reused_pairs=len(items) - len(pending), seconds=time.monotonic() - started))


def verify_generation(output):
    contract = read_json(output / "run_contract.json")
    items = list(items_for(contract))
    for variant in VARIANTS:
        expected = {item["filename"] for item in items}
        actual = {file.name for file in (output / variant / "videos").glob("*.mp4") if not file.name.startswith(".")}
        if actual != expected:
            raise ValueError(f"Missing/unexpected videos in {variant}; expected exactly {len(items)}")
    digests = {variant: [] for variant in VARIANTS}
    for item in items:
        records = []
        for variant in VARIANTS:
            folder = output / variant
            value = read_json(folder / "video_metadata" / f"{item['key']}.json")
            if (value.get("sample") != item or value.get("contract_sha256") != fingerprint(contract)
                    or value.get("system") != SYSTEM or value.get("shape") != contract["exports"][variant]
                    or value.get("video_sha256") != sha256(folder / "videos" / item["filename"])):
                raise ValueError(f"Mismatched video metadata: {variant}/{item['key']}")
            records.append(value)
            digests[variant].append(value["video_sha256"])
        if (len(records[0].get("base_frames_sha256", "")) != 64
                or records[0]["base_frames_sha256"] != records[1].get("base_frames_sha256")):
            raise ValueError(f"Export pair does not share generated frames: {item['key']}")
    for variant, hashes in digests.items():
        scores = output / variant / "scores"
        snapshot = scores / "generation_snapshot.json"
        digest = fingerprint(hashes)
        if scores.exists() and (not snapshot.exists() or read_json(snapshot).get("sha256") != digest):
            scores.rename(output / variant / f"retired_scores_{time.time_ns()}")
        atomic_json(snapshot, {"sha256": digest, "contract_sha256": fingerprint(contract)})
    atomic_json(output / "generation_summary.json", dict(status="complete", checkpoint=contract["checkpoint"],
        training_complete=contract["training_complete"], contract_sha256=fingerprint(contract),
        videos_per_variant=len(items), same_generated_frames=True, external_anchor=False, scores_computed=False))


def score_worker(args):
    contract = read_json(args.output_dir / "run_contract.json")
    for variant, _, index in dual.score_assignments(args.workers, [SYSTEM])[args.rank]:
        dimension = DIMENSIONS[index]
        folder = args.output_dir / variant
        print(f"Scoring Stage 2 {variant}/{dimension}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "tools/evaluate_vbench_resume.py"),
                        "--videos", str(folder / "videos"), "--full-info", str(args.output_dir / "vbench_info.json"),
                        "--output", str(folder / "scores" / dimension),
                        "--expected-samples-per-prompt", str(contract["samples_per_prompt"]),
                        "--start", str(index), "--end", str(index + 1)], cwd=args.vbench_repo, check=True)


def summarize(args):
    contract = read_json(args.output_dir / "run_contract.json")
    if contract["max_prompts"] or not contract["training_complete"]:
        raise ValueError("No full Stage-2 score is claimed for a subset or incomplete training checkpoint")
    from evaluate_vbench_resume import tabulate

    verify_generation(args.output_dir)
    scores = {}
    for variant in VARIANTS:
        merged = args.output_dir / variant / "scores/vbench"
        merged.mkdir(parents=True, exist_ok=True)
        for dimension in DIMENSIONS:
            source = args.output_dir / variant / "scores" / dimension / "vbench" / f"{dimension}_eval_results.json"
            if dimension not in read_json(source):
                raise ValueError(f"Missing score: {source}")
            shutil.copyfile(source, merged / source.name)
        scores[variant] = {SYSTEM: tabulate(merged)}
        subprocess.run([sys.executable, str(ROOT / "tools/report_vbench_low_prompts.py"), "--score-dir", str(merged),
                        "--output-prefix", str(args.output_dir / variant / "low_prompts")], check=True)
    report, markdown = dual.ablation_report(scores, {"selected_systems": [SYSTEM]})
    report["protocol"] = contract
    report["evaluation_assets"] = read_json(args.output_dir / "score_assets.json")
    report["scope"] = contract["scope"]
    atomic_json(args.output_dir / "quicksr_ablation.json", report)
    (args.output_dir / "quicksr_ablation.md").write_text(markdown.replace("Identical generated RGB frames and anchors;",
                                                                       "Identical generated RGB frames; no external anchor;"), encoding="utf-8")
    print(f"Completed Stage-2 three-epoch VBench: {args.output_dir}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("run", "prepare", "generation-worker", "warm-scores", "score-worker", "summarize"), default="run")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/vbench_mobileov_stage2_media3epochs_5x_dual")
    parser.add_argument("--vbench-repo", type=Path, default=ROOT / "checkpoints/vbench_repo")
    parser.add_argument("--vbench-info", type=Path)
    parser.add_argument("--vbench-python", default="/proj/cvl/users/x_fahkh2/envs/neo_vbench/bin/python")
    parser.add_argument("--quicksr-cache", type=Path, default=ROOT / "checkpoints/quicksrnet")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--samples-per-prompt", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow-incomplete-checkpoint", action="store_true", help="ONLY for generation smoke with --max-prompts")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8 or not 0 <= args.rank < args.workers or args.max_prompts < 0 or args.seed < 0:
        parser.error("Invalid worker rank/count, prompt limit or seed")
    if args.phase in ("run", "prepare"):
        if not args.max_prompts and args.samples_per_prompt != 5:
            parser.error("Full benchmark requires 5 samples/prompt; use --max-prompts for smoke")
        if args.allow_incomplete_checkpoint and not args.max_prompts:
            parser.error("Incomplete checkpoints are ONLY allowed for generation smoke, never full evaluation")
    if args.expected_sha256 and (len(args.expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.expected_sha256)):
        parser.error("Expected SHA256 must be 64 lowercase hex characters")
    for name in ("checkpoint", "config", "output_dir", "vbench_repo", "quicksr_cache"):
        setattr(args, name, getattr(args, name).resolve())
    args.vbench_info = (args.vbench_info or args.vbench_repo / "vbench/VBench_full_info.json").resolve()
    return args


def main():
    args = parse_args()
    if args.phase == "generation-worker":
        generation_worker(args)
        return
    if args.phase == "warm-scores":
        core.warm_scores(args)
        return
    if args.phase == "score-worker":
        score_worker(args)
        return
    if args.phase == "summarize":
        summarize(args)
        return
    os.chdir(ROOT)
    with output_lock(args.output_dir):
        if args.phase == "run":
            devices = core.gpu_tokens(args.workers)
            if not Path(args.vbench_python).is_file():
                raise FileNotFoundError(args.vbench_python)
        prepare(args)
        if args.phase == "prepare":
            return
        if not args.max_prompts:
            subprocess.run([args.vbench_python, str(Path(__file__).resolve()), "--phase", "warm-scores",
                            "--output-dir", str(args.output_dir)], cwd=args.vbench_repo, check=True)
        previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            commands = [[sys.executable, str(Path(__file__).resolve()), "--phase", "generation-worker",
                         "--output-dir", str(args.output_dir), "--rank", str(rank), "--workers", str(args.workers)]
                        for rank in range(args.workers)]
            core.run_processes(commands, devices, args.output_dir / "logs/generation")
            verify_generation(args.output_dir)
            if args.max_prompts:
                print("Stage-2 dual-export generation smoke passed; no full score claimed.", flush=True)
                return
            commands = [[args.vbench_python, str(Path(__file__).resolve()), "--phase", "score-worker",
                         "--output-dir", str(args.output_dir), "--vbench-repo", str(args.vbench_repo),
                         "--rank", str(rank), "--workers", str(args.workers)] for rank in range(args.workers)]
            core.run_processes(commands, devices, args.output_dir / "logs/scoring", cwd=args.vbench_repo)
            subprocess.run([args.vbench_python, str(Path(__file__).resolve()), "--phase", "summarize",
                            "--output-dir", str(args.output_dir)], cwd=args.vbench_repo, check=True)
        finally:
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
