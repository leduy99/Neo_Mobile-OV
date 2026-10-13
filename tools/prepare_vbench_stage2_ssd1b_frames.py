#!/usr/bin/env python
"""Generate a verified, shared native SSD1B image cache before Stage-2 VBench."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import benchmark_mobileov_stage2_media3epochs_dual as base  # noqa: E402
from vbench_joint_native_protocol import (  # noqa: E402
    atomic_json, bind_contract, fingerprint, load_prompt_rows, output_lock,
    read_json, sample_items, sha256, shard_items,
)

FORMAT = "mobileov_stage2_shared_ssd1b_frames_v1"
FINAL_SHA256 = "a74fd3069d17437b674415268c5157a85d94223509121220572e2d2fcdb68053"
DEFAULT_ROOT = ROOT / "output/vbench_stage2_epoch3_ssd1b_frames_5x"
COMPONENTS = ("ssd_1b_unet", "ssd_1b_vae", "ssd_1b_text_encoder", "ssd_1b_text_encoder_2",
              "ssd_1b_tokenizer", "ssd_1b_tokenizer_2")


def items_for(contract):
    return sample_items(contract["prompts"], contract["samples_per_prompt"], contract["seed"], "extendable")


def verify_implementation(contract):
    for name, digest in contract["implementation_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Shared SSD1B code changed after preflight: {name}")


def image_record(output, contract, item):
    """Validate exact PNG bytes, sample identity and dimensions; never regenerate here."""
    from PIL import Image

    path = output / "images" / f"{item['key']}.png"
    record = read_json(output / "records" / f"{item['key']}.json")
    if (record.get("sample") != item or record.get("contract_sha256") != fingerprint(contract)
            or record.get("sha256") != sha256(path)):
        raise ValueError(f"Shared SSD1B image/metadata differs: {path}")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        if image.size != (contract["image"]["width"], contract["image"]["height"]) or image.mode != "RGB":
            raise ValueError(f"Wrong shared image shape/mode: {path}")
    return record


def cached_image(output, contract, item):
    try:
        image_record(output, contract, item)
        return True
    except (OSError, ValueError, KeyError):
        return False


def load_native_ssd(model, device):
    import torch
    from neodragon.first_frame_gen import SSD1B_FirstFrameGeneratorPipeline

    pipe = SSD1B_FirstFrameGeneratorPipeline.from_pretrained(model, torch_dtype=torch.bfloat16).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def prepare(args):
    import torch
    from new_mobile_ov.checkpoints import ensure_neodragon_assets
    from new_mobile_ov.config import load_config

    torch.set_num_threads(4)
    payload, checkpoint_sha, _ = base.read_checkpoint(args.checkpoint, expected_sha=args.expected_sha256)
    if payload["step"] != 283977 or sha256(args.config) != payload["contract"]["config_sha256"]:
        raise ValueError("Expected the final epoch-3 checkpoint and its exact training config")
    del payload
    prompts, rows = load_prompt_rows(args.vbench_info, args.max_prompts)
    if not args.max_prompts and len(prompts) != 944:
        raise ValueError(f"Expected 944 official unique prompts, got {len(prompts)}")
    cfg = load_config(args.config)
    source, _, model = ensure_neodragon_assets(
        repo_path=cfg.backend.extra.get("repo_path"), cache_dir=cfg.backend.extra.get("cache_dir"),
        model_id=cfg.backend.extra.get("model_id", "karnewar/Neodragon"), repo_url=cfg.backend.extra.get("repo_url"))
    source, model = Path(source).resolve(), Path(model).resolve()
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from neodragon.first_frame_gen import DEFAULT_INFERENCE_PARAMS
    from neodragon.utils.generation_utils import DEFAULT_PROMPT_MODIFIER

    files = []
    for component in COMPONENTS:
        selected = sorted(p for p in (model / component).rglob("*") if p.is_file())
        if not selected:
            raise ValueError(f"Missing native SSD1B component: {model / component}")
        files.extend(dict(file=str(p.relative_to(model)), size=p.stat().st_size, sha256=sha256(p)) for p in selected)
    contract = dict(
        format=FORMAT, prompts=prompts, max_prompts=args.max_prompts,
        full_info_sha256=sha256(args.vbench_info), samples_per_prompt=args.samples_per_prompt,
        seed=args.seed, seed_mapping="extendable", seed_policy="image seed = paired video seed; reset independently",
        admission_checkpoint_sha256=checkpoint_sha, config_sha256=sha256(args.config),
        image=dict(width=1024, height=640, native_defaults=dict(DEFAULT_INFERENCE_PARAMS),
                   prompt_modifier=DEFAULT_PROMPT_MODIFIER, image_bridge=False),
        model_path=str(model), native_repo=str(source), native_files=files,
        native_source_sha256={str(p.relative_to(source)): sha256(p) for p in sorted((source / "neodragon").rglob("*.py"))},
        implementation_sha256={name: sha256(ROOT / name) for name in (
            "tools/prepare_vbench_stage2_ssd1b_frames.py", "tools/vbench_joint_native_protocol.py",
            "tools/benchmark_mobileov_stage2_media3epochs_dual.py")},
        library_versions={name: version(name) for name in ("torch", "transformers", "diffusers")},
        vbench_source_commit=subprocess.check_output(["git", "-C", str(args.vbench_repo), "rev-parse", "HEAD"], text=True).strip(),
        scope="Generate SSD1B images once; both Stage-2 evaluation settings consume these exact PNG bytes",
    )
    if not (args.output_dir / "run_contract.json").exists() and any(args.output_dir.glob("images/*.png")):
        raise ValueError("Untracked PNGs in shared output; use a fresh ANCHOR_ROOT")
    bind_contract(args.output_dir, contract)
    atomic_json(args.output_dir / "vbench_info.json", rows)
    atomic_json(args.output_dir / "sample_plan.json", dict(contract_sha256=fingerprint(contract), samples=list(items_for(contract))))
    print(f"Shared SSD1B preflight: {len(prompts)} prompts x {args.samples_per_prompt}; final checkpoint verified.", flush=True)
    return contract


def generation_worker(args, *, device=None):
    import torch
    from PIL import Image

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("SSD1B generation requires a SLURM GPU allocation")
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("Each SSD1B worker requires an allocated GPU")
        device = torch.device("cuda:0")
    torch.set_num_threads(4)
    contract = read_json(args.output_dir / "run_contract.json")
    if contract.get("format") != FORMAT:
        raise ValueError("Wrong shared SSD1B contract")
    verify_implementation(contract)
    if {name: version(name) for name in contract["library_versions"]} != contract["library_versions"]:
        raise ValueError("SSD1B generation libraries changed after preflight")
    items = shard_items(list(items_for(contract)), args.rank, args.workers)
    pending = [item for item in items if not cached_image(args.output_dir, contract, item)]
    started = time.monotonic()
    print(f"SSD1B rank={args.rank}: assigned={len(items)}, pending={len(pending)}", flush=True)
    if pending:
        model, source = Path(contract["model_path"]), Path(contract["native_repo"])
        for record in contract["native_files"]:
            if sha256(model / record["file"]) != record["sha256"]:
                raise ValueError(f"Native SSD1B weights changed: {record['file']}")
        for name, digest in contract["native_source_sha256"].items():
            if sha256(source / name) != digest:
                raise ValueError(f"Native SSD1B source changed: {name}")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        pipe = load_native_ssd(model, device)
        for number, item in enumerate(pending, 1):
            torch.manual_seed(item["anchor_seed"])
            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
                image = pipe(prompt=item["prompt"] + contract["image"]["prompt_modifier"], num_images_per_prompt=1,
                             generator=torch.Generator(device=device).manual_seed(item["anchor_seed"])).images[0].convert("RGB")
            if image.size != (1024, 640):
                raise ValueError(f"Unexpected SSD1B image size: {image.size}")
            path = args.output_dir / "images" / f"{item['key']}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.stem + f".tmp.{os.getpid()}.png")
            try:
                image.save(temporary)
                with Image.open(temporary) as check:
                    check.verify()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            atomic_json(args.output_dir / "records" / f"{item['key']}.json", dict(
                sample=item, contract_sha256=fingerprint(contract), sha256=sha256(path)))
            if number == 1 or number % 100 == 0:
                print(f"SSD1B rank={args.rank}: generated={number}/{len(pending)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    atomic_json(args.output_dir / "worker_receipts" / f"rank_{args.rank}.json", dict(
        status="complete", contract_sha256=fingerprint(contract), assigned=len(items),
        generated_images=len(pending), reused_images=len(items)-len(pending), seconds=time.monotonic()-started))


def verify_cache(output, *, require_complete=True):
    contract = read_json(output / "run_contract.json")
    if contract.get("format") != FORMAT:
        raise ValueError("Not a shared SSD1B image cache")
    items = list(items_for(contract))
    if {p.name for p in (output / "images").glob("*.png") if ".tmp." not in p.name} != {f"{i['key']}.png" for i in items}:
        raise ValueError("Shared SSD1B cache has missing/unexpected images")
    manifest = [dict(key=item["key"], sha256=image_record(output, contract, item)["sha256"]) for item in items]
    if require_complete:
        summary = read_json(output / "generation_summary.json")
        if (summary.get("status") != "complete" or summary.get("contract_sha256") != fingerprint(contract)
                or summary.get("image_manifest_sha256") != fingerprint(manifest)):
            raise ValueError("Shared SSD1B cache is incomplete or changed; finish the image job first")
    return contract, manifest


def summarize(output):
    contract, manifest = verify_cache(output, require_complete=False)
    atomic_json(output / "image_manifest.json", manifest)
    atomic_json(output / "generation_summary.json", dict(status="complete", contract_sha256=fingerprint(contract),
                image_manifest_sha256=fingerprint(manifest), images=len(manifest), regeneration_by_eval_jobs=False))
    print(f"Shared SSD1B cache complete: {len(manifest)} verified PNGs in {output}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("run", "prepare", "generation-worker", "verify"), default="run")
    parser.add_argument("--checkpoint", type=Path, default=base.DEFAULT_CHECKPOINT)
    parser.add_argument("--expected-sha256", default=FINAL_SHA256)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--vbench-repo", type=Path, default=ROOT / "checkpoints/vbench_repo")
    parser.add_argument("--vbench-info", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--samples-per-prompt", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8 or not 0 <= args.rank < args.workers or args.max_prompts < 0 or args.seed < 0:
        parser.error("Invalid workers/rank, prompt limit or seed")
    if args.phase in ("run", "prepare") and not args.max_prompts and args.samples_per_prompt != 5:
        parser.error("Full image preparation requires five samples/prompt")
    if len(args.expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.expected_sha256):
        parser.error("Expected SHA256 must be 64 lowercase hex characters")
    for name in ("checkpoint", "config", "output_dir", "vbench_repo"):
        setattr(args, name, getattr(args, name).resolve())
    args.vbench_info = (args.vbench_info or args.vbench_repo / "vbench/VBench_full_info.json").resolve()
    return args


def main():
    args = parse_args()
    if args.phase == "run" and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Submit the shared SSD1B preparation job with sbatch")
    if args.phase == "generation-worker":
        generation_worker(args)
        return
    if args.phase == "verify":
        verify_cache(args.output_dir)
        print("Shared image cache verified; no image generation.", flush=True)
        return
    os.chdir(ROOT)
    with output_lock(args.output_dir):
        devices = base.core.gpu_tokens(args.workers) if args.phase == "run" else None
        prepare(args)
        if args.phase == "prepare":
            return
        previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            commands = [[sys.executable, str(Path(__file__).resolve()), "--phase", "generation-worker",
                         "--output-dir", str(args.output_dir), "--rank", str(rank), "--workers", str(args.workers)]
                        for rank in range(args.workers)]
            base.core.run_processes(commands, devices, args.output_dir / "logs/generation")
            summarize(args.output_dir)
        finally:
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
