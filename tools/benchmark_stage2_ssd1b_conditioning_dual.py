#!/usr/bin/env python
"""Separate SSD1B latent anchors from image conditioning; dual-export VBench."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
from importlib.metadata import version
import io
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import benchmark_mobileov_stage2_media3epochs_dual as base  # noqa: E402
import prepare_vbench_stage2_ssd1b_frames as shared  # noqa: E402
from vbench_joint_native_protocol import (  # noqa: E402
    DIMENSIONS, atomic_json, bind_contract, cached_video, fingerprint, output_lock,
    read_json, sha256, shard_items,
)

FORMAT = "mobileov_stage2_ssd1b_conditioning_vbench_dual_v1"
MODES = {
    "ssd1b_anchor": dict(system="stage2_ssd1b_anchor", image_to_smolvlm2=False, external_anchor=True),
    "image_condition": dict(system="stage2_image_condition", image_to_smolvlm2=True, external_anchor=False),
}


def route(mode):
    if mode not in MODES:
        raise ValueError(f"Unknown conditioning mode: {mode}")
    return dict(MODES[mode])


def positive_condition(encoder, connector, prompt, image, mode):
    if route(mode)["image_to_smolvlm2"]:
        if image is None:
            raise ValueError("Image conditioning requires the shared SSD1B image")
        return connector(*encoder(prompt, image, allow_text_image=True))
    return connector(*encoder(prompt))


def frame_item(item):
    return {k: v for k, v in item.items() if k != "seed"} | dict(anchor_seed=item["seed"], video_seed=item["seed"])


def validate_shared_protocol(template, images):
    for name in ("prompts", "max_prompts", "full_info_sha256", "samples_per_prompt", "seed",
                 "seed_mapping", "config_sha256", "library_versions", "vbench_source_commit"):
        if template.get(name) != images.get(name):
            raise ValueError(f"Shared SSD1B protocol differs from video protocol: {name}")
    if template["checkpoint"]["sha256"] != images["admission_checkpoint_sha256"]:
        raise ValueError("Shared frame preparation admitted a different Stage-2 checkpoint")


def prepare(args):
    images, manifest = shared.verify_cache(args.frames_root)
    asset_args = copy.copy(args)
    asset_args.output_dir = args.output_dir / "_asset_preflight"
    template = base.prepare(asset_args)
    if template["checkpoint"]["step"] != 283977:
        raise ValueError("Use latest completed epoch 3, step 283977; no earlier checkpoint fallback")
    validate_shared_protocol(template, images)
    selected = route(args.mode)
    contract = copy.deepcopy(template)
    contract.update(format=FORMAT, mode=args.mode, **selected,
                    first_frame_generated=not selected["external_anchor"],
                    shared_ssd1b=dict(path=str(args.frames_root), contract_sha256=fingerprint(images),
                                     image_manifest_sha256=fingerprint(manifest)),
                    image_processing=dict(width=512, height=320, resampling="PIL.LANCZOS", pillow_version=version("Pillow")),
                    vae_seed_policy="video seed + 228; private generator; video noise reset independently",
                    conditioning=("literal text + shared image through SmolVLM2; empty negative; no latent anchor"
                                  if selected["image_to_smolvlm2"] else
                                  "literal text ONLY through SmolVLM2; empty negative; shared SSD1B image through VAE ONLY"))
    for name in ("tools/benchmark_stage2_ssd1b_conditioning_dual.py", "tools/prepare_vbench_stage2_ssd1b_frames.py"):
        contract["implementation_sha256"][name] = sha256(ROOT / name)
    contract["scope"] = (
        f"{selected['system']}: final three-epoch Stage-2 MCP+monolithic DiT, step 283977. "
        + ("Text+image condition via SmolVLM2; NO supplied first latent; frame zero is generated. "
           if selected["image_to_smolvlm2"] else
           "Text-only condition via SmolVLM2; SSD1B image supplies ONLY the first VAE latent, not vision tokens. ")
        + "Read exact PNGs from a separately verified shared SSD1B cache; never generate images in this job. "
          "Raw and QuickSR share generated RGB frames. Raw 512x320 vs SR 1024x640 combines output-resolution and SR effects. "
          "No DreamLite, native video conditioner, Hybrid step-distillation or model training."
    )
    bind_contract(args.output_dir, contract)
    atomic_json(args.output_dir / "vbench_info.json", read_json(asset_args.output_dir / "vbench_info.json"))
    atomic_json(args.output_dir / "sample_plan.json", dict(contract_sha256=fingerprint(contract), samples=list(base.items_for(contract))))
    atomic_json(args.output_dir / "checkpoint_audit.json", read_json(asset_args.output_dir / "checkpoint_audit.json"))
    print(f"Actual route: {args.mode}; image_to_smolvlm2={selected['image_to_smolvlm2']}; "
          f"external_anchor={selected['external_anchor']}; shared SSD1B cache verified.", flush=True)
    return contract


def checked_shared(contract):
    output = Path(contract["shared_ssd1b"]["path"])
    images = read_json(output / "run_contract.json")
    manifest = read_json(output / "image_manifest.json")
    summary = read_json(output / "generation_summary.json")
    expected = contract["shared_ssd1b"]
    if (fingerprint(images) != expected["contract_sha256"] or summary.get("status") != "complete"
            or summary.get("contract_sha256") != expected["contract_sha256"]
            or fingerprint(manifest) != expected["image_manifest_sha256"]
            or summary.get("image_manifest_sha256") != expected["image_manifest_sha256"]):
        raise ValueError("Shared SSD1B cache/manifest changed after video preflight")
    return output, images, {row["key"]: row["sha256"] for row in manifest}


def load_shared_image(path, expected_sha):
    from PIL import Image

    # Decode the exact verified bytes, rather than reopening a mutable path after hashing.
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError(f"Shared SSD1B image changed before use: {path}")
    with Image.open(io.BytesIO(data)) as source:
        if source.mode != "RGB" or source.size != (1024, 640):
            raise ValueError("Unexpected shared SSD1B source image")
        return source.resize((512, 320), Image.Resampling.LANCZOS)


def pair_records(output, contract, item, image_sha, *, decode=False):
    values = []
    expected_route = route(contract["mode"])
    for variant in base.VARIANTS:
        folder = output / variant
        video = folder / "videos" / item["filename"]
        metadata = folder / "video_metadata" / f"{item['key']}.json"
        if decode and not cached_video(video, metadata, item, contract["system"], fingerprint(contract), contract["exports"][variant]):
            raise ValueError(f"Invalid cached video: {video}")
        value = read_json(metadata)
        if (value.get("sample") != item or value.get("contract_sha256") != fingerprint(contract)
                or value.get("system") != contract["system"] or value.get("shape") != contract["exports"][variant]
                or value.get("video_sha256") != sha256(video) or value.get("shared_image_sha256") != image_sha
                or any(value.get(name) != expected_route[name] for name in ("image_to_smolvlm2", "external_anchor"))
                or value.get("first_frame_generated") != (not expected_route["external_anchor"])):
            raise ValueError(f"Wrong video route/metadata: {metadata}")
        if bool(value.get("unit_statistics", [{}])[0].get("supplied_anchor", False)) != expected_route["external_anchor"]:
            raise ValueError("Generated first-frame history contradicts the requested route")
        values.append(value)
    for name in ("base_frames_sha256", "resized_image_sha256", "conditioning_sha256"):
        if len(values[0].get(name, "")) != 64 or values[0][name] != values[1].get(name):
            raise ValueError(f"Raw/QuickSR pair differs: {name}")
    return values


def cached_pair(output, contract, item, image_sha):
    try:
        pair_records(output, contract, item, image_sha, decode=True)
        return True
    except (OSError, ValueError, KeyError, IndexError):
        return False


def export_pair(frames, *, output, contract, item, model, device, statistics, image_sha, pixels_sha, condition_sha):
    # Retain the existing numerical RGB -> raw/QuickSR implementation; augment only provenance.
    base.export_pair(frames, output=output, contract=contract, item=item, model=model, device=device, statistics=statistics)
    selected = route(contract["mode"])
    for variant in base.VARIANTS:
        path = output / variant / "video_metadata" / f"{item['key']}.json"
        value = read_json(path)
        value.update(**selected, first_frame_generated=not selected["external_anchor"],
                     shared_image_sha256=image_sha, resized_image_sha256=pixels_sha, conditioning_sha256=condition_sha)
        atomic_json(path, value)


def generation_worker(args, *, device=None):
    import numpy as np
    import torch
    from new_mobile_ov.generation.quicksrnet import load_quicksrnet
    from tools.infer_mobileov_stage1 import autocast, encode_anchor, generate_frames, pad_condition

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Stage-2 video generation requires a SLURM GPU allocation")
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("Each worker requires an allocated GPU")
        device = torch.device("cuda:0")
    torch.set_num_threads(4)
    contract = read_json(args.output_dir / "run_contract.json")
    if contract.get("format") != FORMAT or contract.get("mode") != args.mode:
        raise ValueError("Wrong Stage-2 conditioning run/worker mode")
    if any(contract.get(name) != expected for name, expected in route(args.mode).items()):
        raise ValueError("Run contract contradicts the requested conditioning route")
    for name, digest in contract["implementation_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Code changed during benchmark: {name}")
    if version("Pillow") != contract["image_processing"]["pillow_version"]:
        raise ValueError("Image preprocessing library changed")
    if {name: version(name) for name in contract["library_versions"]} != contract["library_versions"]:
        raise ValueError("Video generation libraries changed after preflight")
    image_root, images, image_shas = checked_shared(contract)
    items = shard_items(list(base.items_for(contract)), args.rank, args.workers)
    for item in items:
        if shared.image_record(image_root, images, frame_item(item))["sha256"] != image_shas[item["key"]]:
            raise ValueError("Shared image bytes differ from the verified manifest")
    pending = [item for item in items if not cached_pair(args.output_dir, contract, item, image_shas[item["key"]])]
    started = time.monotonic()
    audit_path = args.output_dir / "weight_audits" / f"rank_{args.rank}.json"
    print(f"{args.mode} rank={args.rank}: assigned={len(items)}, pending={len(pending)}; no SSD1B generation here.", flush=True)
    if pending:
        payload, _, _ = base.read_checkpoint(Path(contract["checkpoint"]["path"]), expected_sha=contract["checkpoint"]["sha256"])
        encoder, connector, dit, vae, scheduler, audit = base.stage2.load_models(SimpleNamespace(config=Path(contract["config"])), payload, device)
        atomic_json(audit_path, dict(audit, contract_sha256=fingerprint(contract)))
        del payload
        gc.collect()
        if sha256(contract["quicksr"]["path"]) != contract["quicksr"]["sha256"]:
            raise ValueError("QuickSR weights changed after preflight")
        model, _ = load_quicksrnet(variant="medium", scale=2, checkpoint=contract["quicksr"]["path"], device=device, dtype=torch.float16)
        with torch.inference_mode():
            with autocast(device):
                negative = connector(*encoder("", drop_condition=True))
            for number, item in enumerate(pending, 1):
                image = load_shared_image(image_root / "images" / f"{item['key']}.png", image_shas[item["key"]])
                pixels = np.array(image, copy=True)
                pixels_sha = hashlib.sha256(memoryview(np.ascontiguousarray(pixels))).hexdigest()
                with autocast(device):
                    positive = positive_condition(encoder, connector, item["prompt"], image, args.mode)
                    length = max(positive[0].shape[1], negative[0].shape[1])
                    condition = tuple(torch.cat([neg, pos]) for neg, pos in zip(
                        pad_condition(negative, length), pad_condition(positive, length)))
                digest = hashlib.sha256()
                for value in condition:
                    tensor = value.detach().cpu().contiguous()
                    digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode("ascii"))
                    digest.update(tensor.view(torch.uint8).numpy().tobytes())
                anchor = None
                if contract["external_anchor"]:
                    video = torch.from_numpy(pixels).permute(2, 0, 1)[None, :, None]
                    video = (video.to(device=device, dtype=torch.float32) / 127.5 - 1).to(torch.bfloat16)
                    anchor = encode_anchor(vae, video, torch.Generator(device=device).manual_seed(item["seed"] + 228),
                                           window_size=contract["training_contract"].get("vae_window_size", 8))
                frames, statistics = generate_frames(dit, vae, scheduler, condition, device=device, seed=item["seed"],
                    height=320, width=512, num_frames=49, first_steps=20, video_steps=10, guidance=7., video_guidance=5., anchor_latent=anchor)
                if bool(statistics[0].get("supplied_anchor", False)) != contract["external_anchor"]:
                    raise ValueError("Sampler did not follow the requested first-frame route")
                export_pair(frames, output=args.output_dir, contract=contract, item=item, model=model, device=device,
                            statistics=statistics, image_sha=image_shas[item["key"]], pixels_sha=pixels_sha, condition_sha=digest.hexdigest())
                if number == 1 or number % 20 == 0:
                    print(f"{args.mode} rank={args.rank}: pairs={number}/{len(pending)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    elif items and (not audit_path.is_file() or read_json(audit_path).get("contract_sha256") != fingerprint(contract)):
        raise ValueError("Missing trained/frozen-stack audit for cached videos")
    atomic_json(args.output_dir / "worker_receipts" / f"rank_{args.rank}.json", dict(
        status="complete", contract_sha256=fingerprint(contract), assigned=len(items), generated_pairs=len(pending),
        reused_pairs=len(items)-len(pending), seconds=time.monotonic()-started))


def verify_generation(output):
    contract = read_json(output / "run_contract.json")
    image_root, images, image_shas = checked_shared(contract)
    _, manifest = shared.verify_cache(image_root)
    if fingerprint(manifest) != contract["shared_ssd1b"]["image_manifest_sha256"]:
        raise ValueError("Shared image manifest changed")
    items = list(base.items_for(contract))
    for variant in base.VARIANTS:
        actual = {p.name for p in (output / variant / "videos").glob("*.mp4") if not p.name.startswith(".")}
        if actual != {item["filename"] for item in items}:
            raise ValueError(f"Missing/unexpected videos in {variant}")
    hashes = {variant: [] for variant in base.VARIANTS}
    for item in items:
        records = pair_records(output, contract, item, image_shas[item["key"]])
        for variant, record in zip(base.VARIANTS, records):
            hashes[variant].append(record["video_sha256"])
    for variant, values in hashes.items():
        scores = output / variant / "scores"
        snapshot = scores / "generation_snapshot.json"
        digest = fingerprint(values)
        if scores.exists() and (not snapshot.exists() or read_json(snapshot).get("sha256") != digest):
            scores.rename(output / variant / f"retired_scores_{time.time_ns()}")
        atomic_json(snapshot, dict(sha256=digest, contract_sha256=fingerprint(contract)))
    atomic_json(output / "generation_summary.json", dict(status="complete", checkpoint=contract["checkpoint"],
                contract_sha256=fingerprint(contract), videos_per_variant=len(items), same_generated_frames=True,
                mode=contract["mode"], image_to_smolvlm2=contract["image_to_smolvlm2"], external_anchor=contract["external_anchor"],
                shared_ssd1b=contract["shared_ssd1b"], image_generation=False, scores_computed=False))


def score_worker(args):
    contract = read_json(args.output_dir / "run_contract.json")
    for variant, _, index in base.dual.score_assignments(args.workers, [contract["system"]])[args.rank]:
        folder = args.output_dir / variant
        subprocess.run([sys.executable, str(ROOT / "tools/evaluate_vbench_resume.py"),
                        "--videos", str(folder / "videos"), "--full-info", str(args.output_dir / "vbench_info.json"),
                        "--output", str(folder / "scores" / DIMENSIONS[index]),
                        "--expected-samples-per-prompt", str(contract["samples_per_prompt"]),
                        "--start", str(index), "--end", str(index + 1)], cwd=args.vbench_repo, check=True)


def summarize(args):
    from evaluate_vbench_resume import tabulate

    contract = read_json(args.output_dir / "run_contract.json")
    if contract["max_prompts"] or not contract["training_complete"] or contract["samples_per_prompt"] != 5:
        raise ValueError("No full score for a subset or incomplete training checkpoint")
    verify_generation(args.output_dir)
    scores = {}
    for variant in base.VARIANTS:
        merged = args.output_dir / variant / "scores/vbench"
        merged.mkdir(parents=True, exist_ok=True)
        for dimension in DIMENSIONS:
            source = args.output_dir / variant / "scores" / dimension / "vbench" / f"{dimension}_eval_results.json"
            if dimension not in read_json(source):
                raise ValueError(f"Missing score: {source}")
            shutil.copyfile(source, merged / source.name)
        scores[variant] = {contract["system"]: tabulate(merged)}
        subprocess.run([sys.executable, str(ROOT / "tools/report_vbench_low_prompts.py"), "--score-dir", str(merged),
                        "--output-prefix", str(args.output_dir / variant / "low_prompts")], check=True)
    report, markdown = base.dual.ablation_report(scores, {"selected_systems": [contract["system"]]})
    old_scope = report["scope"]
    report.update(protocol=contract, scope=contract["scope"], evaluation_assets=read_json(args.output_dir / "score_assets.json"))
    atomic_json(args.output_dir / "quicksr_ablation.json", report)
    (args.output_dir / "quicksr_ablation.md").write_text(markdown.replace(old_scope, contract["scope"]), encoding="utf-8")
    print(f"Completed {contract['system']}: {args.output_dir}", flush=True)


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--help" in argv or "-h" in argv:
        print("Required: --mode {ssd1b_anchor,image_condition}; optional: --frames-root PATH\n" + __doc__)
        return base.parse_args(["--help"])
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--mode", choices=tuple(MODES), required=True)
    extra.add_argument("--frames-root", type=Path, default=shared.DEFAULT_ROOT)
    known, rest = extra.parse_known_args(argv)
    args = base.parse_args(rest)
    if args.allow_incomplete_checkpoint:
        raise ValueError("This benchmark only accepts completed epoch 3, including smoke runs")
    args.mode, args.frames_root = known.mode, known.frames_root.resolve()
    if args.expected_sha256 is None:
        args.expected_sha256 = shared.FINAL_SHA256
    if not any(v == "--output-dir" or v.startswith("--output-dir=") for v in rest):
        suffix = f"_smoke{args.max_prompts}" if args.max_prompts else ""
        args.output_dir = ROOT / f"output/vbench_stage2_epoch3_{args.mode}_{args.samples_per_prompt}x_dual{suffix}"
    if args.output_dir == args.frames_root:
        raise ValueError("Image cache and benchmark outputs must be separate")
    return args


def main():
    args = parse_args()
    if args.phase == "run" and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Submit the Stage-2 conditioning benchmark with sbatch")
    if args.phase == "generation-worker":
        generation_worker(args)
        return
    if args.phase == "warm-scores":
        base.core.warm_scores(args)
        return
    if args.phase == "score-worker":
        score_worker(args)
        return
    if args.phase == "summarize":
        summarize(args)
        return
    os.chdir(ROOT)
    with output_lock(args.output_dir):
        devices = base.core.gpu_tokens(args.workers) if args.phase == "run" else None
        if args.phase == "run" and not Path(args.vbench_python).is_file():
            raise FileNotFoundError(args.vbench_python)
        prepare(args)
        if args.phase == "prepare":
            return
        if not args.max_prompts:
            subprocess.run([args.vbench_python, str(Path(__file__).resolve()), "--phase", "warm-scores",
                            "--mode", args.mode, "--output-dir", str(args.output_dir)], cwd=args.vbench_repo, check=True)
        previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            commands = [[sys.executable, str(Path(__file__).resolve()), "--phase", "generation-worker", "--mode", args.mode,
                         "--output-dir", str(args.output_dir), "--rank", str(rank), "--workers", str(args.workers)]
                        for rank in range(args.workers)]
            base.core.run_processes(commands, devices, args.output_dir / "logs/generation")
            verify_generation(args.output_dir)
            if args.max_prompts:
                print(f"{args.mode} dual-export smoke passed; no full score claimed.", flush=True)
                return
            commands = [[args.vbench_python, str(Path(__file__).resolve()), "--phase", "score-worker", "--mode", args.mode,
                         "--output-dir", str(args.output_dir), "--vbench-repo", str(args.vbench_repo),
                         "--rank", str(rank), "--workers", str(args.workers)] for rank in range(args.workers)]
            base.core.run_processes(commands, devices, args.output_dir / "logs/scoring", cwd=args.vbench_repo)
            subprocess.run([args.vbench_python, str(Path(__file__).resolve()), "--phase", "summarize", "--mode", args.mode,
                            "--output-dir", str(args.output_dir)], cwd=args.vbench_repo, check=True)
        finally:
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
