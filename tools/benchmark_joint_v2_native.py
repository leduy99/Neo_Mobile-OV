#!/usr/bin/env python
"""Run a resume-safe multi-GPU, paired end-to-end VBench comparison."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import importlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vbench_joint_native_protocol import (  # noqa: E402
    DIMENSIONS, FORMAT, IMAGE_FILE, IMAGE_SHA, JOINT_FILE, JOINT_SHA, NATIVE_REVISION,
    SYSTEMS, SEED_MAPPINGS, aggregate_scores, atomic_json, bind_contract, ensure_checkpoint, fingerprint,
    load_prompt_rows, output_lock, read_json, sample_items, score_assignments, sha256,
)


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def native_snapshot(args):
    if args.native_model_dir:
        path = args.native_model_dir.resolve()
    else:
        from huggingface_hub import snapshot_download

        try:
            path = Path(snapshot_download("karnewar/Neodragon", revision=NATIVE_REVISION,
                                         cache_dir=str(args.native_cache), local_files_only=True))
        except Exception:
            try:
                path = Path(snapshot_download("karnewar/Neodragon", revision=NATIVE_REVISION,
                                             cache_dir=str(args.native_cache)))
            except Exception as error:
                raise RuntimeError(
                    "Native NeoDragon is missing from the pinned local cache and download failed. "
                    "The old HF endpoint may be unavailable. Preserve existing native caches; "
                    "set NATIVE_MODEL_DIR to an intact native model directory."
                ) from error
    required = ["causal_video_vae"]
    if "joint_v2" in args.systems:
        required.append("diffusion_transformer_320p_multistep_t2v")
    if "native_hybrid" in args.systems:
        required.extend(("diffusion_transformer_320p", "context_adapter", "text_encoder", "text_encoder_2",
                         "text_encoder_3", "tokenizer", "tokenizer_2", "tokenizer_3", "ssd_1b_unet", "ssd_1b_vae",
                         "ssd_1b_text_encoder", "ssd_1b_text_encoder_2", "ssd_1b_tokenizer", "ssd_1b_tokenizer_2"))
    missing = [name for name in required if not (path / name).is_dir()]
    if missing:
        raise ValueError(f"Native model directory is incomplete: {missing}")
    records = []
    for name in required:
        files = sorted(p for p in (path / name).rglob("*") if p.is_file())
        if name == "diffusion_transformer_320p_multistep_t2v":
            # Joint V2 already supplies the full DiT weights, not the release weights.
            files = [path / name / "config.json"] if (path / name / "config.json").is_file() else []
        if not files or not any(p.suffix in (".json", ".safetensors", ".bin", ".model") for p in files):
            raise ValueError(f"Empty native component: {name}")
        for file in files:
            records.append({"file": str(file.relative_to(path)), "size": file.stat().st_size, "sha256": sha256(file)})
    return path.resolve(), records


def prepare(args):
    import gc
    import torch
    from new_mobile_ov.checkpoints import ensure_dreamlite_checkpoint, ensure_smolvlm2_checkpoint
    from new_mobile_ov.config import load_config
    from new_mobile_ov.generation.quicksrnet import load_quicksrnet

    torch.set_num_threads(4)
    prompts, rows = load_prompt_rows(args.vbench_info, args.max_prompts)
    if not args.max_prompts and len(prompts) != 944:
        raise ValueError(f"Expected all 944 official unique prompts, got {len(prompts)}")
    items = list(sample_items(prompts, args.samples_per_prompt, args.seed, args.seed_mapping))
    image, joint = None, None
    if "joint_v2" in args.systems:
        image = ensure_checkpoint(args.image_checkpoint, IMAGE_FILE, IMAGE_SHA, "image")
        joint = ensure_checkpoint(args.joint_checkpoint, JOINT_FILE, JOINT_SHA, "joint")
    if not (args.native_repo / "neodragon").is_dir():
        raise FileNotFoundError(f"Native source checkout missing: {args.native_repo}. Restore it before running.")
    model, native_files = native_snapshot(args)
    image_config, video_config = load_config(args.image_config), load_config(args.video_config)
    smol, dream, dream_files, bridge_loads = None, None, [], []
    if "joint_v2" in args.systems:
        smol = Path(ensure_smolvlm2_checkpoint(video_config.bridge.smolvlm2_ckpt_path)).resolve()
        image_smol = Path(ensure_smolvlm2_checkpoint(image_config.bridge.smolvlm2_ckpt_path)).resolve()
        if file_record(smol)["sha256"] != file_record(image_smol)["sha256"]:
            raise ValueError("Image/video configs do not use the same SmolVLM2 weights")
        dream = Path(ensure_dreamlite_checkpoint(image_config.dreamlite.checkpoint_dir,
                                              model_id=image_config.dreamlite.model_id,
                                              revision=image_config.dreamlite.revision)).resolve()
        dream_files = [p for name in ("unet", "vae", "scheduler", "tokenizer")
                       for p in sorted((dream / name).rglob("*")) if p.is_file()]
        bridge_loads = [("image", image["path"], image_config), ("joint", joint["path"], video_config)]
    qsr, qsr_path = load_quicksrnet(variant="medium", scale=2, cache_dir=str(args.quicksr_cache),
                                  device=torch.device("cpu"), dtype=torch.float32)
    del qsr
    # Check trained bridge states before launching GPU workers.
    from new_mobile_ov.bridge import MobileOVDreamLiteImageBridge, MobileOVNeodragonTextBridge
    for kind, checkpoint, config in bridge_loads:
        payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
        if kind == "image":
            bridge = MobileOVDreamLiteImageBridge(config.bridge, config.dreamlite_bridge,
                                                 device=torch.device("cpu"), dtype=torch.bfloat16).eval()
            bridge.load_trainable_state_dict(payload["bridge"])
        else:
            bridge = MobileOVNeodragonTextBridge(config.bridge, device=torch.device("cpu"), dtype=torch.bfloat16).eval()
            bridge.load_state_dict(payload["bridge"], strict=True)
        del bridge, payload
        gc.collect()
    native_git = subprocess.check_output(["git", "-C", str(args.native_repo), "rev-parse", "HEAD"], text=True).strip()
    descriptions = {
        "joint_v2": "DreamLite V11 bridge + frozen Bridge V2 + fine-tuned monolithic DiT step 10K; CFG 7/5; 20/10 steps per stage",
        "native_hybrid": "Native SSD1B + native text conditioner + released Hybrid; 1-1-1, no CFG",
    }
    contract = {
        "format": FORMAT, "prompts": prompts, "full_info_sha256": sha256(args.vbench_info),
        "selected_systems": list(args.systems), "protocol_name": args.protocol,
        "samples_per_prompt": args.samples_per_prompt, "seed": args.seed,
        "seed_mapping": args.seed_mapping,
        "seed_policy": ("seed + global_prompt_index + sample_index * 1000003; sample-0 preserves the completed 1x seeds; reset independently before image and video generation"
                        if args.seed_mapping == "extendable" else "seed + global_prompt_index * samples_per_prompt + sample_index; reset independently before image and video generation"),
        "anchor": {"width": 1024, "height": 640, "steps": 4, "time_id_width": 1280, "time_id_height": 800},
        "base_video": {"width": 512, "height": 320, "frames": 49, "fps": 24},
        "output": {"width": 1024, "height": 640, "frames": 49, "fps": 24},
        "quicksr": {"variant": "medium", "scale": 2, "dtype": "fp16", "batch_frames": 4},
        "weights": {"joint": joint, "image": image, "smolvlm2": file_record(smol) if smol else None,
                    "quicksr": file_record(qsr_path)},
        "assets": {"native_repo_path": str(args.native_repo.resolve()), "native_source_commit": native_git,
                   "native_source_sha256": fingerprint([{"file": str(p.relative_to(args.native_repo)), "sha256": sha256(p)}
                       for p in sorted((args.native_repo / "neodragon").rglob("*.py"))]),
                   "native_model_dir": str(model), "native_files": native_files,
                   "dreamlite_dir": str(dream) if dream else None,
                   "dreamlite_files": [{"file": str(p.relative_to(dream)), "sha256": sha256(p)} for p in dream_files]},
        "image_config": file_record(args.image_config), "video_config": file_record(args.video_config),
        "generation_code_sha256": sha256(ROOT / "tools/generate_vbench_joint_v2_native.py"),
        "protocol_code_sha256": sha256(ROOT / "tools/vbench_joint_native_protocol.py"),
        "compat_code_sha256": sha256(ROOT / "new_mobile_ov/generation/neodragon_compat.py"),
        "score_code_sha256": sha256(ROOT / "tools/evaluate_vbench_resume.py"),
        "vbench_source_commit": subprocess.check_output(["git", "-C", str(args.vbench_repo), "rev-parse", "HEAD"], text=True).strip(),
        "systems": {system: descriptions[system] for system in args.systems},
        "scope": ("Same prompts/seeds, output shape, FPS, VBench implementation and QuickSR. Different image models and generator schedules are intentional end-to-end system differences. Not exact paper replication or a same-backbone conditioner ablation."
                  if len(args.systems) > 1 else f"Only {args.systems[0]} is generated and evaluated. No other model baseline is executed. A 1x run is a single-seed diagnostic, not the official five-sample paper protocol."),
    }
    contract_sha = bind_contract(args.output_dir, contract)
    atomic_json(args.output_dir / "vbench_info.json", rows)
    atomic_json(args.output_dir / "sample_plan.json", {"contract_sha256": contract_sha, "samples": items})
    if args.reuse_1x_run:
        from reuse_vbench_1x_samples import reuse_samples

        reuse_samples(args.output_dir, contract, args.reuse_1x_run)
    print(f"Preflight passed: {len(prompts)} prompts, {len(items)} videos/system; selected={args.systems}; QuickSR 2x.", flush=True)
    return contract


def gpu_tokens(workers):
    import torch

    if torch.cuda.device_count() < workers:
        raise RuntimeError(f"Requested {workers} workers but only {torch.cuda.device_count()} visible CUDA devices")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    tokens = visible.split(",") if visible else [str(index) for index in range(torch.cuda.device_count())]
    if len(tokens) < workers or len(set(tokens[:workers])) != workers:
        raise ValueError(f"Cannot assign independent GPUs from CUDA_VISIBLE_DEVICES={visible!r}")
    return tokens[:workers]


def run_processes(commands, devices, log_dir, *, cwd=ROOT):
    log_dir.mkdir(parents=True, exist_ok=True)
    children = []
    with ExitStack() as stack:
        try:
            for rank, command in enumerate(commands):
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = devices[rank]
                env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "4"
                log = stack.enter_context((log_dir / f"rank_{rank}.log").open("a"))
                child = subprocess.Popen(command, env=env, cwd=cwd, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                children.append(child)
            while True:
                statuses = [child.poll() for child in children]
                failed = [(rank, status) for rank, status in enumerate(statuses) if status not in (None, 0)]
                if failed:
                    for rank, _ in failed:
                        tail = (log_dir / f"rank_{rank}.log").read_text(errors="replace").splitlines()[-30:]
                        print(f"Failed worker {rank}, last log lines:\n" + "\n".join(tail), flush=True)
                    raise RuntimeError(f"Worker failed: {failed}; inspect {log_dir}")
                if all(status == 0 for status in statuses):
                    return
                time.sleep(.5)
        finally:
            for child in children:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for child in children:
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


def verify_generation(output, contract):
    contract_sha = fingerprint(contract)
    items = list(sample_items(contract["prompts"], contract["samples_per_prompt"], contract["seed"],
                             contract.get("seed_mapping", "legacy")))
    selected = contract.get("selected_systems", SYSTEMS)
    for system in selected:
        hashes = []
        folder = output / system
        expected_names = {item["filename"] for item in items}
        actual_names = {p.name for p in (folder / "videos").glob("*.mp4") if not p.name.startswith(".")}
        if actual_names != expected_names:
            raise ValueError(f"Unexpected/missing video coverage for {system}; expected exactly {len(items)} final videos")
        for item in items:
            record = read_json(folder / "records" / f"{item['key']}.json")
            anchor = folder / "anchors" / f"{item['key']}.png"
            video = folder / "videos" / item["filename"]
            if (record.get("sample") != item or record.get("system") != system
                    or record.get("contract_sha256") != contract_sha or record.get("shape") != contract["output"]
                    or record.get("video_sha256") != sha256(video) or record.get("anchor_sha256") != sha256(anchor)):
                raise ValueError(f"Incomplete/wrong generation result: {system}/{item['key']}")
            hashes.append(record["video_sha256"])
        score_dir = output / system / "scores"
        digest = fingerprint(hashes)
        snapshot_file = score_dir / "generation_snapshot.json"
        if score_dir.exists() and (not snapshot_file.exists() or read_json(snapshot_file).get("sha256") != digest):
            retired = output / system / f"retired_scores_{time.time_ns()}"
            score_dir.rename(retired)
            print(f"Changed input videos: moved stale scores to {retired}", flush=True)
        atomic_json(snapshot_file, {"contract_sha256": contract_sha, "videos": len(items), "sha256": digest})
    atomic_json(output / "generation_summary.json", {"status": "complete", "contract_sha256": contract_sha,
                "videos_per_system": len(items), "systems": list(selected), "all_exports_decode_checked_by_workers": True})


def score_worker(args):
    contract = read_json(args.output_dir / "run_contract.json")
    for system, index in score_assignments(args.workers, contract.get("selected_systems", SYSTEMS))[args.rank]:
        dimension = DIMENSIONS[index]
        command = [sys.executable, str(ROOT / "tools/evaluate_vbench_resume.py"),
                   "--videos", str(args.output_dir / system / "videos"),
                   "--full-info", str(args.output_dir / "vbench_info.json"),
                   "--output", str(args.output_dir / system / "scores" / dimension),
                   "--expected-samples-per-prompt", str(contract["samples_per_prompt"]),
                   "--start", str(index), "--end", str(index+1)]
        print(f"Scoring {system}/{dimension}", flush=True)
        subprocess.run(command, cwd=args.vbench_repo, check=True)


def warm_scores(args):
    """Serialize upstream wget/git downloads before independent GPU evaluators."""
    backend = importlib.import_module("vbench")
    for dimension in DIMENSIONS:
        try:
            importlib.import_module(f"vbench.{dimension}")
        except ImportError as error:
            raise RuntimeError(f"Incomplete/incompatible VBench backend at {backend.__file__}: {dimension}") from error
    from vbench.utils import CACHE_DIR, init_submodules
    from vbench.aesthetic_quality import get_aesthetic_model

    cache = Path(CACHE_DIR)
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".joint_native_download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        submodules = init_submodules(list(DIMENSIONS), local=True, read_frame=False)
        get_aesthetic_model(submodules["aesthetic_quality"][1])
        paths = set()

        def inspect(value):
            if isinstance(value, dict):
                for item in value.values():
                    inspect(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    inspect(item)
            elif isinstance(value, str) and value.endswith((".pt", ".pth")):
                path = Path(value)
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"VBench model download is incomplete: {path}")
                paths.add(path)

        inspect(submodules)
        paths.add(Path(submodules["aesthetic_quality"][1]) / "sa_0_4_vit_l_14_linear.pth")
        records = [file_record(path) for path in sorted(paths)]
        target = args.output_dir / "score_assets.json"
        if target.exists() and read_json(target) != records:
            raise ValueError("Evaluation model assets changed; use a new RUN_ROOT")
        atomic_json(target, records)
    print("VBench model cache prepared once before parallel scoring.", flush=True)


def summarize(args):
    from evaluate_vbench_resume import tabulate

    contract = read_json(args.output_dir / "run_contract.json")
    verify_generation(args.output_dir, contract)
    summaries = {}
    selected = contract.get("selected_systems", SYSTEMS)
    for system in selected:
        merged = args.output_dir / system / "scores/vbench"
        merged.mkdir(parents=True, exist_ok=True)
        for dimension in DIMENSIONS:
            source = args.output_dir / system / "scores" / dimension / "vbench" / f"{dimension}_eval_results.json"
            value = read_json(source)
            if dimension not in value:
                raise ValueError(f"Missing dimension: {source}")
            shutil.copyfile(source, merged / source.name)
        summaries[system] = tabulate(merged)
        subprocess.run([sys.executable, str(ROOT / "tools/report_vbench_low_prompts.py"),
                        "--score-dir", str(merged), "--output-prefix", str(args.output_dir / system / "low_prompts")], check=True)
    if len(selected) == 1:
        system = selected[0]
        report = {"status": "complete", "protocol": contract, "system": system,
                  "scores": summaries[system], "evaluation_assets": read_json(args.output_dir / "score_assets.json"),
                  "native_reference_evaluated": system == "native_hybrid",
                  "scope": "Single-system evaluation. A 1x run is not the official five-sample paper protocol. No scores are claimed for any unselected system."}
        atomic_json(args.output_dir / "result.json", report)
        lines = [f"# {system} / {contract.get('protocol_name', 'standard5x')} / QuickSR 2x", "",
                 "| Score | Percent |", "| --- | ---: |"]
        for key in ("quality_score", "semantic_score", "total_score"):
            lines.append(f"| {key} | {100*summaries[system][key]:.2f} |")
        for key in DIMENSIONS:
            lines.append(f"| {key} (raw) | {100*summaries[system]['raw'][key]:.2f} |")
        (args.output_dir / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Completed {system} benchmark: {args.output_dir / 'result.json'}", flush=True)
        return
    report = aggregate_scores(summaries, contract)
    report["evaluation_assets"] = read_json(args.output_dir / "score_assets.json")
    atomic_json(args.output_dir / "comparison.json", report)
    lines = ["# Joint V2 vs Released Native NeoDragon", "", contract["scope"], "",
             "| Score (%) | Joint V2 + QuickSR | Native Hybrid + QuickSR | Difference (points) |",
             "| --- | ---: | ---: | ---: |"]
    for key in ("quality_score", "semantic_score", "total_score"):
        lines.append(f"| {key} | {100*summaries['joint_v2'][key]:.2f} | {100*summaries['native_hybrid'][key]:.2f} | {report['joint_minus_native_percentage_points'][key]:+.2f} |")
    for key in DIMENSIONS:
        lines.append(f"| {key} (raw) | {100*summaries['joint_v2']['raw'][key]:.2f} | {100*summaries['native_hybrid']['raw'][key]:.2f} | {report['joint_minus_native_percentage_points']['raw_dimensions'][key]:+.2f} |")
    (args.output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Completed paired benchmark: {args.output_dir / 'comparison.json'}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("run", "prepare", "warm-scores", "score-worker", "summarize"), default="run")
    parser.add_argument("--systems", choices=SYSTEMS, nargs="+", default=list(SYSTEMS))
    parser.add_argument("--protocol", choices=("standard5x", "paired1x"), default="standard5x")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/vbench_joint_v2_vs_native_quicksr_5x")
    parser.add_argument("--vbench-repo", type=Path, default=ROOT / "checkpoints/vbench_repo")
    parser.add_argument("--vbench-info", type=Path)
    parser.add_argument("--vbench-python", default=os.environ.get("VBENCH_PYTHON", "/proj/cvl/users/x_fahkh2/envs/neo_vbench/bin/python"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--samples-per-prompt", type=int, choices=range(1, 6))
    parser.add_argument("--max-prompts", type=int, default=0, help="Small generation-only smoke; use a separate RUN_ROOT")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seed-mapping", choices=SEED_MAPPINGS, default="legacy")
    parser.add_argument("--reuse-1x-run", action="append", default=[], metavar="SYSTEM=RUN_DIR",
                        help="Validate and import an existing sample-0 without regenerating it")
    parser.add_argument("--image-config", type=Path, default=ROOT / "configs/mobile_ov_dreamlite_compact_v9.yaml")
    parser.add_argument("--video-config", type=Path, default=ROOT / "configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--joint-checkpoint", type=Path, default=ROOT / "checkpoints/hf_mobile_ov" / JOINT_FILE)
    parser.add_argument("--image-checkpoint", type=Path, default=ROOT / "checkpoints/hf_mobile_ov" / IMAGE_FILE)
    parser.add_argument("--native-repo", type=Path, default=ROOT / "checkpoints/neodragon_repo")
    parser.add_argument("--native-cache", type=Path, default=ROOT / "checkpoints/neodragon")
    parser.add_argument("--native-model-dir", type=Path)
    parser.add_argument("--quicksr-cache", type=Path, default=ROOT / "checkpoints/quicksrnet")
    args = parser.parse_args(argv)
    if args.samples_per_prompt is None:
        args.samples_per_prompt = 1 if args.protocol == "paired1x" else 5
    if len(set(args.systems)) != len(args.systems):
        parser.error("Do not repeat a selected system")
    if not 1 <= args.workers <= 8 or not 0 <= args.rank < args.workers or args.max_prompts < 0 or args.seed < 0:
        parser.error("Invalid worker rank/count, max-prompts or seed")
    if args.phase in ("run", "prepare") and not args.max_prompts and args.protocol == "standard5x" and args.samples_per_prompt != 5:
        parser.error("Full standard VBench requires 5 samples/prompt; use --max-prompts for a separate small smoke run")
    if args.protocol == "paired1x" and args.samples_per_prompt != 1:
        parser.error("paired1x requires exactly one sample per prompt")
    if args.reuse_1x_run and (args.samples_per_prompt <= 1 or args.seed_mapping != "extendable"):
        parser.error("Reusing 1x samples requires multiple samples with --seed-mapping extendable")
    args.output_dir = args.output_dir.resolve()
    args.vbench_repo = args.vbench_repo.resolve()
    if args.vbench_info is None:
        candidates = [args.vbench_repo / "vbench/VBench_full_info.json", args.vbench_repo / "VBench_full_info.json"]
        args.vbench_info = next((p for p in candidates if p.is_file()), candidates[0])
    return args


def main():
    args = parse_args()
    if args.phase == "warm-scores":
        warm_scores(args)
        return
    if args.phase == "score-worker":
        score_worker(args)
        return
    if args.phase == "summarize":
        summarize(args)
        return
    os.chdir(ROOT)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    with output_lock(args.output_dir):
        if args.phase == "run":
            devices = gpu_tokens(args.workers)
            if not Path(args.vbench_python).is_file():
                raise FileNotFoundError(args.vbench_python)
        contract = prepare(args)
        if args.phase == "prepare":
            return
        if not args.max_prompts:
            subprocess.run([args.vbench_python, str(Path(__file__).resolve()), "--phase", "warm-scores",
                            "--output-dir", str(args.output_dir)], cwd=args.vbench_repo, check=True)
        previous_sigterm = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            for system in args.systems:
                commands = [[sys.executable, str(ROOT / "tools/generate_vbench_joint_v2_native.py"),
                             "--output-dir", str(args.output_dir), "--system", system,
                             "--rank", str(rank), "--workers", str(args.workers)] for rank in range(args.workers)]
                print(f"Generating {system} on {args.workers} GPUs; logs under {args.output_dir}/logs/{system}", flush=True)
                run_processes(commands, devices, args.output_dir / "logs" / system)
            verify_generation(args.output_dir, contract)
            if args.max_prompts:
                print(f"Smoke generation passed for {args.systems}; full VBench scoring intentionally skipped.", flush=True)
                return
            commands = [[args.vbench_python, str(Path(__file__).resolve()), "--phase", "score-worker",
                         "--output-dir", str(args.output_dir), "--vbench-repo", str(args.vbench_repo),
                         "--rank", str(rank), "--workers", str(args.workers)] for rank in range(args.workers)]
            print(f"Scoring selected systems: {len(args.systems)*len(DIMENSIONS)} tasks across {args.workers} GPUs.", flush=True)
            run_processes(commands, devices, args.output_dir / "logs/scoring", cwd=args.vbench_repo)
            subprocess.run([args.vbench_python, str(Path(__file__).resolve()), "--phase", "summarize",
                            "--output-dir", str(args.output_dir)], cwd=args.vbench_repo, check=True)
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
