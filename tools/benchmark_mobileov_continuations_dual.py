#!/usr/bin/env python
"""Completed Job B / uniform-history: matched text-only VBench, raw and QuickSR."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import gc
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import benchmark_mobileov_job_a_dual as previous  # noqa: E402
from vbench_joint_native_protocol import (  # noqa: E402
    atomic_json, bind_contract, fingerprint, output_lock, read_json, sha256,
)

base = previous.base
PARENT_STEP, FINAL_STEP = previous.PARENT_STEP, previous.FINAL_STEP
PARENT_SHA256 = previous.PARENT_SHA256
SYSTEMS = {"job_b": "stage2_job_b_lora", "uniform_history": "stage2_uniform_history"}
DEFAULT_JOB_B = ROOT / "output/mobileov_stage2_job_b_lora/17783291/job_b_dit_connector_lora_latest.pt"
HISTORY_ROOT = ROOT / "output/mobileov_stage2_uniform_history"
IMPLEMENTATION = ("tools/benchmark_mobileov_continuations_dual.py", "tools/benchmark_mobileov_job_a_dual.py",
                  "tools/benchmark_mobileov_stage2_epoch_dual.py", "new_mobile_ov/training/job_b_model.py",
                  "new_mobile_ov/training/job_b_qa.py", "new_mobile_ov/training/stage2_history.py")


def check_checkpoint(payload, experiment):
    import torch
    from new_mobile_ov.training import job_b_model as job
    from new_mobile_ov.training.stage2_history import MAX_HISTORY_SIGMA, history_contract

    training = payload.get("contract", {})
    parent = training.get("parent_contract", {})
    plan = training.get("media_epochs", {})
    if (training.get("init_step") != PARENT_STEP or training.get("init_stage2_sha256") != PARENT_SHA256
            or parent.get("steps") != PARENT_STEP or parent.get("media_epochs", {}).get("epochs") != 3
            or parent.get("media_epochs", {}).get("steps") != PARENT_STEP):
        raise ValueError("Expected a fork of the exact completed epoch-3 step283977 parent")
    expected = dict(recipe="media_epochs_i2v_v1", epochs=1, start_epoch=3, end_epoch=4,
                    images=984636, videos=529902, world_size=8, accumulation=2,
                    global_batch=16, steps_per_epoch=94659, steps=94659)
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("Continuation media-pass/data budget differs from the completed production recipe")
    if training.get("steps") != FINAL_STEP or payload.get("step") != FINAL_STEP:
        raise ValueError(f"Checkpoint is NOT complete: expected step={FINAL_STEP}, got {payload.get('step')}")
    for name in ("data_summary_sha256", "processor_sha256", "frozen_weights_sha256", "config_sha256", "max_tokens"):
        if not parent.get(name) or training.get(name) != parent[name]:
            raise ValueError(f"Continuation changed the parent data/base conditioning stack: {name}")
    if training.get("ema") is not False:
        raise ValueError("Expected non-EMA trained weights")
    if experiment == "uniform_history":
        if (payload.get("format") != base.stage2.FORMAT
                or training.get("format") != base.stage2.FORMAT
                or training.get("scope") != "uniform_history_continuation_public_data_not_exact_UniVideo"
                or training.get("initialization") != "stage2_weights_only_fork_fresh_optimizer"
                or training.get("history_corruption") != history_contract(MAX_HISTORY_SIGMA)
                or training.get("auxiliary_reconstruction", {}).get("weight") != .1
                or training.get("video_units") != "T2V/reference:uniform0_to6; anchor:uniform1_to6; teacher_forced_corrupted_history"
                or training.get("trainable_components") != ["connector", "dit"]
                or training.get("frozen_components") != ["smolvlm2", "vae"]
                or "first_frame_focus" in training or "smolvlm2_lora" in payload):
            raise ValueError("Not the completed uniform-sampling/reconstruction-.1/history-corruption recipe")
        base.stage2.check_payload(payload, FINAL_STEP)
    elif experiment == "job_b":
        spec = dict(rank=16, alpha=32, dropout=0, targets=list(job.TARGETS))
        if (payload.get("format") != job.FORMAT or training.get("format") != job.FORMAT
                or training.get("integration_smoke") is not False
                or training.get("lora_spec") != spec or payload.get("lora_spec") != spec
                or training.get("trainable_components") != ["LLM_LoRA", "MCP", "DiT"]
                or training.get("reconstruction_weight") != .2 or training.get("t2v_first_probability") != .5
                or training.get("history_noise") != 0 or training.get("kl") is not False
                or training.get("qa_every") != 8 or training.get("qa_weight") != .1):
            raise ValueError("Not the completed Job-B LoRA + QA production recipe")
        state = payload.get("smolvlm2_lora", {})
        if not state or any(not key.endswith((".lora_A", ".lora_B")) for key in state):
            raise ValueError("Missing/invalid Job-B LoRA weights; never infer with the unadapted encoder")
        for key, value in state.items():
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.ndim != 2
                    or not bool(torch.isfinite(value).all())):
                raise ValueError(f"Invalid/non-finite LoRA tensor: {key}")
            if key.endswith(".lora_A"):
                other = state.get(key[:-1] + "B")
                if (value.shape[0] != spec["rank"] or not isinstance(other, torch.Tensor)
                        or other.ndim != 2 or other.shape[1] != spec["rank"]):
                    raise ValueError("LoRA A/B pair or rank differs")
            elif key[:-1] + "A" not in state:
                raise ValueError("Missing LoRA A/B pair")
        if not any(bool(value.count_nonzero()) for key, value in state.items() if key.endswith(".lora_B")):
            raise ValueError("All LoRA B matrices are zero; adapters were not trained")
        # Check the shared trained tensors without pretending the on-disk format is Stage 2.
        shared_contract = dict(parent, **training)
        for key in ("smolvlm2_sha256", "initial_weights_sha256"):
            shared_contract[key] = parent[key]
        base.stage2.check_payload(dict(payload, format=base.stage2.FORMAT, contract=shared_contract), FINAL_STEP)
    else:
        raise ValueError(f"Unknown continuation: {experiment}")
    tasks = {"t2i", "t2v", "i2v_reference", "i2v_anchor", "image_reconstruction"}
    if experiment == "uniform_history" and not tasks.issubset(training.get("tasks", [])):
        raise ValueError("Missing continuation training tasks")
    return FINAL_STEP


def read_checkpoint(path, *, experiment, allow_incomplete=False, expected_sha=None):
    import torch

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed {experiment} checkpoint: {path}; no parent/HF fallback")
    before = path.stat()
    digest = sha256(path)
    if expected_sha and digest != expected_sha:
        raise ValueError("Checkpoint SHA256 changed")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    target = check_checkpoint(payload, experiment)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Checkpoint changed during preflight; wait for the final save")
    return payload, digest, target


def load_job_b_models(args, payload, device):
    import torch
    from new_mobile_ov.training import job_b_model as job
    from tools.infer_mobileov_stage1 import load_stack, verify_inference_stack

    check_checkpoint(payload, "job_b")
    training, parent = payload["contract"], payload["contract"]["parent_contract"]
    if sha256(args.config) != training["config_sha256"]:
        raise ValueError("Inference config differs from Job-B training")
    args.processor, args.max_tokens = training["processor"], training["max_tokens"]
    cfg, frozen, connector, dit, vae, scheduler = load_stack(args, device)
    audit = verify_inference_stack(cfg, frozen, dit, vae,
                                  dict(parent, frozen_weights_sha256=parent["initial_weights_sha256"]))
    owner = job.load_inference(payload, frozen, connector, dit)
    actual = job.lora_state(owner.encoder)
    if any(not torch.equal(actual[key], value) for key, value in payload["smolvlm2_lora"].items()):
        raise ValueError("Loaded LoRA weights differ from the trained checkpoint")
    # Keep the same BF16 DiT inference precision as the parent and Job A.
    owner.dit.to(dtype=torch.bfloat16)
    vae.eval().requires_grad_(False)
    audit.update(trained_lora_loaded=True, lora_tensors=len(actual), lora_spec=payload["lora_spec"],
                 loaded_lora_exact=True, trained_connector_loaded=True, trained_dit_loaded=True,
                 checkpoint_step=payload["step"], target_stack="multistep", dit_parameter_dtype="bfloat16",
                 lora_parameter_dtype="float32", understanding_evaluated=False)
    return owner.encoder, owner.connector, owner.dit, vae, scheduler, audit


@contextmanager
def continuation_policy(experiment):
    original_reader, original_system, original_loader = base.read_checkpoint, base.SYSTEM, base.stage2.load_models
    base.read_checkpoint = lambda path, **kwargs: read_checkpoint(path, experiment=experiment, **kwargs)
    base.SYSTEM = SYSTEMS[experiment]
    if experiment == "job_b":
        base.stage2.load_models = load_job_b_models
    try:
        yield
    finally:
        base.read_checkpoint, base.SYSTEM, base.stage2.load_models = original_reader, original_system, original_loader


def find_history_checkpoint():
    candidates = [path.parent / "stage2_dit_connector_latest.pt"
                  for path in sorted(HISTORY_ROOT.glob("*/training_complete.json"))
                  if read_json(path).get("step") == FINAL_STEP and (path.parent / "stage2_dit_connector_latest.pt").is_file()]
    if len(candidates) != 1:
        raise ValueError("Need exactly one completed uniform-history run, or pass --checkpoint explicitly; "
                         f"found {len(candidates)}: {candidates}")
    return candidates[0]


def prepare(args):
    if args.output_dir.resolve() == args.reference_run.resolve():
        raise ValueError("Use a separate output directory; preserve the epoch-3 benchmark")
    reference_data = None if args.max_prompts else previous.read_reference(args.reference_run)
    asset_args = copy.copy(args)
    asset_args.output_dir = args.output_dir / "_asset_preflight"
    with continuation_policy(args.experiment):
        contract = base.prepare(asset_args)
    helper_audit = previous.compare_protocol(reference_data[0], contract) if reference_data else None
    contract["continuation_evaluation"] = dict(
        experiment=args.experiment, parent_step=PARENT_STEP, final_step=FINAL_STEP,
        additional_updates=FINAL_STEP - PARENT_STEP, parent_checkpoint_sha256=PARENT_SHA256,
        reference_run=str(args.reference_run), reference_verified=reference_data is not None,
        reference_contract_sha256=fingerprint(reference_data[0]) if reference_data else None,
        reference_report_sha256=fingerprint(reference_data[1]) if reference_data else None,
        imported_inference_helpers=helper_audit, understanding_evaluated=False,
        history_corruption_at_inference=False, lora_required=args.experiment == "job_b")
    contract["implementation_sha256"].update({name: sha256(ROOT / name) for name in IMPLEMENTATION})
    contract["scope"] = (
        f"Completed {args.experiment}, step378636 vs epoch-3 parent step283977. Text-only, every frame generated; "
        "no SSD1B, DreamLite, native conditioner or supplied first latent. "
        + ("Load all trained LLM LoRA + MCP + monolithic DiT; this is generation evaluation, not understanding retention. "
           if args.experiment == "job_b" else
           "Frozen SmolVLM2 + trained MCP/monolithic DiT; training history corruption is OFF during inference. ")
        + "Same prompts/seeds, BF16 DiT, 20/10 steps per stage, CFG 7/5 and evaluator assets as epoch 3. "
        "Raw 512x320 vs QuickSR 1024x640 combines output-resolution and SR effects. "
        "Not an isolated loss ablation, few-step or mobile-latency evaluation. "
        + ("Generation smoke only; no full scores." if args.max_prompts else "Reuse parent scores; never regenerate the parent."))
    digest = bind_contract(args.output_dir, contract)
    for name in ("vbench_info.json", "checkpoint_audit.json"):
        value = read_json(asset_args.output_dir / name)
        if name == "checkpoint_audit.json":
            value["continuation_evaluation"] = contract["continuation_evaluation"]
        atomic_json(args.output_dir / name, value)
    samples = read_json(asset_args.output_dir / "sample_plan.json")["samples"]
    atomic_json(args.output_dir / "sample_plan.json", dict(contract_sha256=digest, samples=samples))
    return contract


def checked_contract(output):
    contract = read_json(output / "run_contract.json")
    policy = contract.get("continuation_evaluation", {})
    experiment = policy.get("experiment")
    if (experiment not in SYSTEMS or contract.get("system") != SYSTEMS[experiment]
            or not contract.get("training_complete") or contract.get("checkpoint", {}).get("step") != FINAL_STEP
            or policy.get("final_step") != FINAL_STEP):
        raise ValueError("Not a completed continuation benchmark contract")
    return contract, experiment


def verified_reference(contract):
    policy = contract["continuation_evaluation"]
    if not policy["reference_verified"]:
        raise ValueError("No full comparison for a generation smoke")
    reference, report, assets = previous.read_reference(Path(policy["reference_run"]))
    if (fingerprint(reference) != policy["reference_contract_sha256"]
            or fingerprint(report) != policy["reference_report_sha256"]):
        raise ValueError("Epoch-3 reference changed after preflight")
    return reference, report, assets


def verify_score_assets(output, contract):
    expected = verified_reference(contract)[2]
    actual = read_json(output / "score_assets.json")
    if sorted(row["sha256"] for row in actual) != sorted(row["sha256"] for row in expected):
        raise ValueError("VBench evaluation assets differ from epoch 3")


def comparison(output, contract):
    _, parent, _ = verified_reference(contract)
    report = read_json(output / "quicksr_ablation.json")
    experiment = contract["continuation_evaluation"]["experiment"]
    system = SYSTEMS[experiment]
    metrics = ("quality_score", "semantic_score", "total_score")
    lines = [f"# {experiment} vs Epoch-3 Parent", "", contract["scope"], "",
             "| Score (%) | Parent Raw | New Raw | Change (pp) | Parent QuickSR | New QuickSR | Change (pp) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    rows = {}
    for name in (*metrics, *base.DIMENSIONS):
        row, values = {}, []
        for variant in ("no_quicksr", "quicksr"):
            a = parent["scores"][variant][previous.PARENT_SYSTEM]
            b = report["scores"][variant][system]
            a, b = (a[name], b[name]) if name in metrics else (a["raw"][name], b["raw"][name])
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (a, b)):
                raise ValueError(f"Non-finite comparison score: {variant}/{name}")
            row[variant] = dict(parent=a, continuation=b, difference_pp=100 * (b - a))
            values.extend((f"{100*a:.2f}", f"{100*b:.2f}", f"{100*(b-a):+.2f}"))
        rows[name] = row
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    atomic_json(output / f"{experiment}_vs_epoch3.json", dict(status="complete", rows=rows,
                continuation_evaluation=contract["continuation_evaluation"]))
    (output / f"{experiment}_vs_epoch3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generation_worker(args, *, device=None):
    _, experiment = checked_contract(args.output_dir)
    with continuation_policy(experiment):
        base.generation_worker(args, device=device)


def summarize(args):
    contract, experiment = checked_contract(args.output_dir)
    if contract["max_prompts"] or contract["samples_per_prompt"] != 5:
        raise ValueError("No full score for a subset/single-sample smoke")
    verify_score_assets(args.output_dir, contract)
    with continuation_policy(experiment):
        base.summarize(args)
    comparison(args.output_dir, contract)


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--experiment", choices=tuple(SYSTEMS))
    parser.add_argument("--reference-run", type=Path, default=previous.REFERENCE)
    extra, rest = parser.parse_known_args(argv)
    if "--help" in rest or "-h" in rest:
        print(__doc__ + "\n" + parser.format_help())
    args = base.parse_args(rest)
    args.experiment, args.reference_run = extra.experiment, extra.reference_run.resolve()
    if args.allow_incomplete_checkpoint:
        parser.error("Both continuations require completed step378636 weights, including for smoke")
    if args.phase in ("run", "prepare"):
        if not args.experiment:
            parser.error("Specify --experiment job_b or uniform_history")
        if not any(value == "--checkpoint" or value.startswith("--checkpoint=") for value in rest):
            args.checkpoint = DEFAULT_JOB_B if args.experiment == "job_b" else find_history_checkpoint()
        if not any(value == "--output-dir" or value.startswith("--output-dir=") for value in rest):
            suffix = f"_smoke{args.max_prompts}" if args.max_prompts else ""
            args.output_dir = ROOT / f"output/vbench_mobileov_{SYSTEMS[args.experiment].removeprefix('stage2_')}_{args.samples_per_prompt}x_dual{suffix}"
    return args


def main():
    args = parse_args()
    if args.phase == "warm-scores":
        base.core.warm_scores(args)
        return
    if args.phase == "generation-worker":
        generation_worker(args)
        return
    if args.phase == "score-worker":
        _, experiment = checked_contract(args.output_dir)
        with continuation_policy(experiment):
            base.score_worker(args)
        return
    if args.phase == "summarize":
        summarize(args)
        return
    os.chdir(ROOT)
    with output_lock(args.output_dir):
        if args.phase == "run":
            devices = base.core.gpu_tokens(args.workers)
            if not Path(args.vbench_python).is_file():
                raise FileNotFoundError(args.vbench_python)
        contract = prepare(args)
        gc.collect()
        if args.phase == "prepare":
            return
        if not args.max_prompts:
            subprocess.run([args.vbench_python, str(Path(__file__).resolve()), "--phase", "warm-scores",
                            "--output-dir", str(args.output_dir), "--vbench-repo", str(args.vbench_repo)],
                           cwd=args.vbench_repo, check=True)
            verify_score_assets(args.output_dir, contract)
        previous_signal = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            commands = [[sys.executable, str(Path(__file__).resolve()), "--phase", "generation-worker",
                         "--output-dir", str(args.output_dir), "--rank", str(rank), "--workers", str(args.workers)]
                        for rank in range(args.workers)]
            base.core.run_processes(commands, devices, args.output_dir / "logs/generation")
            with continuation_policy(args.experiment):
                base.verify_generation(args.output_dir)
            if args.max_prompts:
                print("Continuation generation smoke passed; no full VBench score claimed.", flush=True)
                return
            commands = [[args.vbench_python, str(Path(__file__).resolve()), "--phase", "score-worker",
                         "--output-dir", str(args.output_dir), "--vbench-repo", str(args.vbench_repo),
                         "--rank", str(rank), "--workers", str(args.workers)] for rank in range(args.workers)]
            base.core.run_processes(commands, devices, args.output_dir / "logs/scoring", cwd=args.vbench_repo)
            subprocess.run([args.vbench_python, str(Path(__file__).resolve()), "--phase", "summarize",
                            "--output-dir", str(args.output_dir)], cwd=args.vbench_repo, check=True)
        finally:
            signal.signal(signal.SIGTERM, previous_signal)


if __name__ == "__main__":
    main()
