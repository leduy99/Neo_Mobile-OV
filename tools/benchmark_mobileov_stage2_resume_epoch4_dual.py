#!/usr/bin/env python
"""Pinned epoch-4 VBench for the unchanged Adam-resumed five-epoch Stage-2 run."""
from __future__ import annotations

from contextlib import contextmanager
import argparse
import copy
import math
import os
from pathlib import Path
import shutil
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
SYSTEM = "stage2_resume_epoch4"
PARENT_STEP, EPOCH_STEP, TARGET_STEP = 283977, 378636, 473295
DEFAULT_CHECKPOINT = ROOT / "output/mobileov_stage2_resume5epochs16gpu/17892008/stage2_dit_connector_step378636.pt"
DEFAULT_OUTPUT = ROOT / "output/vbench_mobileov_stage2_resume_epoch4_step378636_5x_dual"
IMPLEMENTATION = ("tools/benchmark_mobileov_stage2_resume_epoch4_dual.py",
                  "tools/benchmark_mobileov_job_a_dual.py", "tools/benchmark_mobileov_stage2_epoch_dual.py",
                  "new_mobile_ov/training/stage2_epoch_extension.py",
                  "new_mobile_ov/training/stage2_media_epochs.py")


def check_checkpoint(payload, *, allow_incomplete=False):
    # Only the generation environment has training/data dependencies such as PyAV.
    from new_mobile_ov.training.stage2_epoch_extension import validate_extension
    from new_mobile_ov.training.stage2_media_epochs import MediaEpochPlan

    base.stage2.check_payload(payload, EPOCH_STEP)
    training = payload["contract"]
    plan = MediaEpochPlan(984636, 529902, 16, 1, epochs=5, allow_partial_cycle=True, rng_world_size=8)
    if (training.get("format") != base.stage2.FORMAT or training.get("media_epochs") != plan.contract()
            or training.get("steps") != TARGET_STEP or training.get("world_size") != 16
            or training.get("accumulation") != 1 or "optimizer" in payload or "smolvlm2_lora" in payload):
        raise ValueError("Expected the archived epoch-4 inference weights of the 16-GPU five-epoch resume, not latest/resume/Job A/B/history")
    # Admission checks the canonical old layout; full evaluation also checks the actual parent below.
    original = dict(training, steps=PARENT_STEP, world_size=8, accumulation=2,
                    media_epochs=MediaEpochPlan(984636, 529902, 8, 2).contract())
    validate_extension(original, training, PARENT_STEP, repartition=True)
    return TARGET_STEP


@contextmanager
def epoch4_policy():
    # Keep numerical inference/export unchanged; admit only this exact archived epoch.
    original_check, original_system = base.check_media_checkpoint, base.SYSTEM
    base.check_media_checkpoint, base.SYSTEM = check_checkpoint, SYSTEM
    try:
        yield
    finally:
        base.check_media_checkpoint, base.SYSTEM = original_check, original_system


def compare_protocol(reference, current):
    from new_mobile_ov.training.stage2_epoch_extension import validate_extension

    validate_extension(reference["training_contract"], current["training_contract"],
                       PARENT_STEP, repartition=True)
    # Reuse the existing settings/asset/AST audit without adding a fictitious parent to the checkpoint.
    view = copy.copy(current)
    view["training_contract"] = dict(current["training_contract"], parent_contract=reference["training_contract"])
    return previous.compare_protocol(reference, view)


def prepare(args):
    if args.output_dir.resolve() == args.reference_run.resolve():
        raise ValueError("Use a separate epoch-4 output; preserve the epoch-3 reference")
    reference_data = None if args.max_prompts else previous.read_reference(args.reference_run)
    asset_args = copy.copy(args)
    asset_args.output_dir = args.output_dir / "_asset_preflight"
    with epoch4_policy():
        contract = base.prepare(asset_args)
    helpers = compare_protocol(reference_data[0], contract) if reference_data else None
    contract["epoch4_evaluation"] = dict(
        kind="completed_epoch4_of_ongoing_five_epoch_resume", epoch=4, expected_step=EPOCH_STEP,
        planned_final_step=TARGET_STEP, parent_step=PARENT_STEP, additional_updates=EPOCH_STEP - PARENT_STEP,
        reference_run=str(args.reference_run), reference_verified=reference_data is not None,
        reference_contract_sha256=fingerprint(reference_data[0]) if reference_data else None,
        reference_report_sha256=fingerprint(reference_data[1]) if reference_data else None,
        imported_inference_helpers=helpers,
    )
    contract["implementation_sha256"].update({name: sha256(ROOT / name) for name in IMPLEMENTATION})
    contract["scope"] = (
        "Exact completed media epoch 4, step378636, from the unchanged-recipe Adam continuation of epoch 3. "
        "Planned final step473295 is still incomplete; this does not prevent evaluating the archived epoch. "
        "Frozen SmolVLM2 + trained MCP/monolithic DiT; every frame generated from text, including frame zero. "
        "No SSD1B/DreamLite anchor, vision reference, LoRA, history corruption or Hybrid inference. "
        "Same parent prompts/seeds, 20/10 steps per stage, CFG7/5, BF16 DiT and evaluator weights. "
        "Raw 512x320 and QuickSR 1024x640 share generated frames; this combines resolution and SR effects. "
        + ("Generation smoke only; no full scores." if args.max_prompts else
           "Reuse the existing epoch-3 benchmark; do not regenerate or overwrite it."))
    digest = bind_contract(args.output_dir, contract)
    for name in ("vbench_info.json", "checkpoint_audit.json"):
        value = read_json(asset_args.output_dir / name)
        if name == "checkpoint_audit.json":
            value["epoch4_evaluation"] = contract["epoch4_evaluation"]
        atomic_json(args.output_dir / name, value)
    samples = read_json(asset_args.output_dir / "sample_plan.json")["samples"]
    atomic_json(args.output_dir / "sample_plan.json", dict(contract_sha256=digest, samples=samples))
    return contract


def checked_contract(output):
    contract = read_json(output / "run_contract.json")
    checkpoint, policy = contract.get("checkpoint", {}), contract.get("epoch4_evaluation", {})
    if (contract.get("system") != SYSTEM or contract.get("training_complete") is not False
            or checkpoint.get("step") != EPOCH_STEP or checkpoint.get("final_step") != TARGET_STEP
            or policy.get("epoch") != 4 or policy.get("expected_step") != EPOCH_STEP):
        raise ValueError("Not the pinned epoch-4 resume benchmark")
    return contract


def verified_reference(contract):
    policy = contract["epoch4_evaluation"]
    if not policy["reference_verified"]:
        raise ValueError("No full reference comparison for a generation smoke")
    reference, report, assets = previous.read_reference(Path(policy["reference_run"]))
    if (fingerprint(reference) != policy["reference_contract_sha256"]
            or fingerprint(report) != policy["reference_report_sha256"]):
        raise ValueError("Epoch-3 reference changed after preflight")
    return reference, report, assets


def verify_score_assets(output, contract):
    expected = verified_reference(contract)[2]
    actual = read_json(output / "score_assets.json")
    if sorted(row["sha256"] for row in actual) != sorted(row["sha256"] for row in expected):
        raise ValueError("VBench evaluation weights differ from epoch 3")


def generation_worker(args, *, device=None):
    checked_contract(args.output_dir)
    with epoch4_policy():
        base.generation_worker(args, device=device)


def comparison(output, contract, scores):
    _, parent, _ = verified_reference(contract)
    metrics = ("quality_score", "semantic_score", "total_score")
    lines = ["# Epoch 4 vs Epoch 3: Unchanged Training Recipe", "", contract["scope"], "",
             "| Score (%) | Epoch 3 Raw | Epoch 4 Raw | Change (pp) | Epoch 3 QuickSR | Epoch 4 QuickSR | Change (pp) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    rows = {}
    for name in (*metrics, *base.DIMENSIONS):
        row, values = {}, []
        for variant in ("no_quicksr", "quicksr"):
            a, b = parent["scores"][variant][previous.PARENT_SYSTEM], scores[variant][SYSTEM]
            a, b = (a[name], b[name]) if name in metrics else (a["raw"][name], b["raw"][name])
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (a, b)):
                raise ValueError(f"Non-finite comparison score: {variant}/{name}")
            row[variant] = dict(epoch3=a, epoch4=b, difference_pp=100 * (b - a))
            values.extend((f"{100*a:.2f}", f"{100*b:.2f}", f"{100*(b-a):+.2f}"))
        rows[name] = row
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    atomic_json(output / "epoch4_vs_epoch3.json", dict(status="complete", rows=rows,
                epoch4_evaluation=contract["epoch4_evaluation"]))
    (output / "epoch4_vs_epoch3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(args):
    from evaluate_vbench_resume import tabulate

    contract = checked_contract(args.output_dir)
    if contract["max_prompts"] or contract["samples_per_prompt"] != 5:
        raise ValueError("No full score for a subset/single-sample smoke")
    verify_score_assets(args.output_dir, contract)
    with epoch4_policy():
        base.verify_generation(args.output_dir)
    scores = {}
    for variant in base.VARIANTS:
        merged = args.output_dir / variant / "scores/vbench"
        merged.mkdir(parents=True, exist_ok=True)
        for dimension in base.DIMENSIONS:
            source = args.output_dir / variant / "scores" / dimension / "vbench" / f"{dimension}_eval_results.json"
            if dimension not in read_json(source):
                raise ValueError(f"Missing score: {source}")
            shutil.copyfile(source, merged / source.name)
        scores[variant] = {SYSTEM: tabulate(merged)}
        subprocess.run([sys.executable, str(ROOT / "tools/report_vbench_low_prompts.py"), "--score-dir", str(merged),
                        "--output-prefix", str(args.output_dir / variant / "low_prompts")], check=True)
    report, markdown = base.dual.ablation_report(scores, {"selected_systems": [SYSTEM]})
    report.update(protocol=contract, evaluation_assets=read_json(args.output_dir / "score_assets.json"), scope=contract["scope"])
    atomic_json(args.output_dir / "quicksr_ablation.json", report)
    markdown = markdown.replace("Identical generated RGB frames and anchors;", "Identical generated RGB frames; no external anchor;")
    (args.output_dir / "quicksr_ablation.md").write_text("# Stage-2 Resume: Exact Epoch 4\n\n" + markdown, encoding="utf-8")
    comparison(args.output_dir, contract, scores)
    print(f"Completed epoch-4 VBench: {args.output_dir}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reference-run", type=Path, default=previous.REFERENCE)
    extra, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if "--help" in rest or "-h" in rest:
        print(__doc__ + "\n" + parser.format_help())
    args = base.parse_args(rest)
    args.reference_run = extra.reference_run.resolve()
    if args.phase in ("run", "prepare"):
        if not any(item == "--checkpoint" or item.startswith("--checkpoint=") for item in rest):
            args.checkpoint = DEFAULT_CHECKPOINT
        if not any(item == "--output-dir" or item.startswith("--output-dir=") for item in rest):
            suffix = f"_{args.samples_per_prompt}x_dual_smoke{args.max_prompts}" if args.max_prompts else "_5x_dual"
            args.output_dir = DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.name.removesuffix("_5x_dual") + suffix)
    return args


def main():
    args = parse_args()
    if args.phase == "generation-worker":
        generation_worker(args)
        return
    if args.phase == "warm-scores":
        base.core.warm_scores(args)
        return
    if args.phase == "score-worker":
        checked_contract(args.output_dir)
        with epoch4_policy():
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
        if args.phase == "prepare":
            return
        if not args.max_prompts:
            subprocess.run([args.vbench_python, str(Path(__file__).resolve()), "--phase", "warm-scores",
                            "--output-dir", str(args.output_dir), "--vbench-repo", str(args.vbench_repo)],
                           cwd=args.vbench_repo, check=True)
            verify_score_assets(args.output_dir, contract)
        previous_handler = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            commands = [[sys.executable, str(Path(__file__).resolve()), "--phase", "generation-worker",
                         "--output-dir", str(args.output_dir), "--rank", str(rank), "--workers", str(args.workers)]
                        for rank in range(args.workers)]
            base.core.run_processes(commands, devices, args.output_dir / "logs/generation")
            with epoch4_policy():
                base.verify_generation(args.output_dir)
            if args.max_prompts:
                print("Epoch-4 generation smoke passed; no full scores claimed.", flush=True)
                return
            commands = [[args.vbench_python, str(Path(__file__).resolve()), "--phase", "score-worker",
                         "--output-dir", str(args.output_dir), "--vbench-repo", str(args.vbench_repo),
                         "--rank", str(rank), "--workers", str(args.workers)] for rank in range(args.workers)]
            base.core.run_processes(commands, devices, args.output_dir / "logs/scoring", cwd=args.vbench_repo)
            subprocess.run([args.vbench_python, str(Path(__file__).resolve()), "--phase", "summarize",
                            "--output-dir", str(args.output_dir)], cwd=args.vbench_repo, check=True)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
