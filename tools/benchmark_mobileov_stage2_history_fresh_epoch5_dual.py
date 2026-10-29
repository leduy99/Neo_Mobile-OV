#!/usr/bin/env python
"""Completed fresh-history epoch 5, run17892009: text-only raw/QuickSR VBench."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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

import benchmark_mobileov_stage2_history_fresh_epochs_dual as history  # noqa: E402
import benchmark_mobileov_stage2_resume_epoch4_dual as clean  # noqa: E402
from vbench_joint_native_protocol import (  # noqa: E402
    atomic_json, bind_contract, fingerprint, output_lock, read_json, sha256,
)

base = history.base
SYSTEM, FINAL_STEP = "stage2_history_fresh_epoch5", 473295
DEFAULT_CHECKPOINT = history.TRAINING_ROOT / "stage2_dit_connector_step473295.pt"
DEFAULT_OUTPUT = ROOT / "output/vbench_mobileov_stage2_history_fresh_epoch5_step473295_5x_dual"
IMPLEMENTATION = tuple(dict.fromkeys((*history.IMPLEMENTATION, *clean.IMPLEMENTATION,
    "tools/benchmark_mobileov_stage2_history_fresh_epoch5_dual.py")))


def check_checkpoint(payload, *, allow_incomplete=False):
    from new_mobile_ov.training.stage2_history import MAX_HISTORY_SIGMA, history_contract

    training = payload.get("contract", {})
    if (training.get("scope") != "fresh_multimodal_stage2_with_history_corruption; public_data; not_exact_UniVideo_recipe"
            or training.get("init_step") != 15000
            or training.get("history_corruption") != history_contract(MAX_HISTORY_SIGMA)
            or training.get("video_units") != "T2V/reference:uniform0_to6; anchor:uniform1_to6; teacher_forced_corrupted_history"):
        raise ValueError("Expected fresh-history run17892009 epoch 5, not clean resume or a weights-only fork")
    # Validate the common five-epoch recipe only after verifying the actual history policy.
    # The real step and weights are unchanged; the saved training contract is never rewritten.
    shared = dict(payload, contract=history.clean_projection(training))
    with clean.evaluation_epoch(5):
        return clean.check_checkpoint(shared, allow_incomplete=allow_incomplete)


@contextmanager
def epoch5_policy():
    original_check, original_system = base.check_media_checkpoint, base.SYSTEM
    base.check_media_checkpoint, base.SYSTEM = check_checkpoint, SYSTEM
    try:
        yield
    finally:
        base.check_media_checkpoint, base.SYSTEM = original_check, original_system


def prepare(args):
    if args.output_dir.resolve() == args.reference_run.resolve():
        raise ValueError("Use a separate fresh-history epoch-5 output; preserve the clean epoch-3 reference")
    reference = None if args.max_prompts else history.previous.read_reference(args.reference_run)
    asset_args = copy.copy(args)
    asset_args.output_dir = args.output_dir / "_asset_preflight"
    with epoch5_policy():
        contract = base.prepare(asset_args)
    helpers = history.compare_protocol(reference[0], contract) if reference else None
    contract["history_fresh_evaluation"] = dict(
        kind="fresh_stage2_with_history_corruption_from_initialization", training_run=history.TRAINING_RUN,
        epoch=5, expected_step=FINAL_STEP, planned_final_step=FINAL_STEP,
        reference_kind="independent_clean_history_epoch3; not_a_parent_or_same_epoch5_control",
        reference_run=str(args.reference_run), reference_verified=reference is not None,
        reference_contract_sha256=fingerprint(reference[0]) if reference else None,
        reference_report_sha256=fingerprint(reference[1]) if reference else None,
        imported_inference_helpers=helpers, history_corruption_at_inference=False,
    )
    contract["implementation_sha256"].update({name: sha256(ROOT / name) for name in IMPLEMENTATION})
    contract["scope"] = (
        "Fresh-history run17892009, completed epoch 5, exact step473295. "
        "Stage 2 initialized from alignment15k connector and released multistep DiT, NOT the trained epoch-3 model. "
        "History corruption was enabled during training from initialization; it is OFF during inference. "
        "Frozen SmolVLM2 + trained MCP/monolithic DiT; every frame generated from text, including frame zero. "
        "No SSD1B/DreamLite anchor, vision reference, LoRA or Hybrid inference. "
        "Matched prompts/seeds, 20/10 steps per stage, CFG7/5, BF16 DiT and evaluator assets. "
        "Raw 512x320 and QuickSR 1024x640 share generated frames; this combines resolution and SR effects. "
        "Reference is independently trained clean epoch 3, not a training parent or a same-epoch control. "
        "This comparison changes exposure and history corruption; it is not an isolated corruption ablation. "
        + ("Generation smoke only; no full scores or reference comparison." if args.max_prompts else
           "Reuse the completed clean epoch-3 report without regenerating or overwriting it."))
    digest = bind_contract(args.output_dir, contract)
    for name in ("vbench_info.json", "checkpoint_audit.json"):
        value = read_json(asset_args.output_dir / name)
        if name == "checkpoint_audit.json":
            value["history_fresh_evaluation"] = contract["history_fresh_evaluation"]
        atomic_json(args.output_dir / name, value)
    samples = read_json(asset_args.output_dir / "sample_plan.json")["samples"]
    atomic_json(args.output_dir / "sample_plan.json", dict(contract_sha256=digest, samples=samples))
    return contract


def checked_contract(output):
    contract = read_json(output / "run_contract.json")
    checkpoint, policy = contract.get("checkpoint", {}), contract.get("history_fresh_evaluation", {})
    if (contract.get("system") != SYSTEM or contract.get("training_complete") is not True
            or checkpoint.get("step") != FINAL_STEP or checkpoint.get("final_step") != FINAL_STEP
            or policy.get("epoch") != 5 or policy.get("expected_step") != FINAL_STEP
            or policy.get("training_run") != history.TRAINING_RUN
            or policy.get("history_corruption_at_inference") is not False):
        raise ValueError("Not the completed fresh-history epoch-5 benchmark from run17892009")
    return contract


def generation_worker(args, *, device=None):
    checked_contract(args.output_dir)
    with epoch5_policy():
        base.generation_worker(args, device=device)


def comparison(output, contract, scores):
    _, reference, _ = history.verified_reference(contract)
    metrics, rows = ("quality_score", "semantic_score", "total_score"), {}
    lines = ["# Fresh-History Epoch 5 vs Original Clean Epoch 3", "", contract["scope"], "",
             "| Score (%) | Clean Ep3 Raw | History Ep5 Raw | Change (pp) | Clean Ep3 QuickSR | History Ep5 QuickSR | Change (pp) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in (*metrics, *base.DIMENSIONS):
        row, values = {}, []
        for variant in ("no_quicksr", "quicksr"):
            a, b = reference["scores"][variant][history.previous.PARENT_SYSTEM], scores[variant][SYSTEM]
            a, b = (a[name], b[name]) if name in metrics else (a["raw"][name], b["raw"][name])
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (a, b)):
                raise ValueError(f"Non-finite comparison score: {variant}/{name}")
            row[variant] = dict(clean_epoch3=a, history_fresh=b, difference_pp=100 * (b - a))
            values.extend((f"{100*a:.2f}", f"{100*b:.2f}", f"{100*(b-a):+.2f}"))
        rows[name] = row
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    atomic_json(output / "history_fresh_vs_clean_epoch3.json", dict(status="complete", rows=rows,
                history_fresh_evaluation=contract["history_fresh_evaluation"]))
    (output / "history_fresh_vs_clean_epoch3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(args):
    from evaluate_vbench_resume import tabulate

    contract = checked_contract(args.output_dir)
    if contract["max_prompts"] or contract["samples_per_prompt"] != 5:
        raise ValueError("No full score for a subset/single-sample smoke")
    history.verify_score_assets(args.output_dir, contract)
    with epoch5_policy():
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
    (args.output_dir / "quicksr_ablation.md").write_text("# Fresh-History Stage 2: Completed Epoch 5\n\n" + markdown, encoding="utf-8")
    comparison(args.output_dir, contract, scores)
    print(f"Completed fresh-history epoch-5 VBench: {args.output_dir}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reference-run", type=Path, default=history.previous.REFERENCE)
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
        with epoch5_policy():
            base.score_worker(args)
        return
    if args.phase == "summarize":
        summarize(args)
        return
    os.chdir(ROOT)
    entry = str(Path(__file__).resolve())
    with output_lock(args.output_dir):
        if args.phase == "run":
            devices = base.core.gpu_tokens(args.workers)
            if not Path(args.vbench_python).is_file():
                raise FileNotFoundError(args.vbench_python)
        contract = prepare(args)
        if args.phase == "prepare":
            return
        if not args.max_prompts:
            subprocess.run([args.vbench_python, entry, "--phase", "warm-scores",
                            "--output-dir", str(args.output_dir), "--vbench-repo", str(args.vbench_repo)],
                           cwd=args.vbench_repo, check=True)
            history.verify_score_assets(args.output_dir, contract)
        previous_handler = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            commands = [[sys.executable, entry, "--phase", "generation-worker",
                         "--output-dir", str(args.output_dir), "--rank", str(rank), "--workers", str(args.workers)]
                        for rank in range(args.workers)]
            base.core.run_processes(commands, devices, args.output_dir / "logs/generation")
            with epoch5_policy():
                base.verify_generation(args.output_dir)
            if args.max_prompts:
                print("Fresh-history epoch-5 generation smoke passed; no full scores claimed.", flush=True)
                return
            commands = [[args.vbench_python, entry, "--phase", "score-worker",
                         "--output-dir", str(args.output_dir), "--vbench-repo", str(args.vbench_repo),
                         "--rank", str(rank), "--workers", str(args.workers)] for rank in range(args.workers)]
            base.core.run_processes(commands, devices, args.output_dir / "logs/scoring", cwd=args.vbench_repo)
            subprocess.run([args.vbench_python, entry, "--phase", "summarize",
                            "--output-dir", str(args.output_dir)], cwd=args.vbench_repo, check=True)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
