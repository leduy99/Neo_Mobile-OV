#!/usr/bin/env python
"""Fresh-history Stage-2 epochs 3/4: matched text-only VBench, raw and QuickSR."""
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

import benchmark_mobileov_job_a_dual as previous  # noqa: E402
from vbench_joint_native_protocol import (  # noqa: E402
    atomic_json, bind_contract, fingerprint, output_lock, read_json, sha256,
)

base = previous.base
TRAINING_RUN = "17892009"
TRAINING_ROOT = ROOT / f"output/mobileov_stage2_history_fresh5epochs16gpu/{TRAINING_RUN}"
EPOCH_STEPS = {3: 283977, 4: 378636}
TARGET_STEP = 473295
IMPLEMENTATION = (
    "tools/benchmark_mobileov_stage2_history_fresh_epochs_dual.py",
    "tools/benchmark_mobileov_job_a_dual.py", "tools/benchmark_mobileov_stage2_epoch_dual.py",
    "new_mobile_ov/training/stage2_epoch_extension.py", "new_mobile_ov/training/stage2_media_epochs.py",
    "new_mobile_ov/training/stage2_history.py",
)


def system(epoch):
    if epoch not in EPOCH_STEPS:
        raise ValueError("Fresh-history evaluation requires epoch 3 or 4")
    return f"stage2_history_fresh_epoch{epoch}"


def epoch_paths(epoch):
    system(epoch)
    step = EPOCH_STEPS[epoch]
    return (TRAINING_ROOT / f"stage2_dit_connector_step{step:06d}.pt",
            ROOT / f"output/vbench_mobileov_stage2_history_fresh_epoch{epoch}_step{step}_5x_dual")


def clean_projection(training):
    # Compare shared recipe fields without claiming this independently trained run has a parent.
    result = dict(training)
    result.pop("history_corruption", None)
    result.update(scope="fresh_multimodal_stage2; public_data; not_exact_UniVideo_recipe",
                  video_units="T2V/reference:0_to6; anchor:1_to6; teacher_forced_causal_history")
    return result


def check_checkpoint(payload, epoch, *, allow_incomplete=False):
    # Scoring uses a separate environment: keep all training/data imports generation-only.
    from new_mobile_ov.training.stage2_epoch_extension import validate_extension
    from new_mobile_ov.training.stage2_history import MAX_HISTORY_SIGMA, history_contract
    from new_mobile_ov.training.stage2_media_epochs import MediaEpochPlan

    system(epoch)
    base.stage2.check_payload(payload, EPOCH_STEPS[epoch])
    training = payload["contract"]
    plan = MediaEpochPlan(984636, 529902, 16, 1, epochs=5, allow_partial_cycle=True, rng_world_size=8)
    if (training.get("format") != base.stage2.FORMAT
            or training.get("scope") != "fresh_multimodal_stage2_with_history_corruption; public_data; not_exact_UniVideo_recipe"
            or training.get("init_step") != 15000
            or training.get("history_corruption") != history_contract(MAX_HISTORY_SIGMA)
            or training.get("video_units") != "T2V/reference:uniform0_to6; anchor:uniform1_to6; teacher_forced_corrupted_history"
            or training.get("media_epochs") != plan.contract() or training.get("steps") != TARGET_STEP
            or training.get("world_size") != 16 or training.get("accumulation") != 1
            or any(key in payload for key in ("optimizer", "smolvlm2_lora"))):
        raise ValueError("Expected the pinned fresh-history five-epoch archive from run17892009; "
                         "not clean resume, Job A/B, or the old one-epoch uniform-history fork")
    clean = clean_projection(training)
    original = dict(clean, steps=previous.PARENT_STEP, world_size=8, accumulation=2,
                    media_epochs=MediaEpochPlan(984636, 529902, 8, 2).contract())
    validate_extension(original, clean, previous.PARENT_STEP, repartition=True)
    return TARGET_STEP


@contextmanager
def history_policy(epoch):
    original_check, original_system = base.check_media_checkpoint, base.SYSTEM
    base.check_media_checkpoint = lambda payload, **kwargs: check_checkpoint(payload, epoch, **kwargs)
    base.SYSTEM = system(epoch)
    try:
        yield
    finally:
        base.check_media_checkpoint, base.SYSTEM = original_check, original_system


def compare_protocol(reference, current):
    from new_mobile_ov.training.stage2_epoch_extension import validate_extension

    validate_extension(reference["training_contract"], clean_projection(current["training_contract"]),
                       previous.PARENT_STEP, repartition=True)
    view = copy.copy(current)
    view["training_contract"] = dict(current["training_contract"], parent_contract=reference["training_contract"])
    return previous.compare_protocol(reference, view)


def prepare(args):
    if args.output_dir.resolve() == args.reference_run.resolve():
        raise ValueError("Use a separate fresh-history output; preserve the clean epoch-3 reference")
    reference_data = None if args.max_prompts else previous.read_reference(args.reference_run)
    asset_args = copy.copy(args)
    asset_args.output_dir = args.output_dir / "_asset_preflight"
    with history_policy(args.epoch):
        contract = base.prepare(asset_args)
    helpers = compare_protocol(reference_data[0], contract) if reference_data else None
    contract["history_fresh_evaluation"] = dict(
        kind="fresh_stage2_with_history_corruption_from_initialization", training_run=TRAINING_RUN,
        epoch=args.epoch, expected_step=EPOCH_STEPS[args.epoch], planned_final_step=TARGET_STEP,
        reference_kind="independent_clean_history_epoch3; not_a_parent_or_same_epoch4_control",
        reference_run=str(args.reference_run), reference_verified=reference_data is not None,
        reference_contract_sha256=fingerprint(reference_data[0]) if reference_data else None,
        reference_report_sha256=fingerprint(reference_data[1]) if reference_data else None,
        imported_inference_helpers=helpers, history_corruption_at_inference=False,
    )
    contract["implementation_sha256"].update({name: sha256(ROOT / name) for name in IMPLEMENTATION})
    contract["scope"] = (
        f"Fresh-history run17892009, exact epoch {args.epoch}, step{EPOCH_STEPS[args.epoch]}. "
        "Stage 2 initialized from alignment15k connector and released multistep DiT, NOT the trained epoch-3 model. "
        "History corruption was enabled during training from initialization; it is OFF during inference. "
        "This archive is intermediate within a five-epoch budget; it is eligible for checkpoint evaluation. "
        "Frozen SmolVLM2 + trained MCP/monolithic DiT; every frame generated from text, including frame zero. "
        "No SSD1B/DreamLite anchor, vision reference, LoRA or Hybrid inference. "
        "944 prompts x 5 samples, matched seeds, 20/10 steps per stage, CFG7/5 and BF16 DiT. "
        "Raw 512x320 and QuickSR 1024x640 share generated frames; this combines resolution and SR effects. "
        "Reference is the independently trained clean-history epoch-3 run, not a training parent. "
        "Epoch 4 vs clean epoch 3 also changes training exposure, not only history corruption. "
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


def checked_contract(output, epoch):
    contract = read_json(output / "run_contract.json")
    checkpoint, policy = contract.get("checkpoint", {}), contract.get("history_fresh_evaluation", {})
    if (contract.get("system") != system(epoch) or contract.get("training_complete") is not False
            or checkpoint.get("step") != EPOCH_STEPS[epoch] or checkpoint.get("final_step") != TARGET_STEP
            or policy.get("epoch") != epoch or policy.get("expected_step") != EPOCH_STEPS[epoch]
            or policy.get("training_run") != TRAINING_RUN
            or policy.get("history_corruption_at_inference") is not False):
        raise ValueError(f"Not the pinned fresh-history epoch-{epoch} benchmark from run17892009")
    return contract


def verified_reference(contract):
    policy = contract["history_fresh_evaluation"]
    if not policy["reference_verified"]:
        raise ValueError("No full reference comparison for a generation smoke")
    reference, report, assets = previous.read_reference(Path(policy["reference_run"]))
    if (fingerprint(reference) != policy["reference_contract_sha256"]
            or fingerprint(report) != policy["reference_report_sha256"]):
        raise ValueError("Clean epoch-3 reference changed after preflight")
    return reference, report, assets


def verify_score_assets(output, contract):
    expected = verified_reference(contract)[2]
    actual = read_json(output / "score_assets.json")
    if sorted(row["sha256"] for row in actual) != sorted(row["sha256"] for row in expected):
        raise ValueError("VBench evaluation weights differ from the clean epoch-3 reference")


def generation_worker(args, *, device=None):
    checked_contract(args.output_dir, args.epoch)
    with history_policy(args.epoch):
        base.generation_worker(args, device=device)


def comparison(output, contract, scores):
    _, reference, _ = verified_reference(contract)
    epoch = contract["history_fresh_evaluation"]["epoch"]
    metrics, rows = ("quality_score", "semantic_score", "total_score"), {}
    lines = [f"# Fresh-History Epoch {epoch} vs Original Clean Epoch 3", "", contract["scope"], "",
             f"| Score (%) | Clean Ep3 Raw | History Ep{epoch} Raw | Change (pp) | Clean Ep3 QuickSR | History Ep{epoch} QuickSR | Change (pp) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in (*metrics, *base.DIMENSIONS):
        row, values = {}, []
        for variant in ("no_quicksr", "quicksr"):
            a, b = reference["scores"][variant][previous.PARENT_SYSTEM], scores[variant][system(epoch)]
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

    contract = checked_contract(args.output_dir, args.epoch)
    if contract["max_prompts"] or contract["samples_per_prompt"] != 5:
        raise ValueError("No full score for a subset/single-sample smoke")
    verify_score_assets(args.output_dir, contract)
    with history_policy(args.epoch):
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
        scores[variant] = {system(args.epoch): tabulate(merged)}
        subprocess.run([sys.executable, str(ROOT / "tools/report_vbench_low_prompts.py"), "--score-dir", str(merged),
                        "--output-prefix", str(args.output_dir / variant / "low_prompts")], check=True)
    report, markdown = base.dual.ablation_report(scores, {"selected_systems": [system(args.epoch)]})
    report.update(protocol=contract, evaluation_assets=read_json(args.output_dir / "score_assets.json"), scope=contract["scope"])
    atomic_json(args.output_dir / "quicksr_ablation.json", report)
    markdown = markdown.replace("Identical generated RGB frames and anchors;", "Identical generated RGB frames; no external anchor;")
    (args.output_dir / "quicksr_ablation.md").write_text(f"# Fresh-History Stage 2: Epoch {args.epoch}\n\n" + markdown, encoding="utf-8")
    comparison(args.output_dir, contract, scores)
    print(f"Completed fresh-history epoch-{args.epoch} VBench: {args.output_dir}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--epoch", type=int, choices=(3, 4), required=True)
    parser.add_argument("--reference-run", type=Path, default=previous.REFERENCE)
    extra, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if "--help" in rest or "-h" in rest:
        print(__doc__ + "\n" + parser.format_help())
    args = base.parse_args(rest)
    args.epoch, args.reference_run = extra.epoch, extra.reference_run.resolve()
    if args.phase in ("run", "prepare"):
        checkpoint, output = epoch_paths(args.epoch)
        if not any(item == "--checkpoint" or item.startswith("--checkpoint=") for item in rest):
            args.checkpoint = checkpoint
        if not any(item == "--output-dir" or item.startswith("--output-dir=") for item in rest):
            suffix = f"_{args.samples_per_prompt}x_dual_smoke{args.max_prompts}" if args.max_prompts else "_5x_dual"
            args.output_dir = output.with_name(output.name.removesuffix("_5x_dual") + suffix)
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
        checked_contract(args.output_dir, args.epoch)
        with history_policy(args.epoch):
            base.score_worker(args)
        return
    if args.phase == "summarize":
        summarize(args)
        return
    os.chdir(ROOT)
    entry = [str(Path(__file__).resolve()), "--epoch", str(args.epoch)]
    with output_lock(args.output_dir):
        if args.phase == "run":
            devices = base.core.gpu_tokens(args.workers)
            if not Path(args.vbench_python).is_file():
                raise FileNotFoundError(args.vbench_python)
        contract = prepare(args)
        if args.phase == "prepare":
            return
        if not args.max_prompts:
            subprocess.run([args.vbench_python, *entry, "--phase", "warm-scores",
                            "--output-dir", str(args.output_dir), "--vbench-repo", str(args.vbench_repo)],
                           cwd=args.vbench_repo, check=True)
            verify_score_assets(args.output_dir, contract)
        previous_handler = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            commands = [[sys.executable, *entry, "--phase", "generation-worker",
                         "--output-dir", str(args.output_dir), "--rank", str(rank), "--workers", str(args.workers)]
                        for rank in range(args.workers)]
            base.core.run_processes(commands, devices, args.output_dir / "logs/generation")
            with history_policy(args.epoch):
                base.verify_generation(args.output_dir)
            if args.max_prompts:
                print("Fresh-history generation smoke passed; no full scores claimed.", flush=True)
                return
            commands = [[args.vbench_python, *entry, "--phase", "score-worker",
                         "--output-dir", str(args.output_dir), "--vbench-repo", str(args.vbench_repo),
                         "--rank", str(rank), "--workers", str(args.workers)] for rank in range(args.workers)]
            base.core.run_processes(commands, devices, args.output_dir / "logs/scoring", cwd=args.vbench_repo)
            subprocess.run([args.vbench_python, *entry, "--phase", "summarize",
                            "--output-dir", str(args.output_dir)], cwd=args.vbench_repo, check=True)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
