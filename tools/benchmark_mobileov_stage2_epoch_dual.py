#!/usr/bin/env python
"""Evaluate a pinned intermediate checkpoint against the completed three-epoch run."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import benchmark_mobileov_stage2_media3epochs_dual as base  # noqa: E402
from vbench_joint_native_protocol import (  # noqa: E402
    DIMENSIONS, atomic_json, bind_contract, fingerprint, output_lock, read_json, sha256,
)

REFERENCE = ROOT / "output/vbench_mobileov_stage2_media3epochs_5x_dual"


def validate_checkpoint_policy(payload, expected_step, epoch, training_sha=None):
    if payload["step"] != expected_step:
        raise ValueError(f"Wrong checkpoint: expected step={expected_step}, got {payload['step']}")
    plan = payload["contract"]["media_epochs"]
    boundary = epoch * plan["steps_per_epoch"]
    if epoch not in (1, 2) or not boundary <= expected_step < boundary + plan["steps_per_epoch"]:
        raise ValueError(f"Step {expected_step} is not after epoch {epoch} and before the next epoch")
    if training_sha and fingerprint(payload["contract"]) != training_sha:
        raise ValueError("Checkpoint belongs to a different training contract/data/initialization")


@contextmanager
def checkpoint_policy(expected_step, epoch, training_sha=None):
    # Change admission only; retain the tested numerical inference/export implementation.
    original = base.read_checkpoint

    def read(path, **kwargs):
        kwargs["allow_incomplete"] = True
        payload, digest, target = original(path, **kwargs)
        validate_checkpoint_policy(payload, expected_step, epoch, training_sha)
        return payload, digest, target

    base.read_checkpoint = read
    try:
        yield
    finally:
        base.read_checkpoint = original


def read_reference(folder):
    for name in ("run_contract.json", "quicksr_ablation.json", "score_assets.json"):
        if not (folder / name).is_file():
            raise FileNotFoundError(f"Missing completed epoch-3 reference: {folder / name}. Do not rerun or replace it silently.")
    contract = read_json(folder / "run_contract.json")
    report = read_json(folder / "quicksr_ablation.json")
    assets = read_json(folder / "score_assets.json")
    if (contract.get("format") != "mobileov_stage2_media3epochs_vbench_dual_v1"
            or not contract.get("training_complete") or contract.get("max_prompts")
            or contract.get("samples_per_prompt") != 5 or len(contract.get("prompts", [])) != 944):
        raise ValueError("Reference must be the completed full Stage-2 three-epoch benchmark, not a smoke/Joint V2 run")
    if (report.get("status") != "complete" or report.get("protocol") != contract
            or report.get("evaluation_assets") != assets):
        raise ValueError("Epoch-3 scores do not match their run contract/evaluation assets")
    plan = contract["training_contract"]["media_epochs"]
    if (plan.get("epochs") != 3 or plan.get("steps") != 3 * plan.get("steps_per_epoch", 0)
            or contract["checkpoint"]["step"] != plan["steps"]
            or contract["checkpoint"]["final_step"] != plan["steps"]):
        raise ValueError("Reference checkpoint is not the final three-epoch checkpoint")
    for variant in base.VARIANTS:
        score = report["scores"][variant][base.SYSTEM]
        if any(key not in score for key in ("quality_score", "semantic_score", "total_score")) or any(
                dimension not in score.get("raw", {}) for dimension in DIMENSIONS):
            raise ValueError(f"Reference scores incomplete: {variant}")
    return contract, report, assets


def compare_protocol(reference, current):
    if reference["training_contract"] != current["training_contract"]:
        raise ValueError("Intermediate and final checkpoints have different training contracts")
    names = ("config_sha256", "connector_spec", "prompts", "full_info_sha256", "samples_per_prompt", "seed",
             "seed_mapping", "seed_policy", "exports", "first_steps_per_stage", "video_steps_per_stage",
             "guidance", "video_guidance", "external_anchor", "native_conditioner", "conditioning", "dtype",
             "library_versions", "vbench_source_commit", "implementation_sha256")
    for name in names:
        if reference.get(name) != current.get(name):
            raise ValueError(f"Evaluation differs from completed epoch 3: {name}")
    if {k: v for k, v in reference["quicksr"].items() if k != "path"} != {
            k: v for k, v in current["quicksr"].items() if k != "path"}:
        raise ValueError("QuickSR weights/settings differ from epoch 3")


def prepare(args):
    if args.output_dir.resolve() == args.reference_run.resolve():
        raise ValueError("Use a separate output directory; preserve the completed epoch-3 benchmark")
    reference = None if args.max_prompts else read_reference(args.reference_run)[0]
    training_sha = fingerprint(reference["training_contract"]) if reference else None
    asset_args = copy.copy(args)
    asset_args.output_dir = args.output_dir / "_asset_preflight"
    with checkpoint_policy(args.expected_step, args.epoch, training_sha):
        contract = base.prepare(asset_args)
    if reference:
        compare_protocol(reference, contract)
    plan = contract["training_contract"]["media_epochs"]
    progress = contract["checkpoint"]["step"] / plan["steps_per_epoch"]
    contract["checkpoint_evaluation"] = dict(
        kind="intermediate_checkpoint", expected_step=args.expected_step, requested_epoch=args.epoch,
        exact_epoch_boundary=args.epoch * plan["steps_per_epoch"], media_epoch_progress=progress,
        training_contract_sha256=fingerprint(contract["training_contract"]),
        reference_run=str(args.reference_run),
        reference_contract_sha256=fingerprint(reference) if reference else None,
        reference_verified=reference is not None,
    )
    contract["implementation_sha256"]["tools/benchmark_mobileov_stage2_epoch_dual.py"] = sha256(Path(__file__))
    contract["scope"] = (f"Intermediate step {args.expected_step}, approximately {progress:.3f} media epochs; "
                         "training_complete remains false. Same numerical inference, prompts/seeds, raw/QuickSR "
                         "exports and evaluation as completed epoch 3. No external first-frame generator. "
                         "Checkpoint files are pinned, never selected by filename alone. " +
                         ("Generation-only smoke; final-reference comparison skipped." if args.max_prompts else
                          "The completed final benchmark is reused, not regenerated."))
    digest = bind_contract(args.output_dir, contract)
    atomic_json(args.output_dir / "vbench_info.json", read_json(asset_args.output_dir / "vbench_info.json"))
    plan_json = read_json(asset_args.output_dir / "sample_plan.json")
    atomic_json(args.output_dir / "sample_plan.json", {**plan_json, "contract_sha256": digest})
    atomic_json(args.output_dir / "checkpoint_audit.json", {
        **read_json(asset_args.output_dir / "checkpoint_audit.json"),
        "checkpoint_evaluation": contract["checkpoint_evaluation"],
    })
    print(f"Intermediate checkpoint verified: step={args.expected_step}; epochs={progress:.3f}; OUT={args.output_dir}", flush=True)
    return contract


def generation_worker(args):
    contract = read_json(args.output_dir / "run_contract.json")
    policy = contract["checkpoint_evaluation"]
    with checkpoint_policy(policy["expected_step"], policy["requested_epoch"], policy["training_contract_sha256"]):
        base.generation_worker(args)


def verified_reference(contract):
    policy = contract["checkpoint_evaluation"]
    if not policy["reference_verified"]:
        raise ValueError("No full reference comparison for a generation smoke")
    reference, report, assets = read_reference(Path(policy["reference_run"]))
    if fingerprint(reference) != policy["reference_contract_sha256"]:
        raise ValueError("Epoch-3 reference changed after preflight")
    return reference, report, assets


def verify_score_assets(output, contract):
    _, _, expected = verified_reference(contract)
    actual = read_json(output / "score_assets.json")
    if sorted(row["sha256"] for row in actual) != sorted(row["sha256"] for row in expected):
        raise ValueError("VBench evaluation weights differ from epoch 3")


def epoch_comparison(output, contract, scores):
    reference, final, _ = verified_reference(contract)
    policy = contract["checkpoint_evaluation"]
    metrics = ("quality_score", "semantic_score", "total_score")
    lines = ["# Stage-2 Checkpoint vs Completed Epoch 3", "",
             f"Step {policy['expected_step']} (~{policy['media_epoch_progress']:.3f} media epochs) "
             f"vs final step {reference['checkpoint']['step']}. Same evaluation protocol; no external anchor.", "",
             "| Score (%) | Intermediate Raw | Final Raw | Intermediate - Final (pp) | Intermediate QuickSR | Final QuickSR | Intermediate - Final (pp) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    rows = {}
    for name in (*metrics, *DIMENSIONS):
        row, values = {}, []
        for variant in ("no_quicksr", "quicksr"):
            a, b = scores[variant][base.SYSTEM], final["scores"][variant][base.SYSTEM]
            middle = a[name] if name in metrics else a["raw"][name]
            end = b[name] if name in metrics else b["raw"][name]
            row[variant] = dict(intermediate=middle, final=end, difference_pp=100 * (middle - end))
            values.extend((f"{100*middle:.2f}", f"{100*end:.2f}", f"{100*(middle-end):+.2f}"))
        rows[name] = row
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    atomic_json(output / "epoch_comparison.json", dict(status="complete", checkpoint_evaluation=policy, rows=rows))
    (output / "epoch_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(args):
    from evaluate_vbench_resume import tabulate

    contract = read_json(args.output_dir / "run_contract.json")
    if contract["max_prompts"] or contract["samples_per_prompt"] != 5:
        raise ValueError("No full score for a subset/single-sample generation smoke")
    verify_score_assets(args.output_dir, contract)
    base.verify_generation(args.output_dir)
    scores = {}
    for variant in base.VARIANTS:
        merged = args.output_dir / variant / "scores/vbench"
        merged.mkdir(parents=True, exist_ok=True)
        for dimension in DIMENSIONS:
            source = args.output_dir / variant / "scores" / dimension / "vbench" / f"{dimension}_eval_results.json"
            if dimension not in read_json(source):
                raise ValueError(f"Missing score: {source}")
            shutil.copyfile(source, merged / source.name)
        scores[variant] = {base.SYSTEM: tabulate(merged)}
        subprocess.run([sys.executable, str(ROOT / "tools/report_vbench_low_prompts.py"), "--score-dir", str(merged),
                        "--output-prefix", str(args.output_dir / variant / "low_prompts")], check=True)
    report, markdown = base.dual.ablation_report(scores, {"selected_systems": [base.SYSTEM]})
    report.update(protocol=contract, evaluation_assets=read_json(args.output_dir / "score_assets.json"), scope=contract["scope"])
    atomic_json(args.output_dir / "quicksr_ablation.json", report)
    policy = contract["checkpoint_evaluation"]
    title = f"# Stage-2 Step {policy['expected_step']}: ~{policy['media_epoch_progress']:.3f} Media Epochs\n\n"
    markdown = markdown.replace("Identical generated RGB frames and anchors;", "Identical generated RGB frames; no external anchor;")
    (args.output_dir / "quicksr_ablation.md").write_text(title + markdown, encoding="utf-8")
    epoch_comparison(args.output_dir, contract, scores)
    print(f"Completed intermediate Stage-2 VBench: {args.output_dir}", flush=True)


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--epoch", type=int, choices=(1, 2))
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--reference-run", type=Path, default=REFERENCE)
    extra, rest = parser.parse_known_args(argv)
    if "--help" in rest or "-h" in rest:
        print(__doc__ + "\n" + parser.format_help())
    args = base.parse_args(rest)
    args.epoch, args.expected_step, args.reference_run = extra.epoch, extra.expected_step, extra.reference_run.resolve()
    if args.phase in ("run", "prepare"):
        if args.epoch is None or args.expected_step is None or args.expected_step <= 0:
            parser.error("Run/prepare requires --epoch 1|2 and a positive --expected-step")
        if not any(x == "--checkpoint" or x.startswith("--checkpoint=") for x in rest):
            args.checkpoint = base.DEFAULT_CHECKPOINT.parent / f"stage2_dit_connector_step{args.expected_step:06d}.pt"
        if not any(x == "--output-dir" or x.startswith("--output-dir=") for x in rest):
            args.output_dir = ROOT / f"output/vbench_mobileov_stage2_media3epochs_epoch{args.epoch}_step{args.expected_step}_5x_dual"
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
        previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            commands = [[sys.executable, str(Path(__file__).resolve()), "--phase", "generation-worker",
                         "--output-dir", str(args.output_dir), "--rank", str(rank), "--workers", str(args.workers)]
                        for rank in range(args.workers)]
            base.core.run_processes(commands, devices, args.output_dir / "logs/generation")
            base.verify_generation(args.output_dir)
            if args.max_prompts:
                print("Intermediate checkpoint generation smoke passed; no full benchmark score claimed.", flush=True)
                return
            commands = [[args.vbench_python, str(Path(__file__).resolve()), "--phase", "score-worker",
                         "--output-dir", str(args.output_dir), "--vbench-repo", str(args.vbench_repo),
                         "--rank", str(rank), "--workers", str(args.workers)] for rank in range(args.workers)]
            base.core.run_processes(commands, devices, args.output_dir / "logs/scoring", cwd=args.vbench_repo)
            subprocess.run([args.vbench_python, str(Path(__file__).resolve()), "--phase", "summarize",
                            "--output-dir", str(args.output_dir)], cwd=args.vbench_repo, check=True)
        finally:
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
