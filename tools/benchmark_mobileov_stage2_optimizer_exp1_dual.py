#!/usr/bin/env python
"""Four pinned Exp-1 RAW/EMA snapshots; matched text-only raw/QuickSR VBench."""
from contextlib import contextmanager
import argparse
import copy
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import benchmark_mobileov_job_a_dual as previous
from vbench_joint_native_protocol import atomic_json, bind_contract, fingerprint, output_lock, read_json, sha256

base = previous.base
POLICY = "epoch3_adam_resume_batch64_linear_decay_ema_v1"
METRICS = ("quality_score", "semantic_score", "total_score")


def check_checkpoint(payload, update, variant, *, allow_incomplete=False):
    # Scoring never imports the trainer/PyAV; only the generation environment validates tensors.
    training = payload.get("contract", {})
    spec = training.get("optimizer_exp1", {})
    expected = 283977 + update
    if (update not in (10000, 20000) or variant not in ("raw", "ema")
            or payload.get("exp1_update") != update or payload.get("step") != expected
            or payload.get("weight_variant") != variant or payload.get("ema_updates") != update
            or training.get("scope") != POLICY or spec.get("policy") != POLICY
            or spec.get("parent_step") != 283977 or spec.get("global_batch") != 64
            or spec.get("total_updates") != 20000
            or training.get("ema", {}).get("decay") != .9995
            or training.get("lr_schedule", {}).get("decay_steps") != 20000
            or training.get("parent_contract", {}).get("steps") != 283977
            or training.get("world_size") != 16 or training.get("accumulation") != 4
            or training.get("warmup") != 0 or training.get("lr") != 2e-5
            or training.get("auxiliary_reconstruction", {}).get("weight") != .1
            or any(key in training for key in ("history_corruption", "first_frame_focus", "lora_spec"))
            or "optimizer" in payload):
        raise ValueError("Wrong Exp-1 snapshot/update/weight variant; never fall back to latest or raw weights")
    base.stage2.check_payload(payload, expected)
    return expected  # This is an explicit evaluation milestone, not a completed media epoch.


@contextmanager
def case_policy(update, variant):
    saved = base.SYSTEM, base.check_media_checkpoint
    base.SYSTEM = f"stage2_exp1_{variant}_{update}"
    base.check_media_checkpoint = lambda payload, **kw: check_checkpoint(payload, update, variant, **kw)
    try:
        yield
    finally:
        base.SYSTEM, base.check_media_checkpoint = saved


def worker_policy(output):
    contract = read_json(output / "run_contract.json")
    policy = contract["exp1_evaluation"]
    update, variant = policy["additional_updates"], policy["weights"]
    if (update not in (10000, 20000) or variant not in ("raw", "ema")
            or contract["system"] != f"stage2_exp1_{variant}_{update}"):
        raise ValueError("Invalid Exp-1 worker contract")
    return update, variant


def verify_score_assets(output, contract):
    policy = contract["exp1_evaluation"]
    reference, report, assets = previous.read_reference(Path(policy["reference_run"]))
    if (not policy["reference_verified"] or fingerprint(reference) != policy["reference_contract_sha256"]
            or fingerprint(report) != policy["reference_report_sha256"]):
        raise ValueError("Epoch-3 reference changed after Exp-1 preflight")
    actual = read_json(output / "score_assets.json")
    if sorted(row["sha256"] for row in actual) != sorted(row["sha256"] for row in assets):
        raise ValueError("VBench scorer weights differ from the epoch-3 baseline")


def prepare_case(args, update, variant, reference):
    manifest = read_json(args.exp1_run / "evaluation_checkpoints.json")
    item = manifest.get("checkpoints", {}).get(f"{variant}_{update}")
    if (manifest.get("format") != POLICY or not item or item.get("exp1_update") != update
            or item.get("weight_variant") != variant or item.get("step") != 283977 + update):
        raise ValueError(f"Missing pinned {variant}/{update} snapshot; wait for its archive, do not use latest")
    name = Path(item["path"])
    if name.name != str(name):
        raise ValueError("Checkpoint manifest must contain a local filename")
    child = copy.copy(args)
    child.checkpoint = args.exp1_run / name
    child.expected_sha256 = item["sha256"]
    child.output_dir = args.output_dir / f"{variant}_{update}"
    asset_args = copy.copy(child)
    asset_args.output_dir = child.output_dir / "_asset_preflight"
    with case_policy(update, variant):
        contract = base.prepare(asset_args)
    if reference is not None:
        previous.compare_protocol(reference[0], contract)
    contract["exp1_evaluation"] = dict(additional_updates=update, weights=variant,
        milestone_complete=True, experiment_complete=update == 20000,
        reference_run=str(args.reference_run), reference_verified=reference is not None,
        reference_contract_sha256=fingerprint(reference[0]) if reference else None,
        reference_report_sha256=fingerprint(reference[1]) if reference else None)
    contract["implementation_sha256"]["tools/benchmark_mobileov_stage2_optimizer_exp1_dual.py"] = sha256(Path(__file__))
    contract["scope"] = (f"Advisor Exp1: {variant.upper()} weights after {update} additional batch64 updates from epoch3 Adam. "
        "Not media epoch4/5. Same literal prompts/seeds, 20/10 steps per stage, CFG7/5, frozen SmolVLM2/VAE. "
        "No external SSD1B/DreamLite anchor, vision reference, LoRA, history noise or Hybrid inference. "
        "Raw 512x320 vs QuickSR 1024x640 share generated frames and combine resolution/SR effects. "
        "EMA, larger batch and decay change together; not an isolated EMA ablation.")
    digest = bind_contract(child.output_dir, contract)
    for name in ("vbench_info.json", "checkpoint_audit.json"):
        atomic_json(child.output_dir / name, read_json(asset_args.output_dir / name))
    samples = read_json(asset_args.output_dir / "sample_plan.json")["samples"]
    atomic_json(child.output_dir / "sample_plan.json", dict(contract_sha256=digest, samples=samples))
    return child, contract


def summarize_sweep(args, cases, reference):
    lines = ["# Optimizer Exp 1: RAW and EMA vs Epoch-3 Parent", "",
        "Matched text-only generation; no external anchor. Raw and QuickSR also differ in output resolution.", "",
        "| Setting | Output | Quality (%) | Semantic (%) | Total (%) | Total vs Parent (pp) |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    rows = []
    for folder, contract in cases:
        report = read_json(folder / "quicksr_ablation.json")
        if fingerprint(report.get("protocol")) != fingerprint(contract) or report.get("status") != "complete":
            raise ValueError("Incomplete or mismatched Exp-1 score report")
        for variant in ("no_quicksr", "quicksr"):
            scores = report["scores"][variant][contract["system"]]
            parent = reference[1]["scores"][variant][previous.PARENT_SYSTEM]
            values = [100 * scores[name] for name in METRICS]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Non-finite Exp-1 summary score")
            difference = 100 * (scores["total_score"] - parent["total_score"])
            rows.append(dict(setting=contract["system"], output=variant,
                             **dict(zip(METRICS, values)), total_vs_parent_pp=difference))
            lines.append(f"| {contract['system']} | {variant} | " + " | ".join(f"{v:.2f}" for v in values) + f" | {difference:+.2f} |")
    atomic_json(args.output_dir / "comparison.json", dict(status="complete", rows=rows))
    (args.output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--exp1-run", type=Path)
    parser.add_argument("--updates", nargs="+", type=int, choices=(10000, 20000), default=[10000, 20000])
    parser.add_argument("--reference-run", type=Path, default=previous.REFERENCE)
    extra, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    args = base.parse_args(rest)
    args.exp1_run = extra.exp1_run.resolve() if extra.exp1_run else None
    args.updates = sorted(set(extra.updates))
    args.reference_run = extra.reference_run.resolve()
    if args.phase in ("run", "prepare"):
        if args.exp1_run is None:
            parser.error("Supply --exp1-run pointing to the Exp-1 training output")
        if args.output_dir == base.ROOT / "output/vbench_mobileov_stage2_media3epochs_5x_dual":
            parser.error("Use a separate --output-dir; never overwrite epoch-3 evaluation")
    return args


def main():
    args = parse_args()
    if args.phase == "warm-scores":
        base.core.warm_scores(args)
        return
    if args.phase in ("generation-worker", "score-worker", "summarize"):
        update, variant = worker_policy(args.output_dir)
        with case_policy(update, variant):
            if args.phase == "generation-worker":
                base.generation_worker(args)
            elif args.phase == "score-worker":
                verify_score_assets(args.output_dir, read_json(args.output_dir / "run_contract.json"))
                base.score_worker(args)
            else:
                verify_score_assets(args.output_dir, read_json(args.output_dir / "run_contract.json"))
                base.summarize(args)
        return
    os.chdir(ROOT)
    entry = str(Path(__file__).resolve())
    with output_lock(args.output_dir):
        reference = None if args.max_prompts else previous.read_reference(args.reference_run)
        cases = [prepare_case(args, update, variant, reference) for update in args.updates for variant in ("raw", "ema")]
        if args.phase == "prepare":
            return
        devices = base.core.gpu_tokens(args.workers)
        old_handler = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            for child, contract in cases:
                if not args.max_prompts:
                    subprocess.run([args.vbench_python, entry, "--phase", "warm-scores", "--output-dir", str(child.output_dir),
                                    "--vbench-repo", str(args.vbench_repo)], cwd=args.vbench_repo, check=True)
                    verify_score_assets(child.output_dir, contract)
                commands = [[sys.executable, entry, "--phase", "generation-worker", "--output-dir", str(child.output_dir),
                             "--rank", str(rank), "--workers", str(args.workers)] for rank in range(args.workers)]
                base.core.run_processes(commands, devices, child.output_dir / "logs/generation")
                with case_policy(*worker_policy(child.output_dir)):
                    base.verify_generation(child.output_dir)
                if args.max_prompts:
                    continue
                commands = [[args.vbench_python, entry, "--phase", "score-worker", "--output-dir", str(child.output_dir),
                    "--vbench-repo", str(args.vbench_repo), "--rank", str(rank), "--workers", str(args.workers)]
                    for rank in range(args.workers)]
                base.core.run_processes(commands, devices, child.output_dir / "logs/scoring", cwd=args.vbench_repo)
                subprocess.run([args.vbench_python, entry, "--phase", "summarize", "--output-dir", str(child.output_dir)],
                               cwd=args.vbench_repo, check=True)
            if not args.max_prompts:
                summarize_sweep(args, [(child.output_dir, contract) for child, contract in cases], reference)
        finally:
            signal.signal(signal.SIGTERM, old_handler)


if __name__ == "__main__":
    main()
