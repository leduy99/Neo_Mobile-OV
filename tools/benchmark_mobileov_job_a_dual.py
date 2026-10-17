#!/usr/bin/env python
"""Evaluate completed Job A with matched text-only raw/QuickSR VBench."""
from __future__ import annotations

from contextlib import contextmanager
import argparse
import ast
import copy
import gc
import hashlib
from functools import lru_cache
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import benchmark_mobileov_stage2_media3epochs_dual as base  # noqa: E402
import benchmark_mobileov_stage2_epoch_dual as epochs  # noqa: E402
from vbench_joint_native_protocol import (  # noqa: E402
    atomic_json, bind_contract, fingerprint, output_lock, read_json, sha256,
)

SYSTEM = "stage2_job_a_first_frame_focus"
PARENT_SYSTEM = "stage2_media3epochs"
PARENT_STEP = 283977
FINAL_STEP = 378636
PARENT_SHA256 = "a74fd3069d17437b674415268c5157a85d94223509121220572e2d2fcdb68053"
DEFAULT_CHECKPOINT = ROOT / "output/mobileov_stage2_first_frame_focus/17780021/stage2_dit_connector_latest.pt"
DEFAULT_OUTPUT = ROOT / "output/vbench_mobileov_job_a_first_frame_focus_5x_dual"
REFERENCE = ROOT / "output/vbench_mobileov_stage2_media3epochs_5x_dual"
TRAINING_FILE = "tools/train_mobileov_stage1.py"


def check_job_a_checkpoint(payload):
    training = payload.get("contract", {})
    focus = training.get("first_frame_focus", {})
    plan = training.get("media_epochs", {})
    if (payload.get("format") != base.stage2.FORMAT
            or training.get("scope") != "first_frame_focus_continuation_public_data_not_exact_UniVideo"
            or training.get("initialization") != "stage2_weights_only_fork_fresh_optimizer"
            or training.get("init_step") != PARENT_STEP
            or training.get("init_stage2_sha256") != PARENT_SHA256):
        raise ValueError("Expected Job A fork of the exact completed step283977 parent, not Job B or an older run")
    expected = dict(recipe="media_epochs_i2v_v1", epochs=1, start_epoch=3, end_epoch=4,
                    images=984636, videos=529902, world_size=8, accumulation=2,
                    global_batch=16, steps_per_epoch=94659, steps=94659)
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("Job A media-pass/data budget differs from the agreed one-epoch continuation")
    if training.get("steps") != FINAL_STEP or payload.get("step") != FINAL_STEP:
        raise ValueError(f"Job A is NOT complete: step={payload.get('step')}; expected final step={FINAL_STEP}")
    if (focus.get("integration_smoke") is not False or focus.get("history_noise") != 0
            or focus.get("optimizer_step_origin") != PARENT_STEP
            or focus.get("t2v_first_probability") != .5 or focus.get("reconstruction_weight") != .2
            or training.get("ema") is not False
            or training.get("trainable_components") != ["connector", "dit"]
            or training.get("frozen_components") != ["smolvlm2", "vae"]):
        raise ValueError("Job A training policy differs, or this is an integration-only/LoRA checkpoint")
    parent = training.get("parent_contract", {})
    if (parent.get("steps") != PARENT_STEP or parent.get("media_epochs", {}).get("epochs") != 3
            or parent.get("media_epochs", {}).get("steps") != PARENT_STEP):
        raise ValueError("Job A must record its completed three-epoch parent contract")
    tasks = {"t2i", "t2v", "i2v_reference", "i2v_anchor", "image_reconstruction"}
    if not tasks.issubset(training.get("tasks", [])):
        raise ValueError("Missing Job A training tasks")
    base.stage2.check_payload(payload, FINAL_STEP)
    return FINAL_STEP


def read_checkpoint(path, *, allow_incomplete=False, expected_sha=None):
    import torch

    path = Path(path)
    # The legacy worker passes allow_incomplete for a prompt subset; admission stays final-only.
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed Job A checkpoint: {path}; no parent/HF fallback")
    before = path.stat()
    digest = sha256(path)
    if expected_sha and expected_sha != digest:
        raise ValueError("Job A checkpoint SHA256 changed")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    target = check_job_a_checkpoint(payload)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Job A checkpoint changed during preflight; wait for the final save")
    return payload, digest, target


@contextmanager
def job_a_policy():
    # Reuse the numerical generation/export path; change only checkpoint admission and labels.
    original_reader, original_system = base.read_checkpoint, base.SYSTEM
    base.read_checkpoint, base.SYSTEM = read_checkpoint, SYSTEM
    try:
        yield
    finally:
        base.read_checkpoint, base.SYSTEM = original_reader, original_system


def read_reference(folder):
    original = base.SYSTEM
    base.SYSTEM = PARENT_SYSTEM
    try:
        contract, report, assets = epochs.read_reference(folder)
    finally:
        base.SYSTEM = original
    if contract["checkpoint"]["sha256"] != PARENT_SHA256 or contract["checkpoint"]["step"] != PARENT_STEP:
        raise ValueError("Reference scores must belong to the exact Job A parent step283977")
    return contract, report, assets


def inference_helpers(source):
    names = {"load_stack", "autocast", "encode_anchor"}
    definitions = {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                   for node in ast.parse(source).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and node.name in names}
    if set(definitions) != names:
        raise ValueError("Cannot audit the trainer's imported inference helpers")
    return definitions


@lru_cache(maxsize=16)
def historical_training_source(expected_sha):
    commits = subprocess.check_output(["git", "-C", str(ROOT), "log", "-100", "--format=%H", "--", TRAINING_FILE], text=True).splitlines()
    for commit in commits:
        source = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{commit}:{TRAINING_FILE}"])
        if hashlib.sha256(source).hexdigest() == expected_sha:
            return source.decode("utf-8"), commit
    raise ValueError("Cannot locate the parent's recorded trainer source in git history; fetch its history "
                     "or rerun the parent under the current evaluation code. Do not bypass the inference audit.")


def verify_training_helpers(reference, current):
    before = reference["implementation_sha256"].get(TRAINING_FILE)
    after = current["implementation_sha256"].get(TRAINING_FILE)
    if not before or not after or after != sha256(ROOT / TRAINING_FILE):
        raise ValueError("Missing/mismatched current trainer source fingerprint")
    current_source = (ROOT / TRAINING_FILE).read_text(encoding="utf-8")
    if before == after:
        return dict(policy="identical_trainer_source", source_sha256=after, helpers=inference_helpers(current_source))
    original, commit = historical_training_source(before)
    helpers = inference_helpers(original)
    if helpers != inference_helpers(current_source):
        raise ValueError("Trainer-imported inference helpers differ from the parent; rerun the parent for a matched comparison")
    return dict(policy="changed_training_code_but_identical_imported_inference_helpers",
                parent_source_commit=commit, parent_sha256=before, current_sha256=after, helpers=helpers)


def compare_protocol(reference, current):
    training = current["training_contract"]
    if reference["training_contract"] != training["parent_contract"]:
        raise ValueError("Reference training contract is not Job A's recorded parent")
    for name in ("data_summary_sha256", "processor_sha256", "frozen_weights_sha256"):
        if reference["training_contract"].get(name) != training.get(name):
            raise ValueError(f"Job A changed the parent data/frozen conditioning stack: {name}")
    names = ("config_sha256", "connector_spec", "prompts", "full_info_sha256", "samples_per_prompt",
             "seed", "seed_mapping", "seed_policy", "exports", "first_steps_per_stage", "video_steps_per_stage",
             "guidance", "video_guidance", "external_anchor", "native_conditioner", "conditioning", "dtype",
             "library_versions", "vbench_source_commit")
    for name in names:
        if reference.get(name) != current.get(name):
            raise ValueError(f"Evaluation differs from the epoch-3 parent: {name}")
    if {k: v for k, v in reference["quicksr"].items() if k != "path"} != {
            k: v for k, v in current["quicksr"].items() if k != "path"}:
        raise ValueError("QuickSR weights/settings differ from the parent")
    old, new = reference["implementation_sha256"], current["implementation_sha256"]
    for name in (set(old) | set(new)) - {TRAINING_FILE}:
        if old.get(name) != new.get(name):
            raise ValueError(f"Numerical inference/evaluation code differs from the parent: {name}")
    return verify_training_helpers(reference, current)


def prepare(args):
    if args.output_dir.resolve() == args.reference_run.resolve():
        raise ValueError("Use a separate Job A output; preserve the parent benchmark")
    reference_data = None if args.max_prompts else read_reference(args.reference_run)
    reference = reference_data[0] if reference_data else None
    asset_args = copy.copy(args)
    asset_args.output_dir = args.output_dir / "_asset_preflight"
    with job_a_policy():
        contract = base.prepare(asset_args)
    helper_audit = compare_protocol(reference, contract) if reference else None
    contract["job_a_evaluation"] = dict(
        kind="completed_first_frame_focus_job_a", parent_step=PARENT_STEP, final_step=FINAL_STEP,
        additional_updates=FINAL_STEP - PARENT_STEP, parent_checkpoint_sha256=PARENT_SHA256,
        reference_run=str(args.reference_run), reference_verified=reference is not None,
        reference_contract_sha256=fingerprint(reference) if reference else None,
        reference_report_sha256=fingerprint(reference_data[1]) if reference_data else None,
        imported_inference_helpers=helper_audit)
    contract["implementation_sha256"].update({name: sha256(ROOT / name) for name in (
        "tools/benchmark_mobileov_job_a_dual.py", "tools/benchmark_mobileov_stage2_epoch_dual.py")})
    contract["scope"] = (
        "Completed Job A, step378636: one additional media pass after epoch 3; first-unit emphasis and "
        "reconstruction weight 0.2. Frozen SmolVLM2 + trained MCP/monolithic DiT; no LoRA or external image. "
        "Generate all frames from literal text, including frame zero. Same parent prompts/seeds, 20/10 steps "
        "per stage, CFG 7/5 and evaluator weights. Raw 512x320 vs QuickSR 1024x640 combines resolution/SR. "
        "This is not an isolated reconstruction ablation or a mobile/few-step benchmark. "
        + ("Generation-only smoke; no full score/reference comparison." if args.max_prompts else
           "Reuse the completed parent score report; do not regenerate it."))
    digest = bind_contract(args.output_dir, contract)
    for name in ("vbench_info.json", "checkpoint_audit.json"):
        value = read_json(asset_args.output_dir / name)
        if name == "checkpoint_audit.json":
            value["job_a_evaluation"] = contract["job_a_evaluation"]
        atomic_json(args.output_dir / name, value)
    samples = read_json(asset_args.output_dir / "sample_plan.json")["samples"]
    atomic_json(args.output_dir / "sample_plan.json", dict(contract_sha256=digest, samples=samples))
    print(f"Job A verified: step={FINAL_STEP}; text-only, no external anchor; OUT={args.output_dir}", flush=True)
    return contract


def verified_reference(contract):
    policy = contract["job_a_evaluation"]
    if not policy["reference_verified"]:
        raise ValueError("No full parent comparison for a generation smoke")
    reference, report, assets = read_reference(Path(policy["reference_run"]))
    if (fingerprint(reference) != policy["reference_contract_sha256"]
            or fingerprint(report) != policy["reference_report_sha256"]):
        raise ValueError("Parent reference changed after Job A preflight")
    return reference, report, assets


def verify_score_assets(output, contract):
    expected = verified_reference(contract)[2]
    actual = read_json(output / "score_assets.json")
    if sorted(row["sha256"] for row in actual) != sorted(row["sha256"] for row in expected):
        raise ValueError("VBench evaluation weights differ from the epoch-3 parent")


def comparison(output, contract):
    _, parent, _ = verified_reference(contract)
    report = read_json(output / "quicksr_ablation.json")
    metrics = ("quality_score", "semantic_score", "total_score")
    rows = {}
    lines = ["# Job A vs Epoch-3 Parent", "",
             "Step 378636 vs 283977. Matched text-only inference; no SSD1B/DreamLite anchor. "
             "Changes combine another media pass, first-unit sampling and stronger reconstruction; not an isolated loss ablation.", "",
             "| Score (%) | Parent Raw | Job A Raw | Change (pp) | Parent QuickSR | Job A QuickSR | Change (pp) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in (*metrics, *base.DIMENSIONS):
        row, values = {}, []
        for variant in ("no_quicksr", "quicksr"):
            a = parent["scores"][variant][PARENT_SYSTEM]
            b = report["scores"][variant][SYSTEM]
            a, b = (a[name], b[name]) if name in metrics else (a["raw"][name], b["raw"][name])
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (a, b)):
                raise ValueError(f"Non-finite/missing comparison score: {variant}/{name}")
            row[variant] = dict(parent=a, job_a=b, difference_pp=100 * (b - a))
            values.extend((f"{100*a:.2f}", f"{100*b:.2f}", f"{100*(b-a):+.2f}"))
        rows[name] = row
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    atomic_json(output / "job_a_vs_epoch3.json", dict(status="complete", job_a_evaluation=contract["job_a_evaluation"], rows=rows))
    (output / "job_a_vs_epoch3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generation_worker(args):
    contract = read_json(args.output_dir / "run_contract.json")
    if (contract.get("system") != SYSTEM or not contract.get("training_complete")
            or contract.get("checkpoint", {}).get("step") != FINAL_STEP
            or contract.get("job_a_evaluation", {}).get("final_step") != FINAL_STEP):
        raise ValueError("Not a completed Job A benchmark contract")
    with job_a_policy():
        base.generation_worker(args)


def summarize(args):
    contract = read_json(args.output_dir / "run_contract.json")
    if contract["max_prompts"] or contract["samples_per_prompt"] != 5 or not contract["training_complete"]:
        raise ValueError("No full Job A score for a smoke, incomplete run or single-sample evaluation")
    verify_score_assets(args.output_dir, contract)
    with job_a_policy():
        base.summarize(args)
    comparison(args.output_dir, contract)
    print(f"Completed Job A VBench; comparison: {args.output_dir / 'job_a_vs_epoch3.md'}", flush=True)


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reference-run", type=Path, default=REFERENCE)
    extra, rest = parser.parse_known_args(argv)
    if "--help" in rest or "-h" in rest:
        print(__doc__ + "\n" + parser.format_help())
    args = base.parse_args(rest)
    args.reference_run = extra.reference_run.resolve()
    if args.allow_incomplete_checkpoint:
        parser.error("Job A requires the completed final step378636 checkpoint, including for generation smoke")
    for flag, name, default in (("--checkpoint", "checkpoint", DEFAULT_CHECKPOINT),
                                ("--output-dir", "output_dir", DEFAULT_OUTPUT)):
        if not any(value == flag or value.startswith(flag + "=") for value in rest):
            setattr(args, name, default)
    if args.max_prompts and not any(value == "--output-dir" or value.startswith("--output-dir=") for value in rest):
        args.output_dir = ROOT / f"output/vbench_mobileov_job_a_first_frame_focus_{args.samples_per_prompt}x_dual_smoke{args.max_prompts}"
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
        with job_a_policy():
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
        previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            commands = [[sys.executable, str(Path(__file__).resolve()), "--phase", "generation-worker",
                         "--output-dir", str(args.output_dir), "--rank", str(rank), "--workers", str(args.workers)]
                        for rank in range(args.workers)]
            base.core.run_processes(commands, devices, args.output_dir / "logs/generation")
            with job_a_policy():
                base.verify_generation(args.output_dir)
            if args.max_prompts:
                print("Job A generation smoke passed; no full VBench score claimed.", flush=True)
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
