#!/usr/bin/env python
"""One allocation, five Stage-2 checkpoints, one read-only SSD1B cache, raw + QuickSR."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from importlib.metadata import version
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

import benchmark_stage2_ssd1b_conditioning_dual as anchor  # noqa: E402
import benchmark_mobileov_stage2_resume_epoch4_dual as resumed  # noqa: E402
import benchmark_mobileov_stage2_history_fresh_epochs_dual as history  # noqa: E402
import benchmark_mobileov_stage2_history_fresh_epoch5_dual as history5  # noqa: E402
from vbench_joint_native_protocol import (  # noqa: E402
    atomic_json, bind_contract, fingerprint, output_lock, read_json, sha256,
)

base, shared, previous = anchor.base, anchor.shared, history.previous
FORMAT = "mobileov_stage2_ssd1b_checkpoint_sweep_v1"
MODE = "ssd1b_anchor"
ANCHOR_REFERENCE = ROOT / "output/vbench_stage2_epoch3_ssd1b_conditioning_5x_dual/ssd1b_anchor"
DEFAULT_OUTPUT = ROOT / "output/vbench_stage2_checkpoint_ssd1b_sweep_5x_dual"
CASES = {
    "resume_epoch4": dict(kind="resume", epoch=4, step=378636, system="stage2_resume_epoch4",
        checkpoint="output/mobileov_stage2_resume5epochs16gpu/17892008/stage2_dit_connector_step378636.pt",
        text_reference="output/vbench_mobileov_stage2_resume_epoch4_step378636_5x_dual_v2"),
    "resume_epoch5": dict(kind="resume", epoch=5, step=473295, system="stage2_resume_epoch5",
        checkpoint="output/mobileov_stage2_resume5epochs16gpu/17892008/stage2_dit_connector_step473295.pt",
        text_reference="output/vbench_mobileov_stage2_resume_epoch5_step473295_5x_dual_v2"),
    "history_epoch3": dict(kind="history", epoch=3, step=283977, system="stage2_history_fresh_epoch3",
        checkpoint="output/mobileov_stage2_history_fresh5epochs16gpu/17892009/stage2_dit_connector_step283977.pt",
        text_reference="output/vbench_mobileov_stage2_history_fresh_epoch3_step283977_5x_dual"),
    "history_epoch4": dict(kind="history", epoch=4, step=378636, system="stage2_history_fresh_epoch4",
        checkpoint="output/mobileov_stage2_history_fresh5epochs16gpu/17892009/stage2_dit_connector_step378636.pt",
        text_reference="output/vbench_mobileov_stage2_history_fresh_epoch4_step378636_5x_dual"),
    "history_epoch5": dict(kind="history", epoch=5, step=473295, system="stage2_history_fresh_epoch5",
        checkpoint="output/mobileov_stage2_history_fresh5epochs16gpu/17892009/stage2_dit_connector_step473295.pt",
        text_reference="output/vbench_mobileov_stage2_history_fresh_epoch5_step473295_5x_dual"),
}
NUMERICAL_FILES = (
    "tools/benchmark_mobileov_stage2_media3epochs_dual.py", "tools/generate_vbench_mobileov_stage2.py",
    "tools/infer_mobileov_stage1.py", "tools/train_mobileov_stage1.py",
    "new_mobile_ov/bridge/stage1_conditioner.py", "new_mobile_ov/generation/neodragon_compat.py",
    "tools/apply_quicksrnet_video.py", "tools/evaluate_vbench_resume.py", "tools/vbench_joint_native_protocol.py",
)
ANCHOR_FILES = ("tools/benchmark_stage2_ssd1b_conditioning_dual.py", "tools/prepare_vbench_stage2_ssd1b_frames.py")
IMPLEMENTATION = tuple(dict.fromkeys((*resumed.IMPLEMENTATION, *history.IMPLEMENTATION,
    *history5.IMPLEMENTATION, *ANCHOR_FILES, "tools/benchmark_mobileov_stage2_ssd1b_checkpoint_sweep.py")))
METRICS = ("quality_score", "semantic_score", "total_score")


def selected_system(case):
    return CASES[case]["system"] + "_ssd1b_anchor"


def check_checkpoint(payload, case, *, allow_incomplete=False):
    spec = CASES[case]
    if spec["kind"] == "resume":
        with resumed.evaluation_epoch(spec["epoch"]):
            return resumed.check_checkpoint(payload, allow_incomplete=allow_incomplete)
    if spec["epoch"] == 5:
        return history5.check_checkpoint(payload, allow_incomplete=allow_incomplete)
    return history.check_checkpoint(payload, spec["epoch"], allow_incomplete=allow_incomplete)


@contextmanager
def case_policy(case):
    before = base.check_media_checkpoint, base.SYSTEM, anchor.MODES[MODE]
    base.check_media_checkpoint = lambda payload, **kw: check_checkpoint(payload, case, **kw)
    base.SYSTEM = selected_system(case)
    anchor.MODES[MODE] = dict(system=base.SYSTEM, image_to_smolvlm2=False, external_anchor=True)
    try:
        yield
    finally:
        base.check_media_checkpoint, base.SYSTEM, anchor.MODES[MODE] = before


def score_assets(assets):
    return sorted(row["sha256"] for row in assets)


def read_reference(folder):
    contract = read_json(folder / "run_contract.json")
    report = read_json(folder / "quicksr_ablation.json")
    assets = read_json(folder / "score_assets.json")
    if (report.get("status") != "complete" or fingerprint(report.get("protocol")) != fingerprint(contract)
            or contract.get("max_prompts") != 0 or contract.get("samples_per_prompt") != 5
            or len(contract.get("prompts", [])) != 944
            or score_assets(report.get("evaluation_assets", [])) != score_assets(assets)):
        raise ValueError(f"Reference must be a completed, matched 944-prompt/5-sample report: {folder}")
    for variant in base.VARIANTS:
        values = report["scores"][variant][contract["system"]]
        for name in (*METRICS, *base.DIMENSIONS):
            value = values[name] if name in METRICS else values["raw"][name]
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError(f"Invalid reference score: {folder}/{variant}/{name}")
    return contract, report, assets


def compare_numerical_protocol(reference, current, *, anchored=False):
    # Driver labels differ; the actual inference, preprocessing and scoring code must not.
    if anchored:
        for name in ("mode", "image_to_smolvlm2", "first_frame_generated", "image_processing", "vae_seed_policy"):
            if reference.get(name) != current.get(name):
                raise ValueError(f"Anchor preprocessing/route differs from the existing baseline: {name}")
        for name in ("contract_sha256", "image_manifest_sha256"):
            if reference["shared_ssd1b"][name] != current["shared_ssd1b"][name]:
                raise ValueError(f"Anchor image cache differs from the existing baseline: {name}")
    names = (*NUMERICAL_FILES, *ANCHOR_FILES) if anchored else NUMERICAL_FILES
    before, after = copy.deepcopy(reference), copy.deepcopy(current)
    for value in (before, after):
        recorded = value["implementation_sha256"]
        if any(name not in recorded for name in names):
            raise ValueError("Missing numerical inference source fingerprint")
        value["implementation_sha256"] = {name: recorded[name] for name in names}
    after["training_contract"]["parent_contract"] = before["training_contract"]
    return previous.compare_protocol(before, after)


def validate_image_protocol(template, images):
    if (images.get("admission_checkpoint_sha256") != shared.FINAL_SHA256
            or images.get("image", {}).get("image_bridge") is not False
            or (images["image"].get("width"), images["image"].get("height")) != (1024, 640)):
        raise ValueError("Require the verified native SSD1B cache originally admitted by the final clean epoch-3 checkpoint")
    for name in ("full_info_sha256", "seed", "seed_mapping", "config_sha256", "library_versions", "vbench_source_commit"):
        if template.get(name) != images.get(name):
            raise ValueError(f"SSD1B cache/video protocol mismatch: {name}")
    count = len(template["prompts"])
    if (template["prompts"] != images["prompts"][:count]
            or template["samples_per_prompt"] > images["samples_per_prompt"]
            or (not template["max_prompts"] and (template["prompts"] != images["prompts"]
                or template["samples_per_prompt"] != images["samples_per_prompt"] or images["max_prompts"] != 0))):
        raise ValueError("SSD1B cache does not cover the exact prompts/samples; smoke may only reuse its prefix")
    # Image generation is independent of video weights: retain the original admission SHA,
    # rather than relabel the shared cache as if each new checkpoint generated it.


def frame_cache(args, baseline):
    path = args.frames_root or Path(baseline[0]["shared_ssd1b"]["path"])
    images, manifest = shared.verify_cache(path)
    expected = baseline[0]["shared_ssd1b"]
    if (fingerprint(images) != expected["contract_sha256"]
            or fingerprint(read_json(path / "image_manifest.json")) != fingerprint(manifest)
            or fingerprint(manifest) != expected["image_manifest_sha256"]):
        raise ValueError("Cache is not the exact SSD1B image set used by the epoch-3 anchor baseline")
    return path.resolve(), images, manifest


def prepare_case(args, case, images, manifest, baseline):
    spec = CASES[case]
    output = args.output_dir / case
    local = copy.copy(args)
    local.output_dir, local.checkpoint = output / "_asset_preflight", ROOT / spec["checkpoint"]
    text_root = ROOT / spec["text_reference"]
    text = read_reference(text_root)
    with case_policy(case):
        template = base.prepare(local)
    validate_image_protocol(template, images)
    if (template["checkpoint"]["sha256"] != text[0]["checkpoint"]["sha256"]
            or template["checkpoint"]["step"] != text[0]["checkpoint"]["step"]
            or template["training_contract"] != text[0]["training_contract"]
            or text[0].get("external_anchor") is not False):
        raise ValueError(f"Text-only control does not evaluate these exact weights/recipe: {case}")
    if score_assets(text[2]) != score_assets(baseline[2]):
        raise ValueError(f"Text-only and original anchor references used different VBench weights: {case}")
    text_audit = None if args.max_prompts else compare_numerical_protocol(text[0], template)
    contract = copy.deepcopy(template)
    contract.update(format=anchor.FORMAT, mode=MODE, system=selected_system(case),
        image_to_smolvlm2=False, external_anchor=True, first_frame_generated=False,
        shared_ssd1b=dict(path=str(args.frames_root), contract_sha256=fingerprint(images),
                          image_manifest_sha256=fingerprint(manifest)),
        image_processing=dict(width=512, height=320, resampling="PIL.LANCZOS", pillow_version=version("Pillow")),
        vae_seed_policy="video seed + 228; private generator; video noise reset independently",
        conditioning="literal text ONLY through SmolVLM2; empty negative; shared SSD1B image through VAE ONLY")
    contract["implementation_sha256"].update({name: sha256(ROOT / name) for name in ANCHOR_FILES})
    anchor_audit = None if args.max_prompts else compare_numerical_protocol(baseline[0], contract, anchored=True)
    contract["implementation_sha256"].update({name: sha256(ROOT / name) for name in IMPLEMENTATION})
    contract["ssd1b_checkpoint_sweep"] = dict(
        case=case, kind=spec["kind"], epoch=spec["epoch"], expected_step=spec["step"], eligible_archive=True,
        text_reference=str(text_root), text_contract_sha256=fingerprint(text[0]), text_report_sha256=fingerprint(text[1]),
        text_inference_audit=text_audit, anchor_inference_audit=anchor_audit,
        anchor_reference=str(args.anchor_reference), anchor_contract_sha256=fingerprint(baseline[0]),
        anchor_report_sha256=fingerprint(baseline[1]), image_generation=False, history_noise_at_inference=False)
    contract["scope"] = (
        f"{case}, epoch{spec['epoch']}, step{spec['step']}: trained MCP+monolithic DiT. "
        "Text ONLY through frozen SmolVLM2; SSD1B PNG through VAE ONLY supplies the first latent. "
        "Exact read-only images from the original epoch-3 SSD1B cache; no image generation or vision tokens. "
        "No LoRA, native video conditioner, Hybrid inference, history noise at inference or training. "
        "Raw 512x320 and QuickSR 1024x640 share RGB frames; this combines resolution and SR effects. "
        "Intermediate archived epochs retain training_complete=false within their five-epoch budget. "
        + ("Generation smoke only; no full scores." if args.max_prompts else
           "Same prompts/seeds, sampler and evaluator as the exact-checkpoint text-only control and epoch-3 anchor baseline."))
    digest = bind_contract(output, contract)
    for name in ("vbench_info.json", "checkpoint_audit.json"):
        value = read_json(local.output_dir / name)
        if name == "checkpoint_audit.json":
            value["ssd1b_checkpoint_sweep"] = contract["ssd1b_checkpoint_sweep"]
        atomic_json(output / name, value)
    atomic_json(output / "sample_plan.json", dict(contract_sha256=digest, samples=list(base.items_for(contract))))
    return contract


def prepare_suite(args):
    baseline = read_reference(args.anchor_reference)
    original = baseline[0]
    if (original.get("format") != anchor.FORMAT or original.get("mode") != MODE
            or original.get("training_complete") is not True
            or original.get("image_to_smolvlm2") is not False or original.get("external_anchor") is not True
            or original["checkpoint"]["step"] != 283977 or original["checkpoint"]["sha256"] != shared.FINAL_SHA256):
        raise ValueError("Require the original epoch-3 SSD1B VAE-anchor report, not image-conditioning or text-only scores")
    args.frames_root, images, manifest = frame_cache(args, baseline)
    protected = [args.frames_root, args.anchor_reference, *(ROOT / CASES[case]["text_reference"] for case in args.cases)]
    if any(args.output_dir == path or path in args.output_dir.parents or args.output_dir in path.parents for path in protected):
        raise ValueError("Sweep output must be separate from the read-only image cache and reference benchmarks")
    cases = {case: fingerprint(prepare_case(args, case, images, manifest, baseline)) for case in args.cases}
    contract = dict(format=FORMAT, cases=cases, samples_per_prompt=args.samples_per_prompt,
        max_prompts=args.max_prompts, seed=args.seed, frames_root=str(args.frames_root), image_generation=False,
        shared_contract_sha256=fingerprint(images), shared_image_manifest_sha256=fingerprint(manifest),
        anchor_reference=str(args.anchor_reference), anchor_contract_sha256=fingerprint(baseline[0]),
        anchor_report_sha256=fingerprint(baseline[1]), scope="One 8-GPU allocation; checkpoints run sequentially; original epoch-3 anchor scores reused, not regenerated.")
    bind_contract(args.output_dir, contract)
    print(f"SSD1B sweep prepared: {list(cases)}; read-only cache={args.frames_root}", flush=True)
    return contract


def checked_case(output, case):
    contract = read_json(output / "run_contract.json")
    policy, spec = contract.get("ssd1b_checkpoint_sweep", {}), CASES[case]
    if (contract.get("format") != anchor.FORMAT or contract.get("mode") != MODE
            or contract.get("system") != selected_system(case) or contract.get("external_anchor") is not True
            or contract.get("image_to_smolvlm2") is not False or contract.get("first_frame_generated") is not False
            or contract.get("training_complete") is not (spec["step"] == 473295)
            or contract.get("checkpoint", {}).get("step") != spec["step"]
            or contract["checkpoint"].get("final_step") != 473295
            or policy.get("case") != case or policy.get("eligible_archive") is not True
            or policy.get("image_generation") is not False or policy.get("history_noise_at_inference") is not False):
        raise ValueError(f"Wrong checkpoint/conditioning route in SSD1B sweep: {case}")
    suite = read_json(output.parent / "run_contract.json")
    if suite.get("format") != FORMAT or suite.get("cases", {}).get(case) != fingerprint(contract):
        raise ValueError("Case changed or does not belong to this SSD1B sweep")
    return contract


def verified_controls(contract):
    policy = contract["ssd1b_checkpoint_sweep"]
    results = []
    for prefix in ("text", "anchor"):
        reference = read_reference(Path(policy[f"{prefix}_reference"]))
        if (fingerprint(reference[0]) != policy[f"{prefix}_contract_sha256"]
                or fingerprint(reference[1]) != policy[f"{prefix}_report_sha256"]):
            raise ValueError(f"{prefix} reference changed after preflight")
        results.append(reference)
    return results


def verify_score_assets(output, contract):
    actual = score_assets(read_json(output / "score_assets.json"))
    if any(actual != score_assets(control[2]) for control in verified_controls(contract)):
        raise ValueError("Sweep evaluator weights differ from the existing controls")


def generation_worker(args, *, device=None):
    checked_case(args.output_dir, args.case)
    with case_policy(args.case):
        anchor.generation_worker(args, device=device)


def summarize_case(args):
    from evaluate_vbench_resume import tabulate

    contract = checked_case(args.output_dir, args.case)
    if contract["max_prompts"] or contract["samples_per_prompt"] != 5:
        raise ValueError("No full scores for a subset/single-sample smoke")
    verify_score_assets(args.output_dir, contract)
    with case_policy(args.case):
        anchor.verify_generation(args.output_dir)
    scores = {}
    for variant in base.VARIANTS:
        merged = args.output_dir / variant / "scores/vbench"
        merged.mkdir(parents=True, exist_ok=True)
        for dimension in base.DIMENSIONS:
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


def metric(score, name):
    value = score[name] if name in METRICS else score["raw"][name]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Non-finite score: {name}")
    return value


def summarize_suite(args):
    suite = read_json(args.output_dir / "run_contract.json")
    if suite.get("format") != FORMAT or suite.get("max_prompts") or suite.get("samples_per_prompt") != 5:
        raise ValueError("Need a complete full SSD1B checkpoint sweep")
    baseline = read_reference(Path(suite["anchor_reference"]))
    if (fingerprint(baseline[0]) != suite["anchor_contract_sha256"]
            or fingerprint(baseline[1]) != suite["anchor_report_sha256"]):
        raise ValueError("Original epoch-3 anchor baseline changed")
    reports, rows = {}, []
    for variant in ("no_quicksr", "quicksr"):
        old = baseline[1]["scores"][variant][baseline[0]["system"]]
        rows.append(dict(setting="original_epoch3_anchor (reused)", variant=variant,
                         **{name: metric(old, name) for name in METRICS}, anchor_effect_pp=None, vs_original_anchor_pp=0.))
    for case in suite["cases"]:
        folder = args.output_dir / case
        contract = checked_case(folder, case)
        _, report, _ = read_reference(folder)
        text, original = verified_controls(contract)
        reports[case] = report
        for variant in ("no_quicksr", "quicksr"):
            current = report["scores"][variant][contract["system"]]
            control = text[1]["scores"][variant][text[0]["system"]]
            old = original[1]["scores"][variant][original[0]["system"]]
            rows.append(dict(setting=case, variant=variant, **{name: metric(current, name) for name in METRICS},
                anchor_effect_pp=100 * (current["total_score"] - control["total_score"]),
                vs_original_anchor_pp=100 * (current["total_score"] - old["total_score"])))
    lines = ["# Stage-2 Checkpoints With Identical SSD1B Latent Anchors", "",
             f"Original epoch-3 anchor scores are reused from the verified existing report. {len(suite['cases'])} checkpoints use the same PNG bytes. "
             "SmolVLM2 sees text only; no image regeneration. Raw vs QuickSR also changes resolution.", "",
             "| Setting | Output | Quality (%) | Semantic (%) | Total (%) | Anchor vs Same-Checkpoint Text-Only (pp) | vs Original Epoch-3 Anchor (pp) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        effect = "reference" if row["anchor_effect_pp"] is None else f"{row['anchor_effect_pp']:+.2f}"
        lines.append(f"| {row['setting']} | {row['variant']} | {100*row['quality_score']:.2f} | "
                     f"{100*row['semantic_score']:.2f} | {100*row['total_score']:.2f} | {effect} | {row['vs_original_anchor_pp']:+.2f} |")
    atomic_json(args.output_dir / "comparison.json", dict(status="complete", protocol=suite, rows=rows,
                reused_anchor_reference=baseline[1], reports=reports))
    (args.output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    dimensions = ["# All VBench Dimensions With Identical SSD1B Anchors", ""]
    for variant in ("no_quicksr", "quicksr"):
        dimensions.extend((f"## {variant}", "", "| Score (%) | Original Ep3 Anchor | " + " | ".join(suite["cases"]) + " |",
                           "| --- | ---: | " + " | ".join("---:" for _ in suite["cases"]) + " |"))
        for name in (*METRICS, *base.DIMENSIONS):
            values = [metric(baseline[1]["scores"][variant][baseline[0]["system"]], name)]
            values.extend(metric(reports[case]["scores"][variant][selected_system(case)], name) for case in suite["cases"])
            dimensions.append(f"| {name} | " + " | ".join(f"{100*value:.2f}" for value in values) + " |")
        dimensions.append("")
    (args.output_dir / "dimensions.md").write_text("\n".join(dimensions), encoding="utf-8")
    print(f"SSD1B sweep complete: {args.output_dir / 'comparison.md'}", flush=True)


def parse_args(argv=None):
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--case", choices=tuple(CASES))
    extra.add_argument("--cases", choices=tuple(CASES), nargs="+", default=list(CASES))
    extra.add_argument("--frames-root", type=Path)
    extra.add_argument("--anchor-reference", type=Path, default=ANCHOR_REFERENCE)
    argv = sys.argv[1:] if argv is None else argv
    known, rest = extra.parse_known_args(argv)
    if "--help" in rest or "-h" in rest:
        print(__doc__ + "\n" + extra.format_help())
    args = base.parse_args(rest)
    if any(item == "--checkpoint" or item.startswith("--checkpoint=") for item in rest):
        extra.error("Checkpoints are pinned per case; select --cases instead of overriding --checkpoint")
    if args.allow_incomplete_checkpoint or args.expected_sha256 is not None:
        extra.error("Checkpoints/steps are pinned per case; do not pass global checkpoint admission overrides")
    if len(set(known.cases)) != len(known.cases):
        extra.error("Duplicate checkpoint cases")
    args.case, args.cases, args.mode = known.case, known.cases, MODE
    args.frames_root = known.frames_root.resolve() if known.frames_root else None
    args.anchor_reference = known.anchor_reference.resolve()
    if not any(item == "--output-dir" or item.startswith("--output-dir=") for item in rest):
        suffix = f"_{args.samples_per_prompt}x_dual_smoke{args.max_prompts}" if args.max_prompts else "_5x_dual"
        args.output_dir = DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.name.removesuffix("_5x_dual") + suffix)
    args.suite_root = args.output_dir
    if args.case:
        if args.phase not in ("generation-worker", "score-worker", "summarize"):
            extra.error("--case is for child workers; select --cases for a sweep")
        args.output_dir = args.suite_root / args.case
    elif args.phase in ("generation-worker", "score-worker"):
        extra.error("Child workers require --case")
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
        checked_case(args.output_dir, args.case)
        anchor.score_worker(args)
        return
    if args.phase == "summarize":
        summarize_case(args) if args.case else summarize_suite(args)
        return
    os.chdir(ROOT)
    with output_lock(args.output_dir):
        devices = base.core.gpu_tokens(args.workers) if args.phase == "run" else None
        if args.phase == "run" and not Path(args.vbench_python).is_file():
            raise FileNotFoundError(args.vbench_python)
        suite = prepare_suite(args)
        if args.phase == "prepare":
            return
        entry = str(Path(__file__).resolve())
        if not args.max_prompts:
            subprocess.run([args.vbench_python, entry, "--phase", "warm-scores", "--output-dir", str(args.output_dir),
                            "--vbench-repo", str(args.vbench_repo)], cwd=args.vbench_repo, check=True)
        previous_handler = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            for case in suite["cases"]:
                output = args.output_dir / case
                contract = checked_case(output, case)
                if not args.max_prompts:
                    atomic_json(output / "score_assets.json", read_json(args.output_dir / "score_assets.json"))
                    verify_score_assets(output, contract)
                commands = [[sys.executable, entry, "--phase", "generation-worker", "--case", case,
                             "--output-dir", str(args.output_dir), "--rank", str(rank), "--workers", str(args.workers)]
                            for rank in range(args.workers)]
                base.core.run_processes(commands, devices, output / "logs/generation")
                with case_policy(case):
                    anchor.verify_generation(output)
                if args.max_prompts:
                    continue
                commands = [[args.vbench_python, entry, "--phase", "score-worker", "--case", case,
                             "--output-dir", str(args.output_dir), "--vbench-repo", str(args.vbench_repo),
                             "--rank", str(rank), "--workers", str(args.workers)] for rank in range(args.workers)]
                base.core.run_processes(commands, devices, output / "logs/scoring", cwd=args.vbench_repo)
                subprocess.run([args.vbench_python, entry, "--phase", "summarize", "--case", case,
                                "--output-dir", str(args.output_dir)], cwd=args.vbench_repo, check=True)
            if not args.max_prompts:
                subprocess.run([args.vbench_python, entry, "--phase", "summarize", "--output-dir", str(args.output_dir)],
                               cwd=args.vbench_repo, check=True)
            else:
                print("SSD1B sweep generation smoke passed; no full scores claimed.", flush=True)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
