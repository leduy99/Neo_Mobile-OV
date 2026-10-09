#!/usr/bin/env python
"""Best-effort native release-protocol reproduction, with raw/QuickSR diagnostics."""
from __future__ import annotations

import ast
import copy
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_joint_v2_native as core  # noqa: E402
import benchmark_joint_v2_quicksr_ablation as dual  # noqa: E402
from vbench_joint_native_protocol import (  # noqa: E402
    DIMENSIONS, atomic_json, bind_contract, fingerprint, output_lock, read_json, sha256,
)

WORKER = ROOT / "tools/generate_native_neodragon_release.py"
# Raw dimensions and aggregate percentages from arXiv:2511.06055v1, Table 8.
PAPER_SCORES = {"total_score": 81.61, "quality_score": 83.68, "semantic_score": 73.36,
                "temporal_flickering": 99.27, "aesthetic_quality": 60.71, "imaging_quality": 59.78,
                "object_class": 92.37, "scene": 56.56, "overall_consistency": 28.09}


def released_prompt_modifier(path):
    tree = ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "prompt_modifier"
                                                for target in node.targets)
                and isinstance(node.value, ast.IfExp) and isinstance(node.value.orelse, ast.Constant)
                and isinstance(node.value.orelse.value, str)):
            return node.value.orelse.value
    raise ValueError(f"Cannot identify the released CLI's default prompt modifier in {path}")


def native_contracts(template, modifier, source_script):
    shared = copy.deepcopy(template)
    shared["generation_helper_code_sha256"] = template["generation_code_sha256"]
    shared["generation_code_sha256"] = sha256(WORKER)
    shared["dual_export_helper_sha256"] = sha256(ROOT / "tools/generate_vbench_joint_v2_dual_export.py")
    shared["release_orchestrator_sha256"] = sha256(Path(__file__))
    shared["upstream_inference_script"] = core.file_record(source_script)
    shared["release_policy"] = {
        "prompt_modifier": modifier, "pipeline_call": "NeodragonPipeline.__call__; hybrid",
        "seed_reset": "once before complete SSD1B -> Hybrid call; not between image and video",
        "native_text_conditioner": True, "image_bridge": False, "video_bridge": False,
        "paper_seed_and_sample_count_disclosed": False,
        "public_quicksr_variant_matches_paper_verified": False,
        "generation_dtype": "bf16", "quantized_mobile_pipeline_reproduced": False,
        "benchmark_safety_checker_disabled": True,
        "noise_compatibility_patch": "existing semidefinite 2x2 covariance sampler; CPU RNG",
    }
    shared["seed_policy"] = f"{template['seed_mapping']} mapping; reset once before complete native pipeline call"
    shared["systems"] = {"native_hybrid": "Released native SSD1B + native text conditioner + Hybrid 1-1-1; released CLI prompt modifier"}
    pair_sha = fingerprint(shared)
    variants = {}
    for variant in dual.VARIANTS:
        contract = copy.deepcopy(shared)
        contract["generation_pair_sha256"] = pair_sha
        contract["postprocessing_variant"] = variant
        contract["quicksr"]["enabled"] = variant == "quicksr"
        if variant == "no_quicksr":
            contract["output"] = copy.deepcopy(contract["base_video"])
            contract["quicksr"] = {"enabled": False, "resize": False}
        contract["scope"] = (
            "Best-effort native reproduction using the released CLI modifier and complete native pipeline call. "
            "Fixed per-video seeds/sample count are our reproducibility choices, not a disclosed paper protocol. "
            "Public QuickSR medium 2x and BF16 generation do not reproduce the paper's exact mobile deployment. "
            "Raw 512x320 and QuickSR 1024x640 come from identical generated RGB frames. "
            "No trained Mobile-OV bridge or Joint V2 checkpoint is used."
        )
        variants[variant] = contract
    return variants


def prepare(args):
    asset_args = copy.copy(args)
    asset_args.output_dir = args.output_dir / "_asset_preflight"
    asset_args.reuse_1x_run = []
    template = core.prepare(asset_args)
    source_script = args.native_repo / "scripts/inference_neodragon.py"
    modifier = released_prompt_modifier(source_script)
    variants = native_contracts(template, modifier, source_script)
    contract = {
        "format": "native_neodragon_release_reproduction_v1", "selected_systems": ["native_hybrid"],
        "variants": {name: fingerprint(value) for name, value in variants.items()},
        "generation_pair_sha256": variants["quicksr"]["generation_pair_sha256"],
        "same_generated_frames": True, "samples_per_prompt": template["samples_per_prompt"],
        "release_policy": variants["quicksr"]["release_policy"],
        "paper_reference": {"url": "https://arxiv.org/html/2511.06055v1#S4.T8", "percent": PAPER_SCORES},
    }
    bind_contract(args.output_dir, contract)
    rows = read_json(asset_args.output_dir / "vbench_info.json")
    plan = read_json(asset_args.output_dir / "sample_plan.json")
    for name, value in variants.items():
        folder = args.output_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        digest = bind_contract(folder, value)
        atomic_json(folder / "vbench_info.json", rows)
        atomic_json(folder / "sample_plan.json", {**plan, "contract_sha256": digest})
    print(f"Native release preflight passed: suffix={modifier!r}; one RNG stream through SSD1B and Hybrid.", flush=True)
    print("Generate once; evaluate native raw and public QuickSR 2x. Paper-exact equivalence is NOT asserted.", flush=True)
    return contract


def paper_comparison(output):
    ablation = read_json(output / "quicksr_ablation.json")
    measured = ablation["scores"]["quicksr"]["native_hybrid"]
    rows = {}
    lines = ["# Native NeoDragon: Release-Protocol Reproduction vs Paper", "",
             "Best-effort reproduction, not a paper-exact or quantized-mobile replication. "
             "Our seed plan and public QuickSR medium 2x are explicitly recorded in run_contract.json.", "",
             "Paper reference: https://arxiv.org/html/2511.06055v1#S4.T8", "",
             "| Score (%) | Paper | Native release + QuickSR | Difference (pp) |",
             "| --- | ---: | ---: | ---: |"]
    for key, paper in PAPER_SCORES.items():
        current = 100 * (measured[key] if key.endswith("_score") else measured["raw"][key])
        rows[key] = {"paper_percent": paper, "measured_percent": current, "difference_pp": current - paper}
        lines.append(f"| {key} | {paper:.2f} | {current:.2f} | {current - paper:+.2f} |")
    missing = [key for key in DIMENSIONS if key not in PAPER_SCORES]
    lines.extend(["", "Other paper dimensions are not supplied by Table 8; no values were imputed.",
                  "Raw/no-QuickSR scores are a separate diagnostic in quicksr_ablation.md, not a paper-native baseline."])
    atomic_json(output / "paper_reference_comparison.json", {
        "status": "complete", "scope": "Best-effort released-native reproduction; not paper-exact.",
        "paper_source": "https://arxiv.org/html/2511.06055v1#S4.T8", "rows": rows,
        "unavailable_paper_dimensions": missing, "actual_vbench_results_unmodified": True,
    })
    markdown = "\n".join(lines) + "\n"
    (output / "paper_reference_comparison.md").write_text(markdown, encoding="utf-8")
    print(markdown, flush=True)


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = core.parse_args(argv)
    if any(value == "--systems" or value.startswith("--systems=") for value in argv):
        if args.systems != ["native_hybrid"]:
            raise ValueError("Native release reproduction only runs native_hybrid; no Joint V2 or bridge")
    args.systems = ["native_hybrid"]
    if not any(value == "--output-dir" or value.startswith("--output-dir=") for value in argv):
        args.output_dir = ROOT / "output/vbench_native_neodragon_release_5x"
    if args.reuse_1x_run:
        raise ValueError("Do not reuse previous matched/cinematic clips in a release-protocol reproduction")
    return args


def main():
    args = parse_args()
    if args.phase == "warm-scores":
        core.warm_scores(args)
        return
    if args.phase == "score-worker":
        dual.score_worker(args)
        return
    if args.phase == "summarize":
        dual.summarize(args)
        paper_comparison(args.output_dir)
        return
    os.chdir(ROOT)
    with output_lock(args.output_dir):
        if args.phase == "run":
            devices = core.gpu_tokens(args.workers)
            if not Path(args.vbench_python).is_file():
                raise FileNotFoundError(args.vbench_python)
        prepare(args)
        if args.phase == "prepare":
            return
        if not args.max_prompts:
            subprocess.run([args.vbench_python, str(Path(__file__).resolve()), "--phase", "warm-scores",
                            "--output-dir", str(args.output_dir)], cwd=args.vbench_repo, check=True)
        previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            commands = [[sys.executable, str(WORKER), "--output-dir", str(args.output_dir),
                         "--system", "native_hybrid", "--rank", str(rank), "--workers", str(args.workers)]
                        for rank in range(args.workers)]
            core.run_processes(commands, devices, args.output_dir / "logs/generation")
            dual.verify_generation(args.output_dir)
            if args.max_prompts:
                print("Native release generation smoke passed; scoring intentionally skipped.", flush=True)
                return
            commands = [[args.vbench_python, str(Path(__file__).resolve()), "--phase", "score-worker",
                         "--output-dir", str(args.output_dir), "--vbench-repo", str(args.vbench_repo),
                         "--rank", str(rank), "--workers", str(args.workers)] for rank in range(args.workers)]
            core.run_processes(commands, devices, args.output_dir / "logs/scoring", cwd=args.vbench_repo)
            subprocess.run([args.vbench_python, str(Path(__file__).resolve()), "--phase", "summarize",
                            "--output-dir", str(args.output_dir)], cwd=args.vbench_repo, check=True)
        finally:
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
