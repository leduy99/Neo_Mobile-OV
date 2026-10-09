#!/usr/bin/env python
"""Generate once, evaluate both direct raw and QuickSR exports on allocated GPUs."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_joint_v2_native as core  # noqa: E402
from vbench_joint_native_protocol import (  # noqa: E402
    DIMENSIONS, SCORE_COSTS, atomic_json, bind_contract, fingerprint, output_lock,
    read_json, sample_items, sha256,
)

VARIANTS = ("quicksr", "no_quicksr")
WORKER = ROOT / "tools/generate_vbench_joint_v2_dual_export.py"


def variant_contracts(template):
    shared = copy.deepcopy(template)
    shared["generation_helper_code_sha256"] = template["generation_code_sha256"]
    shared["generation_code_sha256"] = sha256(WORKER)
    shared["ablation_orchestrator_sha256"] = sha256(Path(__file__))
    pair_sha = fingerprint(shared)
    variants = {}
    for name in VARIANTS:
        contract = copy.deepcopy(shared)
        contract["generation_pair_sha256"] = pair_sha
        contract["postprocessing_variant"] = name
        contract["quicksr"]["enabled"] = name == "quicksr"
        if name == "no_quicksr":
            contract["output"] = copy.deepcopy(contract["base_video"])
            contract["quicksr"] = {"enabled": False, "resize": False}
        contract["scope"] = (
            f"{name}: direct export from the same generated 512x320 RGB frames as the other variant. "
            "Raw stays 512x320; QuickSR medium 2x exports 1024x640. Resolution and super-resolution "
            "are intentionally combined in this comparison; it is not a fixed-resolution SR ablation. "
            "Joint and native use different first-frame generators and denoising schedules. "
            "Not an exact paper replication or mobile-latency comparison."
        )
        variants[name] = contract
    return variants


def prepare(args):
    # Reuse the tested asset/weight checks without changing live single-export jobs.
    asset_args = copy.copy(args)
    asset_args.output_dir = args.output_dir / "_asset_preflight"
    asset_args.reuse_1x_run = []
    template = core.prepare(asset_args)
    variants = variant_contracts(template)
    contract = {
        "format": "mobileov_joint_native_dual_export_v1",
        "selected_systems": template["selected_systems"],
        "variants": {name: fingerprint(value) for name, value in variants.items()},
        "generation_pair_sha256": variants["quicksr"]["generation_pair_sha256"],
        "same_generated_frames": True,
        "raw_resolution": template["base_video"],
        "quicksr_resolution": template["output"],
        "samples_per_prompt": template["samples_per_prompt"],
        "seed_mapping": template["seed_mapping"],
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
    print("Dual export: generate once; evaluate QuickSR 1024x640 and unresized raw 512x320 separately.", flush=True)
    return contract


def score_assignments(workers, systems):
    groups, loads = [[] for _ in range(workers)], [0.] * workers
    tasks = [(variant, system, index) for variant in VARIANTS
             for system in systems for index in range(len(DIMENSIONS))]
    for task in sorted(tasks, key=lambda value: -SCORE_COSTS[value[2]]):
        rank = min(range(workers), key=lambda candidate: (loads[candidate], candidate))
        groups[rank].append(task)
        loads[rank] += SCORE_COSTS[task[2]]
    return groups


def verify_generation(output):
    shared = read_json(output / "run_contract.json")
    contracts = {name: read_json(output / name / "run_contract.json") for name in VARIANTS}
    for name, contract in contracts.items():
        if fingerprint(contract) != shared["variants"][name]:
            raise ValueError(f"Changed variant contract: {name}")
    plan = contracts["quicksr"]
    for system in shared["selected_systems"]:
        for item in sample_items(plan["prompts"], plan["samples_per_prompt"], plan["seed"], plan["seed_mapping"]):
            records = [read_json(output / name / system / "records" / f"{item['key']}.json") for name in VARIANTS]
            source = records[0].get("base_frames_sha256", "")
            if (len(source) != 64 or any(record.get("base_frames_sha256") != source for record in records)
                    or records[0].get("anchor_sha256") != records[1].get("anchor_sha256")):
                raise ValueError(f"Exports do not share generated frames/anchor: {system}/{item['key']}")
    for name, contract in contracts.items():
        core.verify_generation(output / name, contract)
    atomic_json(output / "generation_summary.json", {
        "status": "complete", "contract_sha256": fingerprint(shared),
        "same_generated_frames": True, "systems": shared["selected_systems"],
        "videos_per_system_per_variant": len(plan["prompts"]) * plan["samples_per_prompt"],
    })


def score_worker(args):
    shared = read_json(args.output_dir / "run_contract.json")
    contract = read_json(args.output_dir / "quicksr/run_contract.json")
    for variant, system, index in score_assignments(args.workers, shared["selected_systems"])[args.rank]:
        dimension = DIMENSIONS[index]
        folder = args.output_dir / variant
        command = [sys.executable, str(ROOT / "tools/evaluate_vbench_resume.py"),
                   "--videos", str(folder / system / "videos"),
                   "--full-info", str(folder / "vbench_info.json"),
                   "--output", str(folder / system / "scores" / dimension),
                   "--expected-samples-per-prompt", str(contract["samples_per_prompt"]),
                   "--start", str(index), "--end", str(index + 1)]
        print(f"Scoring {variant}/{system}/{dimension}", flush=True)
        subprocess.run(command, cwd=args.vbench_repo, check=True)


def ablation_report(scores, contract):
    systems = contract["selected_systems"]
    metrics = ("quality_score", "semantic_score", "total_score")
    gains = {}
    for system in systems:
        sr, raw = scores["quicksr"][system], scores["no_quicksr"][system]
        gains[system] = {key: 100 * (sr[key] - raw[key]) for key in metrics}
        gains[system]["raw_dimensions"] = {
            key: 100 * (sr["raw"][key] - raw["raw"][key]) for key in DIMENSIONS
        }
    report = {"status": "complete", "protocol": contract, "scores": scores,
              "quicksr_minus_raw_percentage_points": gains,
              "scope": "Identical generated RGB frames and anchors; QuickSR 1024x640 vs unresized raw 512x320. "
                       "This measures the combined output-resolution/postprocessing effect, not SR at fixed resolution."}
    columns = [(variant, system) for system in systems for variant in VARIANTS]
    labels = {"quicksr": "QuickSR 2x (1024x640)", "no_quicksr": "No QuickSR (512x320)"}
    lines = ["# QuickSR / No-QuickSR VBench", "", report["scope"], "",
             "| Score (%) | " + " | ".join(f"{system}: {labels[variant]}" for variant, system in columns) + " |",
             "| --- | " + " | ".join("---:" for _ in columns) + " |"]
    for key in (*metrics, *DIMENSIONS):
        values = [scores[variant][system][key] if key in metrics else scores[variant][system]["raw"][key]
                  for variant, system in columns]
        lines.append(f"| {key} | " + " | ".join(f"{100 * value:.2f}" for value in values) + " |")
    return report, "\n".join(lines) + "\n"


def summarize(args):
    from evaluate_vbench_resume import tabulate
    from vbench_joint_native_protocol import aggregate_scores

    verify_generation(args.output_dir)
    shared = read_json(args.output_dir / "run_contract.json")
    scores = {}
    for variant in VARIANTS:
        folder = args.output_dir / variant
        contract = read_json(folder / "run_contract.json")
        scores[variant] = {}
        for system in shared["selected_systems"]:
            merged = folder / system / "scores/vbench"
            merged.mkdir(parents=True, exist_ok=True)
            for dimension in DIMENSIONS:
                source = folder / system / "scores" / dimension / "vbench" / f"{dimension}_eval_results.json"
                if dimension not in read_json(source):
                    raise ValueError(f"Missing dimension: {source}")
                shutil.copyfile(source, merged / source.name)
            scores[variant][system] = tabulate(merged)
            subprocess.run([sys.executable, str(ROOT / "tools/report_vbench_low_prompts.py"),
                            "--score-dir", str(merged), "--output-prefix", str(folder / system / "low_prompts")], check=True)
        if len(shared["selected_systems"]) == 2:
            report = aggregate_scores(scores[variant], contract)
            report["scope"] = contract["scope"]
            report["evaluation_assets"] = read_json(args.output_dir / "score_assets.json")
            atomic_json(folder / "comparison.json", report)
        else:
            system = shared["selected_systems"][0]
            atomic_json(folder / "result.json", {"status": "complete", "system": system,
                        "scores": scores[variant][system], "protocol": contract, "scope": contract["scope"]})
    report, markdown = ablation_report(scores, shared)
    report["evaluation_assets"] = read_json(args.output_dir / "score_assets.json")
    atomic_json(args.output_dir / "quicksr_ablation.json", report)
    (args.output_dir / "quicksr_ablation.md").write_text(markdown, encoding="utf-8")
    print(markdown, flush=True)


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = core.parse_args(argv)
    if not any(value == "--output-dir" or value.startswith("--output-dir=") for value in argv):
        args.output_dir = ROOT / "output/vbench_joint_v2_vs_native_5x_dual"
    if args.reuse_1x_run:
        raise ValueError("Old QuickSR-only runs did not retain raw RGB exports. A paired raw/QuickSR run must generate both anew.")
    return args


def main():
    args = parse_args()
    if args.phase == "warm-scores":
        core.warm_scores(args)
        return
    if args.phase == "score-worker":
        score_worker(args)
        return
    if args.phase == "summarize":
        summarize(args)
        return
    os.chdir(ROOT)
    with output_lock(args.output_dir):
        if args.phase == "run":
            devices = core.gpu_tokens(args.workers)
            if not Path(args.vbench_python).is_file():
                raise FileNotFoundError(args.vbench_python)
        shared = prepare(args)
        if args.phase == "prepare":
            return
        if not args.max_prompts:
            subprocess.run([args.vbench_python, str(Path(__file__).resolve()), "--phase", "warm-scores",
                            "--output-dir", str(args.output_dir)], cwd=args.vbench_repo, check=True)
        previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            for system in shared["selected_systems"]:
                commands = [[sys.executable, str(WORKER), "--output-dir", str(args.output_dir),
                             "--system", system, "--rank", str(rank), "--workers", str(args.workers)]
                            for rank in range(args.workers)]
                core.run_processes(commands, devices, args.output_dir / "logs" / system)
            verify_generation(args.output_dir)
            if args.max_prompts:
                print("Dual-export smoke passed; full VBench scoring intentionally skipped.", flush=True)
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
