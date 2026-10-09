#!/usr/bin/env python
"""Cross-swap the exact first-frame PNGs from the completed matched 5x baseline."""
from __future__ import annotations

import argparse
import copy
import gc
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_joint_v2_native as core  # noqa: E402
import benchmark_joint_v2_quicksr_ablation as dual  # noqa: E402
import generate_vbench_joint_v2_native as generation  # noqa: E402
import generate_vbench_joint_v2_dual_export as exports  # noqa: E402
from reuse_vbench_1x_samples import install_media  # noqa: E402
from vbench_joint_native_protocol import (  # noqa: E402
    DIMENSIONS, SYSTEMS, atomic_json, bind_contract, fingerprint, output_lock,
    read_json, sample_items, sha256, shard_items,
)

SWAPS = {
    "joint_v2_ssd1b": {"video": "joint_v2", "image": "native_hybrid", "image_label": "native SSD1B"},
    "native_hybrid_dreamlite_v11": {"video": "native_hybrid", "image": "joint_v2", "image_label": "V11 image bridge + DreamLite"},
}
BASELINE = ROOT / "output/vbench_joint_v2_vs_native_quicksr_berzelius_5x_dual"


def verify_baseline_settings(template, parent, baseline):
    if parent.get("format") != "mobileov_joint_native_dual_export_v1" or set(parent.get("selected_systems", [])) != set(SYSTEMS):
        raise ValueError("Use the matched Joint-V2/native dual baseline, NOT native-release 83.35 or a Stage-2 run")
    if fingerprint(baseline) != parent.get("variants", {}).get("quicksr"):
        raise ValueError("Baseline variant contract changed")
    if baseline["prompts"][:len(template["prompts"])] != template["prompts"]:
        raise ValueError("Baseline prompt order differs")
    for name in ("samples_per_prompt", "seed", "seed_mapping", "anchor", "base_video", "output", "full_info_sha256",
                 "compat_code_sha256", "score_code_sha256", "vbench_source_commit", "systems"):
        if baseline.get(name) != template.get(name):
            raise ValueError(f"Matched baseline differs: {name}")
    if {key: value for key, value in baseline["quicksr"].items() if key != "enabled"} != template["quicksr"]:
        raise ValueError("QuickSR settings differ")
    for name in ("image_config", "video_config"):
        if baseline[name]["sha256"] != template[name]["sha256"]:
            raise ValueError(f"Config differs: {name}")
    for name in ("joint", "image", "smolvlm2", "quicksr"):
        if baseline["weights"][name]["sha256"] != template["weights"][name]["sha256"]:
            raise ValueError(f"Weights differ: {name}")
    for name in ("native_source_commit", "native_source_sha256", "native_files", "dreamlite_files"):
        if baseline["assets"][name] != template["assets"][name]:
            raise ValueError(f"Native/DreamLite assets differ: {name}")
    if baseline.get("generation_helper_code_sha256") != template["generation_code_sha256"]:
        raise ValueError("Video generation helper differs from the matched baseline")


def source_anchors(template, baseline_root, case):
    parent = read_json(baseline_root / "run_contract.json")
    baseline = read_json(baseline_root / "quicksr/run_contract.json")
    receipt = read_json(baseline_root / "generation_summary.json")
    if receipt.get("status") != "complete" or receipt.get("contract_sha256") != fingerprint(parent):
        raise ValueError("Baseline generation is not certified complete")
    verify_baseline_settings(template, parent, baseline)
    folder = baseline_root / "quicksr" / SWAPS[case]["image"]
    records = []
    for item in sample_items(template["prompts"], template["samples_per_prompt"], template["seed"], template["seed_mapping"]):
        image = folder / "anchors" / f"{item['key']}.png"
        metadata = folder / "anchor_records" / f"{item['key']}.json"
        if not image.is_file() or not metadata.is_file():
            raise FileNotFoundError(f"Original baseline anchor missing: {image}. Restore baseline anchors; do not regenerate them for this comparison.")
        value = read_json(metadata)
        expected_size = (template["anchor"]["width"], template["anchor"]["height"])
        if not generation.anchor_cached(image, metadata, item, fingerprint(baseline), expected_size):
            raise ValueError(f"Unverified baseline first frame: {image}")
        records.append(dict(key=item["key"], path=str(image), sha256=value["sha256"], source_record_sha256=sha256(metadata)))
    return baseline, records


def prepare(args):
    asset_args = copy.copy(args)
    asset_args.output_dir = args.output_dir / "_asset_preflight"
    asset_args.systems = list(SYSTEMS)
    asset_args.reuse_1x_run = []
    template = core.prepare(asset_args)
    baseline, anchors = source_anchors(template, args.baseline_run, args.swap)
    cell = SWAPS[args.swap]
    shared = copy.deepcopy(template)
    shared.update(selected_systems=[cell["video"]], generation_helper_code_sha256=template["generation_code_sha256"],
                  generation_code_sha256=sha256(Path(__file__)), dual_export_helper_sha256=sha256(Path(exports.__file__)),
                  anchor_swap=dict(case=args.swap, video_system=cell["video"], image_system=cell["image"], image_label=cell["image_label"],
                                   baseline_run=str(args.baseline_run), baseline_contract_sha256=fingerprint(baseline),
                                   source_anchors=anchors, original_anchor_bytes_preserved=True,
                                   image_regeneration=False, video_rng_policy="same independent reset as matched baseline"))
    pair_sha = fingerprint(shared)
    contracts = {}
    for variant in dual.VARIANTS:
        contract = copy.deepcopy(shared)
        contract.update(generation_pair_sha256=pair_sha, postprocessing_variant=variant)
        contract["quicksr"]["enabled"] = variant == "quicksr"
        if variant == "no_quicksr":
            contract["output"] = copy.deepcopy(contract["base_video"])
            contract["quicksr"] = dict(enabled=False, resize=False)
        contract["scope"] = (
            f"{args.swap}: only the first-frame image is replaced with the opposite matched-baseline image. "
            "Video weights, text conditioner, prompt modifier, CFG/schedule and video seed remain unchanged. "
            "This measures the first-frame stack/compatibility effect, not image-bridge error alone. "
            "Raw and QuickSR use identical generated frames. Compare to the matched baseline, NOT the native-release protocol."
        )
        contracts[variant] = contract
    parent = dict(format="mobileov_first_frame_swap_v1", selected_systems=[cell["video"]],
                  variants={name: fingerprint(value) for name, value in contracts.items()},
                  swap=args.swap, baseline_run=str(args.baseline_run), original_anchor_bytes_preserved=True,
                  generation_pair_sha256=pair_sha)
    bind_contract(args.output_dir, parent)
    rows = read_json(asset_args.output_dir / "vbench_info.json")
    plan = read_json(asset_args.output_dir / "sample_plan.json")
    for variant, value in contracts.items():
        folder = args.output_dir / variant
        folder.mkdir(parents=True, exist_ok=True)
        digest = bind_contract(folder, value)
        atomic_json(folder / "vbench_info.json", rows)
        atomic_json(folder / "sample_plan.json", {**plan, "contract_sha256": digest})
    print(f"Anchor swap ready: {cell['image_label']} -> {cell['video']}; {len(anchors)} original PNGs verified; no image regeneration.", flush=True)
    return parent


def supplied_anchor(output, contract, item):
    matches = [row for row in contract["anchor_swap"]["source_anchors"] if row["key"] == item["key"]]
    if len(matches) != 1:
        raise ValueError("Missing/duplicate source anchor mapping")
    source = matches[0]
    image = Path(source["path"])
    if sha256(image) != source["sha256"]:
        raise ValueError(f"Original anchor changed since preflight: {image}")
    folder = output / "quicksr" / contract["anchor_swap"]["video_system"]
    target = folder / "anchors" / f"{item['key']}.png"
    install_media(image, target, source["sha256"])
    atomic_json(folder / "anchor_records" / f"{item['key']}.json", {
        "sample": item, "sha256": source["sha256"], "contract_sha256": fingerprint(contract),
        "source_baseline_anchor": source,
    })
    return target


def generation_worker(args):
    import numpy as np
    from PIL import Image
    import torch
    from new_mobile_ov.bridge import MobileOVNeodragonTextBridge
    from new_mobile_ov.config import load_config
    from new_mobile_ov.generation.quicksrnet import load_quicksrnet

    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Run anchor-swap generation inside a SLURM GPU allocation")
    torch.set_num_threads(4)
    parent = read_json(args.output_dir / "run_contract.json")
    contracts = {name: read_json(args.output_dir / name / "run_contract.json") for name in dual.VARIANTS}
    contract = contracts["quicksr"]
    for name, value in contracts.items():
        if fingerprint(value) != parent["variants"][name]:
            raise ValueError("Variant contract changed")
    for path, key in ((Path(__file__), "generation_code_sha256"), (Path(generation.__file__), "generation_helper_code_sha256"),
                      (Path(exports.__file__), "dual_export_helper_sha256")):
        if sha256(path) != contract[key]:
            raise ValueError(f"Worker/helper code changed: {path}")
    system = contract["anchor_swap"]["video_system"]
    items = shard_items(list(sample_items(contract["prompts"], contract["samples_per_prompt"], contract["seed"], contract["seed_mapping"])),
                        args.rank, args.workers)
    pending = [item for item in items if not exports.paired_cached(args.output_dir, contracts, system, item)]
    print(f"{parent['swap']} rank={args.rank}: assigned={len(items)} pending={len(pending)}", flush=True)
    started = time.monotonic()
    if pending:
        device, dtype = torch.device("cuda:0"), torch.bfloat16
        pipeline = generation.load_pipeline(contract, system, device, dtype)
        # Native Hybrid's __call__ would regenerate SSD1B and ignore a supplied image.
        # Always call generate_frames -> generate(image=...) below; release unused SSD modules.
        pipeline.first_frame_gen_pipeline = None
        gc.collect()
        torch.cuda.empty_cache()
        bridge, negative = None, None
        if system == "joint_v2":
            config = load_config(contract["video_config"]["path"])
            bridge = MobileOVNeodragonTextBridge(config.bridge, device=device, dtype=dtype).eval()
            payload = torch.load(contract["weights"]["joint"]["path"], map_location="cpu", mmap=True, weights_only=False)
            bridge.load_state_dict(payload["bridge"], strict=True)
            del payload
            from neodragon.utils.generation_utils import DEFAULT_NEGATIVE_PROMPT
            with torch.inference_mode():
                negative = bridge.encode([DEFAULT_NEGATIVE_PROMPT])
        model, _ = load_quicksrnet(variant="medium", scale=2, checkpoint=contract["weights"]["quicksr"]["path"],
                                  device=device, dtype=torch.float16)
        for number, item in enumerate(pending, 1):
            path = supplied_anchor(args.output_dir, contract, item)
            with Image.open(path) as image:
                anchor = image.convert("RGB")
            generation.seed_all(item["video_seed"])
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                frames = generation.generate_frames(pipeline, bridge, negative, item["prompt"], anchor, system, device, dtype)
            array = np.stack([np.asarray(frame.convert("RGB")) for frame in frames])
            exports.export_pair(array, output=args.output_dir, contracts=contracts, system=system, item=item,
                                anchor_path=path, model=model, device=device)
            if number == 1 or number % 20 == 0:
                print(f"{parent['swap']} rank={args.rank}: pairs={number}/{len(pending)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    atomic_json(args.output_dir / "worker_receipts" / f"rank_{args.rank}.json", dict(
        status="complete", contract_sha256=fingerprint(parent), assigned=len(items), generated_pairs=len(pending),
        reused_pairs=len(items) - len(pending), seconds=time.monotonic() - started))


def anchor_effect(args):
    current = read_json(args.output_dir / "quicksr_ablation.json")
    contract = read_json(args.output_dir / "quicksr/run_contract.json")
    cell = contract["anchor_swap"]
    source = Path(cell["baseline_run"])
    baseline = read_json(source / "quicksr_ablation.json")
    if read_json(source / "score_assets.json") != read_json(args.output_dir / "score_assets.json"):
        raise ValueError("VBench evaluation weights differ from the baseline")
    system = cell["video_system"]
    metrics = ("quality_score", "semantic_score", "total_score")
    results = {}
    lines = [f"# First-Frame Effect: {cell['case']}", "", contract["scope"], "",
             f"Baseline video stack: {system}. Swapped first frame: {cell['image_label']}.", "",
             "| Score (%) | Baseline Raw | Swapped Raw | Change (pp) | Baseline QuickSR | Swapped QuickSR | Change (pp) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key in (*metrics, *DIMENSIONS):
        row = {}
        values = []
        for variant in ("no_quicksr", "quicksr"):
            a, b = baseline["scores"][variant][system], current["scores"][variant][system]
            old = a[key] if key in metrics else a["raw"][key]
            new = b[key] if key in metrics else b["raw"][key]
            row[variant] = dict(baseline=old, swapped=new, change_pp=100 * (new - old))
            values.extend((f"{100*old:.2f}", f"{100*new:.2f}", f"{100*(new-old):+.2f}"))
        results[key] = row
        lines.append(f"| {key} | " + " | ".join(values) + " |")
    atomic_json(args.output_dir / "anchor_effect.json", dict(case=cell["case"], baseline=str(source),
                scope=contract["scope"], rows=results, original_baseline_anchors_reused=True))
    (args.output_dir / "anchor_effect.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--swap", choices=tuple(SWAPS), default="joint_v2_ssd1b")
    extra.add_argument("--baseline-run", type=Path, default=BASELINE)
    extra.add_argument("--phase", choices=("run", "prepare", "generation-worker", "warm-scores", "score-worker", "summarize"), default="run")
    known, rest = extra.parse_known_args(argv)
    phase = "prepare" if known.phase == "generation-worker" else known.phase
    args = core.parse_args(["--phase", phase, *rest])
    args.phase, args.swap, args.baseline_run = known.phase, known.swap, known.baseline_run.resolve()
    if not any(arg == "--output-dir" or arg.startswith("--output-dir=") for arg in rest):
        args.output_dir = ROOT / f"output/vbench_{args.swap}_5x_dual"
    if args.reuse_1x_run:
        raise ValueError("Only original baseline PNGs are reused, never old swapped videos")
    args.seed_mapping = "extendable"
    return args


def main():
    args = parse_args()
    if args.phase == "generation-worker":
        generation_worker(args)
        return
    if args.phase == "warm-scores":
        core.warm_scores(args)
        return
    if args.phase == "score-worker":
        dual.score_worker(args)
        return
    if args.phase == "summarize":
        dual.summarize(args)
        anchor_effect(args)
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
            commands = [[sys.executable, str(Path(__file__).resolve()), "--phase", "generation-worker",
                         "--output-dir", str(args.output_dir), "--rank", str(rank), "--workers", str(args.workers)] for rank in range(args.workers)]
            core.run_processes(commands, devices, args.output_dir / "logs/generation")
            dual.verify_generation(args.output_dir)
            if args.max_prompts:
                print("Anchor-swap generation smoke passed; no full VBench score claimed.", flush=True)
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
