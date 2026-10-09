#!/usr/bin/env python
"""Native SSD1B -> Hybrid in one released pipeline call; dual raw/QuickSR exports."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_vbench_joint_v2_native as generation  # noqa: E402
import generate_vbench_joint_v2_dual_export as exports  # noqa: E402
from vbench_joint_native_protocol import atomic_json, fingerprint, read_json, sample_items, sha256, shard_items  # noqa: E402


class CaptureFirstFrame:
    """Observe the native image without altering its parameters or consuming RNG."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.image = None
        self.calls = 0

    def __call__(self, *args, **kwargs):
        result = self.pipeline(*args, **kwargs)
        if len(result.images) != 1:
            raise ValueError("Expected one native first-frame image")
        self.image = result.images[0]
        self.calls += 1
        return result

    def __getattr__(self, name):
        return getattr(self.pipeline, name)


def generate_clip(pipeline, capture, item, modifier):
    # Release script uses a single pipeline call. Seed only its entry, never its two components.
    generation.seed_all(item["video_seed"])
    capture.image, capture.calls = None, 0
    frames = pipeline(prompt=item["prompt"], height=320, width=512, num_frames=49,
                      prompt_modifier=modifier, profile=False)
    if capture.calls != 1 or capture.image is None:
        raise ValueError("Native pipeline did not generate exactly one SSD1B first frame")
    return frames, capture.image


def run(args):
    import numpy as np
    import torch
    from new_mobile_ov.generation.quicksrnet import load_quicksrnet

    if args.system != "native_hybrid" or not torch.cuda.is_available():
        raise RuntimeError("Native release generation requires native_hybrid and an allocated GPU")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    shared = read_json(args.output_dir / "run_contract.json")
    if shared.get("format") != "native_neodragon_release_reproduction_v1":
        raise ValueError("This worker cannot consume matched/cinematic benchmark contracts")
    contracts = {name: read_json(args.output_dir / name / "run_contract.json") for name in exports.VARIANTS}
    contract = contracts["quicksr"]
    for name, value in contracts.items():
        if fingerprint(value) != shared["variants"][name]:
            raise ValueError(f"Changed variant contract: {name}")
    for path, key in ((Path(__file__), "generation_code_sha256"),
                      (Path(generation.__file__), "generation_helper_code_sha256"),
                      (Path(exports.__file__), "dual_export_helper_sha256")):
        if sha256(path) != contract[key]:
            raise ValueError(f"Generation code changed after preflight: {path}")
    items = shard_items(list(sample_items(contract["prompts"], contract["samples_per_prompt"], contract["seed"],
                                         contract["seed_mapping"])), args.rank, args.workers)
    pending = [item for item in items if not exports.paired_cached(args.output_dir, contracts, args.system, item)]
    folder = args.output_dir / "quicksr/native_hybrid"
    for variant in exports.VARIANTS:
        for name in ("videos", "records", "anchors", "anchor_records"):
            (args.output_dir / variant / "native_hybrid" / name).mkdir(parents=True, exist_ok=True)
    print(f"Native release rank={args.rank}: assigned={len(items)} pending={len(pending)}", flush=True)
    started = time.monotonic()
    if pending:
        device, dtype = torch.device("cuda:0"), torch.bfloat16
        pipeline = generation.load_pipeline(contract, "native_hybrid", device, dtype)
        capture = CaptureFirstFrame(pipeline.first_frame_gen_pipeline)
        pipeline.first_frame_gen_pipeline = capture
        qsr, _ = load_quicksrnet(variant="medium", scale=2, checkpoint=contract["weights"]["quicksr"]["path"],
                                device=device, dtype=torch.float16)
        modifier = contract["release_policy"]["prompt_modifier"]
        for index, item in enumerate(pending, 1):
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                frames, anchor = generate_clip(pipeline, capture, item, modifier)
            if anchor.size != (contract["anchor"]["width"], contract["anchor"]["height"]):
                raise ValueError(f"Unexpected native first-frame size: {anchor.size}")
            anchor_path = folder / "anchors" / f"{item['key']}.png"
            generation.save_anchor(anchor, anchor_path, folder / "anchor_records" / f"{item['key']}.json",
                                   item, fingerprint(contract))
            array = np.stack([np.asarray(frame.convert("RGB")) for frame in frames])
            exports.export_pair(array, output=args.output_dir, contracts=contracts, system=args.system,
                                item=item, anchor_path=anchor_path, model=qsr, device=device)
            if index == 1 or index % 20 == 0:
                print(f"Native release rank={args.rank}: generated={index}/{len(pending)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    atomic_json(folder / f"worker_{args.rank}.json", {
        "status": "complete", "contract_sha256": fingerprint(contract), "rank": args.rank,
        "assigned": len(items), "generated_pairs": len(pending), "reused_pairs": len(items) - len(pending),
        "seconds": time.monotonic() - started,
    })


if __name__ == "__main__":
    run(generation.parse_args())
