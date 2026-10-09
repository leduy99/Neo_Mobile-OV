#!/usr/bin/env python
"""One GPU worker producing two encodings of each identical generated RGB clip."""
from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_vbench_joint_v2_native as generation  # noqa: E402
from vbench_joint_native_protocol import (  # noqa: E402
    atomic_json, cached_video, decode_video, fingerprint, read_json, sample_items, sha256, shard_items,
)

VARIANTS = ("quicksr", "no_quicksr")


def frames_digest(array):
    import numpy as np

    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(f"{array.dtype}:{array.shape}".encode())
    digest.update(memoryview(array))
    return digest.hexdigest()


def paired_cached(output, contracts, system, item):
    records = []
    try:
        for variant in VARIANTS:
            contract = contracts[variant]
            folder = output / variant / system
            record = folder / "records" / f"{item['key']}.json"
            anchor = folder / "anchors" / f"{item['key']}.png"
            anchor_record = folder / "anchor_records" / f"{item['key']}.json"
            digest = fingerprint(contract)
            size = (contract["anchor"]["width"], contract["anchor"]["height"])
            if (not cached_video(folder / "videos" / item["filename"], record, item, system, digest, contract["output"])
                    or not generation.anchor_cached(anchor, anchor_record, item, digest, size)):
                return False
            value = read_json(record)
            if value.get("anchor_sha256") != sha256(anchor):
                return False
            records.append(value)
        return (len(records[0].get("base_frames_sha256", "")) == 64
                and records[0]["base_frames_sha256"] == records[1].get("base_frames_sha256")
                and records[0]["anchor_sha256"] == records[1]["anchor_sha256"])
    except (OSError, KeyError, ValueError):
        return False


def copy_anchor(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.{os.getpid()}.png")
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def export_pair(array, *, output, contracts, system, item, anchor_path, model, device):
    import numpy as np
    import torch
    from diffusers.utils import export_to_video
    from PIL import Image
    from apply_quicksrnet_video import upscale_frames

    base = contracts["no_quicksr"]["output"]
    expected = (base["frames"], base["height"], base["width"], 3)
    if array.dtype != np.uint8 or array.shape != expected:
        raise ValueError(f"Unexpected base RGB frames: {array.dtype} {array.shape}; expected uint8 {expected}")
    base_sha = frames_digest(array)
    with torch.inference_mode(), torch.autocast(device.type, enabled=False):
        sr, _ = upscale_frames(array, model=model, device=device, dtype=torch.float16,
                               batch_frames=4, reset_peak_memory=False)
    sr_shape = contracts["quicksr"]["output"]
    if sr is None or sr.shape != (sr_shape["frames"], sr_shape["height"], sr_shape["width"], 3):
        raise ValueError("QuickSR output shape does not match the contract")
    anchor_sha = sha256(anchor_path)
    records = []
    # Both encoders receive original arrays; the raw video is never downsampled from SR.
    for variant, frames in (("no_quicksr", array), ("quicksr", sr)):
        contract = contracts[variant]
        folder = output / variant / system
        target_anchor = folder / "anchors" / f"{item['key']}.png"
        if variant == "no_quicksr":
            copy_anchor(anchor_path, target_anchor)
            atomic_json(folder / "anchor_records" / f"{item['key']}.json", {
                "sample": item, "contract_sha256": fingerprint(contract), "sha256": anchor_sha,
            })
        destination = folder / "videos" / item["filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = generation.temporary_video_path(destination, item)
        try:
            export_to_video([Image.fromarray(frame) for frame in frames], temporary, fps=contract["output"]["fps"])
            if not decode_video(temporary, contract["output"]):
                raise ValueError(f"Export/decode check failed: {temporary}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        records.append((folder / "records" / f"{item['key']}.json", {
            "sample": item, "system": system, "contract_sha256": fingerprint(contract),
            "shape": contract["output"], "video_sha256": sha256(destination), "anchor_sha256": anchor_sha,
            "base_frames_sha256": base_sha, "postprocessing_variant": variant,
        }))
    # Certify only after both exports decode correctly. An incomplete pair is regenerated together.
    for path, record in records:
        atomic_json(path, record)


def run(args):
    import numpy as np
    import torch
    from PIL import Image
    from new_mobile_ov.bridge import MobileOVNeodragonTextBridge
    from new_mobile_ov.config import load_config
    from new_mobile_ov.generation.quicksrnet import load_quicksrnet

    if not torch.cuda.is_available():
        raise RuntimeError("Dual-export generation requires an allocated CUDA device")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    contracts = {variant: read_json(args.output_dir / variant / "run_contract.json") for variant in VARIANTS}
    shared = read_json(args.output_dir / "run_contract.json")
    contract = contracts["quicksr"]
    if args.system not in shared["selected_systems"]:
        raise ValueError("Unselected generation system")
    for variant, value in contracts.items():
        if fingerprint(value) != shared["variants"][variant]:
            raise ValueError(f"Changed contract: {variant}")
    if (sha256(Path(__file__)) != contract["generation_code_sha256"]
            or sha256(Path(generation.__file__)) != contract["generation_helper_code_sha256"]):
        raise ValueError("Generation code changed after preflight")
    items = shard_items(list(sample_items(contract["prompts"], contract["samples_per_prompt"], contract["seed"],
                                         contract["seed_mapping"])), args.rank, args.workers)
    pending = [item for item in items if not paired_cached(args.output_dir, contracts, args.system, item)]
    folder = args.output_dir / "quicksr" / args.system
    for variant in VARIANTS:
        for name in ("videos", "records", "anchors", "anchor_records"):
            (args.output_dir / variant / args.system / name).mkdir(parents=True, exist_ok=True)
    print(f"{args.system} rank={args.rank}: assigned={len(items)} pending pairs={len(pending)}", flush=True)
    started = time.monotonic()
    if pending:
        device, dtype = torch.device("cuda:0"), torch.bfloat16
        digest = fingerprint(contract)
        size = (contract["anchor"]["width"], contract["anchor"]["height"])
        if args.system == "joint_v2":
            generation.joint_anchors(contract, pending, folder, digest, device, dtype)
        pipeline = generation.load_pipeline(contract, args.system, device, dtype)
        from neodragon.utils.generation_utils import DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT_MODIFIER

        bridge, negative = None, None
        if args.system == "joint_v2":
            config = load_config(contract["video_config"]["path"])
            bridge = MobileOVNeodragonTextBridge(config.bridge, device=device, dtype=dtype).eval()
            payload = torch.load(contract["weights"]["joint"]["path"], map_location="cpu", mmap=True, weights_only=False)
            bridge.load_state_dict(payload["bridge"], strict=True)
            del payload
            with torch.inference_mode():
                negative = bridge.encode([DEFAULT_NEGATIVE_PROMPT])
        qsr, _ = load_quicksrnet(variant="medium", scale=2, checkpoint=contract["weights"]["quicksr"]["path"],
                                device=device, dtype=torch.float16)
        for index, item in enumerate(pending, 1):
            anchor_path = folder / "anchors" / f"{item['key']}.png"
            anchor_record = folder / "anchor_records" / f"{item['key']}.json"
            if args.system == "native_hybrid" and not generation.anchor_cached(anchor_path, anchor_record, item, digest, size):
                generation.seed_all(item["anchor_seed"])
                with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                    anchor = pipeline.first_frame_gen_pipeline(
                        prompt=item["prompt"] + DEFAULT_PROMPT_MODIFIER, num_images_per_prompt=1,
                        generator=torch.Generator(device=device).manual_seed(item["anchor_seed"]),
                    ).images[0]
                if anchor.size != size:
                    raise ValueError(f"Unexpected native anchor size: {anchor.size}")
                generation.save_anchor(anchor, anchor_path, anchor_record, item, digest)
            with Image.open(anchor_path) as image:
                anchor = image.convert("RGB")
            generation.seed_all(item["video_seed"])
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                frames = generation.generate_frames(pipeline, bridge, negative, item["prompt"], anchor,
                                                     args.system, device, dtype)
            array = np.stack([np.asarray(frame.convert("RGB")) for frame in frames])
            export_pair(array, output=args.output_dir, contracts=contracts, system=args.system,
                        item=item, anchor_path=anchor_path, model=qsr, device=device)
            if index == 1 or index % 20 == 0:
                print(f"{args.system} rank={args.rank}: pairs={index}/{len(pending)} elapsed={time.monotonic()-started:.1f}s", flush=True)
        del qsr, pipeline, bridge
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(folder / f"worker_{args.rank}.json", {
        "status": "complete", "rank": args.rank, "contract_sha256": fingerprint(contract),
        "assigned": len(items), "generated_pairs": len(pending), "reused_pairs": len(items) - len(pending),
        "seconds": time.monotonic() - started,
    })


if __name__ == "__main__":
    run(generation.parse_args())
