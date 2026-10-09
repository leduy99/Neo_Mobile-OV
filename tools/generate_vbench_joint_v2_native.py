#!/usr/bin/env python
"""One GPU worker; native SSD1B+Hybrid or V11+Joint V2, with identical QuickSR."""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vbench_joint_native_protocol import (  # noqa: E402
    SYSTEMS, atomic_json, cached_video, fingerprint, read_json, sample_items, sha256, shard_items,
)


def seed_all(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def anchor_cached(path, record_path, item, contract_sha, size):
    from PIL import Image

    try:
        record = read_json(record_path)
        if record.get("sample") != item or record.get("contract_sha256") != contract_sha or record.get("sha256") != sha256(path):
            return False
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size == size
    except (OSError, ValueError, KeyError):
        return False


def save_anchor(image, path, record_path, item, contract_sha):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f".tmp.{os.getpid()}.png")
    try:
        image.convert("RGB").save(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    atomic_json(record_path, {"sample": item, "contract_sha256": contract_sha, "sha256": sha256(path)})


def temporary_video_path(destination, item):
    # Official prompt filenames can approach NAME_MAX; never append to their stem.
    return destination.parent / f".{item['key']}.tmp.{os.getpid()}.mp4"


def load_pipeline(contract, system, device, dtype):
    import torch

    repo = contract["assets"]["native_repo_path"]
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import neodragon as neo
    from new_mobile_ov.generation.neodragon_compat import install_neodragon_generation_patches

    install_neodragon_generation_patches()
    model = Path(contract["assets"]["native_model_dir"])
    mode = "hybrid" if system == "native_hybrid" else "monolithic"
    dit_id = neo.DIT_ID if mode == "hybrid" else neo.MULTISTEP_DIT_ID
    if system == "joint_v2":
        # The trained checkpoint contains the entire DiT; only reuse its release config.
        dit = neo.PyramidMMDiT.from_config(neo.PyramidMMDiT.load_config(str(model / dit_id)))
        payload = torch.load(contract["weights"]["joint"]["path"], map_location="cpu", mmap=True, weights_only=False)
        dit.load_state_dict(payload["dit"], strict=True)
        dit.to(dtype=dtype)
        del payload
        text, context = None, None
    else:
        dit = neo.PyramidMMDiT.from_pretrained(str(model / dit_id), torch_dtype=dtype)
        text = neo.TextEncoderBundle.from_pretrained(str(model), torch_dtype=dtype)
        context = neo.ContextAdapter.from_pretrained(str(model / neo.CONTEXT_ADAPTER_ID), torch_dtype=dtype)
    vae = neo.AsymmetricCausalVideoVAE.from_pretrained(str(model / neo.VAE_ID), torch_dtype=dtype)
    pipeline = neo.NeodragonPipeline(
        text_encoder_bundle=text, context_adapter=context, dit=dit, vae=vae,
        scheduler=neo.PyramidFlowMatchEulerDiscreteScheduler(), model_path=str(model), mode=mode,
    ).to(device)
    for component in ("text_encoder_bundle", "context_adapter", "dit", "vae", "first_frame_gen_pipeline"):
        module = getattr(pipeline, component, None)
        if hasattr(module, "eval"):
            module.eval()
    return pipeline


def joint_anchors(contract, items, folder, contract_sha, device, dtype):
    import torch
    from new_mobile_ov.config import load_config
    from new_mobile_ov.bridge import MobileOVDreamLiteImageBridge
    from new_mobile_ov.generation.backends import DreamLiteMobileBackend

    size = (contract["anchor"]["width"], contract["anchor"]["height"])
    missing = [item for item in items if not anchor_cached(
        folder / "anchors" / f"{item['key']}.png", folder / "anchor_records" / f"{item['key']}.json",
        item, contract_sha, size)]
    if not missing:
        return
    config = load_config(contract["image_config"]["path"])
    payload = torch.load(contract["weights"]["image"]["path"], map_location="cpu", mmap=True, weights_only=False)
    bridge = MobileOVDreamLiteImageBridge(config.bridge, config.dreamlite_bridge, device=device, dtype=dtype).eval()
    bridge.load_trainable_state_dict(payload["bridge"])
    del payload
    backend = DreamLiteMobileBackend(config.dreamlite, device=device, dtype=dtype, load_vae=True)
    for item in missing:
        seed_all(item["anchor_seed"])
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            condition = bridge([item["prompt"]], mode="generate")
            image = backend.generate_images(
                condition, height=size[1], width=size[0], num_steps=4, seed=item["anchor_seed"],
                time_id_height=contract["anchor"]["time_id_height"],
                time_id_width=contract["anchor"]["time_id_width"],
            )[0]
        if image.size != size:
            raise ValueError(f"Unexpected DreamLite anchor size: {image.size}")
        save_anchor(image, folder / "anchors" / f"{item['key']}.png",
                    folder / "anchor_records" / f"{item['key']}.json", item, contract_sha)
    del bridge, backend, condition, image
    gc.collect()
    torch.cuda.empty_cache()


def generate_frames(pipeline, bridge, negative, prompt, anchor, system, device, dtype):
    import torch
    from torch import nn
    from neodragon.utils.generation_utils import DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT_MODIFIER, generate
    from new_mobile_ov.generation.backends.mobile_ov_neodragon import _PrecomputedNeodragonTextEncoder

    if system == "joint_v2":
        with torch.inference_mode():
            tokens, mask, pooled = bridge.encode([prompt + DEFAULT_PROMPT_MODIFIER])
        text = _PrecomputedNeodragonTextEncoder(
            tokens, mask, pooled, negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            negative_prompt_embeds=negative[0], negative_prompt_mask=negative[1], negative_pooled=negative[2],
        ).to(device)
        context = nn.Identity()
    else:
        text, context = pipeline.text_encoder_bundle, pipeline.context_adapter
    return generate(
        text_encoder_bundle=text, dit=pipeline.dit, context_adapter=context,
        vae=pipeline.vae, scheduler=pipeline.scheduler, prompt=prompt, image=anchor,
        height=320, width=512, num_frames=49, output_type="pil", device=device, dtype=dtype,
        prompt_modifier=DEFAULT_PROMPT_MODIFIER, negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        frames_per_unit=pipeline.config.frames_per_unit, num_stages=len(pipeline.config.stages),
        **pipeline.config.gen_confs[pipeline.mode],
    )


def run(args):
    import numpy as np
    import torch
    from diffusers.utils import export_to_video
    from PIL import Image
    from new_mobile_ov.config import load_config
    from new_mobile_ov.bridge import MobileOVNeodragonTextBridge
    from new_mobile_ov.generation.quicksrnet import load_quicksrnet
    from apply_quicksrnet_video import upscale_frames
    from vbench_joint_native_protocol import decode_video

    if not torch.cuda.is_available():
        raise RuntimeError("Generation requires one allocated CUDA device per worker")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    contract = read_json(args.output_dir / "run_contract.json")
    if args.system not in contract.get("selected_systems", SYSTEMS):
        raise ValueError(f"System {args.system} is not selected in this run")
    contract_sha = fingerprint(contract)
    if sha256(Path(__file__)) != contract["generation_code_sha256"]:
        raise ValueError("Worker code changed after preflight")
    items = shard_items(list(sample_items(contract["prompts"], contract["samples_per_prompt"], contract["seed"],
                                         contract.get("seed_mapping", "legacy"))),
                        args.rank, args.workers)
    folder = args.output_dir / args.system
    for name in ("videos", "records", "anchors", "anchor_records"):
        (folder / name).mkdir(parents=True, exist_ok=True)
    shape = contract["output"]
    size = (contract["anchor"]["width"], contract["anchor"]["height"])
    pending = []
    for item in items:
        video, record = folder / "videos" / item["filename"], folder / "records" / f"{item['key']}.json"
        anchor, anchor_record = folder / "anchors" / f"{item['key']}.png", folder / "anchor_records" / f"{item['key']}.json"
        if not (cached_video(video, record, item, args.system, contract_sha, shape)
                and anchor_cached(anchor, anchor_record, item, contract_sha, size)
                and read_json(record).get("anchor_sha256") == sha256(anchor)):
            pending.append(item)
    print(f"{args.system} rank={args.rank}: assigned={len(items)} pending={len(pending)}", flush=True)
    started = time.monotonic()
    if pending:
        device, dtype = torch.device("cuda:0"), torch.bfloat16
        if args.system == "joint_v2":
            joint_anchors(contract, pending, folder, contract_sha, device, dtype)
        pipeline = load_pipeline(contract, args.system, device, dtype)
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
        qsr, _ = load_quicksrnet(
            variant="medium", scale=2, checkpoint=contract["weights"]["quicksr"]["path"],
            device=device, dtype=torch.float16,
        )
        for index, item in enumerate(pending, 1):
            anchor_path = folder / "anchors" / f"{item['key']}.png"
            anchor_record = folder / "anchor_records" / f"{item['key']}.json"
            if args.system == "native_hybrid" and not anchor_cached(anchor_path, anchor_record, item, contract_sha, size):
                seed_all(item["anchor_seed"])
                with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                    anchor = pipeline.first_frame_gen_pipeline(
                        prompt=item["prompt"] + DEFAULT_PROMPT_MODIFIER, num_images_per_prompt=1,
                        generator=torch.Generator(device=device).manual_seed(item["anchor_seed"]),
                    ).images[0]
                if anchor.size != size:
                    raise ValueError(f"Unexpected native SSD1B anchor size: {anchor.size}")
                save_anchor(anchor, anchor_path, anchor_record, item, contract_sha)
            with Image.open(anchor_path) as image:
                anchor = image.convert("RGB")
            # Image-generation RNG consumption must not change the paired video seed.
            seed_all(item["video_seed"])
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                frames = generate_frames(pipeline, bridge, negative, item["prompt"], anchor, args.system, device, dtype)
            array = np.stack([np.asarray(frame.convert("RGB")) for frame in frames])
            if array.shape != (49, 320, 512, 3):
                raise ValueError(f"Unexpected base video shape: {array.shape}")
            with torch.inference_mode(), torch.autocast("cuda", enabled=False):
                upscaled, _ = upscale_frames(array, model=qsr, device=device, dtype=torch.float16,
                                            batch_frames=4, reset_peak_memory=False)
            if upscaled is None or upscaled.shape != (49, 640, 1024, 3):
                raise ValueError("QuickSR did not produce 49 frames at 1024x640")
            destination = folder / "videos" / item["filename"]
            temporary = temporary_video_path(destination, item)
            try:
                export_to_video([Image.fromarray(frame) for frame in upscaled], temporary, fps=24)
                if not decode_video(temporary, shape):
                    raise ValueError(f"Export/decode check failed: {temporary}")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            atomic_json(folder / "records" / f"{item['key']}.json", {
                "sample": item, "system": args.system, "contract_sha256": contract_sha,
                "shape": shape, "video_sha256": sha256(destination), "anchor_sha256": sha256(anchor_path),
            })
            if index == 1 or index % 20 == 0:
                print(f"{args.system} rank={args.rank}: generated={index}/{len(pending)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    atomic_json(folder / f"worker_{args.rank}.json", {
        "status": "complete", "contract_sha256": contract_sha, "rank": args.rank,
        "assigned": len(items), "generated": len(pending), "reused": len(items)-len(pending),
        "seconds": time.monotonic()-started,
    })


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args(argv)
    if args.workers < 1 or not 0 <= args.rank < args.workers:
        parser.error("Invalid rank/workers")
    args.output_dir = args.output_dir.resolve()
    return args


if __name__ == "__main__":
    run(parse_args())
