#!/usr/bin/env python
# ruff: noqa: E402
"""Generate the synthetic monolithic-teacher latent set used by Pyramidal DMD.

NeoDragon's paper distils a one-step Hybrid DiT from samples produced by the
multi-step teacher.  This program deliberately consumes *prompts only* and
writes the teacher's seven latent units.  The first unit becomes the anchor;
the DMD trainer learns the remaining six autoregressive video units.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.checkpoints import ensure_neodragon_assets
from new_mobile_ov.config import load_config
from new_mobile_ov.training.distributed import (
    barrier,
    cleanup_distributed,
    rank0_print,
    setup_distributed,
)


PROMPT_COLUMNS = (
    "caption_long",
    "caption_medium",
    "caption",
    "prompt",
    "text",
    "caption_short",
)


def dtype_from_name(name: str) -> torch.dtype:
    value = str(name).lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def load_prompts(path: Path, *, prompt_column: str, max_prompts: int, seed: int) -> list[str]:
    if path.suffix.lower() not in {".csv", ".tsv"}:
        values = [normalize_text(line) for line in path.read_text(encoding="utf-8").splitlines()]
        values = [value for value in values if value]
    else:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if reader.fieldnames is None:
                raise ValueError(f"Prompt file has no header: {path}")
            column = prompt_column
            if not column:
                column = next((name for name in PROMPT_COLUMNS if name in reader.fieldnames), "")
            if not column:
                raise ValueError(
                    f"Could not select a prompt column in {path}; available={reader.fieldnames}"
                )
            values = [normalize_text(row.get(column, "")) for row in reader]
            values = [value for value in values if value]

    # The paper uses a curated prompt set. Until that exact set is released,
    # deterministic de-duplication and sampling makes our substitute auditable.
    values = list(dict.fromkeys(values))
    if max_prompts > 0 and len(values) > max_prompts:
        rng = random.Random(seed)
        values = rng.sample(values, int(max_prompts))
    if not values:
        raise ValueError(f"No non-empty prompts found in {path}")
    return values


def load_teacher(cfg, *, device: torch.device, dtype: torch.dtype):
    repo_path, _, local_model_path = ensure_neodragon_assets(
        repo_path=cfg.backend.extra.get("repo_path"),
        cache_dir=cfg.backend.extra.get("cache_dir"),
        model_id=cfg.backend.extra.get("model_id", "karnewar/Neodragon"),
        repo_url=cfg.backend.extra.get("repo_url"),
    )
    repo_path = Path(repo_path).expanduser().resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    from neodragon import MULTISTEP_CONTEXT_ADAPTER_ID, MULTISTEP_DIT_ID, VAE_ID
    from neodragon.asymmetric_causal_video_vae import AsymmetricCausalVideoVAE
    from neodragon.context_adapter import ContextAdapter
    from neodragon.pyramid_mmdit import PyramidMMDiT
    from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler
    from neodragon.text_encoder_bundle import TextEncoderBundle
    from neodragon.utils.generation_utils import (
        DEFAULT_NEGATIVE_PROMPT,
        DEFAULT_PROMPT_MODIFIER,
        generate,
    )

    text = TextEncoderBundle.from_pretrained(local_model_path, torch_dtype=dtype).to(device).eval()
    adapter = ContextAdapter.from_pretrained(
        f"{local_model_path}/{MULTISTEP_CONTEXT_ADAPTER_ID}", torch_dtype=dtype
    ).to(device).eval()
    dit = PyramidMMDiT.from_pretrained(
        f"{local_model_path}/{MULTISTEP_DIT_ID}", torch_dtype=dtype
    ).to(device).eval()
    vae = AsymmetricCausalVideoVAE.from_pretrained(
        f"{local_model_path}/{VAE_ID}", torch_dtype=dtype
    ).to(device).eval()
    for module in (text, adapter, dit, vae):
        module.requires_grad_(False)
    return {
        "text": text,
        "adapter": adapter,
        "dit": dit,
        "vae": vae,
        "scheduler": PyramidFlowMatchEulerDiscreteScheduler(),
        "generate": generate,
        "prompt_modifier": DEFAULT_PROMPT_MODIFIER,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "model_path": str(local_model_path),
    }


def latent_path(output_dir: Path, index: int) -> Path:
    return output_dir / "latents" / f"sample_{index:07d}.pt"


def valid_latent(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        latent = payload.get("latent") if isinstance(payload, dict) else payload
        return isinstance(latent, torch.Tensor) and latent.ndim == 4 and latent.shape[1] >= 7
    except Exception:
        return False


def atomic_save(payload: dict[str, object], destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--prompts", required=True, help="CSV/TSV/text prompt source; no videos are read.")
    parser.add_argument("--prompt-column", default="")
    parser.add_argument("--output-dir", default="data/neodragon_pyramidal_dmd_320p")
    parser.add_argument("--max-prompts", type=int, default=350000)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--first-unit-steps", type=int, default=20)
    parser.add_argument("--video-unit-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = setup_distributed()
    try:
        cfg = load_config(args.config)
        output_dir = Path(args.output_dir)
        output_dir.joinpath("latents").mkdir(parents=True, exist_ok=True)
        prompt_source = Path(args.prompts)
        prompts = load_prompts(
            prompt_source,
            prompt_column=args.prompt_column,
            max_prompts=args.max_prompts,
            seed=args.seed,
        )
        rank_indices = list(range(ctx.rank, len(prompts), ctx.world_size))
        teacher = load_teacher(cfg, device=ctx.device, dtype=dtype_from_name(args.dtype))
        rank0_print(
            ctx,
            "NeoDragon DMD synthetic generation: "
            f"samples={len(prompts)} ranks={ctx.world_size} local_samples={len(rank_indices)} "
            f"teacher_steps={args.first_unit_steps}/{args.video_unit_steps} resolution={args.height}x{args.width}",
        )

        completed = 0
        iterator = tqdm(rank_indices, disable=not ctx.is_main, desc="Monolithic teacher synthetic latents")
        for index in iterator:
            path = latent_path(output_dir, index)
            if not args.overwrite and valid_latent(path):
                completed += 1
                continue
            # generate() does not expose a generator. Isolating the global RNG
            # gives every sample a deterministic seed independent of rank/order.
            with torch.random.fork_rng(devices=[ctx.device]):
                torch.manual_seed(int(args.seed) + index)
                torch.cuda.manual_seed_all(int(args.seed) + index)
                latent = teacher["generate"](
                    teacher["text"],
                    teacher["dit"],
                    teacher["adapter"],
                    teacher["vae"],
                    teacher["scheduler"],
                    prompt=prompts[index],
                    image=None,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    num_inference_steps=[args.first_unit_steps] * 3,
                    video_num_inference_steps=[args.video_unit_steps] * 3,
                    do_classifier_free_guidance=True,
                    guidance_scale=7.0,
                    video_guidance_scale=5.0,
                    prompt_modifier=teacher["prompt_modifier"],
                    negative_prompt=teacher["negative_prompt"],
                    output_type="latent",
                    device=ctx.device,
                    dtype=dtype_from_name(args.dtype),
                )
            latent = latent.detach().cpu().contiguous()
            if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[2] != 7:
                raise RuntimeError(
                    f"Expected teacher latent [1,C,7,H,W], got {tuple(latent.shape)} for index={index}"
                )
            atomic_save(
                {
                    "latent": latent[0],
                    "index": index,
                    "prompt": prompts[index],
                    "condition_prompt": prompts[index] + str(teacher["prompt_modifier"]),
                    "teacher": "released_neodragon_multistep_20_10_cfg",
                    "seed": int(args.seed) + index,
                },
                path,
            )
            completed += 1
            if completed % 25 == 0:
                iterator.set_postfix(saved=completed)

        barrier()
        if ctx.is_main:
            manifest = output_dir / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("index", "latent_path", "prompt", "condition_prompt"),
                )
                writer.writeheader()
                ready = 0
                for index, prompt in enumerate(prompts):
                    path = latent_path(output_dir, index)
                    if valid_latent(path):
                        writer.writerow(
                            {
                                "index": index,
                                "latent_path": str(path.relative_to(output_dir)),
                                "prompt": prompt,
                                "condition_prompt": prompt + str(teacher["prompt_modifier"]),
                            }
                        )
                        ready += 1
            metadata = {
                "objective": "neodragon_pyramidal_dmd_synthetic_teacher_data",
                "prompt_source": str(prompt_source),
                "prompt_count_requested": len(prompts),
                "prompt_count_ready": ready,
                "teacher": "released_neodragon_multistep_20_10_cfg",
                "native_model_path": teacher["model_path"],
                "height": args.height,
                "width": args.width,
                "num_frames": args.num_frames,
                "first_unit_steps": args.first_unit_steps,
                "video_unit_steps": args.video_unit_steps,
                "seed": args.seed,
            }
            (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            rank0_print(ctx, f"Wrote {ready}/{len(prompts)} valid synthetic teacher samples to {manifest}")
        barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
