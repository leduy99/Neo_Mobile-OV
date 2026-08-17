#!/usr/bin/env python
"""Audit the native NeoDragon inference contract for a Pyramidal-DMD student.

This is an inference-only diagnostic.  It deliberately uses the release
``generate(..., output_type='latent')`` path as the reference, then compares
it with an independently written explicit rollout that calls the same public
release helpers.  Every system starts from the same global RNG seed and the
same native text condition.

The audit writes full seven-unit latent videos, per-unit/per-stage endpoints,
decoded videos, and motion measurements.  It does not create training data or
modify any checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from diffusers.utils import export_to_video
from einops import rearrange
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.checkpoints import ensure_neodragon_assets
from new_mobile_ov.config import load_config
from tools import audit_neodragon_dmd_motion as motion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("generate", "decode"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dmd-checkpoint",
        required=True,
        help="Native Pyramidal-DMD student checkpoint at the target training step.",
    )
    parser.add_argument("--dmd-step5k-checkpoint", default="")
    parser.add_argument(
        "--prompt",
        default="A vintage red car drives along a coastal road at golden hour, the camera tracking beside it.",
    )
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--dtype", default="bf16")
    return parser.parse_args()


def require_slurm_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This audit needs a CUDA allocation.")
    if not ("SLURM_JOB_ID" in __import__("os").environ or "SLURM_STEP_ID" in __import__("os").environ):
        raise RuntimeError("Run CUDA inference through srun or sbatch, never directly.")


def resolve_native_paths(config_path: str) -> str:
    cfg = load_config(config_path)
    repo_path, _, local_model_path = ensure_neodragon_assets(
        repo_path=cfg.backend.extra.get("repo_path"),
        cache_dir=cfg.backend.extra.get("cache_dir"),
        model_id=cfg.backend.extra.get("model_id", "karnewar/Neodragon"),
        repo_url=cfg.backend.extra.get("repo_url"),
    )
    repo_path = Path(repo_path).expanduser().resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    return str(Path(local_model_path).resolve())


def dtype_from_name(name: str) -> torch.dtype:
    return motion.dtype_from_name(name)


def reset_rng(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_summary(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    values = motion.tensor_metrics(reference, candidate)
    difference = (candidate.float() - reference.float()).abs()
    values.update(
        max_abs=float(difference.max().cpu()),
        exact_fraction=float((difference == 0).float().mean().cpu()),
    )
    return values


def load_dit(path: str, dtype: torch.dtype) -> torch.nn.Module:
    from neodragon.pyramid_mmdit import PyramidMMDiT

    return PyramidMMDiT.from_pretrained(
        path, torch_dtype=dtype, low_cpu_mem_usage=True, device_map="cuda"
    ).eval()


def load_dmd_student(
    path: Path,
    *,
    local_model_path: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schedule") not in {
        "hybrid_1-1-1_video_units_only",
        "pyramidal_1-1-1_all_native_units",
    }:
        raise ValueError(f"Not a native Pyramidal-DMD checkpoint: {path}")
    state = payload.get("student")
    if not isinstance(state, dict):
        raise ValueError(f"Missing student state in {path}")
    teacher_id = str(payload.get("teacher_dit_id", "diffusion_transformer_320p_multistep_t2v"))
    # The caller passes the released model directory separately.  Loading the
    # released base before applying the saved state avoids relying on optimizer
    # or fake-model content in the large resume checkpoint.
    model = load_dit(f"{local_model_path}/{teacher_id}", dtype)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def encode_condition(
    *,
    text_bundle: torch.nn.Module,
    context_adapter: torch.nn.Module,
    prompt: str,
    device: torch.device,
    use_cfg: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from neodragon.utils.generation_utils import DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT_MODIFIER

    positive, positive_mask, positive_pooled = text_bundle(prompt + DEFAULT_PROMPT_MODIFIER, device)
    if use_cfg:
        negative, negative_mask, negative_pooled = text_bundle(DEFAULT_NEGATIVE_PROMPT, device)
        positive = torch.cat((negative, positive), dim=0)
        positive_mask = torch.cat((negative_mask, positive_mask), dim=0)
        positive_pooled = torch.cat((negative_pooled, positive_pooled), dim=0)
    return context_adapter(positive), positive_mask, positive_pooled


def stage_endpoints(intermediates: list[torch.Tensor], steps: list[int]) -> list[torch.Tensor]:
    if len(intermediates) != sum(steps):
        raise RuntimeError(f"Expected {sum(steps)} denoising states, got {len(intermediates)}")
    result: list[torch.Tensor] = []
    offset = 0
    for count in steps:
        offset += int(count)
        result.append(intermediates[offset - 1].detach().cpu())
    return result


@torch.inference_mode()
def released_generate_capture(
    *,
    dit: torch.nn.Module,
    scheduler: Any,
    text_bundle: torch.nn.Module,
    context_adapter: torch.nn.Module,
    prompt: str,
    args: argparse.Namespace,
    first_steps: list[int],
    video_steps: list[int],
    use_cfg: bool,
) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
    """Use the exact public release entry point and capture its stage outputs."""

    import neodragon.utils.generation_utils as generation

    original = generation._generate_one_unit
    captured: list[list[torch.Tensor]] = []

    def wrapped(*wrapped_args, **wrapped_kwargs):
        wrapped_kwargs["show_denoising"] = True
        final, intermediates = original(*wrapped_args, **wrapped_kwargs)
        current_steps = [int(value) for value in wrapped_kwargs["num_inference_steps"]]
        captured.append(stage_endpoints(intermediates, current_steps))
        return final

    generation._generate_one_unit = wrapped
    try:
        reset_rng(args.seed)
        latents = generation.generate(
            text_encoder_bundle=text_bundle,
            dit=dit,
            context_adapter=context_adapter,
            # output_type='latent' accesses only these two configuration values.
            vae=SimpleNamespace(config=SimpleNamespace(temporal_downsample_scale=8, spatial_downsample_scale=8)),
            scheduler=scheduler,
            prompt=args.prompt,
            image=None,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=first_steps,
            video_num_inference_steps=video_steps,
            do_classifier_free_guidance=use_cfg,
            guidance_scale=7.0,
            video_guidance_scale=5.0,
            output_type="latent",
            device=torch.device("cuda"),
            dtype=dtype_from_name(args.dtype),
        )
    finally:
        generation._generate_one_unit = original
    return latents.detach().cpu(), captured


@torch.inference_mode()
def explicit_release_rollout(
    *,
    dit: torch.nn.Module,
    scheduler: Any,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    pooled: torch.Tensor,
    args: argparse.Namespace,
    first_steps: list[int],
    video_steps: list[int],
    use_cfg: bool,
    initial_anchor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
    """Independent explicit rollout, using only released public helpers.

    Resetting the global RNG makes latent noise and corrective block noise match
    the public ``generate`` call exactly.  This validates our diagnostic
    sampler before it is used to judge DMD behaviour.
    """

    import neodragon.utils.generation_utils as generation

    dtype = dtype_from_name(args.dtype)
    device = torch.device("cuda")
    reset_rng(args.seed)
    latents = generation._prepare_latent_noise(
        1,
        int(dit.config.in_channels),
        7,
        args.height // 8,
        args.width // 8,
        dtype,
        device,
    )
    latents = generation._downsample_noise_2x(latents, 2)
    generated: list[torch.Tensor] = [] if initial_anchor is None else [initial_anchor.to(device=device, dtype=dtype)]
    captured: list[list[torch.Tensor]] = []
    for unit in range(len(generated), 7):
        history = generation._prepare_past_condition_latents(
            generated_latents_list=generated,
            num_stages=3,
            do_classifier_free_guidance=use_cfg,
        )
        steps = first_steps if unit == 0 else video_steps
        final, intermediate = generation._generate_one_unit(
            scheduler=scheduler,
            dit=dit,
            num_stages=3,
            latents=latents[:, :, unit : unit + 1],
            past_conditions=history,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_mask,
            pooled_prompt_embeds=pooled,
            num_inference_steps=steps,
            device=device,
            dtype=dtype,
            do_classifier_free_guidance=use_cfg,
            guidance_scale=7.0,
            video_guidance_scale=5.0,
            show_denoising=True,
        )
        generated.append(final)
        captured.append(stage_endpoints(intermediate, steps))
    return torch.cat(generated, dim=2).detach().cpu(), captured


@torch.inference_mode()
def paper_scaled_one_step_rollout(
    *,
    dit: torch.nn.Module,
    scheduler: Any,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    pooled: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
    """Test Eq. 413's literal global-to-local scale at one Euler step.

    This is a diagnostic only.  The released sampler remains the deployment
    reference.  Running the literal printed factor lets us establish whether
    its missing explicit use in the v1 trainer could explain motion collapse.
    """

    import neodragon.utils.generation_utils as generation

    dtype = dtype_from_name(args.dtype)
    device = torch.device("cuda")
    reset_rng(args.seed)
    latents = generation._prepare_latent_noise(
        1,
        int(dit.config.in_channels),
        7,
        args.height // 8,
        args.width // 8,
        dtype,
        device,
    )
    latents = generation._downsample_noise_2x(latents, 2)
    generated: list[torch.Tensor] = []
    captured: list[list[torch.Tensor]] = []
    for unit in range(7):
        history = generation._prepare_past_condition_latents(generated, 3, True)
        stage_outputs: list[torch.Tensor] = []
        current = latents[:, :, unit : unit + 1]
        for stage in range(3):
            if stage > 0:
                current = generation._upsample_pyramidal_latent(
                    latents=current,
                    orig_sigma=1 - scheduler.orig_start_sigmas[stage],
                    gamma=scheduler.config.gamma,
                    device=device,
                    dtype=dtype,
                )
            timestep = scheduler.get_stage_timesteps(1, stage, device=device)[0].expand(2).to(dtype)
            sigma, sigma_next = scheduler.get_stage_sigmas(1, stage, device=device)[:2]
            prediction = dit(
                sample=[history[stage] + [torch.cat((current, current), dim=0)]],
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_mask,
                pooled_projections=pooled,
                timestep_ratio=timestep,
            )[0]
            unconditional, conditional = prediction.chunk(2)
            guidance = 7.0 if unit == 0 else 5.0
            flow = unconditional + guidance * (conditional - unconditional)
            start_sigma = float(scheduler.start_sigmas[stage])
            end_sigma = float(scheduler.end_sigmas[stage])
            paper_scale = start_sigma / (start_sigma - end_sigma)
            current = scheduler.step(
                model_output=flow * paper_scale,
                sigma=sigma.to(dtype=current.dtype),
                sigma_next=sigma_next.to(dtype=current.dtype),
                sample=current,
            ).prev_sample
            stage_outputs.append(current.detach().cpu())
        generated.append(current)
        captured.append(stage_outputs)
    return torch.cat(generated, dim=2).detach().cpu(), captured


def save_rollout(path: Path, *, name: str, latents: torch.Tensor, stages: list[list[torch.Tensor]], seconds: float) -> None:
    torch.save(
        {
            "name": name,
            "latents": latents,
            "stage_endpoints": stages,
            "inference_seconds": seconds,
        },
        path,
    )


def summarize_stages(reference: list[list[torch.Tensor]], candidate: list[list[torch.Tensor]]) -> dict[str, dict[str, float]]:
    if len(reference) != len(candidate):
        raise ValueError("Unit count differs")
    result: dict[str, dict[str, float]] = {}
    for unit, (reference_unit, candidate_unit) in enumerate(zip(reference, candidate)):
        if len(reference_unit) != len(candidate_unit):
            raise ValueError(f"Stage count differs at unit={unit}")
        for stage, (reference_latent, candidate_latent) in enumerate(zip(reference_unit, candidate_unit)):
            result[f"unit{unit}_stage{stage}"] = tensor_summary(reference_latent, candidate_latent)
    return result


@torch.inference_mode()
def run_generate(args: argparse.Namespace, output_dir: Path) -> None:
    local_model_path = resolve_native_paths(args.config)
    from neodragon import MULTISTEP_CONTEXT_ADAPTER_ID, MULTISTEP_DIT_ID
    from neodragon.context_adapter import ContextAdapter
    from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler
    from neodragon.text_encoder_bundle import TextEncoderBundle

    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    scheduler = PyramidFlowMatchEulerDiscreteScheduler()
    text_bundle = TextEncoderBundle.from_pretrained(local_model_path, torch_dtype=dtype).to(device).eval()
    context_adapter = ContextAdapter.from_pretrained(
        f"{local_model_path}/{MULTISTEP_CONTEXT_ADAPTER_ID}", torch_dtype=dtype
    ).to(device).eval()
    for module in (text_bundle, context_adapter):
        module.requires_grad_(False)
    teacher = load_dit(f"{local_model_path}/{MULTISTEP_DIT_ID}", dtype)
    teacher.requires_grad_(False)

    cfg_condition = encode_condition(
        text_bundle=text_bundle, context_adapter=context_adapter, prompt=args.prompt, device=device, use_cfg=True
    )
    cond_condition = encode_condition(
        text_bundle=text_bundle, context_adapter=context_adapter, prompt=args.prompt, device=device, use_cfg=False
    )
    one_step = [1, 1, 1]
    multistep_first = [20, 20, 20]
    multistep_video = [10, 10, 10]
    records: dict[str, tuple[torch.Tensor, list[list[torch.Tensor]], float]] = {}

    start = time.perf_counter()
    records["teacher_official_111_cfg"] = (*released_generate_capture(
        dit=teacher, scheduler=scheduler, text_bundle=text_bundle, context_adapter=context_adapter,
        prompt=args.prompt, args=args, first_steps=one_step, video_steps=one_step, use_cfg=True,
    ), time.perf_counter() - start)

    start = time.perf_counter()
    records["teacher_manual_111_cfg"] = (*explicit_release_rollout(
        dit=teacher, scheduler=scheduler, prompt_embeds=cfg_condition[0], prompt_mask=cfg_condition[1], pooled=cfg_condition[2],
        args=args, first_steps=one_step, video_steps=one_step, use_cfg=True,
    ), time.perf_counter() - start)

    start = time.perf_counter()
    records["teacher_111_cfg_literal_paper_scale"] = (*paper_scaled_one_step_rollout(
        dit=teacher,
        scheduler=scheduler,
        prompt_embeds=cfg_condition[0],
        prompt_mask=cfg_condition[1],
        pooled=cfg_condition[2],
        args=args,
    ), time.perf_counter() - start)

    start = time.perf_counter()
    records["teacher_manual_111_cond"] = (*explicit_release_rollout(
        dit=teacher, scheduler=scheduler, prompt_embeds=cond_condition[0], prompt_mask=cond_condition[1], pooled=cond_condition[2],
        args=args, first_steps=one_step, video_steps=one_step, use_cfg=False,
    ), time.perf_counter() - start)

    start = time.perf_counter()
    records["teacher_manual_multistep_cfg"] = (*explicit_release_rollout(
        dit=teacher, scheduler=scheduler, prompt_embeds=cfg_condition[0], prompt_mask=cfg_condition[1], pooled=cfg_condition[2],
        args=args, first_steps=multistep_first, video_steps=multistep_video, use_cfg=True,
    ), time.perf_counter() - start)

    for name, (latents, stages, seconds) in records.items():
        save_rollout(output_dir / f"{name}.pt", name=name, latents=latents, stages=stages, seconds=seconds)

    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    dmd_paths = [("dmd10k_manual_111_cond", Path(args.dmd_checkpoint))]
    if args.dmd_step5k_checkpoint:
        dmd_paths.insert(0, ("dmd5k_manual_111_cond", Path(args.dmd_step5k_checkpoint)))
    for name, path in dmd_paths:
        if not path.is_file():
            print(f"Skipping {name}; checkpoint not found: {path}", flush=True)
            continue
        student = load_dmd_student(
            path,
            local_model_path=local_model_path,
            dtype=dtype,
            device=device,
        )
        start = time.perf_counter()
        latents, stages = explicit_release_rollout(
            dit=student, scheduler=scheduler, prompt_embeds=cond_condition[0], prompt_mask=cond_condition[1], pooled=cond_condition[2],
            args=args, first_steps=one_step, video_steps=one_step, use_cfg=False,
        )
        seconds = time.perf_counter() - start
        records[name] = (latents, stages, seconds)
        save_rollout(output_dir / f"{name}.pt", name=name, latents=latents, stages=stages, seconds=seconds)
        # The current 10k trainer omitted unit 0 but teacher-forced the six
        # later units.  This control asks whether those six calls can preserve
        # motion when they receive the exact multi-step teacher first unit.
        if name == "dmd10k_manual_111_cond":
            first_anchor = records["teacher_manual_multistep_cfg"][0][:, :, :1]
            start = time.perf_counter()
            anchored_latents, anchored_stages = explicit_release_rollout(
                dit=student,
                scheduler=scheduler,
                prompt_embeds=cond_condition[0],
                prompt_mask=cond_condition[1],
                pooled=cond_condition[2],
                args=args,
                first_steps=one_step,
                video_steps=one_step,
                use_cfg=False,
                initial_anchor=first_anchor,
            )
            anchored_seconds = time.perf_counter() - start
            anchored_name = "dmd10k_111_cond_teacher_first_anchor"
            records[anchored_name] = (anchored_latents, anchored_stages, anchored_seconds)
            save_rollout(
                output_dir / f"{anchored_name}.pt",
                name=anchored_name,
                latents=anchored_latents,
                stages=anchored_stages,
                seconds=anchored_seconds,
            )
        del student
        gc.collect()
        torch.cuda.empty_cache()

    official = records["teacher_official_111_cfg"]
    manual = records["teacher_manual_111_cfg"]
    report: dict[str, Any] = {
        "protocol": {
            "prompt": args.prompt,
            "seed": args.seed,
            "resolution": [args.height, args.width],
            "num_frames": args.num_frames,
            "native_text_condition": True,
            "external_first_frame": False,
            "teacher_multistep": {"first_unit": multistep_first, "video_units": multistep_video, "cfg": [7.0, 5.0]},
            "student": {"all_units": one_step, "cfg": 0.0},
        },
        "sampler_equivalence": {
            "full_video": tensor_summary(official[0], manual[0]),
            "per_unit_stage": summarize_stages(official[1], manual[1]),
        },
        "systems": {},
    }
    teacher_multistep = records["teacher_manual_multistep_cfg"][0]
    teacher_one_cond = records["teacher_manual_111_cond"][0]
    for name, (latents, stages, seconds) in records.items():
        del stages
        report["systems"][name] = {
            "inference_seconds": seconds,
            "latent_motion": motion.latent_motion_metrics(latents),
            "relative_to_teacher_multistep": motion.tensor_metrics(teacher_multistep, latents),
            "relative_to_teacher_111_cond": motion.tensor_metrics(teacher_one_cond, latents),
        }
    (output_dir / "generation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def decode_latents(vae: torch.nn.Module, latents: torch.Tensor, device: torch.device, dtype: torch.dtype) -> np.ndarray:
    return motion.decode_latents(vae, latents.to(device=device, dtype=dtype))


def make_contact_sheet(output_dir: Path, systems: list[tuple[str, list[Image.Image]]]) -> Path:
    columns = 7
    tile_width, tile_height = 256, 160
    header = 30
    canvas = Image.new("RGB", (columns * tile_width, len(systems) * (tile_height + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, (name, frames) in enumerate(systems):
        y = row * (tile_height + header)
        draw.text((4, y + 5), name, fill="black")
        for column, frame in enumerate(frames[:columns]):
            canvas.paste(frame.resize((tile_width, tile_height)), (column * tile_width, y + header))
    path = output_dir / "comparison_contact_sheet.jpg"
    canvas.save(path, quality=92)
    return path


@torch.inference_mode()
def run_decode(args: argparse.Namespace, output_dir: Path) -> None:
    local_model_path = resolve_native_paths(args.config)
    from neodragon.asymmetric_causal_video_vae import AsymmetricCausalVideoVAE

    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    print("Decode phase: loading only the VAE...", flush=True)
    vae = AsymmetricCausalVideoVAE.from_pretrained(
        f"{local_model_path}/causal_video_vae", torch_dtype=dtype
    ).to(device).eval()
    vae.requires_grad_(False)
    decoded: list[tuple[str, list[Image.Image]]] = []
    report: dict[str, Any] = {"systems": {}}
    for path in sorted(output_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "latents" not in payload or "name" not in payload:
            continue
        name = str(payload["name"])
        frames_np = decode_latents(vae, payload["latents"], device, dtype)
        frames = [Image.fromarray(frame) for frame in frames_np]
        video_path = output_dir / f"{name}.mp4"
        export_to_video(frames, video_path, fps=args.fps)
        decoded.append((name, frames))
        report["systems"][name] = {
            "decoded_motion": motion.decoded_motion_metrics(frames_np),
            "video": str(video_path.resolve()),
            "inference_seconds": float(payload.get("inference_seconds", 0.0)),
        }
        print(f"Decoded {name}: {video_path}", flush=True)
    if not decoded:
        raise FileNotFoundError(f"No generated rollout .pt files in {output_dir}")
    contact_sheet = make_contact_sheet(output_dir, decoded)
    report["contact_sheet"] = str(contact_sheet.resolve())
    (output_dir / "decode_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if (args.height, args.width, args.num_frames) != (320, 512, 49):
        raise ValueError("This audit is intentionally fixed to NeoDragon's 320x512, 49-frame native protocol.")
    require_slurm_cuda()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "generate":
        run_generate(args, output_dir)
    else:
        run_decode(args, output_dir)


if __name__ == "__main__":
    main()
