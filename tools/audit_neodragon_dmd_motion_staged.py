#!/usr/bin/env python
"""Run the DMD motion audit in memory-bounded local phases.

Each phase holds only the modules it needs. This is intentionally equivalent
to ``audit_neodragon_dmd_motion.py`` but avoids loading teacher, student,
SSD1B, and VAE in one cgroup-constrained process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.checkpoints import ensure_neodragon_assets
from new_mobile_ov.config import load_config
from new_mobile_ov.training.neodragon_pyramidal_dmd import DMDCondition
from tools import audit_neodragon_dmd_motion as audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("teacher", "student", "ssd", "hybrid", "decode"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--prompt",
        default="An astronaut walks slowly across the red surface of Mars as dust blows behind them.",
    )
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--first-steps", type=int, default=20)
    parser.add_argument("--video-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--dtype", default="bf16")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The DMD motion audit requires CUDA.")
    if (args.height, args.width, args.num_frames) != (320, 512, 49):
        raise ValueError("This audit is pinned to NeoDragon's released 320x512, 49-frame protocol.")


def native_paths(cfg) -> tuple[Path, str]:
    repo_path, _, local_model_path = ensure_neodragon_assets(
        repo_path=cfg.backend.extra.get("repo_path"),
        cache_dir=cfg.backend.extra.get("cache_dir"),
        model_id=cfg.backend.extra.get("model_id", "karnewar/Neodragon"),
        repo_url=cfg.backend.extra.get("repo_url"),
    )
    repo_path = Path(repo_path).expanduser().resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    return repo_path, str(local_model_path)


def checkpoint_info(path: Path) -> tuple[dict[str, object], dict[str, torch.Tensor] | None]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schedule") != "hybrid_1-1-1_video_units_only":
        raise ValueError(f"Not a Pyramidal-DMD student checkpoint: {path}")
    info = {
        "step": int(payload.get("step", -1)),
        "adapter_id": str(payload["context_adapter_id"]),
        "dit_id": str(payload["teacher_dit_id"]),
    }
    return info, payload.get("student")


def save_rollout(path: Path, rollout: audit.Rollout, *, prompt: str, metadata: dict[str, object]) -> None:
    trace = []
    for (unit, stage), value in rollout.trace.items():
        trace.append(
            {
                "unit": unit,
                "stage": stage,
                "start": value.start.cpu(),
                "end": value.end.cpu(),
                "history": [item.cpu() for item in value.history],
                "update_rms": value.update_rms,
                "flow_rms": value.flow_rms,
            }
        )
    torch.save({"prompt": prompt, "metadata": metadata, "latents": rollout.latents.cpu(), "trace": trace}, path)


def load_rollout(path: Path, device: torch.device, dtype: torch.dtype) -> tuple[str, dict[str, object], audit.Rollout]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    trace = {}
    for row in payload["trace"]:
        trace[(int(row["unit"]), int(row["stage"]))] = audit.Trace(
            start=row["start"].to(device=device, dtype=dtype),
            end=row["end"].to(device=device, dtype=dtype),
            history=tuple(item.to(device=device, dtype=dtype) for item in row["history"]),
            update_rms=float(row["update_rms"]),
            flow_rms=float(row["flow_rms"]),
        )
    return str(payload["prompt"]), dict(payload["metadata"]), audit.Rollout(
        latents=payload["latents"].to(device=device, dtype=dtype), trace=trace
    )


def save_student_rollouts(path: Path, values: dict[str, audit.Rollout]) -> None:
    torch.save({name: value.latents.cpu() for name, value in values.items()}, path)


def load_text_and_adapter(local_model_path: str, adapter_id: str, device: torch.device, dtype: torch.dtype):
    from neodragon.context_adapter import ContextAdapter
    from neodragon.text_encoder_bundle import TextEncoderBundle

    text = TextEncoderBundle.from_pretrained(local_model_path, torch_dtype=dtype).to(device).eval()
    adapter = ContextAdapter.from_pretrained(
        f"{local_model_path}/{adapter_id}", torch_dtype=dtype
    ).to(device).eval()
    text.requires_grad_(False)
    adapter.requires_grad_(False)
    return text, adapter


@torch.inference_mode()
def run_teacher(args: argparse.Namespace, output_dir: Path, device: torch.device, dtype: torch.dtype) -> None:
    _, local_model_path = native_paths(load_config(args.config))
    checkpoint, _ = checkpoint_info(Path(args.checkpoint))
    from neodragon.pyramid_mmdit import PyramidMMDiT
    from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler
    from neodragon.utils.generation_utils import DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT_MODIFIER

    print("Teacher phase: loading text/context stack...", flush=True)
    text, adapter = load_text_and_adapter(local_model_path, checkpoint["adapter_id"], device, dtype)
    print("Teacher phase: loading multistep DiT...", flush=True)
    teacher = PyramidMMDiT.from_pretrained(
        f"{local_model_path}/{checkpoint['dit_id']}", torch_dtype=dtype
    ).to(device).eval()
    teacher.requires_grad_(False)

    prompt = args.prompt + DEFAULT_PROMPT_MODIFIER
    first = audit.make_condition(
        text=text, adapter=adapter, prompt=prompt, negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        guidance=7.0, device=device,
    )
    video = audit.make_condition(
        text=text, adapter=adapter, prompt=prompt, negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        guidance=5.0, device=device,
    )
    full_noise = torch.randn(
        1, 16, 7, args.height // 8, args.width // 8,
        device=device, dtype=dtype, generator=torch.Generator(device=device).manual_seed(args.seed),
    )
    rollout = audit.run_rollout(
        dit=teacher,
        scheduler=PyramidFlowMatchEulerDiscreteScheduler(),
        full_noise=full_noise,
        condition_for_unit=lambda unit: first if unit == 0 else video,
        first_steps=args.first_steps,
        video_steps=args.video_steps,
        transition_seed=args.seed + 1_000_000,
        initial_anchor=None,
    )
    teacher_report = {
        "checkpoint_step": checkpoint["step"],
        "prompt": args.prompt,
        "latent_motion": audit.latent_motion_metrics(rollout.latents),
        "target_mismatch": audit.teacher_target_mismatch(rollout),
    }
    save_rollout(
        output_dir / "teacher_rollout.pt",
        rollout,
        prompt=args.prompt,
        metadata={"checkpoint": checkpoint, "teacher_report": teacher_report},
    )
    (output_dir / "teacher_report.json").write_text(json.dumps(teacher_report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(teacher_report["target_mismatch"]["endpoint"]["mean"], indent=2), flush=True)


@torch.inference_mode()
def run_student(args: argparse.Namespace, output_dir: Path, device: torch.device, dtype: torch.dtype) -> None:
    _, local_model_path = native_paths(load_config(args.config))
    checkpoint, student_state = checkpoint_info(Path(args.checkpoint))
    if student_state is None:
        raise ValueError("DMD checkpoint does not contain a student state dictionary.")
    from neodragon.pyramid_mmdit import PyramidMMDiT
    from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler
    from neodragon.utils.generation_utils import DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT_MODIFIER

    prompt, teacher_metadata, teacher = load_rollout(output_dir / "teacher_rollout.pt", device, dtype)
    print("Student phase: loading text/context stack...", flush=True)
    text, adapter = load_text_and_adapter(local_model_path, checkpoint["adapter_id"], device, dtype)
    print("Student phase: loading DMD DiT...", flush=True)
    student = PyramidMMDiT.from_pretrained(
        f"{local_model_path}/{checkpoint['dit_id']}", torch_dtype=dtype
    ).to(device).eval()
    checkpoint_drift = audit.checkpoint_update_metrics(
        reference=student,
        candidate=student_state,
    )
    student.load_state_dict(student_state, strict=True)
    del student_state
    student.requires_grad_(False)

    condition = audit.make_condition(
        text=text,
        adapter=adapter,
        prompt=prompt + DEFAULT_PROMPT_MODIFIER,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        guidance=0.0,
        device=device,
    )
    full_noise = torch.randn(
        1, 16, 7, args.height // 8, args.width // 8,
        device=device, dtype=dtype, generator=torch.Generator(device=device).manual_seed(args.seed),
    )
    scheduler = PyramidFlowMatchEulerDiscreteScheduler()
    teacher_forced = audit.run_rollout(
        dit=student, scheduler=scheduler, full_noise=full_noise,
        condition_for_unit=lambda _: condition, first_steps=1, video_steps=1,
        transition_seed=args.seed + 1_000_000,
        initial_anchor=teacher.latents[:, :, :1], teacher_history=teacher.latents,
    )
    self_history = audit.run_rollout(
        dit=student, scheduler=scheduler, full_noise=full_noise,
        condition_for_unit=lambda _: condition, first_steps=1, video_steps=1,
        transition_seed=args.seed + 1_000_000,
        initial_anchor=teacher.latents[:, :, :1],
    )
    systems = {"teacher_multistep": teacher, "dmd_teacher_forced": teacher_forced, "dmd_teacher_anchor_self_history": self_history}
    report = {
        "checkpoint_step": checkpoint["step"],
        "prompt": prompt,
        "teacher_metadata": teacher_metadata,
        "checkpoint_drift_from_multistep_init": checkpoint_drift,
        "oracle_local_dmd": audit.oracle_local_dmd_metrics(
            teacher=teacher, student=student, condition=condition, scheduler=scheduler
        ),
        "rollout_relative_to_teacher": {
            name: audit.tensor_metrics(teacher.latents[:, :, 1:], value.latents[:, :, 1:])
            for name, value in systems.items() if name != "teacher_multistep"
        },
        "systems": {name: {"latent_motion": audit.latent_motion_metrics(value.latents)} for name, value in systems.items()},
    }
    save_student_rollouts(output_dir / "student_rollouts.pt", systems)
    (output_dir / "student_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["oracle_local_dmd"]["mean"], indent=2), flush=True)


@torch.inference_mode()
def run_ssd(args: argparse.Namespace, output_dir: Path, device: torch.device, dtype: torch.dtype) -> None:
    _, local_model_path = native_paths(load_config(args.config))
    checkpoint, student_state = checkpoint_info(Path(args.checkpoint))
    if student_state is None:
        raise ValueError("DMD checkpoint does not contain a student state dictionary.")
    from neodragon.asymmetric_causal_video_vae import AsymmetricCausalVideoVAE
    from neodragon.first_frame_gen import SSD1B_FirstFrameGeneratorPipeline
    from neodragon.pyramid_mmdit import PyramidMMDiT
    from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler
    from neodragon.utils.generation_utils import DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT_MODIFIER

    prompt, _, teacher = load_rollout(output_dir / "teacher_rollout.pt", device, dtype)
    print("SSD phase: loading text/context stack and DMD DiT...", flush=True)
    text, adapter = load_text_and_adapter(local_model_path, checkpoint["adapter_id"], device, dtype)
    student = PyramidMMDiT.from_pretrained(
        f"{local_model_path}/{checkpoint['dit_id']}", torch_dtype=dtype
    ).to(device).eval()
    student.load_state_dict(student_state, strict=True)
    del student_state
    student.requires_grad_(False)
    print("SSD phase: loading VAE and SSD1B anchor pipeline...", flush=True)
    vae = AsymmetricCausalVideoVAE.from_pretrained(
        f"{local_model_path}/causal_video_vae", torch_dtype=dtype
    ).to(device).eval()
    first_frame = SSD1B_FirstFrameGeneratorPipeline.from_pretrained(
        local_model_path, torch_dtype=dtype
    ).to(device)
    vae.requires_grad_(False)

    condition = audit.make_condition(
        text=text,
        adapter=adapter,
        prompt=prompt + DEFAULT_PROMPT_MODIFIER,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        guidance=0.0,
        device=device,
    )
    anchor, image = audit.encode_ssd_anchor(
        pipeline=first_frame,
        vae=vae,
        prompt=prompt + DEFAULT_PROMPT_MODIFIER,
        height=args.height,
        width=args.width,
        seed=args.seed,
        device=device,
        dtype=dtype,
    )
    full_noise = torch.randn(
        1, 16, 7, args.height // 8, args.width // 8,
        device=device, dtype=dtype, generator=torch.Generator(device=device).manual_seed(args.seed),
    )
    dmd_ssd = audit.run_rollout(
        dit=student,
        scheduler=PyramidFlowMatchEulerDiscreteScheduler(),
        full_noise=full_noise,
        condition_for_unit=lambda _: condition,
        first_steps=1,
        video_steps=1,
        transition_seed=args.seed + 1_000_000,
        initial_anchor=anchor,
    )
    path = output_dir / "student_rollouts.pt"
    systems = torch.load(path, map_location="cpu", weights_only=False)
    systems["dmd_ssd_anchor_self_history"] = dmd_ssd.latents.cpu()
    torch.save(systems, path)
    torch.save(anchor.cpu(), output_dir / "ssd_anchor_latent.pt")
    image.save(output_dir / "ssd1b_anchor.png")
    report = {
        "checkpoint_step": checkpoint["step"],
        "prompt": prompt,
        "anchor_latent_difference_from_teacher": audit.tensor_metrics(teacher.latents[:, :, :1], anchor),
        "rollout_relative_to_teacher": audit.tensor_metrics(
            teacher.latents[:, :, 1:], dmd_ssd.latents[:, :, 1:]
        ),
        "latent_motion": audit.latent_motion_metrics(dmd_ssd.latents),
    }
    (output_dir / "ssd_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["latent_motion"], indent=2), flush=True)


@torch.inference_mode()
def run_hybrid(args: argparse.Namespace, output_dir: Path, device: torch.device, dtype: torch.dtype) -> None:
    """Render the released Hybrid with the exact SSD anchor/noise control."""

    _, local_model_path = native_paths(load_config(args.config))
    from neodragon.pyramid_mmdit import PyramidMMDiT
    from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler
    from neodragon.utils.generation_utils import DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT_MODIFIER

    prompt, _, teacher = load_rollout(output_dir / "teacher_rollout.pt", device, dtype)
    print("Hybrid phase: loading native released text/context/DiT stack...", flush=True)
    text, native_adapter = load_text_and_adapter(local_model_path, "context_adapter", device, dtype)
    hybrid = PyramidMMDiT.from_pretrained(
        f"{local_model_path}/diffusion_transformer_320p", torch_dtype=dtype
    ).to(device).eval()
    native_adapter.requires_grad_(False)
    hybrid.requires_grad_(False)

    condition = audit.make_condition(
        text=text,
        adapter=native_adapter,
        prompt=prompt + DEFAULT_PROMPT_MODIFIER,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        guidance=0.0,
        device=device,
    )
    anchor = torch.load(output_dir / "ssd_anchor_latent.pt", map_location="cpu", weights_only=False)
    anchor = anchor.to(device=device, dtype=dtype)
    full_noise = torch.randn(
        1, 16, 7, args.height // 8, args.width // 8,
        device=device, dtype=dtype, generator=torch.Generator(device=device).manual_seed(args.seed),
    )
    rollout = audit.run_rollout(
        dit=hybrid,
        scheduler=PyramidFlowMatchEulerDiscreteScheduler(),
        full_noise=full_noise,
        condition_for_unit=lambda _: condition,
        first_steps=1,
        video_steps=1,
        transition_seed=args.seed + 1_000_000,
        initial_anchor=anchor,
    )
    path = output_dir / "student_rollouts.pt"
    systems = torch.load(path, map_location="cpu", weights_only=False)
    systems["released_hybrid_ssd_anchor_self_history"] = rollout.latents.cpu()
    torch.save(systems, path)
    report = {
        "prompt": prompt,
        "rollout_relative_to_teacher": audit.tensor_metrics(teacher.latents[:, :, 1:], rollout.latents[:, :, 1:]),
        "latent_motion": audit.latent_motion_metrics(rollout.latents),
        "condition_stack": "released hybrid context_adapter + released hybrid DiT",
    }
    (output_dir / "hybrid_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["latent_motion"], indent=2), flush=True)


@torch.inference_mode()
def run_decode(args: argparse.Namespace, output_dir: Path, device: torch.device, dtype: torch.dtype) -> None:
    _, local_model_path = native_paths(load_config(args.config))
    from neodragon.asymmetric_causal_video_vae import AsymmetricCausalVideoVAE

    print("Decode phase: loading causal VAE...", flush=True)
    vae = AsymmetricCausalVideoVAE.from_pretrained(
        f"{local_model_path}/causal_video_vae", torch_dtype=dtype
    ).to(device).eval()
    vae.requires_grad_(False)
    values = torch.load(output_dir / "student_rollouts.pt", map_location="cpu", weights_only=False)
    report = {"systems": {}, "videos": {}}
    for name, latent in values.items():
        frames = audit.decode_latents(vae, latent.to(device=device, dtype=dtype))
        destination = output_dir / f"{name}.mp4"
        export_to_video([Image.fromarray(frame) for frame in frames], destination, fps=args.fps)
        report["systems"][name] = audit.decoded_motion_metrics(frames)
        report["videos"][name] = str(destination.resolve())
    teacher_report = json.loads((output_dir / "teacher_report.json").read_text(encoding="utf-8"))
    student_report = json.loads((output_dir / "student_report.json").read_text(encoding="utf-8"))
    ssd_report_path = output_dir / "ssd_report.json"
    hybrid_report_path = output_dir / "hybrid_report.json"
    full = {
        "protocol": "staged memory-bounded DMD motion audit",
        "teacher": teacher_report,
        "student": student_report,
        "ssd": json.loads(ssd_report_path.read_text(encoding="utf-8")) if ssd_report_path.is_file() else None,
        "hybrid": json.loads(hybrid_report_path.read_text(encoding="utf-8")) if hybrid_report_path.is_file() else None,
        "decoded": report,
    }
    (output_dir / "motion_audit.json").write_text(json.dumps(full, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["systems"], indent=2), flush=True)


def main() -> None:
    args = parse_args()
    validate(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = audit.dtype_from_name(args.dtype)
    if args.phase == "teacher":
        run_teacher(args, output_dir, device, dtype)
    elif args.phase == "student":
        run_student(args, output_dir, device, dtype)
    elif args.phase == "ssd":
        run_ssd(args, output_dir, device, dtype)
    elif args.phase == "hybrid":
        run_hybrid(args, output_dir, device, dtype)
    else:
        run_decode(args, output_dir, device, dtype)


if __name__ == "__main__":
    main()
