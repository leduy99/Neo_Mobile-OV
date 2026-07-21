from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistributedContext:
    """Initialize torch.distributed from torchrun or SLURM environment."""
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))

    if torch.cuda.is_available():
        visible = torch.cuda.device_count()
        if visible <= 0:
            raise RuntimeError("CUDA is available but torch.cuda.device_count() returned 0.")
        local_rank = local_rank % visible
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        timeout_minutes = int(os.environ.get("TORCH_DIST_TIMEOUT_MINUTES", "60"))
        print(
            f"[dist] init rank={rank}/{world_size} local_rank={local_rank} device={device} backend={backend} "
            f"timeout={timeout_minutes}m",
            flush=True,
        )
        if device.type == "cuda" and os.environ.get("TORCH_DIST_INIT_DEVICE_ID", "0") == "1":
            try:
                dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_minutes), device_id=device)
            except TypeError:
                dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_minutes))
        else:
            dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_minutes))
        print(f"[dist] ready rank={rank}/{world_size} local_rank={local_rank} device={device}", flush=True)

    return DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size, device=device)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available() and dist.get_backend() == "nccl":
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def rank0_print(ctx: DistributedContext, *values: object, **kwargs: object) -> None:
    if ctx.is_main:
        kwargs.setdefault("flush", True)
        print(*values, **kwargs)


def reduce_mean(value: torch.Tensor, ctx: DistributedContext) -> torch.Tensor:
    if not ctx.is_distributed:
        return value.detach()
    out = value.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out /= ctx.world_size
    return out


def scalar_mean(value: torch.Tensor, ctx: DistributedContext) -> float:
    return float(reduce_mean(value.float(), ctx).detach().cpu())


def build_deepspeed_config(
    *,
    micro_batch_size: int,
    world_size: int,
    gradient_accumulation_steps: int,
    zero_stage: int,
    dtype: torch.dtype,
) -> dict[str, Any]:
    if zero_stage not in {0, 1, 2}:
        raise ValueError("This trainer writes consolidated checkpoints, so use DeepSpeed ZeRO stage 0, 1, or 2.")
    train_batch_size = int(micro_batch_size) * max(int(world_size), 1) * int(gradient_accumulation_steps)
    return {
        "train_micro_batch_size_per_gpu": int(micro_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "train_batch_size": train_batch_size,
        "zero_optimization": {
            "stage": int(zero_stage),
            "overlap_comm": bool(zero_stage > 0),
            "contiguous_gradients": bool(zero_stage > 0),
        },
        "bf16": {"enabled": dtype is torch.bfloat16},
        "fp16": {"enabled": dtype is torch.float16},
        "steps_per_print": 0,
        "wall_clock_breakdown": False,
    }


def write_deepspeed_config(config: dict[str, Any], output_dir: str | Path) -> Path:
    path = Path(output_dir) / "deepspeed_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def full_state_dict(model: torch.nn.Module, parallel: str) -> dict[str, torch.Tensor]:
    """Collect a checkpointable state dict on rank 0.

    FSDP full-state gathering is collective, so all ranks should call this
    function. For DDP and DeepSpeed ZeRO-1/2, rank 0 can save the returned dict.
    """
    parallel = str(parallel).lower()
    if parallel == "fsdp":
        from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            return model.state_dict()
    module = getattr(model, "module", model)
    return module.state_dict()


def full_optimizer_state_dict(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    parallel: str,
) -> dict[str, Any]:
    """Collect a portable optimizer state on rank 0.

    FSDP optimizer states are sharded at runtime. The returned full state can
    therefore be empty on non-zero ranks, matching ``full_state_dict``.
    """
    if str(parallel).lower() == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        return FSDP.full_optim_state_dict(model, optimizer, rank0_only=True)
    return optimizer.state_dict()


def load_full_optimizer_state_dict(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any] | None,
    parallel: str,
    *,
    is_main: bool,
) -> None:
    """Restore a full optimizer checkpoint into the current parallel layout."""
    if str(parallel).lower() == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        sharded = FSDP.scatter_full_optim_state_dict(
            state if is_main else None,
            model,
            optim=optimizer,
        )
        optimizer.load_state_dict(sharded)
        return
    if state is None:
        raise ValueError("Optimizer state is required on every rank outside FSDP.")
    optimizer.load_state_dict(state)
