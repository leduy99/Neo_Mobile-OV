# Development Note

Date: 2026-06-23

This note summarizes the current state of `New-Mobile-OV`, what was added today, and the distributed-training issues we fixed while bringing up FSDP/DeepSpeed-style training for the new Mobile-OV + Neodragon branch.

## Current Repository Scope

`New-Mobile-OV` is a self-contained research repo for the next Mobile-OV generation branch. The repo keeps the Mobile-OV understanding side intact and experiments with multiple generation branches:

- `mobile_ov_current`: reference path matching the existing SmolVLM2 + Bridge v2 + SANA-video direction.
- `mobile_o_sana_0_5b`: lightweight Mobile-O / SANA-image option for image/anchor generation experiments.
- `mobile_ov_neodragon`: Neodragon video generation branch conditioned by Mobile-OV Bridge v2 outputs.

The important Neodragon conditioning design is:

```text
prompt
  -> SmolVLM2 + Bridge v2
  -> prompt_embeds [B, 128, 1536]
  -> pooled_prompt_embeds [B, 2048]
  -> Neodragon DiT
```

The bridge is distilled from frozen Neodragon text conditioning:

```text
Neodragon TextEncoderBundle + ContextAdapter
  -> teacher token condition [B, 128, 1536]

Mobile-OV SmolVLM2 + Bridge v2
  -> student token condition [B, 128, 1536]
```

## Existing Major Components

- `new_mobile_ov/bridge/`: SmolVLM2 + Bridge v2 code, including the Neodragon-shaped bridge.
- `new_mobile_ov/generation/`: generation backends for current Mobile-OV, Mobile-O/SANA-0.5B, and Neodragon.
- `new_mobile_ov/motion/`: Latent Motion Weaver experiments.
- `new_mobile_ov/training/`: shared training utilities, losses, and distributed helpers.
- `tools/`: inference, data preparation, bridge distillation, DiT bridge training, WanVAE anchor checks, and LMW training/eval scripts.
- `scripts/`: SLURM entrypoints for smoke tests, bridge distillation, DiT training, and recaption-data training.
- `environment/neo_mobileov.yaml`: reproducible environment spec for the main local env.

## Added Today

### Automatic Checkpoint Preparation

The repo now prepares missing runtime assets under `./checkpoints/` instead of depending on hard-coded local paths:

```text
checkpoints/smolvlm2_500m/smolvlm2_500m.pt
checkpoints/neodragon_repo/
checkpoints/neodragon/
```

`new_mobile_ov/checkpoints.py` handles:

- Converted SmolVLM2 checkpoint discovery.
- Optional copy/download from `MOBILEOV_SMOLVLM2_CKPT_SOURCE`.
- Optional Hugging Face file download via `MOBILEOV_SMOLVLM2_CKPT_HF_REPO` and `MOBILEOV_SMOLVLM2_CKPT_HF_FILE`.
- Fallback conversion from `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`.
- Neodragon source clone from `https://github.com/Qualcomm-AI-research/neodragon.git`.
- Neodragon weight download from `karnewar/Neodragon`.

The helper uses lock files and ready markers so distributed launches do not let every GPU rank download or convert the same asset at once.

### OpenVid100 Neodragon Data Preparation

Added:

```text
tools/prepare_neodragon_openvid100.py
scripts/prepare_neodragon_openvid100.sh
```

This creates a small local training subset:

```text
data/neodragon_openvid100/
  manifest.csv
  prompts.txt
  videos/
  summary.json
```

The manifest has clean columns such as:

```text
video_path,prompt,caption,source_video_path,source_row,source_id
```

This was used for quick trainer bring-up and smoke tests.

### Distributed Training Utility

Added:

```text
new_mobile_ov/training/distributed.py
```

It centralizes:

- `torch.distributed` initialization from `torchrun`/SLURM env vars.
- rank/local-rank/world-size tracking.
- rank-0 printing.
- safe distributed barriers.
- scalar metric reduction.
- FSDP full-state checkpoint collection.
- DeepSpeed config generation for ZeRO 0/1/2.

### FSDP/DeepSpeed-Aware Bridge Distillation

Updated:

```text
tools/train_neodragon_text_bridge.py
```

It now supports:

```bash
--parallel none|ddp|fsdp|deepspeed
```

The tested path today is FSDP. DeepSpeed support is wired through the same trainer and config generation path, but it still needs a longer dedicated smoke run before we call it fully validated.

### FSDP/DeepSpeed-Aware Neodragon DiT Bridge Training

Updated:

```text
tools/train_neodragon_dit_bridge.py
scripts/train_neodragon_dit_bridge_distributed.sbatch
```

This gives the DiT bridge trainer the same distributed interface. The immediate validated training today was text-bridge distillation; DiT training has the distributed path prepared for the next stage.

### Recaption CSV Training Support

Updated:

```text
tools/train_neodragon_text_bridge.py
scripts/train_neodragon_text_bridge_recaption_multinode.sbatch
```

The trainer can now read OpenVid recaption manifests with:

```text
caption_short
caption_medium
caption_long
```

and randomly sample among them during distillation. The default remote input path is:

```text
/proj/cvl/users/x_fahkh2/Mobile-OV_Alpha/download_data/data/openvid/manifests/openvid_all_recaptions.csv
```

Relevant env controls:

```bash
MOBILEOV_CAPTION_AUG=1
MOBILEOV_CAPTION_AUG_COLUMNS=caption_short,caption_medium,caption_long
MOBILEOV_CAPTION_AUG_WEIGHTS=1,1,1
MOBILEOV_CAPTION_FALLBACK_COLUMN=caption
```

The multi-node script launches one `torchrun` per node through `srun`, so it can scale from one node to multiple nodes:

```bash
sbatch --nodes=4 --gres=gpu:A100-SXM4-80GB:8 \
  scripts/train_neodragon_text_bridge_recaption_multinode.sbatch
```

## Distributed/NCCL Issue And Fix

### Symptom

Initial FSDP jobs were allocated by SLURM but appeared stuck. One failed after a long NCCL timeout:

```text
Watchdog caught collective operation timeout
WorkNCCL(... OpType=ALLREDUCE, NumelIn=1, NumelOut=1 ...)
```

This happened before useful training progress. The job held GPUs, showed GPU utilization, but produced no new training logs.

### Why It Stuck

FSDP, DDP, and DeepSpeed all rely on `torch.distributed` collectives. A collective operation such as `barrier()` or `all_reduce()` is synchronous across all ranks:

```text
rank 0 enters all_reduce
rank 1 enters all_reduce
NCCL must connect both ranks
all ranks wait until communication succeeds
```

If NCCL transport cannot communicate correctly between the allocated GPUs, every rank waits. Python code looks stuck because no normal exception is raised immediately. Eventually the NCCL watchdog kills the job after the distributed timeout.

This affects the system because the job still holds the allocated GPUs while waiting. That can block other SLURM jobs even though useful training is not happening.

### Important SLURM Lesson

`nvidia-smi` showing a GPU as idle does not necessarily mean SLURM can allocate it. Earlier, GPU 1 and GPU 2 looked idle, but SLURM showed they were already allocated to another interactive `bash` job:

```text
GRES=gpu:nvidia_h200_nvl:2(IDX:1-2)
```

So the scheduler correctly kept our job pending. We should trust SLURM allocation state first, then use `nvidia-smi` to inspect utilization inside allocations.

### Fixes Applied

1. Added explicit NCCL-safe barrier routing:

```python
dist.barrier(device_ids=[torch.cuda.current_device()])
```

This avoids PyTorch guessing the wrong device for NCCL barriers.

2. Added rank-level distributed init logs:

```text
[dist] init rank=0/2 local_rank=0 device=cuda:0 backend=nccl
[dist] ready rank=0/2 local_rank=0 device=cuda:0
```

These logs make it obvious whether the job is stuck at process-group init, model loading, or the training loop.

3. Added shorter debug timeout support:

```bash
TORCH_DIST_TIMEOUT_MINUTES=10
```

This prevents smoke jobs from wasting an hour when distributed communication is broken.

4. Ran a minimal NCCL smoke test:

```text
init_process_group("nccl")
all_reduce(torch.ones(1))
barrier()
```

The smoke test passed only after disabling NCCL P2P/IB transport:

```bash
NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
```

5. Set those env vars by default in distributed SLURM scripts:

```bash
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
```

This fix applies to FSDP and DeepSpeed because both ultimately use NCCL collectives underneath.

6. Fixed `srun` environment drift.

One smoke run accidentally used base conda Python 3.13:

```text
/share_0/conda/bin/python3.13
```

That caused unrelated `diffusers`/`torchao` import errors. The script now pins:

```bash
PYTHON_BIN=${CONDA_ENV}/bin/python
TORCHRUN_BIN=${CONDA_ENV}/bin/torchrun
```

and uses the explicit `torchrun` path inside `srun`.

## Validation Done Today

### FSDP text-bridge smoke

Job `1042`:

```text
2 GPUs
1 training step
COMPLETED
ExitCode=0:0
checkpoint saved
```

### FSDP text-bridge short run

Job `1043`:

```text
2 GPUs
20 training steps
COMPLETED
ExitCode=0:0
```

Loss moved in the right direction:

```text
step 1:  loss ~= 2.0108
step 20: loss ~= 1.1034
```

Checkpoint:

```text
output/neodragon_text_bridge_distill/fsdp_latest/bridge/neodragon_text_bridge_latest.pt
```

### Recaption multi-node script smoke

Job `1045`:

```text
script: scripts/train_neodragon_text_bridge_recaption_multinode.sbatch
1 node, 2 GPUs
local mock recaption CSV
caption augmentation enabled
COMPLETED
ExitCode=0:0
checkpoint saved
```

Checkpoint:

```text
output/neodragon_text_bridge_recaption_distill/fsdp_latest/bridge/neodragon_text_bridge_latest.pt
```

## Current Recommended Commands

### One-node smoke with local recaption CSV

```bash
RECAPTION_CSV=/path/to/openvid_all_recaptions.csv \
STEPS=1 SAVE_EVERY=1 LOG_EVERY=1 MAX_PROMPTS=2 \
BATCH_SIZE=1 PARALLEL=fsdp \
sbatch scripts/train_neodragon_text_bridge_recaption_multinode.sbatch
```

### Remote full recaption distillation

```bash
CONDA_ENV=/proj/cvl/users/x_fahkh2/envs/mobileov \
NEO_ROOT=/proj/cvl/users/x_fahkh2/neodragon \
RECAPTION_CSV=/proj/cvl/users/x_fahkh2/Mobile-OV_Alpha/download_data/data/openvid/manifests/openvid_all_recaptions.csv \
STEPS=10000 SAVE_EVERY=500 LOG_EVERY=20 BATCH_SIZE=1 PARALLEL=fsdp \
sbatch --nodes=4 --gres=gpu:A100-SXM4-80GB:8 \
  scripts/train_neodragon_text_bridge_recaption_multinode.sbatch
```

### DeepSpeed variant

```bash
PARALLEL=deepspeed DEEPSPEED_ZERO_STAGE=1 \
sbatch --nodes=4 --gres=gpu:A100-SXM4-80GB:8 \
  scripts/train_neodragon_text_bridge_recaption_multinode.sbatch
```

## Remaining Notes

- FSDP text-bridge distillation is validated with smoke and short run.
- DeepSpeed path is implemented but still should be smoke-tested end-to-end.
- Neodragon DiT bridge distributed training is wired, but the main validated path today is text-bridge distillation.
- The current distillation checkpoint is only a smoke/short-run artifact, not a final trained bridge.
