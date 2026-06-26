#!/usr/bin/env bash
#SBATCH --job-name=neo-recap-8gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -euo pipefail

# Submit from the repo root:
#   sbatch scripts/train_neodragon_text_bridge_recaption_1node8gpu_200k.sbatch
#
# Useful overrides:
#   CONDA_ENV=/proj/cvl/users/x_fahkh2/envs/neo_mobileov \
#   RECAPTION_CSV=/path/to/openvid_all_recaptions.csv \
#   STEPS=200000 SAVE_EVERY=10000 \
#   sbatch scripts/train_neodragon_text_bridge_recaption_1node8gpu_200k.sbatch

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ ! -d "${ROOT}/new_mobile_ov" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT}"
mkdir -p logs output/cache output/neodragon_text_bridge_recaption_distill checkpoints

# Keep env handling intentionally simple for SLURM batch shells.
# Do not source conda.sh here: some cluster conda scripts read PS1 and fail
# under `set -u` with "PS1: unbound variable".
CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
if [[ ! -x "${CONDA_ENV}/bin/python" ]]; then
  echo "Missing python at CONDA_ENV=${CONDA_ENV}" >&2
  echo "Create it first or submit with CONDA_ENV=/path/to/env." >&2
  exit 1
fi
export PATH="${CONDA_ENV}/bin:${PATH}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV}/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_ENV}/bin/torchrun}"
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "Missing torchrun at ${TORCHRUN_BIN}" >&2
  exit 1
fi

RECAPTION_CSV="${RECAPTION_CSV:-/proj/cvl/users/x_fahkh2/Mobile-OV_Alpha/download_data/data/openvid/manifests/openvid_all_recaptions.csv}"
if [[ ! -f "${RECAPTION_CSV}" ]]; then
  echo "Missing RECAPTION_CSV=${RECAPTION_CSV}" >&2
  echo "Set RECAPTION_CSV=/path/to/openvid_all_recaptions.csv before sbatch." >&2
  exit 1
fi

NNODES="${SLURM_NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}"
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST:-localhost}" | head -n 1)}"
MASTER_PORT="${MASTER_PORT:-29577}"
RDZV_ID="${RDZV_ID:-neo_recap_8gpu_${SLURM_JOB_ID:-local}}"

CONFIG="${CONFIG:-configs/mobile_ov_neodragon.yaml}"
PARALLEL="${PARALLEL:-fsdp}"
STEPS="${STEPS:-200000}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
LOG_EVERY="${LOG_EVERY:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LR="${LR:-1e-4}"
MAX_PROMPTS="${MAX_PROMPTS:--1}"

RUN_NAME="${RUN_NAME:-1node8gpu_200k_${SLURM_JOB_ID:-local}}"
OUT="${OUT:-${ROOT}/output/neodragon_text_bridge_recaption_distill/${RUN_NAME}}"
LATEST="${LATEST:-${ROOT}/output/neodragon_text_bridge_recaption_distill/1node8gpu_latest/bridge}"
mkdir -p "${OUT}/bridge" "${LATEST}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export FSDP_USE_ORIG_PARAMS=true
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-60}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-${ROOT}/checkpoints/hf_home}"
export TORCH_HOME="${TORCH_HOME:-${ROOT}/checkpoints/torch_home}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${ROOT}/output/cache/triton}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${ROOT}/output/cache/pip}"
export TMPDIR="${TMPDIR:-${ROOT}/output/cache/tmp}"
export MOBILEOV_CAPTION_AUG="${MOBILEOV_CAPTION_AUG:-1}"
export MOBILEOV_CAPTION_AUG_COLUMNS="${MOBILEOV_CAPTION_AUG_COLUMNS:-caption_short,caption_medium,caption_long}"
export MOBILEOV_CAPTION_AUG_WEIGHTS="${MOBILEOV_CAPTION_AUG_WEIGHTS:-1,1,1}"
export MOBILEOV_CAPTION_FALLBACK_COLUMN="${MOBILEOV_CAPTION_FALLBACK_COLUMN:-caption}"
export KEEP_STEP_CHECKPOINTS="${KEEP_STEP_CHECKPOINTS:-1}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}" "${PIP_CACHE_DIR}" "${TMPDIR}"

HEARTBEAT_PID=""
start_gpu_heartbeat() {
  "${PYTHON_BIN}" - <<'PY' &
import os
import time

import torch

devices = list(range(torch.cuda.device_count()))
if not devices:
    print("[gpu-heartbeat] no cuda devices visible", flush=True)
    while True:
        time.sleep(15)

tensors = [torch.randn((1024, 1024), device=f"cuda:{idx}", dtype=torch.float16) for idx in devices]
print(f"[gpu-heartbeat] active devices={devices}", flush=True)
while True:
    for idx, x in zip(devices, tensors):
        y = x @ x
        torch.cuda.synchronize(idx)
    print(f"[gpu-heartbeat] tick devices={devices}", flush=True)
    time.sleep(15)
PY
  HEARTBEAT_PID=$!
}

stop_gpu_heartbeat() {
  if [[ -n "${HEARTBEAT_PID}" ]] && kill -0 "${HEARTBEAT_PID}" 2>/dev/null; then
    kill "${HEARTBEAT_PID}" 2>/dev/null || true
    wait "${HEARTBEAT_PID}" 2>/dev/null || true
  fi
  HEARTBEAT_PID=""
}
trap stop_gpu_heartbeat EXIT

echo "ROOT=${ROOT}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "TORCHRUN_BIN=${TORCHRUN_BIN}"
echo "CONFIG=${CONFIG}"
echo "RECAPTION_CSV=${RECAPTION_CSV}"
echo "NNODES=${NNODES} GPUS_PER_NODE=${GPUS_PER_NODE} WORLD_SIZE=${WORLD_SIZE}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} RDZV_ID=${RDZV_ID}"
echo "PARALLEL=${PARALLEL} STEPS=${STEPS} SAVE_EVERY=${SAVE_EVERY} BATCH_SIZE=${BATCH_SIZE} GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS} LR=${LR}"
echo "OUT=${OUT}"
echo "LATEST=${LATEST}"
echo "Caption aug: enabled=${MOBILEOV_CAPTION_AUG} columns=${MOBILEOV_CAPTION_AUG_COLUMNS} weights=${MOBILEOV_CAPTION_AUG_WEIGHTS}"
date
nvidia-smi || true

if [[ "${PREPARE_CHECKPOINTS:-1}" != "0" ]]; then
  echo "Preparing checkpoints with GPU heartbeat enabled..."
  start_gpu_heartbeat
  "${PYTHON_BIN}" tools/prepare_checkpoints.py --config "${CONFIG}"
  stop_gpu_heartbeat
  echo "Checkpoint preparation finished."
else
  echo "Skipping checkpoint preparation because PREPARE_CHECKPOINTS=0."
fi

"${TORCHRUN_BIN}" \
  --standalone \
  --nnodes="${NNODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  tools/train_neodragon_text_bridge.py \
  --config "${CONFIG}" \
  --prompts "${RECAPTION_CSV}" \
  --output-dir "${OUT}/bridge" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --max-prompts "${MAX_PROMPTS}" \
  --parallel "${PARALLEL}" \
  --deepspeed-zero-stage "${DEEPSPEED_ZERO_STAGE:-1}" \
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
  --log-every "${LOG_EVERY}" \
  --save-every "${SAVE_EVERY}" \
  --caption-variant-columns "${MOBILEOV_CAPTION_AUG_COLUMNS}" \
  --caption-variant-weights "${MOBILEOV_CAPTION_AUG_WEIGHTS}" \
  --caption-fallback-column "${MOBILEOV_CAPTION_FALLBACK_COLUMN}" \
  --keep-step-checkpoints \
  ${CAPTION_AUG_FLAG:-} \
  ${EXTRA_ARGS:-}

cp -f "${OUT}/bridge/neodragon_text_bridge_latest.pt" "${LATEST}/neodragon_text_bridge_latest.pt"
cp -f "${OUT}/bridge/history.json" "${LATEST}/history.json"
if [[ -f "${OUT}/bridge/deepspeed_config.json" ]]; then
  cp -f "${OUT}/bridge/deepspeed_config.json" "${LATEST}/deepspeed_config.json"
fi

find "${OUT}" -maxdepth 3 -type f -print -exec ls -lh {} \;
find "${LATEST}" -maxdepth 1 -type f -print -exec ls -lh {} \;
date
