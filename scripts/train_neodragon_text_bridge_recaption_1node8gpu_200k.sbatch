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

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_ENV}/bin/torchrun}"
RECAPTION_CSV="${RECAPTION_CSV:-/proj/cvl/users/x_fahkh2/Mobile-OV_Alpha/download_data/data/openvid/manifests/openvid_all_recaptions.csv}"

CONFIG="${CONFIG:-configs/mobile_ov_neodragon.yaml}"
OUT="${OUT:-output/neo_bridge_8gpu_200k/${SLURM_JOB_ID:-local}}"
STEPS="${STEPS:-200000}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
LOG_EVERY="${LOG_EVERY:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LR="${LR:-1e-4}"
PARALLEL="${PARALLEL:-fsdp}"

mkdir -p logs "${OUT}"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export FSDP_USE_ORIG_PARAMS=true
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-60}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-${PWD}/checkpoints/hf_home}"
export TORCH_HOME="${TORCH_HOME:-${PWD}/checkpoints/torch_home}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PWD}/output/cache/triton}"

echo "PWD=${PWD}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "RECAPTION_CSV=${RECAPTION_CSV}"
echo "OUT=${OUT}"
echo "STEPS=${STEPS} SAVE_EVERY=${SAVE_EVERY} BATCH_SIZE=${BATCH_SIZE} LR=${LR} PARALLEL=${PARALLEL}"
nvidia-smi || true

"${TORCHRUN_BIN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  tools/train_neodragon_text_bridge.py \
  --config "${CONFIG}" \
  --prompts "${RECAPTION_CSV}" \
  --output-dir "${OUT}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --parallel "${PARALLEL}" \
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
  --log-every "${LOG_EVERY}" \
  --save-every "${SAVE_EVERY}" \
  --caption-variant-columns caption_short,caption_medium,caption_long \
  --caption-variant-weights 1,1,1 \
  --caption-fallback-column caption \
  --keep-step-checkpoints \
  ${EXTRA_ARGS:-}

find "${OUT}" -maxdepth 1 -type f -print -exec ls -lh {} \;
