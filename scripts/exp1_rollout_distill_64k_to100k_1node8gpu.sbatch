#!/usr/bin/env bash
#SBATCH --job-name=neo-exp1-rollout
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/neo-exp1-rollout-%j.out
#SBATCH --error=logs/neo-exp1-rollout-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:8
##SBATCH --gres=gpu:A100-SXM4-80GB:8

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

export PYTHONPATH="./:${PYTHONPATH:-}"
export HF_HOME=/proj/cvl/users/x_fahkh2/caches
export TORCH_HOME=/proj/cvl/users/x_fahkh2/caches
export PIP_CACHE_DIR=/proj/cvl/users/x_fahkh2/caches
export TMPDIR=/proj/cvl/users/x_fahkh2/caches
export TRITON_CACHE_DIR=/proj/cvl/users/x_fahkh2/caches

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_ENV}/bin/torchrun}"
CONFIG="${CONFIG:-configs/mobile_ov_neodragon.yaml}"
MANIFEST="${MANIFEST:-data/openvid_neodragon_2s_latents/latent_manifest.csv}"
REMOTE_EXP1="output/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt"
LOCAL_EXP1="checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt"
if [[ -z "${INIT_CHECKPOINT:-}" ]]; then
  if [[ -f "${REMOTE_EXP1}" ]]; then
    INIT_CHECKPOINT="${REMOTE_EXP1}"
  else
    INIT_CHECKPOINT="${LOCAL_EXP1}"
  fi
fi

OUT="${OUT:-output/neo_exp1_rollout_64k_to100k}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

mkdir -p logs "${OUT}"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-60}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for required in "${MANIFEST}" "${INIT_CHECKPOINT}" "${TORCHRUN_BIN}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 1
  fi
done

echo "Exp1 full-rollout continuation: validated 64k bridge -> 100k"
echo "CONFIG=${CONFIG}"
echo "MANIFEST=${MANIFEST}"
echo "INIT_CHECKPOINT=${INIT_CHECKPOINT}"
echo "OUT=${OUT} RESUME=${RESUME:-auto}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE} BATCH_SIZE=${BATCH_SIZE:-4} GLOBAL_BATCH=$((NPROC_PER_NODE * ${BATCH_SIZE:-4}))"
echo "LR=${LR:-1e-5} TARGET_STEP=${TARGET_STEP:-100000}"
echo "LATEST_EVERY=${SAVE_LATEST_EVERY:-5000} ARCHIVE_EVERY=${SAVE_ARCHIVE_EVERY:-10000}"
nvidia-smi || true

"${TORCHRUN_BIN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train_neodragon_bridge_rollout_distill.py \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  --resume "${RESUME:-auto}" \
  --output-dir "${OUT}" \
  --target-step "${TARGET_STEP:-100000}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --functional-batch-size "${FUNCTIONAL_BATCH_SIZE:-1}" \
  --parallel ddp \
  --lr "${LR:-1e-5}" \
  --lr-warmup-steps "${LR_WARMUP_STEPS:-500}" \
  --lr-final-scale "${LR_FINAL_SCALE:-0.1}" \
  --rollout-ramp-steps "${ROLLOUT_RAMP_STEPS:-2000}" \
  --rollout-mse-weight "${ROLLOUT_MSE_WEIGHT:-1.0}" \
  --rollout-cos-weight "${ROLLOUT_COS_WEIGHT:-0.1}" \
  --raw-token-weight 0.25 \
  --normalized-token-weight 1.0 \
  --cos-weight 0.5 \
  --token-norm-weight 0.1 \
  --pooled-weight 0.25 \
  --pooled-cos-weight 0.2 \
  --relational-weight 0.1 \
  --caption-variant-columns caption_short,caption_medium,caption_long \
  --caption-variant-weights 1,1,1 \
  --log-every "${LOG_EVERY:-20}" \
  --save-latest-every "${SAVE_LATEST_EVERY:-5000}" \
  --save-archive-every "${SAVE_ARCHIVE_EVERY:-10000}" \
  --archive-start-step 70000 \
  ${EXTRA_ARGS:-}

find "${OUT}" -maxdepth 1 -type f -print -exec ls -lh {} \;
