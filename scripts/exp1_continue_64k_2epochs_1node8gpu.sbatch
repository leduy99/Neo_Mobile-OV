#!/usr/bin/env bash
#SBATCH --job-name=neo-exp1-cont
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/neo-exp1-cont-%j.out
#SBATCH --error=logs/neo-exp1-cont-%j.out
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
PROMPTS="${PROMPTS:-download_data/data/openvid/manifests/openvid_all_recaptions_merged.csv}"
REMOTE_EXP1="output/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt"
LOCAL_EXP1="checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt"
if [[ -z "${INIT_CHECKPOINT:-}" ]]; then
  if [[ -f "${REMOTE_EXP1}" ]]; then
    INIT_CHECKPOINT="${REMOTE_EXP1}"
  else
    INIT_CHECKPOINT="${LOCAL_EXP1}"
  fi
fi

OUT="${OUT:-output/neo_exp1_continue_64k_2epochs/${SLURM_JOB_ID:-local}}"
ADDITIONAL_STEPS="${ADDITIONAL_STEPS:-63748}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

mkdir -p logs "${OUT}"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export FSDP_USE_ORIG_PARAMS=true
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-60}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for required_file in "${PROMPTS}" "${INIT_CHECKPOINT}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required file: ${required_file}" >&2
    exit 1
  fi
done
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "Missing torchrun: ${TORCHRUN_BIN}" >&2
  exit 1
fi

echo "Experiment 1 continuation: 64k checkpoint plus two OpenVid epochs"
echo "INIT_CHECKPOINT=${INIT_CHECKPOINT}"
echo "PROMPTS=${PROMPTS}"
echo "OUT=${OUT}"
echo "ADDITIONAL_STEPS=${ADDITIONAL_STEPS} LR=${LR:-1e-5}"
nvidia-smi || true

"${TORCHRUN_BIN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train_neodragon_text_bridge.py \
  --config "${CONFIG}" \
  --prompts "${PROMPTS}" \
  --output-dir "${OUT}" \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  --step-offset -1 \
  --steps "${ADDITIONAL_STEPS}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --lr "${LR:-1e-5}" \
  --parallel fsdp \
  --caption-variant-columns caption_short,caption_medium,caption_long \
  --caption-variant-weights 1,1,1 \
  --raw-token-weight 0.25 \
  --normalized-token-weight 1.0 \
  --cos-weight 0.5 \
  --token-norm-weight 0.1 \
  --pooled-weight 0.25 \
  --pooled-cos-weight 0.2 \
  --relational-weight 0.1 \
  --functional-weight 1.0 \
  --functional-cos-weight 0.1 \
  --functional-start-step 1 \
  --functional-ramp-steps 2000 \
  --functional-every 1 \
  --functional-batch-size 1 \
  --log-every 20 \
  --save-every "${SAVE_EVERY:-5000}" \
  --keep-step-checkpoints \
  ${EXTRA_ARGS:-}

find "${OUT}" -maxdepth 1 -type f -print -exec ls -lh {} \;
