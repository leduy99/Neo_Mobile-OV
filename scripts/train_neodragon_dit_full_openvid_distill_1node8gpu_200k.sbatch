#!/usr/bin/env bash
#SBATCH --job-name=neo-dit-distill
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

# Full-weight NeoDragon DiT training with a frozen original-NeoDragon functional teacher.
# Submit from repo root after preparing data:
#   bash scripts/prepare_openvid_neodragon_2s.sh
#   sbatch scripts/train_neodragon_dit_full_openvid_distill_1node8gpu_200k.sbatch

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_ENV}/bin/torchrun}"

CONFIG="${CONFIG:-configs/mobile_ov_neodragon.yaml}"
MANIFEST="${MANIFEST:-data/openvid_neodragon_2s_latents/latent_manifest.csv}"
BRIDGE_CKPT="${BRIDGE_CKPT:-checkpoints/hf_mobile_ov/neo_bridge_8gpu_200k_merged_caps/17002399/neodragon_text_bridge_latest.pt}"
OUT="${OUT:-output/neo_dit_distill/${SLURM_JOB_ID:-local}}"

STEPS="${STEPS:-200000}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
LOG_EVERY="${LOG_EVERY:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LR="${LR:-5e-6}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PARALLEL="${PARALLEL:-fsdp}"
DISTILL_WEIGHT="${DISTILL_WEIGHT:-0.05}"
DISTILL_COS_WEIGHT="${DISTILL_COS_WEIGHT:-0.01}"

mkdir -p logs "${OUT}"

if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "Missing torchrun: ${TORCHRUN_BIN}" >&2
  exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing MANIFEST=${MANIFEST}" >&2
  echo "Run first:" >&2
  echo "  bash scripts/prepare_openvid_neodragon_2s.sh" >&2
  echo "  sbatch scripts/encode_neodragon_vae_latents_1node8gpu.sbatch" >&2
  exit 1
fi
if [[ ! -f "${BRIDGE_CKPT}" ]]; then
  echo "Missing BRIDGE_CKPT=${BRIDGE_CKPT}" >&2
  exit 1
fi

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
echo "MANIFEST=${MANIFEST}"
echo "BRIDGE_CKPT=${BRIDGE_CKPT}"
echo "OUT=${OUT}"
echo "STEPS=${STEPS} SAVE_EVERY=${SAVE_EVERY} BATCH_SIZE=${BATCH_SIZE} LR=${LR} PARALLEL=${PARALLEL}"
echo "DISTILL_WEIGHT=${DISTILL_WEIGHT} DISTILL_COS_WEIGHT=${DISTILL_COS_WEIGHT}"
nvidia-smi || true

"${TORCHRUN_BIN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train_neodragon_dit_bridge.py \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --bridge-ckpt "${BRIDGE_CKPT}" \
  --output-dir "${OUT}/dit" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --train-last-n-blocks 0 \
  --parallel "${PARALLEL}" \
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
  --log-every "${LOG_EVERY}" \
  --save-every "${SAVE_EVERY}" \
  --target-fps 24 \
  --use-precomputed-latents \
  --caption-aug \
  --caption-variant-columns caption_short,caption_medium,caption_long \
  --caption-variant-weights 1,1,1 \
  --caption-fallback-column caption \
  --distill-weight "${DISTILL_WEIGHT}" \
  --distill-cos-weight "${DISTILL_COS_WEIGHT}" \
  --keep-step-checkpoints \
  ${EXTRA_ARGS:-}

find "${OUT}" -maxdepth 3 -type f -print -exec ls -lh {} \;
