#!/usr/bin/env bash
#SBATCH --job-name=neo-exp2-joint
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/neo-exp2-%j.out
#SBATCH --error=logs/neo-exp2-%j.out
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
CONFIG="${CONFIG:-configs/mobile_ov_neodragon_bridge_v2.yaml}"
MANIFEST="${MANIFEST:-data/openvid_neodragon_2s_latents/latent_manifest.csv}"
BRIDGE_CKPT="${BRIDGE_CKPT:-}"
OUT="${OUT:-output/neo_exp2_joint_flow_distill/${SLURM_JOB_ID:-local}}"
STEPS="${STEPS:-28000}"
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

if [[ -z "${BRIDGE_CKPT}" || ! -f "${BRIDGE_CKPT}" ]]; then
  echo "Set BRIDGE_CKPT to Experiment 1 neodragon_text_bridge_latest.pt" >&2
  exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing MANIFEST=${MANIFEST}" >&2
  exit 1
fi
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "Missing torchrun=${TORCHRUN_BIN}" >&2
  exit 1
fi

echo "Experiment 2: joint bridge + full DiT flow matching and teacher distillation"
echo "CONFIG=${CONFIG} MANIFEST=${MANIFEST} BRIDGE_CKPT=${BRIDGE_CKPT} OUT=${OUT} STEPS=${STEPS}"
nvidia-smi || true

"${TORCHRUN_BIN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train_neodragon_dit_bridge.py \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --bridge-ckpt "${BRIDGE_CKPT}" \
  --output-dir "${OUT}" \
  --steps "${STEPS}" \
  --batch-size 1 \
  --parallel fsdp \
  --train-bridge \
  --train-last-n-blocks 0 \
  --lr "${DIT_LR:-3e-6}" \
  --bridge-lr "${BRIDGE_LR:-1e-5}" \
  --dit-warmup-steps 2000 \
  --flow-weight 0.3 \
  --flow-final-weight "${FLOW_FINAL_WEIGHT:-0.1}" \
  --cooldown-steps 4000 \
  --distill-weight 1.0 \
  --distill-cos-weight 0.1 \
  --preservation-weight 0.5 \
  --preservation-cos-weight 0.05 \
  --preservation-every 4 \
  --bridge-repr-weight 0.1 \
  --caption-aug \
  --caption-variant-columns caption_short,caption_medium,caption_long \
  --caption-variant-weights 5,4,1 \
  --log-every 20 \
  --save-every 1000 \
  --keep-step-checkpoints \
  ${EXTRA_ARGS:-}

find "${OUT}" -maxdepth 1 -type f -print -exec ls -lh {} \;
