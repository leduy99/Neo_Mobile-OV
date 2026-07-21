#!/usr/bin/env bash
#SBATCH --job-name=neo-exp5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/neo-exp5-%j.out
#SBATCH --error=logs/neo-exp5-%j.out
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
OUT="${OUT:-output/neo_exp5_staged}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

if [[ -z "${BRIDGE_CKPT:-}" ]]; then
  REMOTE_EXP1="output/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt"
  LOCAL_EXP1="checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt"
  if [[ -f "${REMOTE_EXP1}" ]]; then
    BRIDGE_CKPT="${REMOTE_EXP1}"
  else
    BRIDGE_CKPT="${LOCAL_EXP1}"
  fi
fi

mkdir -p logs "${OUT}"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export FSDP_USE_ORIG_PARAMS=true
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-60}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for required in "${MANIFEST}" "${BRIDGE_CKPT}" "${TORCHRUN_BIN}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 1
  fi
done

echo "Experiment 5: protected bridge refinement plus full NeoDragon DiT adaptation"
echo "CONFIG=${CONFIG}"
echo "MANIFEST=${MANIFEST}"
echo "BRIDGE_CKPT=${BRIDGE_CKPT}"
echo "OUT=${OUT} RESUME=${RESUME:-auto} NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "STEPS=${STEPS:-255000} PHASE_A_STEPS=${PHASE_A_STEPS:-10000} PHASE_B_STEPS=${PHASE_B_STEPS:-120000}"
nvidia-smi || true

"${TORCHRUN_BIN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train_neodragon_exp5.py \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --bridge-ckpt "${BRIDGE_CKPT}" \
  --output-dir "${OUT}" \
  --resume "${RESUME:-auto}" \
  --steps "${STEPS:-255000}" \
  --phase-a-steps "${PHASE_A_STEPS:-10000}" \
  --phase-b-steps "${PHASE_B_STEPS:-120000}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --parallel fsdp \
  --dit-lr "${DIT_LR:-3e-6}" \
  --bridge-lr "${BRIDGE_LR:-1e-6}" \
  --dit-warmup-steps "${DIT_WARMUP_STEPS:-2000}" \
  --bridge-warmup-steps "${BRIDGE_WARMUP_STEPS:-2000}" \
  --bridge-cooldown-steps "${BRIDGE_COOLDOWN_STEPS:-10000}" \
  --final-cooldown-steps "${FINAL_COOLDOWN_STEPS:-20000}" \
  --flow-start-weight "${FLOW_START_WEIGHT:-0.05}" \
  --flow-weight "${FLOW_WEIGHT:-0.3}" \
  --flow-final-weight "${FLOW_FINAL_WEIGHT:-0.1}" \
  --distill-weight "${DISTILL_WEIGHT:-1.0}" \
  --distill-cos-weight "${DISTILL_COS_WEIGHT:-0.1}" \
  --preservation-weight "${PRESERVATION_WEIGHT:-0.5}" \
  --preservation-cos-weight "${PRESERVATION_COS_WEIGHT:-0.05}" \
  --preservation-every "${PRESERVATION_EVERY:-2}" \
  --bridge-repr-weight "${BRIDGE_REPR_WEIGHT:-0.5}" \
  --bridge-raw-token-weight "${BRIDGE_RAW_TOKEN_WEIGHT:-0.25}" \
  --bridge-normalized-token-weight "${BRIDGE_NORMALIZED_TOKEN_WEIGHT:-1.0}" \
  --bridge-cos-weight "${BRIDGE_COS_WEIGHT:-0.5}" \
  --bridge-token-norm-weight "${BRIDGE_TOKEN_NORM_WEIGHT:-0.1}" \
  --bridge-pooled-weight "${BRIDGE_POOLED_WEIGHT:-0.25}" \
  --bridge-pooled-cos-weight "${BRIDGE_POOLED_COS_WEIGHT:-0.2}" \
  --bridge-relational-weight "${BRIDGE_RELATIONAL_WEIGHT:-0.1}" \
  --bridge-functional-weight "${BRIDGE_FUNCTIONAL_WEIGHT:-1.0}" \
  --bridge-functional-cos-weight "${BRIDGE_FUNCTIONAL_COS_WEIGHT:-0.1}" \
  --bridge-functional-every "${BRIDGE_FUNCTIONAL_EVERY:-2}" \
  --bridge-functional-batch-size "${BRIDGE_FUNCTIONAL_BATCH_SIZE:-1}" \
  --caption-variant-columns caption_short,caption_medium,caption_long \
  --caption-variant-weights 1,1,1 \
  --diagnostic-every "${DIAGNOSTIC_EVERY:-500}" \
  --log-every "${LOG_EVERY:-100}" \
  --save-latest-every "${SAVE_LATEST_EVERY:-5000}" \
  --save-archive-every "${SAVE_ARCHIVE_EVERY:-20000}" \
  ${EXTRA_ARGS:-}

find "${OUT}" -maxdepth 1 -type f -print -exec ls -lh {} \;
