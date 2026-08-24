#!/usr/bin/env bash
#SBATCH --job-name=neo-jv3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/neo-jv3-%j.out
#SBATCH --error=logs/neo-jv3-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:8

# Controlled Joint V3 continuation from the validated fixed-bridge V2 step 10k.
# VARIANT=distill keeps the released teacher trust region and bridge contracts.
# VARIANT=flow_only removes all teacher objectives while keeping every other
# training setting identical.
set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module load buildenv-nvhpc/24.5-cuda12.4
fi

VARIANT="${VARIANT:-distill}"
case "${VARIANT}" in
  distill|flow_only) ;;
  *)
    echo "ERROR: VARIANT must be distill or flow_only, got ${VARIANT}" >&2
    exit 2
    ;;
esac

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_ENV}/bin/torchrun}"
HF_BIN="${HF_BIN:-${CONDA_ENV}/bin/hf}"
CONFIG="${CONFIG:-configs/mobile_ov_neodragon.yaml}"
MANIFEST="${MANIFEST:-data/openvid_neodragon_2s_latents/latent_manifest.csv}"
HF_REPO="${HF_REPO:-Amshaker/Mobile-OV}"
HF_FILE="mobileov_monolithic_pyramid_cfg_dit/mobileov_monolithic_pyramid_cfg_dit_17346642/neodragon_dit_bridge_step010000.pt"
OUTPUT_INIT="output/mobileov_monolithic_pyramid_cfg_dit/mobileov_monolithic_pyramid_cfg_dit_17346642/neodragon_dit_bridge_step010000.pt"
CACHE_INIT="checkpoints/hf_mobile_ov/${HF_FILE}"
RUN_NAME="${RUN_NAME:-joint_v3_${VARIANT}_${SLURM_JOB_ID:-local}}"
OUT="${OUT:-output/mobileov_monolithic_joint_v3/${VARIANT}/${RUN_NAME}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

cd "${ROOT}"
mkdir -p logs "${OUT}"

if [[ -n "${INIT_JOINT_CKPT:-}" ]]; then
  INIT_CKPT="${INIT_JOINT_CKPT}"
elif [[ -s "${OUTPUT_INIT}" ]]; then
  INIT_CKPT="${OUTPUT_INIT}"
elif [[ -s "${CACHE_INIT}" ]]; then
  INIT_CKPT="${CACHE_INIT}"
else
  if [[ ! -x "${HF_BIN}" ]]; then
    echo "ERROR: V2 step-10k checkpoint is missing and hf CLI is unavailable: ${HF_BIN}" >&2
    exit 1
  fi
  mkdir -p checkpoints/hf_mobile_ov
  "${HF_BIN}" download "${HF_REPO}" "${HF_FILE}" --local-dir checkpoints/hf_mobile_ov
  INIT_CKPT="${CACHE_INIT}"
fi

for required in "${MANIFEST}" "${INIT_CKPT}"; do
  if [[ ! -s "${required}" ]]; then
    echo "ERROR: required input is missing or empty: ${required}" >&2
    exit 1
  fi
done
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "ERROR: torchrun is missing or not executable: ${TORCHRUN_BIN}" >&2
  exit 1
fi

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export FSDP_USE_ORIG_PARAMS=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TORCH_HOME="${TORCH_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"
export TMPDIR="${TMPDIR:-/proj/cvl/users/x_fahkh2/caches}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-180}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

COMMON_ARGS=(
  --config "${CONFIG}"
  --target-stack multistep
  --manifest "${MANIFEST}"
  --init-joint-ckpt "${INIT_CKPT}"
  --output-dir "${OUT}"
  --steps "${STEPS:-4000}"
  --batch-size "${BATCH_SIZE:-1}"
  --lr "${DIT_LR:-1e-6}"
  --bridge-lr "${BRIDGE_LR:-1e-7}"
  --train-bridge
  --bridge-start-step 0
  --bridge-warmup-steps "${BRIDGE_WARMUP_STEPS:-500}"
  --train-last-n-blocks 0
  --parallel fsdp
  --dit-trainable-fp32
  --bridge-trainable-fp32
  --use-precomputed-latents
  --flow-contract pyramid
  --stage-sampling-ratio 1,1,1
  --history-corrupt-max "${HISTORY_CORRUPT_MAX:-0.3333333333333333}"
  --flow-weight "${FLOW_WEIGHT:-1.0}"
  --flow-final-weight "${FLOW_FINAL_WEIGHT:-0.5}"
  --cooldown-steps "${COOLDOWN_STEPS:-1000}"
  --dit-warmup-steps "${DIT_WARMUP_STEPS:-200}"
  --clip-grad-norm "${CLIP_GRAD_NORM:-1.0}"
  --diagnostic-every "${DIAGNOSTIC_EVERY:-50}"
  --log-every "${LOG_EVERY:-10}"
  --save-every "${SAVE_EVERY:-500}"
  --archive-every "${ARCHIVE_EVERY:-2000}"
  --seed "${SEED:-20260824}"
)

if [[ "${VARIANT}" == "distill" ]]; then
  TEACHER_OBJECTIVES="enabled"
  OBJECTIVE_ARGS=(
    --objective-mode joint-distill
    --distill-weight "${DISTILL_WEIGHT:-1.0}"
    --distill-cos-weight "${DISTILL_COS_WEIGHT:-0.1}"
    --preservation-weight "${PRESERVATION_WEIGHT:-0.5}"
    --preservation-cos-weight "${PRESERVATION_COS_WEIGHT:-0.05}"
    --preservation-every "${PRESERVATION_EVERY:-2}"
    --bridge-repr-weight "${BRIDGE_REPR_WEIGHT:-0.05}"
    --bridge-repr-final-weight "${BRIDGE_REPR_FINAL_WEIGHT:-0.02}"
    --bridge-raw-token-weight 0.25
    --bridge-normalized-token-weight 1.0
    --bridge-cos-weight 0.5
    --bridge-token-norm-weight 0.1
    --bridge-pooled-weight 0.25
    --bridge-pooled-cos-weight 0.2
    --bridge-relational-weight 0.1
    --bridge-negative-repr-weight "${BRIDGE_NEGATIVE_REPR_WEIGHT:-0.01}"
    --bridge-cfg-token-delta-weight "${BRIDGE_CFG_TOKEN_DELTA_WEIGHT:-0.025}"
    --bridge-cfg-token-delta-cos-weight "${BRIDGE_CFG_TOKEN_DELTA_COS_WEIGHT:-0.025}"
    --bridge-cfg-pooled-delta-weight "${BRIDGE_CFG_POOLED_DELTA_WEIGHT:-0.01}"
    --bridge-cfg-pooled-delta-cos-weight "${BRIDGE_CFG_POOLED_DELTA_COS_WEIGHT:-0.02}"
    --bridge-functional-weight "${BRIDGE_FUNCTIONAL_WEIGHT:-0.25}"
    --bridge-functional-cos-weight "${BRIDGE_FUNCTIONAL_COS_WEIGHT:-0.025}"
    --bridge-functional-every "${BRIDGE_FUNCTIONAL_EVERY:-2}"
    --bridge-functional-batch-size 1
    --cfg-distill-weight "${CFG_DISTILL_WEIGHT:-0.5}"
    --cfg-distill-cos-weight "${CFG_DISTILL_COS_WEIGHT:-0.1}"
    --bridge-cfg-functional-weight "${BRIDGE_CFG_FUNCTIONAL_WEIGHT:-0.1}"
    --bridge-cfg-functional-cos-weight "${BRIDGE_CFG_FUNCTIONAL_COS_WEIGHT:-0.02}"
    --cfg-every "${CFG_EVERY:-2}"
    --cfg-scale "${CFG_SCALE:-5.0}"
  )
else
  TEACHER_OBJECTIVES="disabled"
  OBJECTIVE_ARGS=(--objective-mode flow-only)
fi

echo "Mobile-OV Joint V3 variant=${VARIANT}"
echo "INIT_CKPT=${INIT_CKPT} MANIFEST=${MANIFEST} OUT=${OUT}"
echo "Contract: external anchor, units 1..6, Pyramidal Flow, train bridge + full DiT"
echo "Steps=${STEPS:-4000} global_batch=$((NPROC_PER_NODE * ${BATCH_SIZE:-1})) DiT/bridge LR=${DIT_LR:-1e-6}/${BRIDGE_LR:-1e-7}"
echo "Only controlled difference: protected teacher objectives are ${TEACHER_OBJECTIVES}"
nvidia-smi || true

"${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train_neodragon_dit_bridge.py \
  "${COMMON_ARGS[@]}" \
  "${OBJECTIVE_ARGS[@]}" \
  ${EXTRA_ARGS:-}

echo "Completed Joint V3 variant=${VARIANT}: ${OUT}"
find "${OUT}" -maxdepth 1 -type f -print -exec ls -lh {} \;
