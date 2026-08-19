#!/usr/bin/env bash
# Re-distill the Mobile-OV bridge for NeoDragon's released multistep stack.
# This version explicitly matches the fixed negative prompt, native CFG
# direction, and frozen-DiT CFG response discovered by the four-way ablation.
#
# Default: warm-start the validated 64K monolithic video-units bridge.
# Scratch ablation: INIT_CHECKPOINT=none sbatch <this-script>
#SBATCH --job-name=neo-mono-cfg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/neo-mono-cfg-%j.out
#SBATCH --error=logs/neo-mono-cfg-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:8

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_ENV}/bin/torchrun}"
CONFIG="${CONFIG:-configs/mobile_ov_neodragon.yaml}"
PROMPTS="${PROMPTS:-download_data/data/openvid/manifests/openvid_all_recaptions_merged.csv}"
DEFAULT_INIT="output/neo_monolithic_video_units_text_bridge/monolithic_video_units_exp1_functional_17318011/neodragon_text_bridge_latest.pt"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${DEFAULT_INIT}}"
RUN_NAME="${RUN_NAME:-monolithic_cfg_bridge_v2_${SLURM_JOB_ID}}"
OUT="${OUT:-output/neo_monolithic_cfg_bridge_v2/${RUN_NAME}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

test -f "${PROMPTS}"
test -x "${TORCHRUN_BIN}"
mkdir -p logs "${OUT}"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export FSDP_USE_ORIG_PARAMS=true
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TORCH_HOME="${TORCH_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"
export TMPDIR="${TMPDIR:-/proj/cvl/users/x_fahkh2/caches}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-90}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

INIT_ARGS=()
if [[ "${INIT_CHECKPOINT}" != "none" ]]; then
  test -f "${INIT_CHECKPOINT}"
  INIT_ARGS+=(--init-checkpoint "${INIT_CHECKPOINT}" --target-step "${TARGET_STEP:-96000}")
else
  INIT_ARGS+=(--steps "${STEPS:-96000}" --step-offset 0)
fi

echo "NeoDragon monolithic CFG-aware bridge distillation"
echo "PROMPTS=${PROMPTS} INIT_CHECKPOINT=${INIT_CHECKPOINT} OUT=${OUT}"
echo "Contract: released multistep stack, external anchor, video units 1..6, CFG scale=${CFG_SCALE:-5.0}"
nvidia-smi || true

"${TORCHRUN_BIN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train_neodragon_text_bridge.py \
  --config "${CONFIG}" \
  --prompts "${PROMPTS}" \
  --output-dir "${OUT}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --lr "${LR:-1e-5}" \
  --lr-warmup-steps "${LR_WARMUP_STEPS:-500}" \
  --lr-final-scale "${LR_FINAL_SCALE:-0.2}" \
  --parallel fsdp \
  --target-stack multistep \
  --caption-variant-columns caption_short,caption_medium,caption_long \
  --caption-variant-weights 1,1,1 \
  --raw-token-weight 0.25 \
  --normalized-token-weight 1.0 \
  --cos-weight 0.5 \
  --token-norm-weight 0.1 \
  --pooled-weight 0.25 \
  --pooled-cos-weight 0.2 \
  --relational-weight 0.1 \
  --negative-repr-weight "${NEGATIVE_REPR_WEIGHT:-0.1}" \
  --cfg-token-delta-weight "${CFG_TOKEN_DELTA_WEIGHT:-0.5}" \
  --cfg-token-delta-cos-weight "${CFG_TOKEN_DELTA_COS_WEIGHT:-0.5}" \
  --cfg-pooled-delta-weight "${CFG_POOLED_DELTA_WEIGHT:-0.1}" \
  --cfg-pooled-delta-cos-weight "${CFG_POOLED_DELTA_COS_WEIGHT:-0.2}" \
  --functional-weight 1.0 \
  --functional-cos-weight 0.1 \
  --cfg-functional-weight "${CFG_FUNCTIONAL_WEIGHT:-0.25}" \
  --cfg-functional-cos-weight "${CFG_FUNCTIONAL_COS_WEIGHT:-0.05}" \
  --cfg-scale "${CFG_SCALE:-5.0}" \
  --functional-start-step 1 \
  --functional-ramp-steps "${FUNCTIONAL_RAMP_STEPS:-1000}" \
  --functional-every 1 \
  --functional-batch-size 1 \
  --no-functional-include-first-unit \
  --functional-unit-policy cycle \
  --trainable-fp32 \
  --validation-size "${VALIDATION_SIZE:-256}" \
  --best-every "${BEST_EVERY:-2000}" \
  --save-every "${SAVE_EVERY:-5000}" \
  --archive-every "${ARCHIVE_EVERY:-10000}" \
  --archive-start-step "${ARCHIVE_START_STEP:-70000}" \
  --log-every "${LOG_EVERY:-20}" \
  "${INIT_ARGS[@]}" \
  ${EXTRA_ARGS:-}

find "${OUT}" -maxdepth 1 -type f -print -exec ls -lh {} \;
