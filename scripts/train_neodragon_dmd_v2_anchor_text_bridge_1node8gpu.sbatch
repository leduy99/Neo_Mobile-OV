#!/usr/bin/env bash
#SBATCH --job-name=neo-dmd-v2a-bridge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/neo-dmd-v2a-bridge-%j.out
#SBATCH --error=logs/neo-dmd-v2a-bridge-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:8

# Exp1-style bridge distillation for the deployed DMD-v2 anchor alternative.
# Native multistep tokens remain the representation target, while functional
# supervision comes directly from the frozen DMD student that will be used at
# inference. Unit zero is excluded from the bridge functional cycle as well.
set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_ENV}/bin/torchrun}"
CONFIG="${CONFIG:-configs/mobile_ov_neodragon.yaml}"
PROMPTS="${PROMPTS:-download_data/data/openvid/manifests/openvid_all_recaptions_merged.csv}"
DMD_CHECKPOINT="${DMD_CHECKPOINT:-output/neodragon_pyramidal_dmd_reproduction/dmd_repro_v2_anchor_alt_10k/neodragon_pyramidal_dmd_student_step010000.pt}"
RUN_NAME="${RUN_NAME:-dmd_v2_anchor_exp1_functional_${SLURM_JOB_ID}}"
OUT="${OUT:-output/neo_dmd_v2_anchor_text_bridge/${RUN_NAME}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

test -f "${PROMPTS}"
test -f "${DMD_CHECKPOINT}"
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

echo "DMD-v2 anchor bridge: multistep representation + deployed-DMD functional distillation"
echo "PROMPTS=${PROMPTS} DMD_CHECKPOINT=${DMD_CHECKPOINT}"
echo "OUT=${OUT} STEPS=${STEPS:-64000} GLOBAL_BATCH=$((NPROC_PER_NODE * ${BATCH_SIZE:-4}))"
echo "Functional coverage: deployed DMD student, video units 1..6, one call per update"
nvidia-smi || true

INIT_ARGS=()
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  test -f "${INIT_CHECKPOINT}"
  INIT_ARGS+=(--init-checkpoint "${INIT_CHECKPOINT}")
fi

"${TORCHRUN_BIN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train_neodragon_text_bridge.py \
  --config "${CONFIG}" \
  --prompts "${PROMPTS}" \
  --output-dir "${OUT}" \
  --steps "${STEPS:-64000}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --lr "${LR:-5e-5}" \
  --parallel fsdp \
  --target-stack multistep \
  --functional-dit-checkpoint "${DMD_CHECKPOINT}" \
  --functional-dit-required-schedule pyramidal_1-1-1_external_anchor_video_units \
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
  --no-functional-include-first-unit \
  --functional-unit-policy cycle \
  --trainable-fp32 \
  --validation-size "${VALIDATION_SIZE:-256}" \
  --best-every "${BEST_EVERY:-2000}" \
  --save-every "${SAVE_EVERY:-5000}" \
  --archive-every "${ARCHIVE_EVERY:-10000}" \
  --log-every "${LOG_EVERY:-20}" \
  "${INIT_ARGS[@]}" \
  ${EXTRA_ARGS:-}

find "${OUT}" -maxdepth 1 -type f -print -exec ls -lh {} \;
