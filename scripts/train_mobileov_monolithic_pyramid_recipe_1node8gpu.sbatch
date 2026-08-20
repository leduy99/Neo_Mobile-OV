#!/usr/bin/env bash
# Evidence-driven Mobile-OV + NeoDragon multistep adaptation on Berzelius.
# Phase 1 keeps the validated bridge frozen while adapting the DiT on the
# correct Pyramidal-Flow contract. Phase 2 resumes the full phase-1 checkpoint
# and conservatively opens the bridge under frozen-teacher protection.
#SBATCH --job-name=neo-pfm-joint
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/neo-pfm-joint-%j.out
#SBATCH --error=logs/neo-pfm-joint-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:8

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_ENV}/bin/torchrun}"
CONFIG="${CONFIG:-configs/mobile_ov_neodragon.yaml}"
MANIFEST="${MANIFEST:-data/openvid_neodragon_2s_latents/latent_manifest.csv}"
DEFAULT_BRIDGE_CKPT="output/neo_monolithic_cfg_bridge_v2/monolithic_cfg_bridge_v2_17327720/neodragon_text_bridge_latest.pt"
BRIDGE_CKPT="${BRIDGE_CKPT:-${DEFAULT_BRIDGE_CKPT}}"
RUN_NAME="${RUN_NAME:-mobileov_monolithic_pyramid_${SLURM_JOB_ID}}"
OUT="${OUT:-output/mobileov_monolithic_pyramid_recipe/${RUN_NAME}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
PHASE1_STEPS="${PHASE1_STEPS:-300}"
PHASE2_STEPS="${PHASE2_STEPS:-700}"

cd "${ROOT}"
test -s "${MANIFEST}"
test -s "${BRIDGE_CKPT}"
test -x "${TORCHRUN_BIN}"
mkdir -p logs "${OUT}/phase1" "${OUT}/phase2"

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
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-90}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

COMMON_ARGS=(
  --config "${CONFIG}"
  --target-stack multistep
  --manifest "${MANIFEST}"
  --bridge-ckpt "${BRIDGE_CKPT}"
  --batch-size "${BATCH_SIZE}"
  --max-samples "${MAX_SAMPLES}"
  --use-precomputed-latents
  --train-last-n-blocks 0
  --parallel fsdp
  --dit-trainable-fp32
  --flow-contract pyramid
  --stage-sampling-ratio 1,1,1
  --history-corrupt-max 0.3333333333333333
  --flow-weight 1.0
  --distill-weight 1.0
  --distill-cos-weight 0.1
  --preservation-weight 0.5
  --preservation-cos-weight 0.05
  --preservation-every 1
  --clip-grad-norm 1.0
  --diagnostic-every 50
  --log-every 10
)

echo "Mobile-OV corrected Pyramidal-Flow training on Berzelius"
echo "MANIFEST=${MANIFEST} BRIDGE_CKPT=${BRIDGE_CKPT} OUT=${OUT}"
echo "world_size=${NPROC_PER_NODE} batch_per_gpu=${BATCH_SIZE} global_batch=$((NPROC_PER_NODE * BATCH_SIZE))"
nvidia-smi || true

echo "Phase 1/${PHASE1_STEPS}: bridge frozen, correct PFM plus teacher trust region"
"${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train_neodragon_dit_bridge.py \
  "${COMMON_ARGS[@]}" \
  --output-dir "${OUT}/phase1" \
  --steps "${PHASE1_STEPS}" \
  --lr "${PHASE1_LR:-3e-5}" \
  --dit-warmup-steps "${PHASE1_WARMUP_STEPS:-50}" \
  --no-train-bridge \
  --save-every "${PHASE1_SAVE_EVERY:-100}" \
  --archive-every "${PHASE1_ARCHIVE_EVERY:-300}" \
  ${PHASE1_EXTRA_ARGS:-}

PHASE1_CKPT="${OUT}/phase1/neodragon_dit_bridge_latest.pt"
test -s "${PHASE1_CKPT}"

echo "Phase 2/${PHASE2_STEPS}: resume phase 1 and open bridge conservatively"
"${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train_neodragon_dit_bridge.py \
  "${COMMON_ARGS[@]}" \
  --init-joint-ckpt "${PHASE1_CKPT}" \
  --output-dir "${OUT}/phase2" \
  --steps "${PHASE2_STEPS}" \
  --lr "${PHASE2_DIT_LR:-3e-6}" \
  --dit-warmup-steps "${PHASE2_DIT_WARMUP_STEPS:-100}" \
  --bridge-lr "${PHASE2_BRIDGE_LR:-1e-6}" \
  --train-bridge \
  --bridge-trainable-fp32 \
  --bridge-start-step 0 \
  --bridge-warmup-steps "${PHASE2_BRIDGE_WARMUP_STEPS:-100}" \
  --bridge-repr-weight 0.25 \
  --bridge-functional-weight 0.5 \
  --bridge-functional-cos-weight 0.05 \
  --bridge-functional-every 1 \
  --save-every "${PHASE2_SAVE_EVERY:-100}" \
  --archive-every "${PHASE2_ARCHIVE_EVERY:-200}" \
  ${PHASE2_EXTRA_ARGS:-}

echo "Completed corrected Pyramidal-Flow recipe: ${OUT}"
find "${OUT}" -maxdepth 2 -type f -print -exec ls -lh {} \;
