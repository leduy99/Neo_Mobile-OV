#!/usr/bin/env bash
#SBATCH --job-name=neo-dmd-v2a
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/neo-dmd-v2a-%j.out
#SBATCH --error=logs/neo-dmd-v2a-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:8

# Controlled alternative to DMD-v2. It keeps the same teacher data, optimizer,
# losses, and budget, but excludes native unit zero from optimization. Stored
# unit zero remains teacher-forced causal history; deployment supplies an
# SSD1B, DreamLite, or source-image first-frame anchor.
set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_ENV}/bin/torchrun}"
CONFIG="${CONFIG:-configs/mobile_ov_neodragon.yaml}"
MANIFEST="${MANIFEST:-data/neodragon_pyramidal_dmd_320p/manifest_partial.csv}"
RUN_NAME="${RUN_NAME:-dmd_repro_v2_anchor_alt_10k}"
OUT="${OUT:-output/neodragon_pyramidal_dmd_reproduction/${RUN_NAME}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

test -f "${MANIFEST}"
test -x "${TORCHRUN_BIN}"
mkdir -p logs "${OUT}"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TORCH_HOME="${TORCH_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"
export TMPDIR="${TMPDIR:-/proj/cvl/users/x_fahkh2/caches}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-720}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "NeoDragon Pyramidal-DMD v2 anchor alternative: video units 1..6 only"
echo "MANIFEST=${MANIFEST} OUT=${OUT} RUN_NAME=${RUN_NAME} RESUME=${RESUME:-none}"
echo "STEPS=${STEPS:-10000} GLOBAL_BATCH=$((NPROC_PER_NODE * ${BATCH_SIZE:-1})) student:fake=1:${FAKE_UPDATES:-2}"
echo "Controlled delta from V2: exclude unit zero; all DMD hyperparameters unchanged"
nvidia-smi || true

"${TORCHRUN_BIN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train_neodragon_pyramidal_dmd.py \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --output-dir "${OUT}" \
  --resume "${RESUME:-none}" \
  --steps "${STEPS:-10000}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --student-lr "${STUDENT_LR:-1e-6}" \
  --fake-lr "${FAKE_LR:-1e-6}" \
  --fake-updates "${FAKE_UPDATES:-2}" \
  --dmd-weight "${DMD_WEIGHT:-1.0}" \
  --cauchy-weight "${CAUCHY_WEIGHT:-0.5}" \
  --max-dmd-weight "${MAX_DMD_WEIGHT:-100.0}" \
  --no-include-first-unit \
  --external-anchor-alternative \
  --teacher-video-guidance "${TEACHER_VIDEO_GUIDANCE:-5.0}" \
  --save-every "${SAVE_EVERY:-500}" \
  --archive-every "${ARCHIVE_EVERY:-5000}" \
  --log-every "${LOG_EVERY:-25}" \
  --dtype "${DTYPE:-bf16}" \
  ${EXTRA_ARGS:-}

find "${OUT}" -maxdepth 1 -type f -print -exec ls -lh {} \;
