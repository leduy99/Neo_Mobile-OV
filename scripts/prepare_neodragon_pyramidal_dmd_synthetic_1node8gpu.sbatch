#!/usr/bin/env bash
#SBATCH --job-name=neo-dmd-data
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/neo-dmd-data-%j.out
#SBATCH --error=logs/neo-dmd-data-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:8
##SBATCH --gres=gpu:A100-SXM4-80GB:8

# Build the prompt-only synthetic data set required by NeoDragon Pyramidal DMD.
# It deliberately does not encode OpenVid videos: the frozen multi-step teacher
# generates the latent targets that define the student distribution.
set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_ENV}/bin/torchrun}"
CONFIG="${CONFIG:-configs/mobile_ov_neodragon.yaml}"
OUT="${OUT:-data/neodragon_pyramidal_dmd_320p}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MAX_PROMPTS="${MAX_PROMPTS:-350000}"

if [[ -n "${PROMPTS:-}" ]]; then
  PROMPTS="${PROMPTS}"
else
  for candidate in \
    "download_data/data/openvid/manifests/openvid_all_recaptions_merged.csv" \
    "download_data/data/openvid/manifests/openvid_all_recaptions.csv" \
    "data/openvid_neodragon_2s/manifest.csv"; do
    if [[ -f "${candidate}" ]]; then
      PROMPTS="${candidate}"
      break
    fi
  done
fi

if [[ -z "${PROMPTS:-}" || ! -f "${PROMPTS}" ]]; then
  echo "Set PROMPTS=/absolute/or/repo-relative/prompts.csv; no default prompt manifest was found." >&2
  exit 1
fi
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "Missing torchrun: ${TORCHRUN_BIN}" >&2
  exit 1
fi

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

echo "NeoDragon Pyramidal-DMD synthetic teacher dataset"
echo "PROMPTS=${PROMPTS} OUT=${OUT} MAX_PROMPTS=${MAX_PROMPTS} NPROC=${NPROC_PER_NODE}"
echo "Teacher=released monolithic 20/10 CFG; output=[anchor + six video latent units]"
nvidia-smi || true

"${TORCHRUN_BIN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  tools/prepare_neodragon_dmd_synthetic_data.py \
  --config "${CONFIG}" \
  --prompts "${PROMPTS}" \
  --output-dir "${OUT}" \
  --max-prompts "${MAX_PROMPTS}" \
  --first-unit-steps "${FIRST_UNIT_STEPS:-20}" \
  --video-unit-steps "${VIDEO_UNIT_STEPS:-10}" \
  --dtype "${DTYPE:-bf16}" \
  ${EXTRA_ARGS:-}

test -f "${OUT}/manifest.csv"
test -f "${OUT}/metadata.json"
wc -l "${OUT}/manifest.csv"
