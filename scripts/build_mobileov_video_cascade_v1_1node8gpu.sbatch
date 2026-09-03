#!/usr/bin/env bash
#SBATCH --job-name=mov-data-video
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/mov-data-video-%j.out
#SBATCH --error=logs/mov-data-video-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:8

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${CONDA_ENV}/bin/python"
TORCHRUN_BIN="${CONDA_ENV}/bin/torchrun"
INPUT_MANIFEST="${INPUT_MANIFEST:-data/openvid_neodragon_2s_latents/latent_manifest.csv}"
OUT="${OUT:-data/mobileov_video_cascade_v1}"
CONFIG="${CONFIG:-configs/mobileov_data_v1.yaml}"
MAX_SAMPLES="${MAX_SAMPLES:-500000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

test -x "${PYTHON_BIN}"
test -x "${TORCHRUN_BIN}"
test -s "${INPUT_MANIFEST}"
test -s "${CONFIG}"
mkdir -p logs "${OUT}" "${OUT}/shards"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TORCH_HOME="${TORCH_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TMPDIR="${TMPDIR:-/proj/cvl/users/x_fahkh2/caches}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "MobileOV-Data-v1 video cascade"
echo "INPUT_MANIFEST=${INPUT_MANIFEST} MAX_SAMPLES=${MAX_SAMPLES} OUT=${OUT}"
echo "Gate 1=decode; Gate 2=multi-frame SigLIP2; Gate 3=non-static/non-cut motion"
nvidia-smi

"${TORCHRUN_BIN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  tools/score_mobileov_video_data.py \
  --input-manifest "${INPUT_MANIFEST}" \
  --output-dir "${OUT}" \
  --max-samples "${MAX_SAMPLES}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --loader-workers "${LOADER_WORKERS:-8}" \
  --sample-frames "${SAMPLE_FRAMES:-6}" \
  --dtype "${DTYPE:-bf16}" \
  --resume

"${PYTHON_BIN}" tools/finalize_mobileov_video_data.py \
  --scores "${OUT}/video_scores.csv" \
  --output "${OUT}/video_high_precision.csv" \
  --summary "${OUT}/video_high_precision.summary.json" \
  --alignment-quantile "${ALIGNMENT_QUANTILE:-0.25}" \
  --motion-low-quantile "${MOTION_LOW_QUANTILE:-0.10}" \
  --transition-high-quantile "${TRANSITION_HIGH_QUANTILE:-0.99}" \
  --absolute-min-frame-diff "${ABSOLUTE_MIN_FRAME_DIFF:-0.005}" \
  --absolute-min-optical-flow "${ABSOLUTE_MIN_OPTICAL_FLOW:-0.05}" \
  --absolute-max-transition-diff "${ABSOLUTE_MAX_TRANSITION_DIFF:-0.35}"

PROMPT_BANK="${PROMPT_BANK:-data/mobileov_data_v1/prompt_banks/anchor_teacher_100k.csv}"
PROMPT_SUMMARY="${PROMPT_BANK%.csv}.summary.json"
if [[ "${FORCE_PROMPT_BANK:-0}" == "1" ]]; then
  "${PYTHON_BIN}" tools/build_mobileov_anchor_prompt_bank.py \
    --config "${CONFIG}" \
    --force
elif [[ -s "${PROMPT_BANK}" && -s "${PROMPT_SUMMARY}" ]]; then
  echo "Reusing completed prompt bank: ${PROMPT_BANK}"
else
  "${PYTHON_BIN}" tools/build_mobileov_anchor_prompt_bank.py --config "${CONFIG}"
fi

wc -l "${OUT}/video_high_precision.csv" "${PROMPT_BANK}"
cat "${OUT}/video_high_precision.summary.json"
cat "${PROMPT_SUMMARY}"
