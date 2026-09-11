#!/usr/bin/env bash
# Connector-only alignment on all three tasks, not old caption distillation.
#SBATCH --job-name=mov-s1-align
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-s1-align-%j.out
#SBATCH --error=logs/mov-s1-align-%j.out

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repo root}}"
module load buildenv-nvhpc/24.5-cuda12.4
CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TMPDIR="${TMPDIR:-${HF_HOME}/tmp}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA_ROOT="${DATA_ROOT:-download_data/data/univideo_stage1}"
OUT="${OUT:-output/mobileov_stage1_alignment/${SLURM_JOB_ID}}"
test -s "${DATA_ROOT}/.stage1_data_complete"
test -x "${CONDA_ENV}/bin/torchrun"
mkdir -p logs "${TMPDIR}"
RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
    test -s "${RESUME}"
    RESUME_ARGS+=(--resume "${RESUME}")
fi
echo "Stage1: SmolVLM2 + MCP + frozen multistep NeoDragon; no external image generator"
echo "DATA_ROOT=${DATA_ROOT} OUT=${OUT} STEPS=${STEPS:-15000} ACCUMULATION=${ACCUMULATION:-3}"
"${CONDA_ENV}/bin/torchrun" --standalone --nnodes=1 --nproc_per_node=8 \
    tools/train_mobileov_stage1.py \
    --config "${CONFIG:-configs/mobile_ov_neodragon.yaml}" \
    --data-root "${DATA_ROOT}" --output-dir "${OUT}" \
    --steps "${STEPS:-15000}" --accumulation "${ACCUMULATION:-3}" \
    --lr "${LR:-1e-4}" --warmup 50 \
    --save-every "${SAVE_EVERY:-1000}" --archive-every "${ARCHIVE_EVERY:-5000}" \
    --validate-every "${VALIDATE_EVERY:-500}" --validation-samples "${VALIDATION_SAMPLES:-2}" \
    --workers "${WORKERS:-2}" "${RESUME_ARGS[@]}"
