#!/usr/bin/env bash
# Fresh alignment, same recipe as the 15k run; automatic continuation if close to walltime.
#SBATCH --job-name=mov-s1-100k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-s1-100k-%j.out
#SBATCH --error=logs/mov-s1-100k-%j.out

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repo root}}"
export PROJECT_ROOT="${PWD}"
export PHASE=1
export OUT="${OUT:-output/mobileov_stage1_alignment_100k/${SLURM_JOB_ID}}"
export STEPS="${STEPS:-100000}"
export SAVE_EVERY="${SAVE_EVERY:-10000}"
export ARCHIVE_EVERY="${ARCHIVE_EVERY:-10000}"
export MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-244800}" # 68 h leaves time to validate/save.
if [[ -n "${RESUME:-}" ]]; then
    echo "Continuing the same ${STEPS}-step run from ${RESUME}; no optimizer/LR reset."
else
    echo "Fresh random connector; no initialization from checkpoint 17507068."
fi
bash scripts/train_mobileov_stage1_alignment_1node8gpu.sbatch
if [[ -s "${OUT}/training_paused.json" && ! -e "${OUT}/training_complete.json" ]]; then
    export RESUME="${OUT}/stage1_resume.pt"
    test -s "${RESUME}"
    echo "Time budget reached; submitting continuation with the same output and optimizer."
    sbatch --dependency="afterok:${SLURM_JOB_ID}" --export=ALL \
        scripts/train_mobileov_stage1_alignment_100k_1node8gpu.sbatch
fi
