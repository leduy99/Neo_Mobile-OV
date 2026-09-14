#!/usr/bin/env bash
# T2I/T2V connector + DiT adaptation; SmolVLM2/VAE frozen. Independent of the 100k alignment run.
#SBATCH --job-name=mov-s2-100k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-s2-15k-%j.out
#SBATCH --error=logs/mov-s2-15k-%j.out

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repo root}}"
export PROJECT_ROOT="${PWD}"
export PHASE=2
export OUT="${OUT:-output/mobileov_stage2_from_alignment15k/${SLURM_JOB_ID}}"
CUSTOM_INIT="${INIT_CONNECTOR:-}"
export INIT_CONNECTOR="${INIT_CONNECTOR:-output/mobileov_stage1_alignment/17507068/stage1_connector_latest.pt}"
# Use the already trained remote file, then a local HF cache. Never initialize randomly on a missing file.
if [[ ! -s "${INIT_CONNECTOR}" ]]; then
    CACHE="checkpoints/hf_mobile_ov/mobileov_stage1_alignment/17507068/stage1_connector_latest.pt"
    if [[ -z "${CUSTOM_INIT}" && -s "${CACHE}" ]]; then
        export INIT_CONNECTOR="${CACHE}"
    else
        echo "Missing Stage-1 checkpoint: ${INIT_CONNECTOR}. Restore it or set INIT_CONNECTOR." >&2
        exit 1
    fi
fi
export STEPS="${STEPS:-100000}"
export ACCUMULATION="${ACCUMULATION:-2}" # 8 T2I + 8 T2V examples per optimizer update.
export LR="${LR:-2e-5}"
export SAVE_EVERY="${SAVE_EVERY:-10000}"
export ARCHIVE_EVERY="${ARCHIVE_EVERY:-10000}"
export VALIDATE_EVERY="${VALIDATE_EVERY:-1000}"
export MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-244800}"
echo "Stage 2 from alignment 17507068: ${STEPS} new updates; fresh optimizer; LR=${LR}."
echo "Public T2I/T2V data, pyramid-flow MSE only, no native distillation/EMA/reconstruction loss."
bash scripts/train_mobileov_stage1_alignment_1node8gpu.sbatch
if [[ -s "${OUT}/training_paused.json" && ! -e "${OUT}/training_complete.json" ]]; then
    export RESUME="${OUT}/stage2_resume.pt"
    test -s "${RESUME}"
    sbatch --dependency="afterok:${SLURM_JOB_ID}" --export=ALL \
        scripts/train_mobileov_stage2_from_alignment15k_1node8gpu.sbatch
fi
