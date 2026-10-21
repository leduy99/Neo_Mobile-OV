#!/usr/bin/env bash
# Extend the original epoch-3 run with its Adam state; never start from A/B/history.
#SBATCH --job-name=mov-s2-extend
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-s2-extend-%j.out
#SBATCH --error=logs/mov-s2-extend-%j.out

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repo root}}"
export PROJECT_ROOT="${PWD}"
module load buildenv-nvhpc/24.5-cuda12.4
CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TMPDIR="${TMPDIR:-${HF_HOME}/tmp}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_ROOT="${DATA_ROOT:-download_data/data/univideo_stage1_expanded650}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
export OUT="${OUT:-output/mobileov_stage2_media${TOTAL_EPOCHS}epochs_extension/${SLURM_JOB_ID}}"
export RESUME="${RESUME:-output/mobileov_stage2_media3epochs/17641364/stage2_resume.pt}"
export AUTO_CONTINUE="${AUTO_CONTINUE:-1}"
if [[ "${TOTAL_EPOCHS}" != 4 && "${TOTAL_EPOCHS}" != 5 ]]; then
    echo "TOTAL_EPOCHS must be 4 or 5 (total passes, not additional passes)." >&2; exit 1
fi
if [[ "${AUTO_CONTINUE}" != 0 && "${AUTO_CONTINUE}" != 1 ]]; then
    echo "AUTO_CONTINUE must be 0 or 1." >&2; exit 1
fi
if [[ -n "${STEPS:-}" || -n "${INIT_CONNECTOR:-}" || -n "${INIT_STAGE2:-}" || -n "${LR:-}" ]]; then
    echo "Keep the original recipe: no STEPS/INIT_CONNECTOR/INIT_STAGE2/LR overrides." >&2; exit 1
fi
if [[ ! -s "${RESUME}" ]]; then
    echo "Missing original optimizer checkpoint: ${RESUME}. No inference-weight fallback." >&2; exit 1
fi
mkdir -p logs "${TMPDIR}"
"${CONDA_ENV}/bin/python" -s tools/plan_mobileov_stage2_epoch_extension.py \
    --resume "${RESUME}" --data-root "${DATA_ROOT}" --output-dir "${OUT}" --epochs "${TOTAL_EPOCHS}"
echo "Resume original weights + Adam; total epochs=${TOTAL_EPOCHS}; LR2e-5 with NO warmup restart."
echo "Unchanged media/tasks, reconstruction .1, uniform units, clean history; frozen SmolVLM2/VAE."
echo "Save every 10k and EXACT epoch boundaries. Preserve original outputs; OUT=${OUT}."
"${CONDA_ENV}/bin/torchrun" --standalone --nnodes=1 --nproc_per_node=8 \
    tools/train_mobileov_stage1.py --phase 2 \
    --config "${CONFIG:-configs/mobile_ov_neodragon.yaml}" \
    --data-root "${DATA_ROOT}" --output-dir "${OUT}" --resume "${RESUME}" \
    --media-epochs "${TOTAL_EPOCHS}" --extend-media-epochs \
    --min-train-videos 500000 --require-expanded-release --accumulation 2 \
    --lr 2e-5 --warmup 50 --reconstruction-weight .1 \
    --save-every 10000 --archive-every 10000 --validate-every "${VALIDATE_EVERY:-2000}" \
    --validation-samples "${VALIDATION_SAMPLES:-2}" --workers "${WORKERS:-2}" \
    --max-runtime-seconds "${MAX_RUNTIME_SECONDS:-244800}"
if [[ -s "${OUT}/training_paused.json" && ! -e "${OUT}/training_complete.json" ]]; then
    export RESUME="${OUT}/stage2_resume.pt"
    test -s "${RESUME}"
    if [[ "${AUTO_CONTINUE}" == 1 ]]; then
        NEXT_JOB="$(sbatch --parsable --dependency="afterok:${SLURM_JOB_ID}" --export=ALL \
            scripts/train_mobileov_stage2_extend_epochs_1node8gpu.sbatch)"
        printf '%s\n' "${NEXT_JOB}" >> "${OUT}/continuation_jobs.txt"
        echo "Queued continuation ${NEXT_JOB}; same OUT, optimizer, LR, sampler position and per-rank RNG."
    else
        echo "Paused safely. Resume with TOTAL_EPOCHS=${TOTAL_EPOCHS} OUT=${OUT} RESUME=${RESUME}."
    fi
fi
