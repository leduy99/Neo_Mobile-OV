#!/usr/bin/env bash
# Continue Stage 2 from 17519025: 100k -> 150k TOTAL updates, with the original optimizer.
#SBATCH --job-name=mov-s2-150k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-s2-150k-%j.out
#SBATCH --error=logs/mov-s2-150k-%j.out

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repo root}}"
export PROJECT_ROOT="${PWD}"
export PHASE=2 STEPS=150000 ACCUMULATION=2 LR=2e-5
export SAVE_EVERY=10000 ARCHIVE_EVERY=10000 VALIDATE_EVERY=1000 VALIDATION_SAMPLES=2
export EXTEND_STEPS=1
export MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-244800}"
export OUT="${OUT:-output/mobileov_stage2_extend150k/${SLURM_JOB_ID}}"
export RESUME="${RESUME:-output/mobileov_stage2_from_alignment15k/17519025/stage2_resume.pt}"
export EXPECTED_RESUME_STEP="${EXPECTED_RESUME_STEP:-100000}"
if [[ ! -s "${RESUME}" ]]; then
    echo "Missing optimizer checkpoint: ${RESUME}" >&2
    echo "Restore stage2_resume.pt from job 17519025. The HF inference checkpoint cannot resume its optimizer." >&2
    exit 1
fi
# This run must never overwrite the 100k baseline. Requeues use this run's own resume file.
BASELINE="output/mobileov_stage2_from_alignment15k/17519025"
if [[ "$(realpath -m "${OUT}")" == "$(realpath -m "${BASELINE}")" ]]; then
    echo "OUT must be a new directory, not the 100k baseline." >&2
    exit 1
fi
if [[ "${EXPECTED_RESUME_STEP}" == 100000 && "$(realpath -m "${OUT}")" == "$(dirname "$(realpath "${RESUME}")")" ]]; then
    echo "Start the extension in a new OUT directory to preserve its source checkpoint." >&2
    exit 1
fi
if [[ -e "${OUT}/run_contract.json" && "$(realpath "${RESUME}")" != "$(realpath -m "${OUT}/stage2_resume.pt")" ]]; then
    echo "OUT already contains a run. Resume its own stage2_resume.pt or choose a new OUT." >&2
    exit 1
fi
CUSTOM_INIT="${INIT_CONNECTOR:-}"
export INIT_CONNECTOR="${INIT_CONNECTOR:-output/mobileov_stage1_alignment/17507068/stage1_connector_latest.pt}"
if [[ ! -s "${INIT_CONNECTOR}" ]]; then
    CACHE="checkpoints/hf_mobile_ov/mobileov_stage1_alignment/17507068/stage1_connector_latest.pt"
    if [[ -z "${CUSTOM_INIT}" && -s "${CACHE}" ]]; then
        export INIT_CONNECTOR="${CACHE}"
    else
        echo "Missing original Stage-1 checkpoint: ${INIT_CONNECTOR}. Restore it or set INIT_CONNECTOR." >&2
        exit 1
    fi
fi
echo "Continue Stage 2: ${EXPECTED_RESUME_STEP} -> 150000 total steps; original optimizer, LR=2e-5."
echo "RESUME=${RESUME}; OUT=${OUT}; archives at 110k/120k/130k/140k/150k."
echo "Unchanged recipe: connector + DiT; frozen SmolVLM2/VAE; T2I:T2V=1:1; pyramid-flow MSE only."
bash scripts/train_mobileov_stage1_alignment_1node8gpu.sbatch
if [[ -s "${OUT}/training_paused.json" && ! -e "${OUT}/training_complete.json" ]]; then
    export RESUME="${OUT}/stage2_resume.pt"
    export EXPECTED_RESUME_STEP
    EXPECTED_RESUME_STEP="$(jq -er '.step' "${OUT}/training_paused.json")"
    test -s "${RESUME}"
    sbatch --dependency="afterok:${SLURM_JOB_ID}" --export=ALL \
        scripts/train_mobileov_stage2_extend150k_1node8gpu.sbatch
fi
