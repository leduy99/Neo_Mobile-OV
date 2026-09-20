#!/usr/bin/env bash
# Separate experiment: 100k -> 150k with paired image reconstruction, not the text-only continuation.
#SBATCH --job-name=mov-s2-rec150k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-s2-rec150k-%j.out
#SBATCH --error=logs/mov-s2-rec150k-%j.out

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repo root}}"
export PROJECT_ROOT="${PWD}"
export PHASE=2 STEPS=150000 ACCUMULATION=2 LR=2e-5
export SAVE_EVERY=10000 ARCHIVE_EVERY=10000 VALIDATE_EVERY=1000 VALIDATION_SAMPLES=2
export RECONSTRUCTION_WEIGHT=0.1 ENABLE_RECONSTRUCTION_ON_RESUME=1 EXTEND_STEPS=1
export MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-244800}"
export OUT="${OUT:-output/mobileov_stage2_reconstruction150k/${SLURM_JOB_ID}}"
export RESUME="${RESUME:-output/mobileov_stage2_from_alignment15k/17519025/stage2_resume.pt}"
export EXPECTED_RESUME_STEP="${EXPECTED_RESUME_STEP:-100000}"
if [[ ! -s "${RESUME}" ]]; then
    echo "Missing optimizer checkpoint: ${RESUME}. Use stage2_resume.pt, not an inference checkpoint." >&2
    exit 1
fi
if [[ "$(realpath -m "${OUT}")" == "$(realpath -m output/mobileov_stage2_from_alignment15k/17519025)" ]]; then
    echo "OUT must not overwrite the 100k baseline." >&2
    exit 1
fi
if [[ "${EXPECTED_RESUME_STEP}" == 100000 && "$(realpath -m "${OUT}")" == "$(dirname "$(realpath "${RESUME}")")" ]]; then
    echo "Start reconstruction in a new OUT directory." >&2
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
echo "Stage 2 + reconstruction: ${EXPECTED_RESUME_STEP} -> 150000; original optimizer; LR=2e-5; new OUT=${OUT}."
echo "Loss = 0.5*T2I + 0.5*T2V + 0.1*image_reconstruction. No extra dataset or VAE encoding."
echo "SmolVLM2/VAE frozen; connector/DiT trainable. Existing text-only jobs are not changed or cancelled."
bash scripts/train_mobileov_stage1_alignment_1node8gpu.sbatch
if [[ -s "${OUT}/training_paused.json" && ! -e "${OUT}/training_complete.json" ]]; then
    export RESUME="${OUT}/stage2_resume.pt"
    export EXPECTED_RESUME_STEP
    EXPECTED_RESUME_STEP="$(jq -er '.step' "${OUT}/training_paused.json")"
    test -s "${RESUME}"
    sbatch --dependency="afterok:${SLURM_JOB_ID}" --export=ALL \
        scripts/train_mobileov_stage2_reconstruction150k_1node8gpu.sbatch
fi
