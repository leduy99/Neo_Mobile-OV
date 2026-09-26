#!/usr/bin/env bash
# Fresh Stage 2 from the Stage-1 connector and released monolithic DiT, not random weights.
# All images/videos get three passes; each selected image has a paired reconstruction loss.
#SBATCH --job-name=mov-s2-media3ep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-s2-media3ep-%j.out
#SBATCH --error=logs/mov-s2-media3ep-%j.out

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
export OUT="${OUT:-output/mobileov_stage2_media3epochs/${SLURM_JOB_ID}}"
CUSTOM_INIT="${INIT_CONNECTOR:-}"
export INIT_CONNECTOR="${INIT_CONNECTOR:-output/mobileov_stage1_alignment/17507068/stage1_connector_latest.pt}"
export ACCUMULATION="${ACCUMULATION:-2}"
export AUTO_CONTINUE="${AUTO_CONTINUE:-1}"
if [[ "${AUTO_CONTINUE}" != 0 && "${AUTO_CONTINUE}" != 1 ]]; then
    echo "AUTO_CONTINUE must be 0 or 1." >&2
    exit 1
fi
if [[ -n "${STEPS:-}" ]]; then
    echo "Do not set STEPS: this job derives three full media epochs from the verified manifests." >&2
    exit 1
fi
if [[ ! -s "${INIT_CONNECTOR}" ]]; then
    CACHE="checkpoints/hf_mobile_ov/mobileov_stage1_alignment/17507068/stage1_connector_latest.pt"
    if [[ -z "${CUSTOM_INIT}" && -s "${CACHE}" ]]; then
        export INIT_CONNECTOR="${CACHE}"
    else
        echo "Missing alignment initialization: ${INIT_CONNECTOR}. Restore the 15k Stage-1 checkpoint or set INIT_CONNECTOR." >&2
        exit 1
    fi
fi
if [[ -e "${OUT}/run_contract.json" && -z "${RESUME:-}" ]]; then
    echo "OUT already contains a run. Set RESUME to its stage2_resume.pt or choose a new OUT." >&2
    exit 1
fi
mkdir -p logs "${TMPDIR}"
"${CONDA_ENV}/bin/python" -s tools/plan_mobileov_stage2_epochs.py \
    --data-root "${DATA_ROOT}" --world-size 8 --accumulation "${ACCUMULATION}" \
    --epochs 3 --min-train-videos 500000 --require-expanded-release --output "${OUT}/epoch_plan.json"
ARGS=()
if [[ -n "${RESUME:-}" ]]; then
    test -s "${RESUME}"
    ARGS+=(--resume "${RESUME}")
fi
if [[ -n "${RESUME:-}" ]]; then
    echo "Continuing this media-epoch run with trained weights + optimizer: ${RESUME}"
else
    echo "Fresh Stage 2: alignment15k connector + released multistep DiT, fresh optimizer; not Stage2-150k."
fi
echo "T2I + paired reconstruction; each video rotates T2V/reference/anchor. No native teacher or DMD."
echo "Three source passes, not three passes per video mode. Exact budget: ${OUT}/epoch_plan.json"
echo "Train connector + DiT; freeze SmolVLM2 + VAE. Checkpoints every 10k; auto-continue=${AUTO_CONTINUE}."
"${CONDA_ENV}/bin/torchrun" --standalone --nnodes=1 --nproc_per_node=8 \
    tools/train_mobileov_stage1.py --phase 2 \
    --config "${CONFIG:-configs/mobile_ov_neodragon.yaml}" \
    --data-root "${DATA_ROOT}" --output-dir "${OUT}" \
    --init-connector "${INIT_CONNECTOR}" --expected-init-step 15000 \
    --media-epochs 3 --min-train-videos 500000 --require-expanded-release \
    --accumulation "${ACCUMULATION}" --lr "${LR:-2e-5}" --warmup 50 \
    --reconstruction-weight 0.1 --save-every 10000 --archive-every 10000 \
    --validate-every "${VALIDATE_EVERY:-2000}" --validation-samples "${VALIDATION_SAMPLES:-2}" \
    --workers "${WORKERS:-2}" --max-runtime-seconds "${MAX_RUNTIME_SECONDS:-244800}" "${ARGS[@]}"
if [[ -s "${OUT}/training_paused.json" && ! -e "${OUT}/training_complete.json" ]]; then
    export RESUME="${OUT}/stage2_resume.pt"
    test -s "${RESUME}"
    if [[ "${AUTO_CONTINUE}" == 1 ]]; then
        NEXT_JOB="$(sbatch --parsable --dependency="afterok:${SLURM_JOB_ID}" --export=ALL \
            scripts/train_mobileov_stage2_media3epochs_1node8gpu.sbatch)"
        printf '%s\n' "${NEXT_JOB}" >> "${OUT}/continuation_jobs.txt"
        echo "Queued continuation job ${NEXT_JOB}; same OUT=${OUT}, same optimizer/sample position."
    else
        echo "Paused safely; auto-continue disabled. Resume with OUT=${OUT} RESUME=${RESUME}."
    fi
fi
