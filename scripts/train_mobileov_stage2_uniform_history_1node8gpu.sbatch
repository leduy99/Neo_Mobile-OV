#!/usr/bin/env bash
# Epoch-3 fork, original unit/loss balance, real stage-resolution history corruption.
#SBATCH --job-name=mov-s2-history
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-s2-history-%j.out
#SBATCH --error=logs/mov-s2-history-%j.out
set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repo root}}"
export PROJECT_ROOT="${PWD}"
module load buildenv-nvhpc/24.5-cuda12.4
CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TMPDIR="${TMPDIR:-${HF_HOME}/tmp}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_ROOT="${DATA_ROOT:-download_data/data/univideo_stage1_expanded650}"
export OUT="${OUT:-output/mobileov_stage2_uniform_history/${SLURM_JOB_ID}}"
CUSTOM_PARENT="${INIT_STAGE2:-}"
export INIT_STAGE2="${INIT_STAGE2:-output/mobileov_stage2_media3epochs/17641364/stage2_dit_connector_latest.pt}"
export AUTO_CONTINUE="${AUTO_CONTINUE:-1}"
PARENT_SHA="a74fd3069d17437b674415268c5157a85d94223509121220572e2d2fcdb68053"
if [[ "${AUTO_CONTINUE}" != 0 && "${AUTO_CONTINUE}" != 1 ]]; then
    echo "AUTO_CONTINUE must be 0 or 1." >&2; exit 1
fi
if [[ -n "${STEPS:-}" || -n "${INIT_CONNECTOR:-}" || -n "${T2V_FIRST_PROBABILITY:-}" ]]; then
    echo "Uniform-history recipe derives one epoch and uses uniform units; do not override its budget/sampling." >&2; exit 1
fi
if [[ ! -s "${INIT_STAGE2}" ]]; then
    CACHE="checkpoints/hf_mobile_ov/mobileov_stage2_media3epochs/17641364/stage2_dit_connector_step283977.pt"
    if [[ -z "${CUSTOM_PARENT}" && -s "${CACHE}" ]]; then
        export INIT_STAGE2="${CACHE}"
    else
        echo "Missing exact epoch-3 step283977 parent; never fall back to Job A or an intermediate." >&2; exit 1
    fi
fi
if [[ -e "${OUT}/run_contract.json" && -z "${RESUME:-}" ]]; then
    echo "OUT contains a run; set RESUME to its own stage2_resume.pt or choose a new OUT." >&2; exit 1
fi
mkdir -p logs "${TMPDIR}"
"${CONDA_ENV}/bin/python" -s tools/plan_mobileov_stage2_history.py \
    --data-root "${DATA_ROOT}" --parent "${INIT_STAGE2}" --parent-sha256 "${PARENT_SHA}" \
    --world-size 8 --accumulation 2 --min-train-videos 500000 --require-expanded-release \
    --output "${OUT}/epoch_plan.json"
ARGS=()
if [[ -n "${RESUME:-}" ]]; then
    test -s "${RESUME}"
    ARGS+=(--resume "${RESUME}")
fi
echo "Uniform-history: step283977 -> step378636; fresh optimizer; uniform T2V/reference units; reconstruction .1."
echo "History noise sigma~U[0,1/3] AFTER temporal/spatial pyramid; independent RNG; observed VAE anchor clean."
echo "Require actual eligible/applied unit counts and changed_rms in history/audit; no LoRA, EMA, teacher or rollout training."
"${CONDA_ENV}/bin/torchrun" --standalone --nnodes=1 --nproc_per_node=8 \
    tools/train_mobileov_stage1.py --phase 2 --config "${CONFIG:-configs/mobile_ov_neodragon.yaml}" \
    --data-root "${DATA_ROOT}" --output-dir "${OUT}" \
    --init-stage2 "${INIT_STAGE2}" --expected-stage2-step 283977 --expected-stage2-sha256 "${PARENT_SHA}" \
    --media-epochs 1 --media-start-epoch 3 --history-corrupt-max 0.3333333333333333 \
    --reconstruction-weight .1 --min-train-videos 500000 --require-expanded-release --accumulation 2 \
    --lr 2e-5 --warmup 50 --save-every 10000 --archive-every 10000 --validate-every 5000 \
    --validation-samples "${VALIDATION_SAMPLES:-2}" --workers "${WORKERS:-2}" \
    --max-runtime-seconds "${MAX_RUNTIME_SECONDS:-244800}" "${ARGS[@]}"
if [[ -s "${OUT}/training_paused.json" && ! -e "${OUT}/training_complete.json" ]]; then
    export RESUME="${OUT}/stage2_resume.pt"
    test -s "${RESUME}"
    if [[ "${AUTO_CONTINUE}" == 1 ]]; then
        NEXT_JOB="$(sbatch --parsable --dependency="afterok:${SLURM_JOB_ID}" --export=ALL \
            scripts/train_mobileov_stage2_uniform_history_1node8gpu.sbatch)"
        printf '%s\n' "${NEXT_JOB}" >> "${OUT}/continuation_jobs.txt"
        echo "Queued ${NEXT_JOB}; same OUT, optimizer, epoch position and per-rank RNG."
    else
        echo "Paused safely. Resume with OUT=${OUT} RESUME=${RESUME}."
    fi
fi
