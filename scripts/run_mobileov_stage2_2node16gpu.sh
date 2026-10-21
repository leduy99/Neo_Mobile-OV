#!/usr/bin/env bash
# One torchrun agent per node; each agent launches eight DDP workers.
set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
if [[ "${1:-}" == --node ]]; then
    cd "${PROJECT_ROOT:?}"
    ARGS=()
    if [[ "${TRAIN_MODE}" == resume ]]; then
        ARGS+=(--resume "${RESUME}" --extend-media-epochs --repartition-media-batch)
    elif [[ "${TRAIN_MODE}" == history ]]; then
        ARGS+=(--init-connector "${INIT_CONNECTOR}" --expected-init-step 15000
               --fresh-media-history --history-corrupt-max 0.3333333333333333)
        if [[ -n "${RESUME:-}" ]]; then ARGS+=(--resume "${RESUME}"); fi
    else
        echo "Unknown training mode" >&2; exit 1
    fi
    exec "${CONDA_ENV}/bin/torchrun" --nnodes=2 --nproc_per_node=8 \
        --node_rank="${SLURM_NODEID:?}" --master_addr="${MASTER_ADDR:?}" --master_port="${MASTER_PORT:?}" \
        --max_restarts=0 tools/train_mobileov_stage1.py --phase 2 \
        --config "${CONFIG:-configs/mobile_ov_neodragon.yaml}" \
        --data-root "${DATA_ROOT}" --output-dir "${OUT}" \
        --media-epochs 5 --min-train-videos 500000 --require-expanded-release \
        --accumulation 1 --media-rng-world-size 8 --lr 2e-5 --warmup 50 \
        --reconstruction-weight .1 --vae-window-size 8 \
        --save-every 10000 --archive-every 10000 --validate-every "${VALIDATE_EVERY:-2000}" \
        --validation-samples "${VALIDATION_SAMPLES:-1}" --workers "${WORKERS:-2}" \
        --max-runtime-seconds "${MAX_RUNTIME_SECONDS:-244800}" "${ARGS[@]}"
fi

cd "${PROJECT_ROOT:?}"
export TRAIN_MODE="${1:?Choose resume or history}"
if [[ "${TRAIN_MODE}" != resume && "${TRAIN_MODE}" != history ]]; then
    echo "Choose resume or history." >&2; exit 1
fi
if [[ "${SLURM_NNODES:-0}" != 2 ]]; then
    echo "This job requires two nodes, eight GPUs per node." >&2; exit 1
fi
if [[ -n "${STEPS:-}" || -n "${LR:-}" || -n "${ACCUMULATION:-}" || -n "${INIT_STAGE2:-}" ]]; then
    echo "Do not override STEPS/LR/ACCUMULATION/INIT_STAGE2: preserve global batch 16 and the selected recipe." >&2; exit 1
fi
module load buildenv-nvhpc/24.5-cuda12.4
export CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TMPDIR="${TMPDIR:-${HF_HOME}/tmp}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_ROOT="${DATA_ROOT:-download_data/data/univideo_stage1_expanded650}"
export AUTO_CONTINUE="${AUTO_CONTINUE:-1}"
if [[ "${AUTO_CONTINUE}" != 0 && "${AUTO_CONTINUE}" != 1 ]]; then
    echo "AUTO_CONTINUE must be 0 or 1." >&2; exit 1
fi
if [[ "${TRAIN_MODE}" == resume ]]; then
    export OUT="${OUT:-output/mobileov_stage2_resume5epochs16gpu/${SLURM_JOB_ID}}"
    export RESUME="${RESUME:-output/mobileov_stage2_media3epochs/17641364/stage2_resume.pt}"
    SUBMIT_SCRIPT=scripts/train_mobileov_stage2_resume_epoch3_to5_2node16gpu.sbatch
    test -s "${RESUME}"
    mkdir -p logs "${TMPDIR}"
    "${CONDA_ENV}/bin/python" -s tools/plan_mobileov_stage2_epoch_extension.py \
        --resume "${RESUME}" --data-root "${DATA_ROOT}" --output-dir "${OUT}" --epochs 5 \
        --world-size 16 --accumulation 1 --media-rng-world-size 8 --repartition-media-batch
else
    export OUT="${OUT:-output/mobileov_stage2_history_fresh5epochs16gpu/${SLURM_JOB_ID}}"
    SUBMIT_SCRIPT=scripts/train_mobileov_stage2_history_fresh5epochs_2node16gpu.sbatch
    CUSTOM_INIT="${INIT_CONNECTOR:-}"
    export INIT_CONNECTOR="${INIT_CONNECTOR:-output/mobileov_stage1_alignment/17507068/stage1_connector_latest.pt}"
    if [[ ! -s "${INIT_CONNECTOR}" ]]; then
        CACHE=checkpoints/hf_mobile_ov/mobileov_stage1_alignment/17507068/stage1_connector_latest.pt
        if [[ -z "${CUSTOM_INIT}" && -s "${CACHE}" ]]; then export INIT_CONNECTOR="${CACHE}"
        else echo "Missing alignment15k connector: ${INIT_CONNECTOR}" >&2; exit 1; fi
    fi
    if [[ -e "${OUT}/run_contract.json" && -z "${RESUME:-}" ]]; then
        echo "OUT already contains a run; set RESUME to its own stage2_resume.pt." >&2; exit 1
    fi
    if [[ -n "${RESUME:-}" ]]; then
        if [[ "${RESUME}" != "${OUT}/stage2_resume.pt" ]]; then
            echo "History run may resume only its OWN optimizer, never epoch-3 weights." >&2; exit 1
        fi
        test -s "${RESUME}"
    fi
    mkdir -p logs "${TMPDIR}"
    "${CONDA_ENV}/bin/python" -s tools/plan_mobileov_stage2_epochs.py \
        --data-root "${DATA_ROOT}" --world-size 16 --accumulation 1 --epochs 5 \
        --allow-partial-cycle --media-rng-world-size 8 --min-train-videos 500000 \
        --require-expanded-release --output "${OUT}/epoch_plan.json"
fi

mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?}")
if [[ "${#HOSTS[@]}" != 2 ]]; then echo "Expected exactly two allocated hosts." >&2; exit 1; fi
export MASTER_ADDR="${HOSTS[0]}"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
echo "Mode=${TRAIN_MODE}; 2 nodes x 8 GPUs; global batch=16, LR=2e-5; OUT=${OUT}."
echo "Epoch checkpoints: 3=283977, 4=378636, 5=473295 for the verified current release."
if [[ "${TRAIN_MODE}" == history ]]; then
    echo "History corruption enabled from initialization; only eligible past video latents are changed."
else
    echo "Clean history; unchanged parent training objective."
fi
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-128}" \
    bash scripts/run_mobileov_stage2_2node16gpu.sh --node
if [[ -s "${OUT}/training_paused.json" && ! -e "${OUT}/training_complete.json" ]]; then
    export RESUME="${OUT}/stage2_resume.pt"
    test -s "${RESUME}"
    if [[ "${AUTO_CONTINUE}" == 1 ]]; then
        NEXT_JOB="$(sbatch --parsable --dependency="afterok:${SLURM_JOB_ID}" --export=ALL "${SUBMIT_SCRIPT}")"
        printf '%s\n' "${NEXT_JOB}" >> "${OUT}/continuation_jobs.txt"
        echo "Queued ${NEXT_JOB}; same OUT, trained weights, Adam state, LR and sampler position."
    else
        echo "Paused safely; resume with OUT=${OUT} RESUME=${RESUME}."
    fi
fi
