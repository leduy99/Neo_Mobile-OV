#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:?}"
export EXP1_NODES="${EXP1_NODES:-2}" EXP1_ACCUMULATION="${EXP1_ACCUMULATION:-4}"
if [[ "${EXP1_NODES}:${EXP1_ACCUMULATION}" != 2:4 && "${EXP1_NODES}:${EXP1_ACCUMULATION}" != 1:8 ]]; then
    echo "Exp 1 supports only 16 GPUs x4 or 8 GPUs x8; batch64 is unchanged." >&2; exit 1
fi
if [[ "${1:-}" == --node ]]; then
    ARGS=()
    if [[ -n "${RESUME:-}" ]]; then ARGS+=(--resume "${RESUME}"); fi
    exec "${CONDA_ENV}/bin/torchrun" --nnodes="${EXP1_NODES}" --nproc_per_node=8 \
        --node_rank="${SLURM_NODEID:?}" --master_addr="${MASTER_ADDR:?}" --master_port="${MASTER_PORT:?}" \
        --max_restarts=0 tools/train_mobileov_stage2_optimizer_exp1.py \
        --parent-resume "${PARENT_RESUME}" --config "${CONFIG:-configs/mobile_ov_neodragon.yaml}" \
        --data-root "${DATA_ROOT}" --output-dir "${OUT}" --steps 20000 --decay-steps 20000 \
        --ema-decay .9995 --accumulation "${EXP1_ACCUMULATION}" --expected-parent-step 283977 \
        --save-every 2000 --evaluate-every 10000 --validate-every "${VALIDATE_EVERY:-2000}" \
        --validation-samples "${VALIDATION_SAMPLES:-1}" --workers "${WORKERS:-2}" \
        --max-runtime-seconds "${MAX_RUNTIME_SECONDS:-244800}" "${ARGS[@]}"
fi
if [[ "${SLURM_NNODES:-0}" != "${EXP1_NODES}" ]]; then
    echo "Exp 1 requires ${EXP1_NODES} node(s) with eight GPUs each for this launcher." >&2; exit 1
fi
if [[ "${EXP1_NODES}" == 1 ]]; then
    export EXP1_SUBMIT_SCRIPT=scripts/train_mobileov_stage2_optimizer_exp1_1node8gpu.sbatch
else
    export EXP1_SUBMIT_SCRIPT=scripts/train_mobileov_stage2_optimizer_exp1_2node16gpu.sbatch
fi
if [[ -n "${LR:-}" || -n "${STEPS:-}" || -n "${ACCUMULATION:-}" || -n "${INIT_STAGE2:-}" ]]; then
    echo "Exp 1 is pinned: epoch-3 Adam, 20K additional updates, batch64, 2e-5->0, EMA .9995." >&2; exit 1
fi
module load buildenv-nvhpc/24.5-cuda12.4
export CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TMPDIR="${TMPDIR:-${HF_HOME}/tmp}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_ROOT="${DATA_ROOT:-download_data/data/univideo_stage1_expanded650}"
export OUT="${OUT:-output/mobileov_stage2_optimizer_exp1/${SLURM_JOB_ID}}"
export PARENT_RESUME="${PARENT_RESUME:-output/mobileov_stage2_media3epochs/17641364/stage2_resume.pt}"
export AUTO_CONTINUE="${AUTO_CONTINUE:-1}"
if [[ "${AUTO_CONTINUE}" != 0 && "${AUTO_CONTINUE}" != 1 ]]; then echo "Invalid AUTO_CONTINUE" >&2; exit 1; fi
if [[ -n "${RESUME:-}" ]]; then
    if [[ "${RESUME}" != "${OUT}/exp1_resume.pt" ]]; then
        echo "Continuation must use its OWN Exp-1 Adam+EMA+scheduler state." >&2; exit 1
    fi
    test -s "${RESUME}"
else
    test -s "${PARENT_RESUME}" || { echo "Missing ORIGINAL epoch-3 stage2_resume.pt: ${PARENT_RESUME}" >&2; exit 1; }
    if [[ -e "${OUT}/run_contract.json" ]]; then echo "Existing run: set RESUME, or use a new OUT." >&2; exit 1; fi
fi
mkdir -p logs "${TMPDIR}"
mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?}")
if [[ "${#HOSTS[@]}" != "${EXP1_NODES}" ]]; then echo "Allocated host count differs from launcher." >&2; exit 1; fi
export MASTER_ADDR="${HOSTS[0]}" MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
echo "Advisor Exp1: $((EXP1_NODES * 8)) GPUs x accumulation${EXP1_ACCUMULATION} = global batch64; 20K additional updates; LR2e-5->0, no warmup."
echo "Original epoch3 Adam is preserved. No data, flow-loss or history-noise changes. OUT=${OUT}"
echo "10K/20K RAW+EMA archives; latest inference=EMA. CPU FP32 EMA avoids another full DiT in GPU RAM."
srun --nodes="${EXP1_NODES}" --ntasks="${EXP1_NODES}" --ntasks-per-node=1 --kill-on-bad-exit=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-128}" \
    bash scripts/run_mobileov_stage2_optimizer_exp1_16gpu.sh --node
if [[ -s "${OUT}/training_paused.json" && ! -e "${OUT}/training_complete.json" ]]; then
    export RESUME="${OUT}/exp1_resume.pt"
    test -s "${RESUME}"
    if [[ "${AUTO_CONTINUE}" == 1 ]]; then
        NEXT_JOB="$(sbatch --parsable --dependency="afterok:${SLURM_JOB_ID}" --export=ALL \
            "${EXP1_SUBMIT_SCRIPT}")"
        printf '%s\n' "${NEXT_JOB}" >> "${OUT}/continuation_jobs.txt"
        echo "Queued ${NEXT_JOB}; same OUT, Adam/EMA/scheduler/RNG/dataset cursor."
    else
        echo "Paused safely. Resume with OUT=${OUT} RESUME=${RESUME}."
    fi
fi
