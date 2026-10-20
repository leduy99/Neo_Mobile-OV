#!/usr/bin/env bash
# Shared setup for the two completed continuation benchmarks; called inside sbatch.
set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the Neo_Mobile-OV repo root}}"
export ROOT="${PWD}"
EXPERIMENT="${1:?Expected job_b or uniform_history}"
case "${EXPERIMENT}" in
    job_b)
        TRAIN_RUN="${TRAIN_RUN:-output/mobileov_stage2_job_b_lora/17783291}"
        CHECKPOINT="${CHECKPOINT:-${TRAIN_RUN}/job_b_dit_connector_lora_latest.pt}"
        LABEL="job_b_lora"
        ;;
    uniform_history)
        if [[ -z "${TRAIN_RUN:-}" && -z "${CHECKPOINT:-}" ]]; then
            test -d output/mobileov_stage2_uniform_history || {
                echo "Missing uniform-history output directory. Set TRAIN_RUN explicitly." >&2; exit 1;
            }
            mapfile -t COMPLETE < <(find output/mobileov_stage2_uniform_history -mindepth 2 -maxdepth 2 \
                -type f -name training_complete.json -print | sort)
            if [[ "${#COMPLETE[@]}" -ne 1 ]]; then
                echo "Expected exactly one completed uniform-history run; set TRAIN_RUN or CHECKPOINT explicitly." >&2
                printf '%s\n' "${COMPLETE[@]}" >&2
                exit 1
            fi
            TRAIN_RUN="$(dirname "${COMPLETE[0]}")"
        fi
        CHECKPOINT="${CHECKPOINT:-${TRAIN_RUN}/stage2_dit_connector_latest.pt}"
        LABEL="uniform_history"
        ;;
    *) echo "Unknown continuation: ${EXPERIMENT}" >&2; exit 1 ;;
esac
REFERENCE_RUN="${REFERENCE_RUN:-output/vbench_mobileov_stage2_media3epochs_5x_dual}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-5}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
if [[ ! "${MAX_PROMPTS}" =~ ^[0-9]+$ || ! "${SAMPLES_PER_PROMPT}" =~ ^[1-5]$ ]]; then
    echo "Invalid prompt limit/sample count." >&2; exit 1
fi
if [[ "${MAX_PROMPTS}" -eq 0 && "${SAMPLES_PER_PROMPT}" -ne 5 ]]; then
    echo "Full continuation evaluation requires 5 videos/prompt." >&2; exit 1
fi
test -s "${CHECKPOINT}" || { echo "Missing completed checkpoint: ${CHECKPOINT}; no parent/HF fallback." >&2; exit 1; }
if [[ "${MAX_PROMPTS}" -eq 0 ]]; then
    for FILE in run_contract.json quicksr_ablation.json score_assets.json; do
        test -s "${REFERENCE_RUN}/${FILE}" || { echo "Missing epoch-3 reference: ${REFERENCE_RUN}/${FILE}" >&2; exit 1; }
    done
fi
SUFFIX=""
if [[ "${MAX_PROMPTS}" -gt 0 ]]; then SUFFIX="_smoke${MAX_PROMPTS}"; fi
RUN_ROOT="${RUN_ROOT:-output/vbench_mobileov_${LABEL}_${SAMPLES_PER_PROMPT}x_dual${SUFFIX}}"
module load buildenv-gcccuda/12.1.1-gcc12.3.0
export CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
export BASE_PYTHON="${CONDA_ENV}/bin/python"
export VBENCH_ENV="${VBENCH_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_vbench}"
export VBENCH_REPO="${VBENCH_REPO:-${ROOT}/checkpoints/vbench_repo}"
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TORCH_HOME="${TORCH_HOME:-${HF_HOME}}" XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HF_HOME}}"
export TMPDIR="${TMPDIR:-${HF_HOME}/tmp}" VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-${HF_HOME}/vbench}"
export PYTHONPATH="${ROOT}:${VBENCH_REPO}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
test -x "${BASE_PYTHON}"
mkdir -p logs "${RUN_ROOT}" "${TMPDIR}" "${VBENCH_CACHE_DIR}" "$(dirname "${VBENCH_ENV}")"
exec 9>"${VBENCH_ENV}.mobileov_setup.lock"
flock 9
bash scripts/install_vbench_env.sh
flock -u 9
exec 9>&-
test -x "${VBENCH_ENV}/bin/python"
test -s "${VBENCH_REPO}/vbench/VBench_full_info.json"
PIN_ARGS=()
if [[ -n "${EXPECTED_SHA256:-}" ]]; then PIN_ARGS+=(--expected-sha256 "${EXPECTED_SHA256}"); fi
echo "${EXPERIMENT}: ${CHECKPOINT}; require completed step378636 and exact epoch-3 parent."
echo "Text-only; 944 prompts x 5 videos; 8 GPU workers; same parent seeds/20-10 steps/CFG7-5."
echo "Generate frame zero; NO SSD1B/DreamLite anchor. History corruption OFF at inference."
echo "Job B loads trained LoRA + MCP + DiT; this job does NOT measure understanding retention."
echo "Raw 512x320 and QuickSR 1024x640 share generated RGB frames. OUT=${RUN_ROOT}"
srun --ntasks=1 --gres=gpu:8 "${BASE_PYTHON}" -u tools/benchmark_mobileov_continuations_dual.py \
    --experiment "${EXPERIMENT}" --checkpoint "${CHECKPOINT}" --reference-run "${REFERENCE_RUN}" \
    --config "${CONFIG:-configs/mobile_ov_neodragon.yaml}" --output-dir "${RUN_ROOT}" \
    --workers 8 --samples-per-prompt "${SAMPLES_PER_PROMPT}" --max-prompts "${MAX_PROMPTS}" \
    --seed "${SEED:-2026}" --vbench-repo "${VBENCH_REPO}" \
    --vbench-python "${VBENCH_ENV}/bin/python" "${PIN_ARGS[@]}"
if [[ "${MAX_PROMPTS}" -eq 0 ]]; then
    test -s "${RUN_ROOT}/${EXPERIMENT}_vs_epoch3.json"
    cat "${RUN_ROOT}/${EXPERIMENT}_vs_epoch3.md"
fi
