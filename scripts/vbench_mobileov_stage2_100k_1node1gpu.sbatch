#!/usr/bin/env bash
# Stage 2 baseline: all 49 frames from the trained monolithic DiT, no DreamLite/SSD1B.
# Default is the full VBench prompt set at 1 video/prompt, not a 5-sample paper reproduction.
#SBATCH --job-name=mov-vb-s2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-vb-s2-%j.out
#SBATCH --error=logs/mov-vb-s2-%j.out

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the Neo_Mobile-OV repo root}}"
ROOT="$(pwd -P)"
test -f tools/generate_vbench_mobileov_stage2.py

# The VBench environment uses Detectron2 compiled against CUDA 12.1.
if [[ "${SKIP_MODULE_LOAD:-0}" != 1 ]]; then
    module load buildenv-gcccuda/12.1.1-gcc12.3.0
fi
CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV}/bin/python}"
export ROOT BASE_PYTHON="${PYTHON_BIN}"
export VBENCH_ENV="${VBENCH_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_vbench}"
export VBENCH_REPO="${VBENCH_REPO:-${ROOT}/checkpoints/vbench_repo}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TORCH_HOME="${TORCH_HOME:-${HF_HOME}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HF_HOME}}"
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-${HF_HOME}/vbench}"
export TMPDIR="${TMPDIR:-${HF_HOME}/tmp}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs "${TMPDIR}"
test -x "${PYTHON_BIN}"

SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
RUN_ROOT="${RUN_ROOT:-output/vbench_mobileov_stage2_100k_${SAMPLES_PER_PROMPT}x}"
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd -P)"
# Lock scoring as well as generation so simultaneous submissions cannot mix results.
exec 9>"${RUN_ROOT}/.benchmark.lock"
flock -n 9 || { echo "Another benchmark is using ${RUN_ROOT}" >&2; exit 1; }

STOP_FILE="${TMPDIR}/mov_s2_vbench_${SLURM_JOB_ID}.stop"
rm -f "${STOP_FILE}"
"${PYTHON_BIN}" tools/utils/gpu_heartbeat.py --all-devices --interval 5 --tensor-mb 4 \
    --work-seconds 1 --stop-file "${STOP_FILE}" --label "mov-s2-vbench-setup-${SLURM_JOB_ID}" &
HEARTBEAT_PID=$!
cleanup_heartbeat() {
    touch "${STOP_FILE}" 2>/dev/null || true
    wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
    rm -f "${STOP_FILE}" 2>/dev/null || true
}
trap cleanup_heartbeat EXIT
if [[ "${SKIP_VBENCH_INSTALL:-0}" != 1 ]]; then
    bash scripts/install_vbench_env.sh
fi
test -x "${VBENCH_ENV}/bin/python"
VBENCH_INFO="${VBENCH_INFO:-${VBENCH_REPO}/vbench/VBench_full_info.json}"
test -s "${VBENCH_INFO}"
VBENCH_INFO="$(realpath "${VBENCH_INFO}")"
VBENCH_ENV="$(cd "${VBENCH_ENV}" && pwd -P)"
VBENCH_REPO="$(cd "${VBENCH_REPO}" && pwd -P)"
PYTHON_BIN="$(realpath -s "${PYTHON_BIN}")"
HF_FILE="${HF_FILE:-mobileov_stage2_from_alignment15k/17519025/stage2_dit_connector_latest.pt}"
if [[ -z "${CHECKPOINT:-}" ]]; then
    CHECKPOINT="output/${HF_FILE}"
    if [[ ! -s "${CHECKPOINT}" ]]; then
        CHECKPOINT="checkpoints/hf_mobile_ov/${HF_FILE}"
    fi
fi
echo "Stage 2 checkpoint=${CHECKPOINT}; expected_step=${EXPECTED_STEP:-100000}"
CHECKPOINT_ARGS=()
if [[ -n "${EXPECTED_SHA256:-}" ]]; then
    CHECKPOINT_ARGS+=(--expected-sha256 "${EXPECTED_SHA256}")
    echo "Pinned checkpoint SHA256=${EXPECTED_SHA256}"
fi
echo "RUN_ROOT=${RUN_ROOT}; full monolithic; trained empty negative; CFG=7/5; 49 frames 320x512 at 24fps"
echo "SAMPLES_PER_PROMPT=${SAMPLES_PER_PROMPT}; MAX_PROMPTS=${MAX_PROMPTS}"
if [[ "${SAMPLES_PER_PROMPT}" -lt 5 ]]; then
    echo "This is a ${SAMPLES_PER_PROMPT}x run; stock VBench may warn about unused indices up to 4. Coverage is checked explicitly."
fi
cleanup_heartbeat
trap - EXIT

"${PYTHON_BIN}" tools/generate_vbench_mobileov_stage2.py \
    --checkpoint "${CHECKPOINT}" --expected-step "${EXPECTED_STEP:-100000}" \
    "${CHECKPOINT_ARGS[@]}" \
    --hf-repo "${HF_REPO:-Amshaker/Mobile-OV}" --hf-file "${HF_FILE}" --hf-revision "${HF_REVISION:-main}" \
    --config "${CONFIG:-configs/mobile_ov_neodragon.yaml}" \
    --vbench-info "${VBENCH_INFO}" --output-dir "${RUN_ROOT}" \
    --samples-per-prompt "${SAMPLES_PER_PROMPT}" --max-prompts "${MAX_PROMPTS}" \
    --seed "${SEED:-20260911}" --first-steps 20 --video-steps 10 --guidance 7 --video-guidance 5

if [[ "${MAX_PROMPTS}" -gt 0 ]]; then
    echo "Smoke generation completed. No full VBench score is claimed for this subset."
    exit 0
fi
# Relative evaluator assets are resolved from the VBench repository, not the project root.
cd "${VBENCH_REPO}"
"${VBENCH_ENV}/bin/python" "${ROOT}/tools/evaluate_vbench_resume.py" \
    --videos "${RUN_ROOT}/videos" --full-info "${VBENCH_INFO}" \
    --output "${RUN_ROOT}/scores" --expected-samples-per-prompt "${SAMPLES_PER_PROMPT}"
"${PYTHON_BIN}" "${ROOT}/tools/report_vbench_low_prompts.py" \
    --score-dir "${RUN_ROOT}/scores/vbench" --output-prefix "${RUN_ROOT}/low_prompts" \
    --failure-dir "${RUN_ROOT}/low_prompt_videos" --top-k 10
echo "Completed Stage 2 VBench: ${RUN_ROOT}"
cat "${RUN_ROOT}/scores/vbench/summary.json"
