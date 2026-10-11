#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the Neo_Mobile-OV repo root}}"
export ROOT="${PWD}"
case "${EVAL_EPOCH:?Set by the epoch job wrapper}" in
  1) EXPECTED_STEP=100000 ;;
  2) EXPECTED_STEP=190000 ;;
  *) echo "EVAL_EPOCH must be 1 or 2" >&2; exit 1 ;;
esac
TRAIN_RUN="${TRAIN_RUN:-output/mobileov_stage2_media3epochs/17641364}"
CHECKPOINT="${CHECKPOINT:-${TRAIN_RUN}/stage2_dit_connector_step${EXPECTED_STEP}.pt}"
REFERENCE_RUN="${REFERENCE_RUN:-output/vbench_mobileov_stage2_media3epochs_5x_dual}"
test -s "${CHECKPOINT}" || { echo "Missing archived checkpoint: ${CHECKPOINT}; no latest/HF fallback." >&2; exit 1; }
for FILE in run_contract.json quicksr_ablation.json score_assets.json; do
  test -s "${REFERENCE_RUN}/${FILE}" || { echo "Missing completed epoch-3 reference: ${REFERENCE_RUN}/${FILE}" >&2; exit 1; }
done
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-5}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
if [[ ! "${MAX_PROMPTS}" =~ ^[0-9]+$ || ! "${SAMPLES_PER_PROMPT}" =~ ^[1-5]$ ]]; then
  echo "Invalid prompt limit/sample count" >&2; exit 1
fi
if [[ "${MAX_PROMPTS}" -eq 0 && "${SAMPLES_PER_PROMPT}" -ne 5 ]]; then
  echo "Full checkpoint comparison requires 5 videos/prompt" >&2; exit 1
fi
SUFFIX=""
if [[ "${MAX_PROMPTS}" -gt 0 ]]; then SUFFIX="_smoke${MAX_PROMPTS}"; fi
RUN_ROOT="${RUN_ROOT:-output/vbench_mobileov_stage2_media3epochs_epoch${EVAL_EPOCH}_step${EXPECTED_STEP}_${SAMPLES_PER_PROMPT}x_dual${SUFFIX}}"
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
PIN_ARGS=()
if [[ -n "${EXPECTED_SHA256:-}" ]]; then PIN_ARGS+=(--expected-sha256 "${EXPECTED_SHA256}"); fi
echo "Intermediate Stage-2 checkpoint: ${CHECKPOINT}; expected step=${EXPECTED_STEP}; approximate epoch=${EVAL_EPOCH}."
echo "944 prompts x 5 samples; 8 GPU workers; unchanged 20/10 steps per stage, CFG 7/5, 49 frames at 24 FPS."
echo "Raw 512x320 + QuickSR 1024x640. Generate every frame; no external anchor."
echo "Compare against existing epoch-3 report: ${REFERENCE_RUN}. Never regenerate the final benchmark."
srun --ntasks=1 --gres=gpu:8 "${BASE_PYTHON}" -u tools/benchmark_mobileov_stage2_epoch_dual.py \
  --epoch "${EVAL_EPOCH}" --expected-step "${EXPECTED_STEP}" --reference-run "${REFERENCE_RUN}" \
  --checkpoint "${CHECKPOINT}" --config "${CONFIG:-configs/mobile_ov_neodragon.yaml}" \
  --output-dir "${RUN_ROOT}" --workers 8 --samples-per-prompt "${SAMPLES_PER_PROMPT}" \
  --max-prompts "${MAX_PROMPTS}" --seed "${SEED:-2026}" \
  --vbench-repo "${VBENCH_REPO}" --vbench-python "${VBENCH_ENV}/bin/python" "${PIN_ARGS[@]}"
if [[ "${MAX_PROMPTS}" -eq 0 ]]; then
  test -s "${RUN_ROOT}/epoch_comparison.json"
  cat "${RUN_ROOT}/epoch_comparison.md"
fi
