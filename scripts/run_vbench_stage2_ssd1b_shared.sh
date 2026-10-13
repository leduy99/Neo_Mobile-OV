#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the Neo_Mobile-OV root}}"
export ROOT="${PWD}"
case "${STAGE2_PHASE:?Set by the job wrapper}" in
  frames|ssd1b_anchor|image_condition) ;;
  *) echo "Invalid STAGE2_PHASE" >&2; exit 1 ;;
esac
CHECKPOINT="${CHECKPOINT:-output/mobileov_stage2_media3epochs/17641364/stage2_dit_connector_latest.pt}"
EXPECTED_SHA256="${EXPECTED_SHA256:-a74fd3069d17437b674415268c5157a85d94223509121220572e2d2fcdb68053}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-5}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
SEED="${SEED:-2026}"
if [[ ! "${MAX_PROMPTS}" =~ ^[0-9]+$ || ! "${SAMPLES_PER_PROMPT}" =~ ^[1-5]$ || ! "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "Invalid prompt limit, sample count or seed" >&2; exit 1
fi
if [[ "${MAX_PROMPTS}" -eq 0 && "${SAMPLES_PER_PROMPT}" -ne 5 ]]; then
  echo "Full evaluation requires five independently generated videos per prompt" >&2; exit 1
fi
SUFFIX=""
if [[ "${MAX_PROMPTS}" -gt 0 ]]; then SUFFIX="_smoke${MAX_PROMPTS}"; fi
PAIR_ROOT="${PAIR_ROOT:-output/vbench_stage2_epoch3_ssd1b_conditioning_${SAMPLES_PER_PROMPT}x_dual${SUFFIX}}"
ANCHOR_ROOT="${ANCHOR_ROOT:-${PAIR_ROOT}/shared_ssd1b}"
RUN_ROOT="${RUN_ROOT:-${PAIR_ROOT}/${STAGE2_PHASE}}"
test -s "${CHECKPOINT}" || { echo "Missing local completed epoch-3 checkpoint: ${CHECKPOINT}; no HF/130K fallback" >&2; exit 1; }
if [[ "${STAGE2_PHASE}" != frames ]]; then
  test -s "${ANCHOR_ROOT}/generation_summary.json" || { echo "Shared SSD1B job has not completed" >&2; exit 1; }
fi
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
mkdir -p logs "${PAIR_ROOT}" "${TMPDIR}" "${VBENCH_CACHE_DIR}" "$(dirname "${VBENCH_ENV}")"
exec 9>"${VBENCH_ENV}.mobileov_setup.lock"
flock 9
bash scripts/install_vbench_env.sh
flock -u 9
exec 9>&-
test -x "${VBENCH_ENV}/bin/python"
test -s "${VBENCH_REPO}/vbench/VBench_full_info.json"
COMMON=(--checkpoint "${CHECKPOINT}" --expected-sha256 "${EXPECTED_SHA256}"
        --config "${CONFIG:-configs/mobile_ov_neodragon.yaml}" --vbench-repo "${VBENCH_REPO}"
        --workers 8 --samples-per-prompt "${SAMPLES_PER_PROMPT}" --max-prompts "${MAX_PROMPTS}" --seed "${SEED}")
if [[ "${STAGE2_PHASE}" == frames ]]; then
  echo "Generate shared native SSD1B images once; verify latest epoch 3, step 283977 before generation."
  srun --ntasks=1 --gres=gpu:8 "${BASE_PYTHON}" -u tools/prepare_vbench_stage2_ssd1b_frames.py \
    "${COMMON[@]}" --output-dir "${ANCHOR_ROOT}"
  test -s "${ANCHOR_ROOT}/generation_summary.json"
else
  echo "Evaluate ${STAGE2_PHASE}; read shared PNGs only; SSD1B is not loaded in either evaluation job."
  echo "Eight GPUs, 944 prompts x 5 videos; raw 512x320 and QuickSR 1024x640 from identical generated frames."
  srun --ntasks=1 --gres=gpu:8 "${BASE_PYTHON}" -u tools/benchmark_stage2_ssd1b_conditioning_dual.py \
    "${COMMON[@]}" --mode "${STAGE2_PHASE}" --frames-root "${ANCHOR_ROOT}" \
    --output-dir "${RUN_ROOT}" --vbench-python "${VBENCH_ENV}/bin/python"
  if [[ "${MAX_PROMPTS}" -eq 0 ]]; then
    test -s "${RUN_ROOT}/quicksr_ablation.json"
    cat "${RUN_ROOT}/quicksr_ablation.md"
  fi
fi
