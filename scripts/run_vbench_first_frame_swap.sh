#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the Neo_Mobile-OV repo root}}"
export ROOT="${PWD}"
case "${SWAP_CASE:?Set by the job wrapper}" in
  joint_v2_ssd1b|native_hybrid_dreamlite_v11) ;;
  *) echo "Unknown first-frame swap: ${SWAP_CASE}" >&2; exit 1 ;;
esac
module load buildenv-gcccuda/12.1.1-gcc12.3.0
export CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
export BASE_PYTHON="${CONDA_ENV}/bin/python" PYTHON_BIN="${CONDA_ENV}/bin/python"
export VBENCH_ENV="${VBENCH_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_vbench}"
export VBENCH_REPO="${VBENCH_REPO:-${ROOT}/checkpoints/vbench_repo}"
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TORCH_HOME="${TORCH_HOME:-${HF_HOME}}" XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HF_HOME}}"
export TMPDIR="${TMPDIR:-${HF_HOME}/tmp}" VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-${HF_HOME}/vbench}"
export PYTHONPATH="${ROOT}:${VBENCH_REPO}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASELINE_RUN="${BASELINE_RUN:-output/vbench_joint_v2_vs_native_quicksr_berzelius_5x_dual}"
test -s "${BASELINE_RUN}/generation_summary.json"
test -s "${BASELINE_RUN}/quicksr/run_contract.json"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-5}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
if [[ ! "${MAX_PROMPTS}" =~ ^[0-9]+$ || ! "${SAMPLES_PER_PROMPT}" =~ ^[1-5]$ ]]; then
  echo "Invalid prompt limit/sample count" >&2; exit 1
fi
if [[ "${MAX_PROMPTS}" -eq 0 && "${SAMPLES_PER_PROMPT}" -ne 5 ]]; then
  echo "Full benchmark requires 5 samples/prompt" >&2; exit 1
fi
if [[ "${MAX_PROMPTS}" -eq 0 ]]; then
  test -s "${BASELINE_RUN}/quicksr_ablation.json"
  test -s "${BASELINE_RUN}/score_assets.json"
fi
SUFFIX=""
if [[ "${MAX_PROMPTS}" -gt 0 ]]; then SUFFIX="_smoke${MAX_PROMPTS}"; fi
RUN_ROOT="${RUN_ROOT:-output/vbench_${SWAP_CASE}_${SAMPLES_PER_PROMPT}x_dual${SUFFIX}}"
REL="mobileov_monolithic_pyramid_cfg_dit/mobileov_monolithic_pyramid_cfg_dit_17346642/neodragon_dit_bridge_step010000.pt"
if [[ -z "${JOINT_CKPT:-}" ]]; then
  if [[ -s "output/${REL}" ]]; then JOINT_CKPT="output/${REL}"; else JOINT_CKPT="checkpoints/hf_mobile_ov/${REL}"; fi
fi
NATIVE_ARGS=()
if [[ -n "${NATIVE_MODEL_DIR:-}" ]]; then NATIVE_ARGS+=(--native-model-dir "${NATIVE_MODEL_DIR}"); fi
mkdir -p logs "${RUN_ROOT}" "${TMPDIR}" "${VBENCH_CACHE_DIR}" "$(dirname "${VBENCH_ENV}")"
test -x "${BASE_PYTHON}"
exec 9>"${VBENCH_ENV}.mobileov_setup.lock"
flock 9
bash scripts/install_dreamlite_dependencies.sh
bash scripts/install_vbench_env.sh
flock -u 9
exec 9>&-
echo "SWAP=${SWAP_CASE}; baseline=${BASELINE_RUN}; OUT=${RUN_ROOT}"
echo "Reuse ORIGINAL baseline PNGs. No image regeneration; original video weights/conditions/schedules/seeds."
echo "Each new clip is exported raw and with QuickSR; both variants receive complete 16-dimension VBench scoring."
echo "Native control is matched 82.85, NOT release-style 83.35. This tests the whole first-frame stack, not bridge error alone."
srun --ntasks=1 --gres=gpu:8 "${BASE_PYTHON}" -u tools/benchmark_first_frame_swap.py \
  --swap "${SWAP_CASE}" --baseline-run "${BASELINE_RUN}" --output-dir "${RUN_ROOT}" \
  --joint-checkpoint "${JOINT_CKPT}" --workers 8 --samples-per-prompt "${SAMPLES_PER_PROMPT}" \
  --max-prompts "${MAX_PROMPTS}" --seed "${SEED:-2026}" --seed-mapping extendable \
  --vbench-repo "${VBENCH_REPO}" --vbench-python "${VBENCH_ENV}/bin/python" "${NATIVE_ARGS[@]}"
if [[ "${MAX_PROMPTS}" -eq 0 ]]; then
  test -s "${RUN_ROOT}/anchor_effect.json"
  cat "${RUN_ROOT}/anchor_effect.md"
fi
