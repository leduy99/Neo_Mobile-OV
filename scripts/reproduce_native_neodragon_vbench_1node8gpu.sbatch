#!/usr/bin/env bash
# Native release protocol ONLY: SSD1B -> Hybrid in one pipeline call, released CLI suffix.
# Public QuickSR and raw exports are evaluated separately; paper-exact equivalence is not claimed.
#SBATCH --job-name=neo-native-repro
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/neo-native-repro-%j.out
#SBATCH --error=logs/neo-native-repro-%j.out

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the Neo_Mobile-OV repo root}}"
export ROOT="${PWD}"
module load buildenv-gcccuda/12.1.1-gcc12.3.0
export CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
export BASE_PYTHON="${CONDA_ENV}/bin/python" PYTHON_BIN="${CONDA_ENV}/bin/python"
export VBENCH_ENV="${VBENCH_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_vbench}"
export VBENCH_REPO="${VBENCH_REPO:-${ROOT}/checkpoints/vbench_repo}"
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TORCH_HOME="${TORCH_HOME:-${HF_HOME}}" XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HF_HOME}}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${HF_HOME}}" TMPDIR="${TMPDIR:-${HF_HOME}}"
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-${HF_HOME}/vbench}"
export PYTHONNOUSERSITE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${ROOT}:${VBENCH_REPO}:${PYTHONPATH:-}"

SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-5}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
if [[ ! "${MAX_PROMPTS}" =~ ^[0-9]+$ || ! "${SAMPLES_PER_PROMPT}" =~ ^[1-5]$ ]]; then
  echo "MAX_PROMPTS must be nonnegative and SAMPLES_PER_PROMPT must be 1-5." >&2
  exit 1
fi
if [[ "${MAX_PROMPTS}" -eq 0 && "${SAMPLES_PER_PROMPT}" -ne 5 ]]; then
  echo "Full evaluation requires 5 samples/prompt. Use MAX_PROMPTS for a generation-only smoke." >&2
  exit 1
fi
SUFFIX=""
if [[ "${MAX_PROMPTS}" -gt 0 ]]; then SUFFIX="_smoke${MAX_PROMPTS}"; fi
RUN_ROOT="${RUN_ROOT:-output/vbench_native_neodragon_release_${SAMPLES_PER_PROMPT}x${SUFFIX}}"
NATIVE_ARGS=(--native-repo "${NATIVE_REPO:-${ROOT}/checkpoints/neodragon_repo}"
             --native-cache "${NATIVE_CACHE:-${ROOT}/checkpoints/neodragon}")
if [[ -n "${NATIVE_MODEL_DIR:-}" ]]; then NATIVE_ARGS+=(--native-model-dir "${NATIVE_MODEL_DIR}"); fi
mkdir -p logs "${RUN_ROOT}" "${TMPDIR}" "${VBENCH_CACHE_DIR}" "$(dirname "${VBENCH_ENV}")"
test -x "${BASE_PYTHON}"
exec 9>"${VBENCH_ENV}.mobileov_setup.lock"
flock 9
bash scripts/install_dreamlite_dependencies.sh
bash scripts/install_vbench_env.sh
flock -u 9
exec 9>&-
test -x "${VBENCH_ENV}/bin/python"
test -s "${VBENCH_REPO}/vbench/VBench_full_info.json"

echo "RUN_ROOT=${RUN_ROOT}; native released SSD1B + Hybrid ONLY; no Mobile-OV bridges or Joint checkpoint."
echo "Released CLI suffix; seed once per complete pipeline call (no image/video reseed)."
echo "8 GPU workers; 944 prompts x 5 samples; 49 frames at 24 FPS; raw and QuickSR 2x."
echo "Seed/sample count and public SR choice are recorded; this is not guaranteed to recover the paper's exact score."
echo "Resubmit with the SAME RUN_ROOT after interruption to resume certified pairs."
srun --ntasks=1 --gres=gpu:8 "${BASE_PYTHON}" -u tools/reproduce_native_neodragon_vbench.py \
  --output-dir "${RUN_ROOT}" --systems native_hybrid --protocol standard5x \
  --workers 8 --samples-per-prompt "${SAMPLES_PER_PROMPT}" --max-prompts "${MAX_PROMPTS}" \
  --seed "${SEED:-2026}" --seed-mapping extendable \
  --vbench-repo "${VBENCH_REPO}" --vbench-python "${VBENCH_ENV}/bin/python" \
  "${NATIVE_ARGS[@]}"
if [[ "${MAX_PROMPTS}" -eq 0 ]]; then
  test -s "${RUN_ROOT}/paper_reference_comparison.json"
  test -s "${RUN_ROOT}/quicksr_ablation.json"
  cat "${RUN_ROOT}/paper_reference_comparison.md"
fi
