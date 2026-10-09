#!/usr/bin/env bash
# Final Stage-2 media3epochs checkpoint: all frames generated; raw + QuickSR evaluated separately.
#SBATCH --job-name=mov-vb-s2-3ep
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-vb-s2-3ep-%j.out
#SBATCH --error=logs/mov-vb-s2-3ep-%j.out

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the Neo_Mobile-OV repo root}}"
export ROOT="${PWD}"
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
CHECKPOINT="${CHECKPOINT:-output/mobileov_stage2_media3epochs/17641364/stage2_dit_connector_latest.pt}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-5}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
if [[ ! "${MAX_PROMPTS}" =~ ^[0-9]+$ || ! "${SAMPLES_PER_PROMPT}" =~ ^[1-5]$ ]]; then
  echo "Invalid prompt limit or sample count." >&2; exit 1
fi
if [[ "${MAX_PROMPTS}" -eq 0 && "${SAMPLES_PER_PROMPT}" -ne 5 ]]; then
  echo "Full evaluation requires 5 videos/prompt." >&2; exit 1
fi
SUFFIX=""
if [[ "${MAX_PROMPTS}" -gt 0 ]]; then SUFFIX="_smoke${MAX_PROMPTS}"; fi
RUN_ROOT="${RUN_ROOT:-output/vbench_mobileov_stage2_media3epochs_${SAMPLES_PER_PROMPT}x_dual${SUFFIX}}"
test -s "${CHECKPOINT}" || { echo "Missing final local checkpoint: ${CHECKPOINT}; no old HF fallback." >&2; exit 1; }
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
echo "Final Stage-2 three-epoch checkpoint: ${CHECKPOINT}"
echo "SmolVLM2 + trained Stage-2 connector/DiT; no DreamLite/SSD1B/native conditioner/observed image."
echo "Eight GPU workers; 944 prompts x 5 samples; first/video steps 20/10 per stage, CFG 7/5."
echo "Generate 49 frames including frame zero once; export raw 512x320 and QuickSR medium 2x 1024x640 at 24 FPS."
echo "Refuse incomplete checkpoints. Resubmit with SAME RUN_ROOT and unchanged weights/code to resume."
srun --ntasks=1 --gres=gpu:8 "${BASE_PYTHON}" -u tools/benchmark_mobileov_stage2_media3epochs_dual.py \
  --checkpoint "${CHECKPOINT}" --config "${CONFIG:-configs/mobile_ov_neodragon.yaml}" \
  --output-dir "${RUN_ROOT}" --workers 8 --samples-per-prompt "${SAMPLES_PER_PROMPT}" \
  --max-prompts "${MAX_PROMPTS}" --seed "${SEED:-2026}" \
  --vbench-repo "${VBENCH_REPO}" --vbench-python "${VBENCH_ENV}/bin/python" "${PIN_ARGS[@]}"
if [[ "${MAX_PROMPTS}" -eq 0 ]]; then
  test -s "${RUN_ROOT}/quicksr_ablation.json"
  cat "${RUN_ROOT}/quicksr_ablation.md"
fi
