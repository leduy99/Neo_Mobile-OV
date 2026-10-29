#!/usr/bin/env bash
# One allocation, sequential checkpoint evaluations, existing SSD1B PNGs only.
#SBATCH --job-name=mov-sanc
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-sanc-%j.out
#SBATCH --error=logs/mov-sanc-%j.out

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repo root}}"
export ROOT="${PWD}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-5}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
if [[ ! "${MAX_PROMPTS}" =~ ^[0-9]+$ || ! "${SAMPLES_PER_PROMPT}" =~ ^[1-5]$ ]]; then
    echo "Invalid prompt limit/sample count." >&2; exit 1
fi
if [[ "${MAX_PROMPTS}" -eq 0 && "${SAMPLES_PER_PROMPT}" -ne 5 ]]; then
    echo "Full evaluation requires 5 videos/prompt." >&2; exit 1
fi
ANCHOR_REFERENCE="${ANCHOR_REFERENCE:-output/vbench_stage2_epoch3_ssd1b_conditioning_5x_dual/ssd1b_anchor}"
test -s "${ANCHOR_REFERENCE}/quicksr_ablation.json" || { echo "Missing original epoch-3 anchor report: ${ANCHOR_REFERENCE}" >&2; exit 1; }
SUFFIX=""
if [[ "${MAX_PROMPTS}" -gt 0 ]]; then SUFFIX="_smoke${MAX_PROMPTS}"; fi
RUN_ROOT="${RUN_ROOT:-output/vbench_stage2_checkpoint_ssd1b_sweep_${SAMPLES_PER_PROMPT}x_dual${SUFFIX}}"
CASE_ARGS=()
if [[ -n "${SWEEP_CASES:-}" ]]; then
    read -r -a SELECTED <<< "${SWEEP_CASES}"
    CASE_ARGS=(--cases "${SELECTED[@]}")
fi
FRAME_ARGS=()
if [[ -n "${ANCHOR_ROOT:-}" ]]; then FRAME_ARGS=(--frames-root "${ANCHOR_ROOT}"); fi
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
echo "One 8-GPU job; sequential SSD1B-anchor sweep. Default: clean-resume epochs4/5 + fresh-history epochs3/4/5."
echo "Reuse the original epoch-3 SSD1B PNG cache and anchor scores. No SSD1B model loading or image generation."
echo "944 prompts x 5 samples/checkpoint; text-only SmolVLM2, VAE first-frame anchor, raw + QuickSR. OUT=${RUN_ROOT}"
srun --ntasks=1 --gres=gpu:8 "${BASE_PYTHON}" -u tools/benchmark_mobileov_stage2_ssd1b_checkpoint_sweep.py \
    --config "${CONFIG:-configs/mobile_ov_neodragon.yaml}" --output-dir "${RUN_ROOT}" \
    --anchor-reference "${ANCHOR_REFERENCE}" --workers 8 \
    --samples-per-prompt "${SAMPLES_PER_PROMPT}" --max-prompts "${MAX_PROMPTS}" --seed "${SEED:-2026}" \
    --vbench-repo "${VBENCH_REPO}" --vbench-python "${VBENCH_ENV}/bin/python" \
    "${CASE_ARGS[@]}" "${FRAME_ARGS[@]}"
if [[ "${MAX_PROMPTS}" -eq 0 ]]; then
    test -s "${RUN_ROOT}/comparison.json"
    cat "${RUN_ROOT}/comparison.md"
fi
