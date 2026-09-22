#!/usr/bin/env bash
# Exercise the new production wrapper and one scoring dimension inside local SLURM.
#SBATCH --job-name=mov-s2rec-vb-test
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/mov-s2rec-vb-test-%j.out
#SBATCH --error=logs/mov-s2rec-vb-test-%j.out

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
ROOT="$(pwd -P)"
export PYTHON_BIN="${PYTHON_BIN:-/share_4/users/duy/.conda/envs/neo_mobileov/bin/python}"
export VBENCH_ENV="${VBENCH_ENV:-/share_4/users/duy/.conda/envs/vbench_cu121_20260515}"
export VBENCH_REPO="${VBENCH_REPO:-/share_4/users/duy/project/unified_video/VBench}"
export VBENCH_INFO="${VBENCH_INFO:-${VBENCH_REPO}/vbench/VBench_full_info.json}"
export HF_HOME="${HF_HOME:-/share_4/users/duy/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/share_4/users/duy/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/share_4/users/duy/.cache}"
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-/share_4/users/duy/.cache/vbench}"
export TMPDIR="${TMPDIR:-/share_4/users/duy/tmp}"
export SKIP_MODULE_LOAD=1 SKIP_VBENCH_INSTALL=1 MAX_PROMPTS=2 SAMPLES_PER_PROMPT=1
export RUN_ROOT="${RUN_ROOT:-${ROOT}/output/stage2_reconstruction_vbench_smoke/${SLURM_JOB_ID}}"
bash scripts/vbench_mobileov_stage2_reconstruction150k_1node1gpu.sbatch
# Completed generation must resume without rendering the two videos again.
bash scripts/vbench_mobileov_stage2_reconstruction150k_1node1gpu.sbatch
jq '.[0:2]' "${VBENCH_INFO}" > "${RUN_ROOT}/smoke_info.json"
cd "${VBENCH_REPO}"
"${VBENCH_ENV}/bin/python" "${ROOT}/tools/evaluate_vbench_resume.py" \
    --videos "${RUN_ROOT}/videos" --full-info "${RUN_ROOT}/smoke_info.json" \
    --output "${RUN_ROOT}/smoke_scores" --expected-samples-per-prompt 1 --start 6 --end 7
"${PYTHON_BIN}" "${ROOT}/tools/report_vbench_low_prompts.py" \
    --score-dir "${RUN_ROOT}/smoke_scores/vbench" --output-prefix "${RUN_ROOT}/low_prompts" \
    --dimension temporal_flickering --top-k 2
echo "Reconstruction 150k VBench smoke passed: generation, resume, temporal_flickering and low prompts; not full VBench."
