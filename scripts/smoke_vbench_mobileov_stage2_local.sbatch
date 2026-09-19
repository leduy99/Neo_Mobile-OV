#!/usr/bin/env bash
# Exercise the production launcher, resume, and one real VBench dimension on local H200.
#SBATCH --job-name=mov-s2-vb-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=logs/mov-s2-vb-smoke-%j.out
#SBATCH --error=logs/mov-s2-vb-smoke-%j.out

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
export SKIP_MODULE_LOAD=1 SKIP_VBENCH_INSTALL=1 MAX_PROMPTS=2
export RUN_ROOT="${RUN_ROOT:-${ROOT}/output/stage2_vbench_smoke/${SLURM_JOB_ID}}"
bash scripts/vbench_mobileov_stage2_100k_1node1gpu.sbatch
# A second invocation must skip both completed videos without loading GPU models.
bash scripts/vbench_mobileov_stage2_100k_1node1gpu.sbatch
# The first two official prompts are temporal_flickering prompts. Do not label this a full score.
jq '.[0:2]' "${VBENCH_INFO}" > "${RUN_ROOT}/smoke_info.json"
cd "${VBENCH_REPO}"
"${VBENCH_ENV}/bin/python" "${ROOT}/tools/evaluate_vbench_resume.py" \
    --videos "${RUN_ROOT}/videos" --full-info "${RUN_ROOT}/smoke_info.json" \
    --output "${RUN_ROOT}/smoke_scores" --expected-samples-per-prompt 1 --start 6 --end 7
"${PYTHON_BIN}" "${ROOT}/tools/report_vbench_low_prompts.py" \
    --score-dir "${RUN_ROOT}/smoke_scores/vbench" --output-prefix "${RUN_ROOT}/low_prompts" \
    --dimension temporal_flickering --top-k 2
echo "Stage 2 generation, resume, VBench temporal_flickering and low-prompt smoke passed: ${RUN_ROOT}"
