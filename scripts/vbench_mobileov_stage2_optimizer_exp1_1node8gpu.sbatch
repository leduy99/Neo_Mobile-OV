#!/usr/bin/env bash
# RAW/EMA after 10K and 20K additional updates. Each gets raw + QuickSR scoring.
#SBATCH --job-name=mov-vb-opt-exp1
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-vb-opt-exp1-%j.out
#SBATCH --error=logs/mov-vb-opt-exp1-%j.out

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repo root}}"
export ROOT="${PWD}"
: "${EXP1_RUN:?Set EXP1_RUN to output/mobileov_stage2_optimizer_exp1/TRAIN_JOB_ID}"
REFERENCE_RUN="${REFERENCE_RUN:-output/vbench_mobileov_stage2_media3epochs_5x_dual}"
RUN_ROOT="${RUN_ROOT:-output/vbench_mobileov_stage2_optimizer_exp1_5x_dual}"
module load buildenv-gcccuda/12.1.1-gcc12.3.0
export CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
export VBENCH_ENV="${VBENCH_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_vbench}"
export VBENCH_REPO="${VBENCH_REPO:-${ROOT}/checkpoints/vbench_repo}"
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TORCH_HOME="${TORCH_HOME:-${HF_HOME}}" XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HF_HOME}}"
export TMPDIR="${TMPDIR:-${HF_HOME}/tmp}" VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-${HF_HOME}/vbench}"
export PYTHONPATH="${ROOT}:${VBENCH_REPO}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs "${RUN_ROOT}" "${TMPDIR}" "${VBENCH_CACHE_DIR}" "$(dirname "${VBENCH_ENV}")"
test -s "${EXP1_RUN}/evaluation_checkpoints.json"
exec 9>"${VBENCH_ENV}.mobileov_setup.lock"
flock 9
bash scripts/install_vbench_env.sh
flock -u 9
exec 9>&-
read -r -a UPDATES <<< "${EVAL_UPDATES:-10000 20000}"
echo "Exp1 RAW+EMA checkpoint sweep; text only, no external anchor; shared raw/QuickSR frames."
srun --ntasks=1 --gres=gpu:8 "${CONDA_ENV}/bin/python" -u tools/benchmark_mobileov_stage2_optimizer_exp1_dual.py \
    --exp1-run "${EXP1_RUN}" --updates "${UPDATES[@]}" --reference-run "${REFERENCE_RUN}" \
    --config "${CONFIG:-configs/mobile_ov_neodragon.yaml}" --output-dir "${RUN_ROOT}" \
    --workers 8 --samples-per-prompt 5 --seed 2026 --vbench-repo "${VBENCH_REPO}" \
    --vbench-python "${VBENCH_ENV}/bin/python"
cat "${RUN_ROOT}/comparison.md"
