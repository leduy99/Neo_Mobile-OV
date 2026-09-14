#!/usr/bin/env bash
# Real 2-GPU local training, checkpoint/resume, and 49-frame generation. Not a quality experiment.
#SBATCH --job-name=mov-s2-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --output=logs/mov-s2-smoke-%j.out
#SBATCH --error=logs/mov-s2-smoke-%j.out

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export HF_HOME="${HF_HOME:-/share_4/users/duy/.cache/huggingface}"
export TMPDIR="${TMPDIR:-/share_4/users/duy/tmp}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON="${PYTHON_BIN:-/share_4/users/duy/.conda/envs/neo_mobileov/bin/python}"
OUT="${OUT:-output/stage2_local_smoke/${SLURM_JOB_ID}}"
DATA_ROOT="${DATA_ROOT:-output/alignment_data_checks/stage1_slurm_release}"
COMMON=(--data-root "${DATA_ROOT}" --workers 1 --validation-samples 1 --log-every 1 --verify-frozen)
mkdir -p "${TMPDIR}"
# Test that a 100k alignment contract can pause/resume without resetting its optimizer or samples.
"${PYTHON}" -s -m torch.distributed.run --standalone --nproc_per_node=2 tools/train_mobileov_stage1.py \
    "${COMMON[@]}" --output-dir "${OUT}/alignment" --steps 100000 --stop-after 1 \
    --save-every 10000 --archive-every 10000 --validate-every 10000
"${PYTHON}" -s -m torch.distributed.run --standalone --nproc_per_node=2 tools/train_mobileov_stage1.py \
    "${COMMON[@]}" --output-dir "${OUT}/alignment" --steps 100000 --stop-after 2 \
    --save-every 10000 --archive-every 10000 --validate-every 10000 --resume "${OUT}/alignment/stage1_resume.pt"
JOINT=(--phase 2 --init-connector checkpoints/hf_mobile_ov/mobileov_stage1_alignment/17507068/stage1_connector_latest.pt
       --steps 6 --save-every 3 --archive-every 6 --validate-every 6 --output-dir "${OUT}/joint")
"${PYTHON}" -s -m torch.distributed.run --standalone --nproc_per_node=2 tools/train_mobileov_stage1.py \
    "${COMMON[@]}" "${JOINT[@]}" --stop-after 3
"${PYTHON}" -s -m torch.distributed.run --standalone --nproc_per_node=2 tools/train_mobileov_stage1.py \
    "${COMMON[@]}" "${JOINT[@]}" --resume "${OUT}/joint/stage2_resume.pt"
"${PYTHON}" -s -u tools/infer_mobileov_stage1.py \
    --checkpoint "${OUT}/joint/stage2_dit_connector_latest.pt" \
    --prompt "A yellow toy car rolls across a wooden table in front of a bookcase." \
    --frames 49 --output-dir "${OUT}/inference"
echo "Completed local 2-GPU Stage-1 resume + Stage-2 train/resume/inference smoke: ${OUT}"
