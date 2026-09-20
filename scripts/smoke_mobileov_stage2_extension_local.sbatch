#!/usr/bin/env bash
# Extend a real completed 2-GPU smoke checkpoint, then resume the extended run after a pause.
#SBATCH --job-name=mov-s2-ext-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --gres=gpu:2
#SBATCH --time=00:45:00
#SBATCH --output=logs/mov-s2-ext-smoke-%j.out
#SBATCH --error=logs/mov-s2-ext-smoke-%j.out

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export HF_HOME="${HF_HOME:-/share_4/users/duy/.cache/huggingface}"
export TMPDIR="${TMPDIR:-/share_4/users/duy/tmp}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 TORCH_DIST_INIT_DEVICE_ID=1 TORCH_DIST_TIMEOUT_MINUTES=3
PYTHON="${PYTHON_BIN:-/share_4/users/duy/.conda/envs/neo_mobileov/bin/python}"
OUT="${OUT:-output/stage2_extension_smoke/${SLURM_JOB_ID}}"
SOURCE="${SOURCE:-output/stage2_local_smoke/5082/joint/stage2_resume.pt}"
test -s "${SOURCE}"
mkdir -p "${TMPDIR}"
COMMON=(--phase 2 --init-connector checkpoints/hf_mobile_ov/mobileov_stage1_alignment/17507068/stage1_connector_latest.pt
        --data-root output/alignment_data_checks/stage1_slurm_release --workers 1 --validation-samples 2
        --log-every 1 --verify-frozen --cuda-memory-limit-gib 36 --output-dir "${OUT}"
        --steps 16 --save-every 10000 --archive-every 10000 --validate-every 10000)
"${PYTHON}" -s -m torch.distributed.run --standalone --nproc_per_node=2 tools/train_mobileov_stage1.py \
    "${COMMON[@]}" --resume "${SOURCE}" --expected-resume-step 12 --extend-steps --stop-after 14
"${PYTHON}" -s -m torch.distributed.run --standalone --nproc_per_node=2 tools/train_mobileov_stage1.py \
    "${COMMON[@]}" --resume "${OUT}/stage2_resume.pt" --expected-resume-step 14
jq -e '.step == 16' "${OUT}/training_complete.json"
jq -e '.previous_target_steps == 12 and .target_steps == 16 and .optimizer_restored' "${OUT}/resume_from_step000012.json"
echo "Passed local real-model 2-GPU extension 12->14->16; this is a resume test, not a quality claim: ${OUT}"
