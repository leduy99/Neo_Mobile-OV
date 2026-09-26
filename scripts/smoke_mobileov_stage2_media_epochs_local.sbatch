#!/usr/bin/env bash
#SBATCH --job-name=mov-s2-epoch-test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --output=logs/mov-s2-epoch-test-%j.out
#SBATCH --error=logs/mov-s2-epoch-test-%j.out
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
OUT="${OUT:-output/stage2_media_epochs_smoke/${SLURM_JOB_ID}}"
mkdir -p "${TMPDIR}"
"${PYTHON}" -s tools/prepare_stage2_media_smoke.py \
    --source output/alignment_data_checks/stage1_slurm_release --output "${OUT}/data"
COMMON=(--phase 2 --media-epochs 3 --accumulation 2 --reconstruction-weight .1
        --init-connector checkpoints/hf_mobile_ov/mobileov_stage1_alignment/17507068/stage1_connector_latest.pt
        --data-root "${OUT}/data" --output-dir "${OUT}/train" --workers 1
        --validation-samples 1 --validate-every 10000 --save-every 10000 --archive-every 10000
        --log-every 1 --verify-frozen --cuda-memory-limit-gib 60)
"${PYTHON}" -s -m torch.distributed.run --standalone --nproc_per_node="${NPROC:-2}" \
    tools/train_mobileov_stage1.py "${COMMON[@]}" --stop-after 4
"${PYTHON}" -s -m torch.distributed.run --standalone --nproc_per_node="${NPROC:-2}" \
    tools/train_mobileov_stage1.py "${COMMON[@]}" --resume "${OUT}/train/stage2_resume.pt"
# Use an unseen video first frame. Inference must denoise unit0 only in reference mode.
"${PYTHON}" -s tools/infer_mobileov_stage1.py \
    --checkpoint "${OUT}/train/stage2_dit_connector_latest.pt" \
    --validation-task t2v --data-root "${OUT}/data" --output-dir "${OUT}/t2v"
PROMPT="$(<"${OUT}/data/heldout_prompt.txt")"
"${PYTHON}" -s tools/infer_mobileov_stage1.py \
    --checkpoint "${OUT}/train/stage2_dit_connector_latest.pt" \
    --prompt "${PROMPT}" --first-frame "${OUT}/data/heldout_first_frame.png" \
    --i2v-mode reference --output-dir "${OUT}/i2v_reference"
"${PYTHON}" -s tools/infer_mobileov_stage1.py \
    --checkpoint "${OUT}/train/stage2_dit_connector_latest.pt" \
    --prompt "${PROMPT}" --first-frame "${OUT}/data/heldout_first_frame.png" \
    --i2v-mode anchor --output-dir "${OUT}/i2v_anchor"
echo "Completed real-model media-epoch smoke (${NPROC:-2} GPU process(es)): ${OUT}"
