#!/usr/bin/env bash
#SBATCH --job-name=dream-v8-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-/share_4/users/duy/.conda/envs/neo_mobileov}"
PYTHON_BIN="${CONDA_ENV}/bin/python"
TORCHRUN_BIN="${CONDA_ENV}/bin/torchrun"
OUT="${OUT:-output/dreamlite_compact_v8_2gpu_smoke/${SLURM_JOB_ID:-local}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
IMAGE_PATH="${PWD}/Docs/assets/training_image_bridge_dreamlite/native_resolution_factors.jpg"
JOURNEYDB_MANIFEST="${OUT}/journeydb_smoke.csv"
SHORT_MANIFEST="${OUT}/short_caption_smoke.csv"

mkdir -p logs "${OUT}"
test -f "${IMAGE_PATH}"
printf 'caption,image_path\nA red panda reads a book beside a window.,%s\n' \
  "${IMAGE_PATH}" > "${JOURNEYDB_MANIFEST}"
printf 'caption,image_path\nAn astronaut walks across a red desert.,%s\n' \
  "${IMAGE_PATH}" > "${SHORT_MANIFEST}"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES=30
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train_dreamlite_image_bridge.py \
  --config configs/mobile_ov_dreamlite_compact_v8.yaml \
  --generation-prompt-manifests "${JOURNEYDB_MANIFEST};${SHORT_MANIFEST}" \
  --generation-source-weights 0.7,0.3 \
  --generation-source-names journeydb,shortcaption \
  --grounded-source-names journeydb,shortcaption \
  --output-dir "${OUT}" \
  --target-step 3 \
  --resume none \
  --batch-size 1 \
  --lr 4e-5 \
  --lr-warmup-steps 1 \
  --lr-final-scale 0.1 \
  --lr-decay-end-step 2 \
  --resolution-buckets 256x256@1024x1024:1 \
  --semantic-prompt-probability 0 \
  --representation-mode direct \
  --content-token-mse-weight 0.5 \
  --content-token-cos-weight 1.0 \
  --wrapper-token-weight 0.1 \
  --content-pooled-cos-weight 0.25 \
  --content-pooled-mse-weight 0.1 \
  --content-token-mean-weight 0.025 \
  --content-token-std-weight 0.025 \
  --semantic-contrastive-weight 0.1 \
  --global-contrastive \
  --projected-weight 1.0 \
  --functional-weight 5.0 \
  --functional-cos-weight 0.1 \
  --functional-start-step 2 \
  --functional-ramp-steps 1 \
  --functional-batch-size 1 \
  --functional-call-weights 0,0,1,0 \
  --grounded-functional-probability 1.0 \
  --grounded-functional-weight 0.5 \
  --grounded-functional-start-step 2 \
  --transition-weight 1.0 \
  --transition-cos-weight 0.05 \
  --student-state-probability 1.0 \
  --student-state-start-step 3 \
  --student-state-ramp-steps 1 \
  --closed-loop-weight 0 \
  --training-version v8 \
  --log-every 1 \
  --save-latest-every 2 \
  --save-archive-every 3

test -s "${OUT}/dreamlite_image_bridge_latest.pt"
test -s "${OUT}/history.jsonl"
grep -q '"phase": "grounded_functional"' "${OUT}/history.jsonl"
"${PYTHON_BIN}" - "${OUT}/dreamlite_image_bridge_latest.pt" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert payload["architecture"] == "MobileOVDreamLiteCompactBridgeV7"
assert payload["config"]["training_version"] == "v8"
assert payload["config"]["lr_decay_end_step"] == 2
print("V8 checkpoint metadata: OK")
PY

echo "DreamLite compact V8 ${NPROC_PER_NODE}-GPU smoke passed: ${OUT}"
