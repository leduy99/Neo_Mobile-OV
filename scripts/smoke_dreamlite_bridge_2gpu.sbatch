#!/usr/bin/env bash
#SBATCH --job-name=dream-bridge-smoke
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
OUT="output/dreamlite_bridge_2gpu_smoke/${SLURM_JOB_ID:-local}"
EDIT_IMAGE="${OUT}/source.png"
EDIT_MANIFEST="${OUT}/edit.csv"

mkdir -p logs "${OUT}"
export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES=30
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${PYTHON_BIN}" - "${EDIT_IMAGE}" "${EDIT_MANIFEST}" <<'PY'
import csv
import sys
from PIL import Image, ImageDraw

image_path, manifest_path = sys.argv[1:]
image = Image.new("RGB", (256, 256), "#9bc9df")
draw = ImageDraw.Draw(image)
draw.rectangle((32, 96, 224, 224), fill="#5f8f45")
draw.ellipse((88, 48, 168, 128), fill="#d86c35")
image.save(image_path)
with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["source_image", "instruction"])
    writer.writeheader()
    writer.writerow({"source_image": image_path, "instruction": "turn the sky into a warm sunset"})
    writer.writerow({"source_image": image_path, "instruction": "make the scene look like a pencil sketch"})
PY

"${TORCHRUN_BIN}" --standalone --nproc_per_node=2 \
  tools/train_dreamlite_image_bridge.py \
  --config configs/mobile_ov_dreamlite.yaml \
  --generation-prompts configs/prompts/dreamlite_smoke.csv \
  --edit-manifest "${EDIT_MANIFEST}" \
  --edit-probability 0.5 \
  --output-dir "${OUT}/train" \
  --target-step 4 \
  --resume none \
  --batch-size 1 \
  --max-samples 2 \
  --height 256 \
  --width 256 \
  --functional-start-step 2 \
  --functional-ramp-steps 1 \
  --functional-batch-size 1 \
  --closed-loop-start-step 3 \
  --closed-loop-ramp-steps 1 \
  --closed-loop-every 1 \
  --closed-loop-batch-size 1 \
  --log-every 1 \
  --save-latest-every 2 \
  --save-archive-every 4

test -s "${OUT}/train/dreamlite_image_bridge_latest.pt"
test -s "${OUT}/train/history.jsonl"
echo "DreamLite two-GPU generation/edit functional and closed-loop smoke passed: ${OUT}"
