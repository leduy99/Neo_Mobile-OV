#!/usr/bin/env bash
#SBATCH --job-name=neo-rollout-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=output/neodragon_rollout_smoke/slurm-%j.out
#SBATCH --error=output/neodragon_rollout_smoke/slurm-%j.out

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-/share_4/users/duy/.conda/envs/neo_mobileov}"
TORCHRUN="${CONDA_ENV}/bin/torchrun"
MANIFEST="${MANIFEST:-output/neodragon_rollout_smoke/data/latent_manifest.csv}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"
OUT="${OUT:-output/neodragon_rollout_smoke/run-${SLURM_JOB_ID}}"
TARGET_STEP="${TARGET_STEP:-64002}"
RESUME="${RESUME:-none}"
EXPECTED_HISTORY_ROWS="${EXPECTED_HISTORY_ROWS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-8}"

mkdir -p output/neodragon_rollout_smoke "${OUT}"
export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-60}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

nvidia-smi

"${TORCHRUN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  tools/train_neodragon_bridge_rollout_distill.py \
  --manifest "${MANIFEST}" \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  --resume "${RESUME}" \
  --output-dir "${OUT}" \
  --target-step "${TARGET_STEP}" \
  --batch-size "${BATCH_SIZE}" \
  --max-samples "${MAX_SAMPLES}" \
  --parallel ddp \
  --lr 1e-5 \
  --lr-warmup-steps 1 \
  --rollout-ramp-steps 1 \
  --log-every 1 \
  --save-latest-every 1 \
  --save-archive-every 1 \
  --archive-start-step 64001

"${CONDA_ENV}/bin/python" - "${OUT}" "${TARGET_STEP}" "${EXPECTED_HISTORY_ROWS}" <<'PY'
import json
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
target_step = int(sys.argv[2])
expected_history_rows = int(sys.argv[3])
latest = torch.load(
    root / "neodragon_rollout_bridge_latest.pt",
    map_location="cpu",
    weights_only=False,
)
assert latest["step"] == target_step, latest["step"]
assert latest["architecture"]["rollout_calls"] == 18
assert latest["architecture"]["differentiable_cross_unit_history"] is True
assert len(latest["history"]) == expected_history_rows
assert (root / f"neodragon_rollout_bridge_step{target_step:06d}.pt").is_file()
summary = {
    "status": "ok",
    "step": latest["step"],
    "history_rows": len(latest["history"]),
    "rollout_calls": latest["architecture"]["rollout_calls"],
    "latest": str(root / "neodragon_rollout_bridge_latest.pt"),
}
(root / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
