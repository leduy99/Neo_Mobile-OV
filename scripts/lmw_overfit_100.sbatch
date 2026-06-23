#!/usr/bin/env bash
#SBATCH --job-name=lmw-overfit100
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH --chdir=/share_4/users/duy/project/unified_video/New-Mobile-OV
#SBATCH --output=output/lmw_overfit_100_slurm_logs/%x_%j.out
#SBATCH --error=output/lmw_overfit_100_slurm_logs/%x_%j.out

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/share_4/users/duy/project/unified_video/New-Mobile-OV}"
if [[ ! -d "${ROOT}/new_mobile_ov" ]]; then
  ROOT="/share_4/users/duy/project/unified_video/New-Mobile-OV"
fi
cd "$ROOT"

source /share_0/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-/share_4/users/duy/.conda/envs/neo_mobileov}"

export PYTHONPATH="${ROOT}:${ROOT}/third_party/sana:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/share_4/users/duy/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/share_4/users/duy/.cache/torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${CONFIG:-configs/mobile_ov_current.yaml}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-data/lmw_feasibility/openvid100_81f/manifest.csv}"
OUT_ROOT="${OUT_ROOT:-output/lmw_overfit_100_${SLURM_JOB_ID:-local}}"
WAN_VAE="${WAN_VAE:-/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/omni_ckpts/sana_video_2b_480p/vae/Wan2.1_VAE.pth}"

STEPS="${STEPS:-2000}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
BLOCKS="${BLOCKS:-10}"
TEXT_SEED="${TEXT_SEED:-0}"
DECODE_VISUAL_COUNT="${DECODE_VISUAL_COUNT:-10}"

SPLIT_DIR="${OUT_ROOT}/splits"
TRAIN_MANIFEST="${SPLIT_DIR}/train80.csv"
VAL_MANIFEST="${SPLIT_DIR}/val10.csv"
VIS_MANIFEST="${SPLIT_DIR}/visual10.csv"

export CONFIG SOURCE_MANIFEST OUT_ROOT WAN_VAE
export STEPS HIDDEN_DIM BLOCKS TEXT_SEED DECODE_VISUAL_COUNT
export SPLIT_DIR TRAIN_MANIFEST VAL_MANIFEST VIS_MANIFEST

mkdir -p "$OUT_ROOT" "$SPLIT_DIR"

echo "ROOT=${ROOT}"
echo "CONFIG=${CONFIG}"
echo "SOURCE_MANIFEST=${SOURCE_MANIFEST}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "STEPS=${STEPS} HIDDEN_DIM=${HIDDEN_DIM} BLOCKS=${BLOCKS}"
nvidia-smi

python - <<'PY'
import csv
import json
import os
import pickle
from pathlib import Path

import torch

from new_mobile_ov.config import load_config
from new_mobile_ov.motion import LatentMotionWeaver

root = Path.cwd()
source = Path(os.environ["SOURCE_MANIFEST"])
if not source.is_absolute():
    source = root / source
out_root = Path(os.environ["OUT_ROOT"])
split_dir = Path(os.environ["SPLIT_DIR"])
split_dir.mkdir(parents=True, exist_ok=True)

with source.open(newline="") as f:
    rows = list(csv.DictReader(f))
if len(rows) < 100:
    raise SystemExit(f"Need at least 100 rows in {source}, found {len(rows)}")

splits = {
    "train80.csv": rows[:80],
    "val10.csv": rows[80:90],
    "visual10.csv": rows[90:100],
}
for name, split_rows in splits.items():
    path = split_dir / name
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(split_rows)
    print(f"Wrote {len(split_rows):3d} rows -> {path}")

checked = []
for idx in [0, 79, 80, 89, 90, 99]:
    row = rows[idx]
    latent_path = Path(row["latent_path"])
    if not latent_path.is_absolute():
        latent_path = source.parent / latent_path
    with latent_path.open("rb") as f:
        payload = pickle.load(f)
    latent = payload.get("latent_feature", payload.get("latent"))
    if not isinstance(latent, torch.Tensor):
        latent = torch.as_tensor(latent)
    checked.append({
        "row": idx,
        "latent_path": str(latent_path),
        "shape": list(latent.shape),
        "prompt_preview": str(row.get("prompt") or row.get("caption") or "")[:120],
    })
    if tuple(latent.shape) != (16, 21, 60, 104):
        raise SystemExit(f"Unexpected latent shape at row {idx}: {tuple(latent.shape)}")

cfg = load_config(os.environ["CONFIG"])
model = LatentMotionWeaver(
    latent_channels=cfg.motion.latent_channels,
    text_dim=cfg.motion.text_dim,
    hidden_dim=int(os.environ["HIDDEN_DIM"]),
    temporal_len=cfg.motion.temporal_len,
    num_blocks=int(os.environ["BLOCKS"]),
    mlp_ratio=cfg.motion.mlp_ratio,
    temporal_kernel=cfg.motion.temporal_kernel,
    temporal_dilation_cycle=cfg.motion.temporal_dilation_cycle,
).float().eval()
sample = torch.zeros(1, 16, 21, 8, 8)
text = torch.zeros(1, cfg.motion.text_dim)
with torch.no_grad():
    pred = model(sample[:, :, 0], text)
copy = sample[:, :, 0:1].repeat(1, 1, 21, 1, 1)
preflight = {
    "source_manifest": str(source),
    "num_rows": len(rows),
    "splits": {"train": 80, "val": 10, "visual": 10},
    "checked_latents": checked,
    "model": {
        "latent_channels": cfg.motion.latent_channels,
        "text_dim": cfg.motion.text_dim,
        "hidden_dim": int(os.environ["HIDDEN_DIM"]),
        "temporal_len": cfg.motion.temporal_len,
        "num_blocks": int(os.environ["BLOCKS"]),
        "temporal_kernel": cfg.motion.temporal_kernel,
        "temporal_dilation_cycle": cfg.motion.temporal_dilation_cycle,
    },
    "loss": {
        "latent_loss": "L1(z_pred[:,:,1:], z_gt[:,:,1:])",
        "motion_loss": "L1(delta_t z_pred, delta_t z_gt)",
        "total": f"latent_loss + {cfg.train.motion_weight} * motion_loss + {cfg.train.accel_weight} * accel_loss",
    },
    "zero_init_copy_max_abs_error": float((pred - copy).abs().max()),
    "anchor_slice_max_abs_error": float((pred[:, :, 0] - sample[:, :, 0]).abs().max()),
}
(out_root / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
print(json.dumps(preflight, indent=2))
PY

run_one_mode() {
  local mode="$1"
  local out="${OUT_ROOT}/${mode}"
  mkdir -p "$out"

  echo "============================================================"
  echo "Training LMW overfit pilot: mode=${mode}, steps=${STEPS}"
  echo "============================================================"

  python tools/train_lmw_projection.py \
    --config "$CONFIG" \
    --manifest "$TRAIN_MANIFEST" \
    --output-dir "$out" \
    --steps "$STEPS" \
    --hidden-dim "$HIDDEN_DIM" \
    --blocks "$BLOCKS" \
    --text-mode "$mode" \
    --text-seed "$TEXT_SEED" \
    --device cuda

  for split in train val visual; do
    local manifest="$TRAIN_MANIFEST"
    local decode_count=0
    if [[ "$split" == "val" ]]; then
      manifest="$VAL_MANIFEST"
      decode_count=0
    elif [[ "$split" == "visual" ]]; then
      manifest="$VIS_MANIFEST"
      decode_count="$DECODE_VISUAL_COUNT"
    fi

    echo "============================================================"
    echo "Evaluating mode=${mode}, split=${split}, decode_count=${decode_count}"
    echo "============================================================"

    python tools/evaluate_lmw_projection.py \
      --config "$CONFIG" \
      --checkpoint "${out}/lmw_latest.pt" \
      --manifest "$manifest" \
      --output-dir "${out}/eval_${split}" \
      --text-mode checkpoint \
      --vae-path "$WAN_VAE" \
      --decode-count "$decode_count" \
      --device cuda
  done
}

run_one_mode zero
run_one_mode stable_random

python - <<'PY'
import json
import os
from pathlib import Path

out_root = Path(os.environ["OUT_ROOT"])
rows = []
for mode in ["zero", "stable_random"]:
    for split in ["train", "val", "visual"]:
        path = out_root / mode / f"eval_{split}" / "eval_summary.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        rows.append({
            "mode": mode,
            "split": split,
            "copy_latent": data["copy"]["latent_loss"],
            "pred_latent": data["pred"]["latent_loss"],
            "latent_improvement": data["improvement"]["latent_loss"],
            "copy_motion": data["copy"]["motion_loss"],
            "pred_motion": data["pred"]["motion_loss"],
            "motion_improvement": data["improvement"]["motion_loss"],
        })

summary_path = out_root / "overfit_summary.tsv"
with summary_path.open("w", encoding="utf-8") as f:
    f.write("mode\tsplit\tcopy_latent\tpred_latent\tlatent_improvement\tcopy_motion\tpred_motion\tmotion_improvement\n")
    for r in rows:
        f.write(
            f"{r['mode']}\t{r['split']}\t"
            f"{r['copy_latent']:.6f}\t{r['pred_latent']:.6f}\t{r['latent_improvement']:.6f}\t"
            f"{r['copy_motion']:.6f}\t{r['pred_motion']:.6f}\t{r['motion_improvement']:.6f}\n"
        )
print(f"Wrote aggregate summary -> {summary_path}")
print(summary_path.read_text())
PY

echo "LMW overfit pilot finished: ${OUT_ROOT}"
