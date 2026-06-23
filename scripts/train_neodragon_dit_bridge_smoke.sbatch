#!/usr/bin/env bash
#SBATCH --job-name=mov-neo-dit
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --chdir=/share_4/users/duy/project/unified_video/New-Mobile-OV
#SBATCH --output=output/neodragon_dit_bridge_smoke/logs/%x_%j.out
#SBATCH --error=output/neodragon_dit_bridge_smoke/logs/%x_%j.out

set -euo pipefail

ROOT="/share_4/users/duy/project/unified_video/New-Mobile-OV"
NEO_ROOT="/share_4/users/duy/project/unified_video/neodragon"
OUT="${ROOT}/output/neodragon_dit_bridge_smoke"
MANIFEST="${OUT}/smoke_manifest.csv"
BRIDGE_CKPT="${BRIDGE_CKPT:-${ROOT}/output/neodragon_text_bridge_smoke/bridge/neodragon_text_bridge_latest.pt}"
mkdir -p "${OUT}/logs"

source /share_0/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-/share_4/users/duy/.conda/envs/neo_mobileov}"

export PYTHONPATH="${ROOT}:${NEO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export FSDP_USE_ORIG_PARAMS=true
export HF_HOME="${NEO_ROOT}/models/hf_home"
export TORCH_HOME="${NEO_ROOT}/models/torch_home"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${ROOT}"
python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("/share_4/users/duy/project/unified_video/New-Mobile-OV")
src = root / "output/wanvae_anchor_similarity_100_910/per_video_metrics.csv"
out = root / "output/neodragon_dit_bridge_smoke/smoke_manifest.csv"
df = pd.read_csv(src)
row = df[df["status"].eq("ok")].iloc[0]
pd.DataFrame(
    {
        "video_path": [row["video_path"]],
        "prompt": ["A short realistic video clip with natural motion."],
    }
).to_csv(out, index=False)
print(out)
PY

date
nvidia-smi

python tools/train_neodragon_dit_bridge.py \
  --config configs/mobile_ov_neodragon.yaml \
  --manifest "${MANIFEST}" \
  --bridge-ckpt "${BRIDGE_CKPT}" \
  --output-dir "${OUT}/dit" \
  --steps "${STEPS:-1}" \
  --batch-size 1 \
  --train-last-n-blocks "${TRAIN_LAST_N_BLOCKS:-1}" \
  --log-every 1 \
  --save-every 1

find "${OUT}" -maxdepth 3 -type f -print -exec ls -lh {} \;
date
