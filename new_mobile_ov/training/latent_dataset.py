from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class WanVAELatentDataset(Dataset):
    """Dataset for cached WanVAE latents with prompt text.

    Manifest columns:
    - ``latent_path``: pickle with key ``latent_feature`` as [C,T,H,W]
    - ``prompt`` or ``caption``: text used by the bridge
    """

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.rows = pd.read_csv(self.manifest_path).to_dict("records")
        if not self.rows:
            raise ValueError(f"Empty latent manifest: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index % len(self.rows)]
        latent_path = Path(str(row["latent_path"]))
        if not latent_path.is_absolute():
            latent_path = self.manifest_path.parent / latent_path
        with open(latent_path, "rb") as f:
            payload = pickle.load(f)
        latent = payload.get("latent_feature", payload.get("latent"))
        if not isinstance(latent, torch.Tensor):
            latent = torch.as_tensor(latent)
        prompt = str(row.get("prompt") or row.get("caption") or payload.get("prompt") or "")
        return {"index": int(index % len(self.rows)), "prompt": prompt, "latent": latent.float(), "latent_path": str(latent_path)}
