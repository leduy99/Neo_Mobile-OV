# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Small checkpoint loader used by the vendored SANA/WanVAE code."""

from __future__ import annotations

import json
import os

import torch
from termcolor import colored

def hf_download_or_fpath(path: str) -> str:
    """Resolve a local path or a simple hf://repo_id/file path."""

    if not path.startswith("hf://"):
        return path

    from huggingface_hub import hf_hub_download

    rel = path[len("hf://") :]
    parts = rel.split("/")
    if len(parts) < 3:
        raise ValueError(f"Expected hf://org/repo/file, got: {path}")
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    return hf_hub_download(repo_id=repo_id, filename=filename)


def find_model(model_name: str):
    """Load a local or hf:// SANA checkpoint on CPU."""

    print(colored(f"[Sana] Loading model from {model_name}", attrs=["bold"]))
    model_name = hf_download_or_fpath(model_name)
    if not os.path.isfile(model_name):
        raise FileNotFoundError(f"Could not find Sana checkpoint at {model_name}")
    print(colored(f"[Sana] Loaded model from {model_name}", attrs=["bold"]))

    if model_name.endswith(".safetensors"):
        import safetensors.torch

        return {"state_dict": safetensors.torch.load_file(model_name, device="cpu")}

    if model_name.endswith(".safetensors.index.json"):
        import safetensors.torch

        with open(model_name, encoding="utf-8") as handle:
            index = json.load(handle)["weight_map"]
        state_dict = {}
        for shard_name in set(index.values()):
            shard_path = os.path.join(os.path.dirname(model_name), shard_name)
            state_dict.update(safetensors.torch.load_file(shard_path, device="cpu"))
        return {"state_dict": state_dict}

    return torch.load(model_name, map_location=lambda storage, loc: storage)
