"""Data and flow contracts shared by Stage-1 training, validation, and tests."""
from __future__ import annotations

from array import array
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.checkpoint import checkpoint

from new_mobile_ov.training.stage1_alignment_data import load_sources, read_alignment_sample
from new_mobile_ov.training.neodragon_pyramid_flow import build_pyramid_flow_state
from tools.data_prepare.download_alignment_images import sha256_file

TASKS = ("t2i", "t2v", "image_reconstruction")


def verify_release(root: Path):
    summary = json.loads((root / "stage1_summary.json").read_text())
    if summary != json.loads((root / ".stage1_data_complete").read_text()):
        raise ValueError("Stage-1 data release is incomplete or modified")
    if summary["status"] != "raw_data_ready" or set(summary["tasks"]) != set(TASKS):
        raise ValueError("Stage-1 requires all three raw-data tasks")
    if sha256_file(root / "sources.json") != summary["sources_sha256"]:
        raise ValueError("Modified source references")
    for name, artifact in summary["manifests"].items():
        path = (root / artifact["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"Invalid manifest: {name}")
    for task in TASKS:
        if min(summary["counts"].get(f"{split}.{task}", 0) for split in ("train", "validation")) < 1:
            raise ValueError(f"Both train and validation must contain {task}")
    return summary


class JsonlRecords:
    """Keep byte offsets rather than millions of Python sample dictionaries."""
    def __init__(self, path):
        self.path = Path(path)
        self.offsets = array("Q")
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.strip():
                    raise ValueError(f"Blank manifest row: {self.path}")
                self.offsets.append(offset)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            return json.loads(stream.readline())


def permuted_index(ordinal, count, seed, task):
    if count < 1:
        raise ValueError("Empty task")
    epoch, position = divmod(ordinal, count)
    digest = hashlib.sha256(f"{seed}:{task}:{epoch}".encode()).digest()
    stride = int.from_bytes(digest[:8], "little") % count or 1
    while math.gcd(stride, count) != 1:
        stride += 1
    return (position * stride + int.from_bytes(digest[8:16], "little")) % count


def prepare_sample(record, sources, short_side=320, long_side=512):
    sample = read_alignment_sample(record, sources)
    w, h = sample["target_frames"][0].size
    square = (int(round(math.sqrt(short_side * long_side) / 64)) * 64,) * 2
    buckets = ((long_side, short_side), (short_side, long_side), square)
    size = min(buckets, key=lambda dims: abs(math.log((w / h) / (dims[0] / dims[1]))))
    frames = [im.resize(size, Image.Resampling.LANCZOS) for im in sample["target_frames"]]
    # Resize the entire view rather than crop away captioned objects. Reconstruction
    # sees the same resized image as the target, only through the frozen VLM.
    sample["image"] = frames[0] if record["task"] == "image_reconstruction" else None
    sample["video"] = torch.from_numpy(np.stack([np.asarray(im) for im in frames])).permute(3, 0, 1, 2).float() / 127.5 - 1
    sample["sample_id"] = record["sample_id"]
    sample.pop("target_frames")
    sample.pop("conditioning_images")
    return sample


class Stage1Dataset(Dataset):
    def __init__(self, root, *, steps, accumulation, rank, world_size, seed,
                 start_step=0, short_side=320, long_side=512):
        self.root, self.sources = Path(root), load_sources(Path(root))
        self.records = {task: JsonlRecords(self.root / "train" / f"{task}.jsonl") for task in TASKS}
        self.steps, self.accumulation = steps, accumulation
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.start_step, self.short_side, self.long_side = start_step, short_side, long_side

    def __len__(self):
        return (self.steps - self.start_step) * self.accumulation

    def __getitem__(self, index):
        micro = index + self.start_step * self.accumulation
        task = TASKS[micro % len(TASKS)]
        ordinal = (micro // len(TASKS)) * self.world_size + self.rank
        records = self.records[task]
        position = permuted_index(ordinal, len(records), self.seed, task)
        sample = prepare_sample(records[position], self.sources, self.short_side, self.long_side)
        if sample["task"] != task:
            raise ValueError("Manifest/task mismatch")
        sample.update(micro=micro, ordinal=ordinal)
        return sample


def flow_loss(dit, connector, layers, mask, latents, scheduler, *, stage, unit,
              generator, gradient_checkpointing=True):
    from neodragon.utils.generation_utils import _prepare_past_condition_latents
    if not 0 <= unit < latents.shape[2]:
        raise ValueError("Invalid latent unit; first unit is 0, not 1")
    clean = latents[:, :, unit:unit + 1].float()
    history = _prepare_past_condition_latents(
        [latents[:, :, i:i + 1] for i in range(unit)], scheduler.config.stages, False)[stage]
    indices = torch.randint(scheduler.config.num_train_timesteps, (latents.shape[0],),
                            device=latents.device, generator=generator)
    sigma = scheduler.sigmas_per_stage[stage].to(latents.device, torch.float32)[indices]
    time = scheduler.timesteps_per_stage[stage].to(latents.device, torch.float32)[indices]
    noise = torch.randn(clean.shape, device=clean.device, generator=generator, dtype=torch.float32)
    state = build_pyramid_flow_state(clean, stage=stage, local_sigma=sigma, noise_high=noise,
                                     start_sigma=scheduler.start_sigmas[stage],
                                     end_sigma=scheduler.end_sigmas[stage], stages=scheduler.config.stages)
    tokens, attention, pooled = connector(layers, mask)

    def predict(tokens, pooled):
        return dit(sample=[history + [state.noisy.to(latents.dtype)]], encoder_hidden_states=tokens,
                   encoder_attention_mask=attention, pooled_projections=pooled,
                   timestep_ratio=time.to(latents.dtype))[0]

    pred = checkpoint(predict, tokens, pooled, use_reentrant=False) if gradient_checkpointing and torch.is_grad_enabled() else predict(tokens, pooled)
    if pred.shape != state.target.shape:
        raise ValueError(f"Flow shape mismatch: {pred.shape} != {state.target.shape}")
    return torch.nn.functional.mse_loss(pred.float(), state.target.float())


def frozen_signatures(modules):
    """Full byte hashes, not just selected weight norms, for smoke freeze checks."""
    signatures = {}
    for name, module in modules.items():
        digest = hashlib.sha256()
        for key, value in module.state_dict().items():
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        signatures[name] = digest.hexdigest()
    return signatures
