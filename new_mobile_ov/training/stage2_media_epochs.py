"""Exact media passes, with paired reconstruction and rotating video conditions.

All ranks process the same source type per update, including zero-weight padding.
This keeps the optional reconstruction backward passes aligned across DDP ranks.
"""
from dataclasses import dataclass
from functools import cached_property
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from new_mobile_ov.training.stage1_alignment import JsonlRecords, permuted_index, prepare_sample, verify_release
from new_mobile_ov.training.stage1_alignment_data import load_sources

VIDEO_MODES = ("t2v", "i2v_reference", "i2v_anchor")
METRIC_TASKS = ("t2i", *VIDEO_MODES, "image_reconstruction")


@dataclass(frozen=True)
class MediaEpochPlan:
    images: int
    videos: int
    world_size: int
    accumulation: int
    epochs: int = 3
    seed: int = 20260911

    def __post_init__(self):
        if min(self.images, self.videos, self.world_size, self.accumulation, self.epochs) < 1:
            raise ValueError("Media counts, batch size and epochs must be positive")
        if self.epochs % 3:
            raise ValueError("Use complete three-epoch cycles so every video sees all three modes equally")

    @property
    def global_batch(self):
        return self.world_size * self.accumulation

    @property
    def image_steps(self):
        return math.ceil(self.images / self.global_batch)

    @property
    def video_steps(self):
        return math.ceil(self.videos / self.global_batch)

    @property
    def steps_per_epoch(self):
        return self.image_steps + self.video_steps

    @property
    def steps(self):
        return self.epochs * self.steps_per_epoch

    @cached_property
    def orders(self):
        orders = []
        for epoch in range(self.epochs):
            order = list(range(self.steps_per_epoch))
            random.Random(self.seed + 104729 * epoch).shuffle(order)
            orders.append(order)
        return orders

    def slot(self, step, micro, rank):
        """Zero-based optimizer step; padding repeats a valid row with zero weight."""
        if not (0 <= step < self.steps and 0 <= micro < self.accumulation and 0 <= rank < self.world_size):
            raise IndexError("Invalid epoch schedule position")
        epoch, update = divmod(step, self.steps_per_epoch)
        block = self.orders[epoch][update]
        source = "t2i" if block < self.image_steps else "t2v"
        block = block if source == "t2i" else block - self.image_steps
        count = self.images if source == "t2i" else self.videos
        position = block * self.global_batch + micro * self.world_size + rank
        padding = position >= count
        index = permuted_index(epoch * count + position % count, count, self.seed, source)
        mode = "t2i" if source == "t2i" else VIDEO_MODES[(index + self.seed + epoch) % 3]
        # Averaged across all updates of one epoch: .5 E[image] + .5 E[video].
        coefficient = 0.0 if padding else self.steps_per_epoch * self.global_batch / (2 * count)
        return dict(source=source, index=index, epoch=epoch, mode=mode, padding=padding,
                    loss_scale=coefficient, micro=step * self.accumulation + micro)

    def contract(self):
        return dict(recipe="media_epochs_i2v_v1", images=self.images, videos=self.videos,
                    epochs=self.epochs, world_size=self.world_size, accumulation=self.accumulation,
                    global_batch=self.global_batch, steps_per_epoch=self.steps_per_epoch, steps=self.steps,
                    image_presentations=self.images * self.epochs, video_presentations=self.videos * self.epochs,
                    paired_reconstruction_presentations=self.images * self.epochs,
                    video_modes=list(VIDEO_MODES), per_video_mode_presentations=self.epochs // 3,
                    presentations_per_video_mode=self.videos * (self.epochs // 3),
                    padding_per_epoch=self.steps_per_epoch * self.global_batch - self.images - self.videos,
                    padding="zero_loss_not_counted_as_exposure", seed=self.seed,
                    sampler="shuffled_source_homogeneous_updates; bijective_media_permutation_each_epoch",
                    objective="0.5*mean_t2i + 0.5*mean_video_modes + reconstruction_weight*mean_paired_reconstruction",
                    condition_dropout="drop_both_text_and_vision; retain_VAE_anchor_in_anchor_mode",
                    anchor="independently_encode_first_frame_with_same_VAE_posterior; replace_history_unit0; supervise_units1_to6",
                    reference="first_frame_to_VLM_only; supervise_units0_to6",
                    history="teacher_forced_past_units; generated_past_at_inference; exposure_bias_not_solved")


def plan_from_release(root, *, world_size, accumulation, epochs=3, seed=20260911,
                      min_videos=1, require_expansion=False, verify=True):
    root = Path(root)
    summary = verify_release(root) if verify else json.loads((root / "stage1_summary.json").read_text())
    if require_expansion:
        report = root / "expansion_report.json"
        if not report.is_file() or json.loads(report.read_text()).get("status") != "complete":
            raise ValueError(f"Expanded video release is not complete: {report}. Resume data preparation first.")
        expansion = json.loads(report.read_text())
        if (expansion.get("total_train_videos") != summary["counts"]["train.t2v"]
                or summary.get("expansion", {}).get("status") != "complete"
                or summary["expansion"].get("total_train_videos") != expansion["total_train_videos"]):
            raise ValueError("Expansion report and verified release disagree on training videos")
    counts = {task: len(JsonlRecords(root / "train" / f"{task}.jsonl")) for task in ("t2i", "t2v")}
    for task, count in counts.items():
        if summary["counts"][f"train.{task}"] != count:
            raise ValueError(f"Manifest row count differs from release summary: {task}")
    if counts["t2v"] < min_videos:
        raise ValueError(f"Only {counts['t2v']} training videos, require {min_videos}; do not silently use the old subset")
    return MediaEpochPlan(counts["t2i"], counts["t2v"], world_size, accumulation, epochs, seed)


def first_frame_image(video):
    if video.ndim != 4 or video.shape[0] != 3 or video.shape[1] < 1:
        raise ValueError("Expected RGB video [3,T,H,W]")
    rgb = ((video[:, 0].float().permute(1, 2, 0) + 1) * 127.5).round().clamp(0, 255).byte().cpu().numpy()
    return Image.fromarray(np.ascontiguousarray(rgb))


def condition_video(sample, mode):
    if sample["task"] != "t2v" or mode not in VIDEO_MODES:
        raise ValueError("Video mode requires a T2V source sample")
    sample = dict(sample, mode=mode)
    sample["image"] = None if mode == "t2v" else first_frame_image(sample["video"])
    return sample


def sample_unit(mode, units, device, generator):
    low = 1 if mode == "i2v_anchor" else 0
    if units <= low:
        raise ValueError("Anchored video needs at least one future unit")
    return int(torch.randint(low, units, (), device=device, generator=generator))


class MediaEpochDataset(Dataset):
    def __init__(self, root, *, plan, rank, start_step=0, short_side=320, long_side=512):
        self.root, self.sources = Path(root), load_sources(Path(root))
        self.records = {task: JsonlRecords(self.root / "train" / f"{task}.jsonl") for task in ("t2i", "t2v")}
        if (len(self.records["t2i"]), len(self.records["t2v"])) != (plan.images, plan.videos):
            raise ValueError("Dataset changed since epoch planning")
        self.plan, self.rank, self.start_step = plan, rank, start_step
        self.short_side, self.long_side = short_side, long_side

    def __len__(self):
        return (self.plan.steps - self.start_step) * self.plan.accumulation

    def __getitem__(self, index):
        step, micro = divmod(index, self.plan.accumulation)
        slot = self.plan.slot(step + self.start_step, micro, self.rank)
        sample = prepare_sample(self.records[slot["source"]][slot["index"]], self.sources,
                                self.short_side, self.long_side)
        if sample["task"] != slot["source"]:
            raise ValueError("Source manifest/task mismatch")
        if sample["task"] == "t2v":
            sample = condition_video(sample, slot["mode"])
        sample.update(slot)
        return sample


def distributed_loss_means(losses, ctx):
    """Missing modes are null, not zero; all ranks execute identical collectives."""
    rows = []
    for task in METRIC_TASKS:
        values = losses.get(task, [])
        total = sum((v.double() for v in values), torch.zeros((), device=ctx.device, dtype=torch.float64))
        rows.append(torch.stack((total, total.new_tensor(len(values)))))
    packed = torch.stack(rows)
    if ctx.is_distributed:
        dist.all_reduce(packed)
    rows = packed.cpu().tolist()
    return ({task: total / count if count else None for task, (total, count) in zip(METRIC_TASKS, rows)},
            {task: int(count) for task, (_, count) in zip(METRIC_TASKS, rows)})
