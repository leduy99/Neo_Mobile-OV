#!/usr/bin/env python
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.bridge import (
    DreamLiteCondition,
    MobileOVDreamLiteImageBridge,
    MobileOVNeodragonTextBridge,
)
from new_mobile_ov.config import load_config
from new_mobile_ov.training.distributed import (
    barrier,
    cleanup_distributed,
    rank0_print,
    scalar_mean,
    setup_distributed,
)
from new_mobile_ov.training.dreamlite_distillation import (
    DreamLiteClosedLoopResult,
    DreamLiteFrozenController,
    DreamLiteFrozenQwenTeacher,
    DreamLiteFunctionalResult,
    DreamLiteResolutionBucket,
    dreamlite_content_aware_representation_losses,
    dreamlite_direct_representation_losses,
    dreamlite_representation_losses,
    parse_dreamlite_resolution_buckets,
)
from new_mobile_ov.training.neodragon_objectives import linear_ramp
from new_mobile_ov.training.prompt_curriculum import (
    CaptionManifestDataset,
    MixedPromptDataset,
    clean_text,
    prompt_example_collate,
)


class CompositionalPromptDataset(Dataset):
    """Large deterministic prompt curriculum without benchmark prompt leakage."""

    COLORS = (
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "white",
        "black",
        "silver",
        "turquoise",
        "magenta",
    )
    OBJECTS = (
        ("astronaut", "astronauts"),
        ("red panda", "red pandas"),
        ("elephant", "elephants"),
        ("rhinoceros", "rhinoceroses"),
        ("golden retriever", "golden retrievers"),
        ("tabby cat", "tabby cats"),
        ("ceramic teapot", "ceramic teapots"),
        ("kingfisher", "kingfishers"),
        ("robot", "robots"),
        ("bicycle", "bicycles"),
        ("sailboat", "sailboats"),
        ("fox", "foxes"),
        ("glass sculpture", "glass sculptures"),
        ("vintage camera", "vintage cameras"),
        ("origami crane", "origami cranes"),
        ("steam locomotive", "steam locomotives"),
        ("grand piano", "grand pianos"),
        ("fire engine", "fire engines"),
        ("wooden chair", "wooden chairs"),
        ("glass bottle", "glass bottles"),
        ("sunflower", "sunflowers"),
        ("lighthouse", "lighthouses"),
    )
    ACTIONS = (
        "standing beside a quiet lake",
        "walking through fresh snow",
        "floating above a futuristic city",
        "resting on a wooden table",
        "crossing a narrow stone bridge",
        "surrounded by wildflowers",
        "under dramatic studio lighting",
        "reflected in a rain-covered window",
        "running across a sandy beach",
        "cooking beside a kitchen counter",
        "reading near a tall bookshelf",
        "waiting on a crowded train platform",
    )
    SCENES = (
        "a bright home kitchen",
        "a quiet library reading room",
        "an underwater aquarium tunnel",
        "a busy airport terminal",
        "a crowded train station platform",
        "a hospital corridor",
        "a science laboratory",
        "a grocery store aisle",
        "a children's classroom",
        "an amusement park at dusk",
        "a snowy mountain village",
        "a tropical coral reef",
        "a city street after rain",
        "a spacious art museum",
        "a small neighborhood cafe",
        "a professional football stadium",
        "a greenhouse full of plants",
        "a rocky desert canyon",
        "a modern office meeting room",
        "a traditional wooden workshop",
    )
    STYLES = (
        "cinematic photograph",
        "detailed product photograph",
        "documentary photograph",
        "watercolor illustration",
        "35mm film photograph",
        "minimalist poster",
    )
    RELATIONS = (
        "to the left of",
        "to the right of",
        "behind",
        "in front of",
        "above",
        "below",
    )
    TEXTS = ("NORTH", "OPEN", "MOON", "CAFE", "2049", "DREAM")
    COUNTS = (("one", 1), ("two", 2), ("three", 3), ("four", 4), ("five", 5))

    def __init__(self, length: int, *, seed: int) -> None:
        self.length = max(int(length), 1)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> str:
        rng = random.Random(self.seed + 1000003 * int(index))
        kind = rng.randrange(8)
        color = rng.choice(self.COLORS)
        singular, plural = rng.choice(self.OBJECTS)
        action = rng.choice(self.ACTIONS)
        style = rng.choice(self.STYLES)
        if kind == 0:
            count_word, count = rng.choice(self.COUNTS)
            noun = singular if count == 1 else plural
            return f"A {style} of {count_word} {color} {noun} {action}."
        if kind == 1:
            other_color = rng.choice([value for value in self.COLORS if value != color])
            other, _ = rng.choice(
                [value for value in self.OBJECTS if value[0] != singular]
            )
            relation = rng.choice(self.RELATIONS)
            return f"A {color} {singular} {relation} a {other_color} {other}, {style}."
        if kind == 2:
            return (
                f"A {style} of a {color} {singular} beside a sign clearly reading "
                f'"{rng.choice(self.TEXTS)}".'
            )
        if kind == 3:
            material = rng.choice(("glass", "wooden", "metal", "porcelain", "paper"))
            return f"A highly detailed {material} {singular} {action}, centered in a {style}."
        if kind == 4:
            return (
                f"Wide composition: a small {color} {singular} {action}; keep the full subject "
                f"visible with generous background space, {style}."
            )
        scene = rng.choice(self.SCENES)
        if kind == 5:
            return f"A {color} {singular} inside {scene}, with the complete scene clearly visible, {style}."
        if kind == 6:
            return f"A wide establishing view of {scene}, realistic layout and recognizable background details."
        other_color = rng.choice([value for value in self.COLORS if value != color])
        other, _ = rng.choice([value for value in self.OBJECTS if value[0] != singular])
        return (
            f"Inside {scene}, a {color} {singular} is {rng.choice(self.RELATIONS)} "
            f"a {other_color} {other}; both objects are fully visible."
        )


class EditDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        *,
        image_column: str,
        instruction_column: str,
        max_samples: int,
    ) -> None:
        path = Path(path)
        if path.suffix.lower() == ".jsonl":
            frame = pd.read_json(path, lines=True)
        else:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            frame = pd.read_csv(path, sep=sep, low_memory=False)
        image_column = (
            image_column
            if image_column in frame.columns
            else next(
                (
                    name
                    for name in ("source_image", "image_path", "image")
                    if name in frame.columns
                ),
                "",
            )
        )
        instruction_column = (
            instruction_column
            if instruction_column in frame.columns
            else next(
                (
                    name
                    for name in ("instruction", "edit_instruction", "prompt", "text")
                    if name in frame.columns
                ),
                "",
            )
        )
        if not image_column or not instruction_column:
            raise ValueError(
                f"{path} needs source image and edit instruction columns; got {list(frame.columns)}"
            )
        self.items = []
        for _, row in frame.iterrows():
            image_path = Path(clean_text(row.get(image_column))).expanduser()
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            instruction = clean_text(row.get(instruction_column))
            if image_path.is_file() and instruction:
                self.items.append((str(image_path), instruction))
        if max_samples > 0:
            self.items = self.items[:max_samples]
        if not self.items:
            raise ValueError(f"No valid edit samples found in {path}.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.items[index]


def edit_collate(items: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    paths, prompts = zip(*items)
    return list(paths), list(prompts)


def dtype_from_name(value: str) -> torch.dtype:
    value = value.lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype={value}")


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def split_paths(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(";") if item.strip()]


def select_condition(
    condition: DreamLiteCondition,
    indices: list[int],
) -> DreamLiteCondition:
    index = torch.as_tensor(
        indices,
        device=condition.prompt_embeds.device,
        dtype=torch.long,
    )
    return DreamLiteCondition(
        condition.prompt_embeds.index_select(0, index),
        condition.attention_mask.index_select(0, index),
    )


def load_grounding_images(
    image_paths: list[str],
    source_names: list[str],
    *,
    allowed_sources: set[str],
    max_images: int,
) -> tuple[list[Image.Image], list[int]]:
    images: list[Image.Image] = []
    indices: list[int] = []
    for index, (path_text, source) in enumerate(zip(image_paths, source_names)):
        if allowed_sources and source not in allowed_sources:
            continue
        path = Path(path_text).expanduser() if path_text else None
        if path is None or not path.is_file():
            continue
        try:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        except (OSError, ValueError):
            continue
        indices.append(index)
        if len(images) >= max_images:
            break
    return images, indices


def unwrap(model):
    return getattr(model, "module", model)


def lr_scale(
    step: int, *, total_steps: int, warmup_steps: int, final_scale: float
) -> float:
    if step <= warmup_steps:
        return max(step / float(max(warmup_steps, 1)), 1e-3)
    progress = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
    progress = min(max(progress, 0.0), 1.0)
    return final_scale + (1.0 - final_scale) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def representation_total(losses: dict[str, torch.Tensor], args) -> torch.Tensor:
    total = (
        args.token_mse_weight * losses["token_normalized_mse"]
        + args.token_cos_weight * losses["token_cosine"]
        + args.token_norm_weight * losses["token_norm"]
        + args.pooled_mse_weight * losses["pooled_normalized_mse"]
        + args.pooled_cos_weight * losses["pooled_cosine"]
        + args.geometry_weight * losses["geometry"]
        + args.variance_weight * losses["variance"]
    )
    if "token_mean" in losses:
        total = total + args.token_mean_weight * losses["token_mean"]
    if "token_std" in losses:
        total = total + args.token_std_weight * losses["token_std"]
    return total


def projected_representation_total(
    losses: dict[str, torch.Tensor],
    args,
) -> torch.Tensor:
    return (
        0.50 * losses["token_normalized_mse"]
        + losses["token_cosine"]
        + args.projected_pooled_cos_weight * losses["pooled_cosine"]
        + args.projected_moment_weight * losses["token_mean"]
        + args.projected_moment_weight * losses["token_std"]
    )


def content_aware_representation_total(
    losses: dict[str, torch.Tensor],
    args,
    *,
    projected: bool,
) -> torch.Tensor:
    scale = args.projected_content_scale if projected else 1.0
    return scale * (
        args.content_token_mse_weight * losses["content_token_normalized_mse"]
        + args.content_token_cos_weight * losses["content_token_cosine"]
        + args.wrapper_token_weight
        * (
            0.5 * losses["wrapper_token_normalized_mse"]
            + losses["wrapper_token_cosine"]
        )
        + args.content_pooled_cos_weight * losses["content_pooled_cosine"]
        + args.content_pooled_mse_weight * losses["content_pooled_normalized_mse"]
        + args.content_token_mean_weight * losses["content_token_mean"]
        + args.content_token_std_weight * losses["content_token_std"]
        + args.semantic_contrastive_weight * losses["semantic_contrastive"]
    )


CONTENT_AWARE_LEGACY_VERSIONS = frozenset(
    {"v5", "v6", "v7", "v8", "v9", "v10", "shared_v1"}
)


def resolve_representation_objective(args: argparse.Namespace) -> str:
    """Keep historical launchers working while making new objectives explicit."""

    if args.representation_objective != "auto":
        return args.representation_objective
    if args.training_version.lower() in CONTENT_AWARE_LEGACY_VERSIONS:
        return "content_aware"
    if args.representation_mode == "direct":
        return "direct"
    return "interpolated"


def validate_representation_objective(args: argparse.Namespace) -> str:
    objective = resolve_representation_objective(args)
    if args.global_contrastive and objective != "content_aware":
        raise ValueError(
            "--global-contrastive requires --representation-objective content_aware."
        )
    return objective


def initialize_bridge_from_checkpoint(bridge, checkpoint_path: str | Path) -> dict[str, object]:
    """Load bridge weights only, deliberately leaving optimizer and step fresh."""

    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Missing bridge initialization checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("bridge", payload)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Bridge initialization checkpoint has no state: {path}")
    bridge.load_trainable_state_dict(state)
    return {
        "checkpoint": str(path),
        "source_step": int(payload.get("step", -1)),
        "source_training_version": payload.get("config", {}).get("training_version"),
        "source_architecture": payload.get("architecture"),
        "optimizer_reset": True,
        "step_reset": True,
    }


def choose_resolution_bucket(
    buckets: list[DreamLiteResolutionBucket],
    *,
    seed: int,
    step: int,
) -> DreamLiteResolutionBucket:
    # The seed does not contain rank so every DDP process runs the same graph.
    rng = random.Random(seed + 104729 * step)
    return rng.choices(buckets, weights=[bucket.weight for bucket in buckets], k=1)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill Qwen3-VL into the Mobile-OV DreamLite bridge."
    )
    parser.add_argument("--config", default="configs/mobile_ov_dreamlite.yaml")
    parser.add_argument(
        "--generation-prompts",
        default="",
        help="Legacy single prompt manifest. Ignored when --generation-prompt-manifests is set.",
    )
    parser.add_argument(
        "--generation-prompt-manifests",
        default="",
        help="Semicolon-separated caption manifests for weighted prompt mixing.",
    )
    parser.add_argument(
        "--generation-source-weights",
        default="",
        help="Comma-separated source-level sampling weights.",
    )
    parser.add_argument(
        "--generation-source-names",
        default="",
        help="Comma-separated stable source labels used by logs and grounding filters.",
    )
    parser.add_argument(
        "--generation-image-columns",
        default="image_path,media_path,video_path",
        help="Candidate raw-image columns, in priority order.",
    )
    parser.add_argument(
        "--image-path-roots",
        default="",
        help="Comma-separated data roots used to resolve relocated manifest image paths.",
    )
    parser.add_argument("--edit-manifest", default=None)
    parser.add_argument("--edit-image-column", default="source_image")
    parser.add_argument("--edit-instruction-column", default="instruction")
    parser.add_argument("--edit-probability", type=float, default=0.25)
    parser.add_argument(
        "--shared-video-bridge-ckpt",
        default="",
        help=(
            "Exp1 NeoDragon bridge checkpoint. When set, run SmolVLM2 once on "
            "the Exp1 token contract and train the DreamLite image head from those "
            "shared features. Generation-only; image editing remains a separate "
            "multimodal conditioning path."
        ),
    )
    parser.add_argument(
        "--shared-video-config",
        default="configs/mobile_ov_neodragon.yaml",
        help=(
            "NeoDragon bridge config that defines the canonical shared SmolVLM2 "
            "token contract. Must match --shared-video-bridge-ckpt."
        ),
    )
    parser.add_argument(
        "--shared-prompt-suffix",
        default="",
        help=(
            "Suffix appended once before the shared Exp1 SmolVLM2 forward. "
            "The DreamLite Qwen teacher continues to receive the original prompt."
        ),
    )
    parser.add_argument("--output-dir", default="output/dreamlite_image_bridge")
    parser.add_argument("--target-step", type=int, default=100000)
    parser.add_argument("--resume", default="auto")
    parser.add_argument(
        "--init-bridge-checkpoint",
        default="",
        help=(
            "Initialize only bridge parameters from this checkpoint while resetting "
            "optimizer state and training step. Mutually exclusive with --resume."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--lr-warmup-steps", type=int, default=1000)
    parser.add_argument("--lr-final-scale", type=float, default=0.1)
    parser.add_argument(
        "--lr-decay-end-step",
        type=int,
        default=-1,
        help=(
            "Step where cosine decay reaches --lr-final-scale. Negative uses "
            "--target-step; later steps keep the final learning rate."
        ),
    )
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument(
        "--resolution-buckets",
        default="",
        help=(
            "Comma-separated actual@logical resolution buckets: "
            "WIDTHxHEIGHT@TIME_WIDTHxTIME_HEIGHT:WEIGHT. "
            "An empty value preserves the legacy --width/--height behavior."
        ),
    )
    parser.add_argument(
        "--caption-variant-columns", default="caption_short,caption_medium,caption_long"
    )
    parser.add_argument("--caption-variant-weights", default="1,1,1")
    parser.add_argument("--caption-fallback-column", default="caption")
    parser.add_argument("--semantic-prompt-probability", type=float, default=0.0)
    parser.add_argument("--semantic-prompt-seed", type=int, default=73129)
    parser.add_argument("--token-mse-weight", type=float, default=0.50)
    parser.add_argument("--token-cos-weight", type=float, default=1.00)
    parser.add_argument("--token-norm-weight", type=float, default=0.05)
    parser.add_argument("--pooled-mse-weight", type=float, default=0.50)
    parser.add_argument("--pooled-cos-weight", type=float, default=1.00)
    parser.add_argument("--geometry-weight", type=float, default=0.25)
    parser.add_argument("--variance-weight", type=float, default=0.05)
    parser.add_argument(
        "--representation-mode",
        choices=("interpolated", "direct"),
        default="interpolated",
    )
    parser.add_argument(
        "--representation-objective",
        choices=("auto", "interpolated", "direct", "content_aware"),
        default="auto",
        help=(
            "Representation-loss family. New experiments should set this explicitly; "
            "auto preserves historical version-based behavior."
        ),
    )
    parser.add_argument("--token-mean-weight", type=float, default=0.0)
    parser.add_argument("--token-std-weight", type=float, default=0.0)
    parser.add_argument("--projected-weight", type=float, default=0.0)
    parser.add_argument("--projected-pooled-cos-weight", type=float, default=0.5)
    parser.add_argument("--projected-moment-weight", type=float, default=0.25)
    parser.add_argument("--content-prefix-tokens", type=int, default=3)
    parser.add_argument("--content-suffix-tokens", type=int, default=5)
    parser.add_argument("--content-token-mse-weight", type=float, default=0.5)
    parser.add_argument("--content-token-cos-weight", type=float, default=1.0)
    parser.add_argument("--wrapper-token-weight", type=float, default=0.25)
    parser.add_argument("--content-pooled-cos-weight", type=float, default=0.1)
    parser.add_argument("--content-pooled-mse-weight", type=float, default=0.0)
    parser.add_argument("--content-token-mean-weight", type=float, default=0.0)
    parser.add_argument("--content-token-std-weight", type=float, default=0.0)
    parser.add_argument("--semantic-contrastive-weight", type=float, default=0.2)
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument(
        "--global-contrastive",
        action="store_true",
        help="Use the full distributed batch as contrastive candidates.",
    )
    parser.add_argument("--projected-content-scale", type=float, default=1.0)
    parser.add_argument("--representation-final-scale", type=float, default=0.25)
    parser.add_argument("--functional-weight", type=float, default=5.0)
    parser.add_argument("--functional-cos-weight", type=float, default=0.5)
    parser.add_argument("--functional-start-step", type=int, default=10001)
    parser.add_argument("--functional-ramp-steps", type=int, default=5000)
    parser.add_argument("--functional-batch-size", type=int, default=1)
    parser.add_argument("--functional-call-weights", default="1,1,1,1")
    parser.add_argument("--grounded-functional-probability", type=float, default=0.0)
    parser.add_argument(
        "--grounded-batch-probability",
        type=float,
        default=0.0,
        help=(
            "Probability of drawing an entire functional batch from a dedicated "
            "image-verified loader. This guarantees grounded supervision instead "
            "of relying on incidental image paths in the main prompt batch."
        ),
    )
    parser.add_argument("--grounded-functional-weight", type=float, default=1.0)
    parser.add_argument("--grounded-functional-start-step", type=int, default=-1)
    parser.add_argument(
        "--grounded-source-names",
        default="",
        help="Comma-separated prompt sources whose raw images may define functional states.",
    )
    parser.add_argument("--transition-weight", type=float, default=0.0)
    parser.add_argument("--transition-cos-weight", type=float, default=0.0)
    parser.add_argument("--student-state-probability", type=float, default=0.0)
    parser.add_argument("--student-state-start-step", type=int, default=30001)
    parser.add_argument("--student-state-ramp-steps", type=int, default=20000)
    parser.add_argument("--closed-loop-weight", type=float, default=2.0)
    parser.add_argument("--closed-loop-prediction-weight", type=float, default=0.5)
    parser.add_argument("--closed-loop-cos-weight", type=float, default=0.1)
    parser.add_argument("--closed-loop-transition-weight", type=float, default=1.0)
    parser.add_argument("--closed-loop-terminal-weight", type=float, default=1.0)
    parser.add_argument("--closed-loop-start-step", type=int, default=25001)
    parser.add_argument("--closed-loop-ramp-steps", type=int, default=5000)
    parser.add_argument("--closed-loop-every", type=int, default=4)
    parser.add_argument("--closed-loop-batch-size", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-latest-every", type=int, default=5000)
    parser.add_argument("--save-archive-every", type=int, default=10000)
    parser.add_argument("--training-version", default="legacy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.init_bridge_checkpoint and args.resume != "none":
        raise ValueError(
            "--init-bridge-checkpoint requires --resume none so optimizer and step "
            "cannot be restored accidentally."
        )
    args.resolved_representation_objective = validate_representation_objective(args)
    context = setup_distributed()
    torch.manual_seed(args.seed + context.rank)
    random.seed(args.seed + context.rank)
    config = load_config(args.config)
    inference_dtype = dtype_from_name(args.dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_video_bridge_ckpt = str(args.shared_video_bridge_ckpt).strip()
    if shared_video_bridge_ckpt and args.edit_manifest:
        raise ValueError(
            "Shared SmolVLM2 image training is generation-only. Do not set --edit-manifest."
        )
    if args.resolution_buckets:
        resolution_buckets = parse_dreamlite_resolution_buckets(args.resolution_buckets)
    else:
        resolution_buckets = [
            DreamLiteResolutionBucket(
                width=args.width,
                height=args.height,
                time_id_width=args.width,
                time_id_height=args.height,
                weight=1.0,
            )
        ]

    columns = split_csv(args.caption_variant_columns)
    weights = [float(value) for value in split_csv(args.caption_variant_weights)]
    if len(columns) != len(weights):
        raise ValueError(
            "caption variant columns and weights must have the same length"
        )
    manifest_paths = split_paths(args.generation_prompt_manifests)
    if not manifest_paths and args.generation_prompts:
        manifest_paths = [args.generation_prompts]
    if not manifest_paths:
        raise ValueError(
            "Set --generation-prompts or --generation-prompt-manifests."
        )
    source_weights = [
        float(value) for value in split_csv(args.generation_source_weights)
    ]
    if not source_weights:
        source_weights = [1.0] * len(manifest_paths)
    source_names = split_csv(args.generation_source_names)
    if not source_names:
        source_names = [Path(path).stem for path in manifest_paths]
    if len(source_weights) != len(manifest_paths):
        raise ValueError(
            "generation source weights must match generation prompt manifests"
        )
    if len(source_names) != len(manifest_paths):
        raise ValueError(
            "generation source names must match generation prompt manifests"
        )
    if len(set(source_names)) != len(source_names):
        raise ValueError("generation source names must be unique")
    grounded_source_names = set(split_csv(args.grounded_source_names))
    unknown_grounded_sources = grounded_source_names - set(source_names)
    if unknown_grounded_sources:
        raise ValueError(
            "grounded source names are not generation sources: "
            + ", ".join(sorted(unknown_grounded_sources))
        )
    image_columns = split_csv(args.generation_image_columns)
    image_path_roots = split_paths(args.image_path_roots)
    generation_sources = [
        CaptionManifestDataset(
            path,
            source_name=name,
            variant_columns=columns,
            variant_weights=weights,
            fallback_column=args.caption_fallback_column,
            image_columns=image_columns,
            max_samples=args.max_samples,
            require_existing_image=name in grounded_source_names,
            image_path_roots=image_path_roots,
        )
        for path, name in zip(manifest_paths, source_names)
    ]
    generation_data = MixedPromptDataset(
        generation_sources,
        source_weights,
        seed=args.seed + 43,
    )
    if not 0.0 <= args.semantic_prompt_probability <= 1.0:
        raise ValueError("semantic prompt probability must be in [0, 1]")
    if not 0.0 <= args.student_state_probability <= 1.0:
        raise ValueError("student state probability must be in [0, 1]")
    if not 0.0 <= args.grounded_functional_probability <= 1.0:
        raise ValueError("grounded functional probability must be in [0, 1]")
    if not 0.0 <= args.grounded_batch_probability <= 1.0:
        raise ValueError("grounded batch probability must be in [0, 1]")
    if (
        args.grounded_functional_probability > 0
        and args.grounded_batch_probability > 0
    ):
        raise ValueError(
            "Use either legacy --grounded-functional-probability or the dedicated "
            "--grounded-batch-probability, not both."
        )
    if args.grounded_functional_weight < 0:
        raise ValueError("grounded functional weight must be non-negative")
    lr_decay_end_step = (
        args.target_step if args.lr_decay_end_step < 0 else args.lr_decay_end_step
    )
    if lr_decay_end_step < args.lr_warmup_steps:
        raise ValueError("lr decay end step must not precede lr warmup")
    grounded_start_step = (
        args.functional_start_step
        if args.grounded_functional_start_step < 0
        else args.grounded_functional_start_step
    )
    eligible_grounded_sources = [
        source
        for source in generation_sources
        if (
            not grounded_source_names or source.source_name in grounded_source_names
        )
        and source.image_candidate_rows > 0
    ]
    if args.grounded_functional_probability > 0 and not eligible_grounded_sources:
        raise ValueError(
            "Grounded functional loss is enabled, but its selected manifests "
            "contain no image paths with a supported extension."
        )
    if args.grounded_batch_probability > 0 and not grounded_source_names:
        raise ValueError(
            "Dedicated grounded batches require --grounded-source-names so their "
            "image-caption distribution is explicit."
        )

    grounded_data = None
    if args.grounded_batch_probability > 0:
        source_weight_by_name = dict(zip(source_names, source_weights))
        grounded_sources = [
            source
            for source in generation_sources
            if source.source_name in grounded_source_names
        ]
        grounded_weights = [
            source_weight_by_name[source.source_name] for source in grounded_sources
        ]
        # A verified grounded-only source may intentionally have zero main
        # sampling weight so it cannot alter the caption distribution. In that
        # case draw uniformly among the selected grounded sources instead.
        if not any(grounded_weights):
            grounded_weights = [1.0] * len(grounded_sources)
        grounded_data = MixedPromptDataset(
            grounded_sources,
            grounded_weights,
            seed=args.seed + 67,
        )
    generation_sampler = (
        DistributedSampler(
            generation_data,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=args.seed,
        )
        if context.is_distributed
        else None
    )
    generation_loader = DataLoader(
        generation_data,
        batch_size=args.batch_size,
        sampler=generation_sampler,
        shuffle=generation_sampler is None,
        num_workers=0,
        drop_last=True,
        collate_fn=prompt_example_collate,
    )
    grounded_sampler = (
        DistributedSampler(
            grounded_data,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=args.seed + 11,
        )
        if grounded_data is not None and context.is_distributed
        else None
    )
    grounded_loader = (
        DataLoader(
            grounded_data,
            batch_size=args.batch_size,
            sampler=grounded_sampler,
            shuffle=grounded_sampler is None,
            num_workers=0,
            drop_last=True,
            collate_fn=prompt_example_collate,
        )
        if grounded_data is not None
        else None
    )
    semantic_data = CompositionalPromptDataset(
        len(generation_data),
        seed=args.semantic_prompt_seed,
    )
    semantic_sampler = (
        DistributedSampler(
            semantic_data,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=args.seed + 31,
        )
        if context.is_distributed
        else None
    )
    semantic_loader = DataLoader(
        semantic_data,
        batch_size=args.batch_size,
        sampler=semantic_sampler,
        shuffle=semantic_sampler is None,
        num_workers=0,
        drop_last=True,
    )
    edit_loader = None
    edit_sampler = None
    if args.edit_manifest:
        edit_data = EditDataset(
            args.edit_manifest,
            image_column=args.edit_image_column,
            instruction_column=args.edit_instruction_column,
            max_samples=args.max_samples,
        )
        edit_sampler = (
            DistributedSampler(
                edit_data,
                num_replicas=context.world_size,
                rank=context.rank,
                shuffle=True,
                seed=args.seed + 17,
            )
            if context.is_distributed
            else None
        )
        edit_loader = DataLoader(
            edit_data,
            batch_size=args.batch_size,
            sampler=edit_sampler,
            shuffle=edit_sampler is None,
            num_workers=0,
            drop_last=True,
            collate_fn=edit_collate,
        )

    rank0_print(context, "Loading Qwen3-VL teacher and DreamLite controller...")
    shared_video_bridge = None
    if shared_video_bridge_ckpt:
        checkpoint_path = Path(shared_video_bridge_ckpt)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing shared video bridge checkpoint: {checkpoint_path}")
        shared_video_config_path = Path(args.shared_video_config)
        if not shared_video_config_path.is_file():
            raise FileNotFoundError(
                f"Missing shared video bridge config: {shared_video_config_path}"
            )
        rank0_print(
            context,
            "Loading frozen Exp1 video bridge as the shared SmolVLM2 encoder...",
        )
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload.get("bridge", payload)
        shared_video_config = load_config(shared_video_config_path)
        shared_video_bridge = MobileOVNeodragonTextBridge(
            shared_video_config.bridge,
            device=context.device,
            dtype=inference_dtype,
        ).eval()
        missing, unexpected = shared_video_bridge.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Shared video bridge checkpoint mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        shared_video_bridge.requires_grad_(False)
        source_dim = int(
            shared_video_bridge.token_bridge.smolvlm2_model.config.text_config.hidden_size
        )
        bridge = MobileOVDreamLiteImageBridge(
            config.bridge,
            config.dreamlite_bridge,
            device=context.device,
            dtype=inference_dtype,
            load_feature_provider=False,
            external_feature_dim=source_dim,
        )
    else:
        rank0_print(context, "Loading frozen local SmolVLM2 image encoder...")
        bridge = MobileOVDreamLiteImageBridge(
            config.bridge,
            config.dreamlite_bridge,
            device=context.device,
            dtype=inference_dtype,
        )
    bridge.promote_trainable_parameters_to_fp32()
    teacher = DreamLiteFrozenQwenTeacher(
        config.dreamlite,
        device=context.device,
        dtype=inference_dtype,
    )
    controller = DreamLiteFrozenController(
        config.dreamlite,
        device=context.device,
        dtype=inference_dtype,
    )
    functional_call_weights = [
        float(value) for value in split_csv(args.functional_call_weights)
    ]
    if (
        len(functional_call_weights) != controller.num_steps
        or any(value < 0 for value in functional_call_weights)
        or sum(functional_call_weights) <= 0
    ):
        raise ValueError(
            f"functional call weights must contain {controller.num_steps} "
            "non-negative values with a positive sum"
        )
    trainable = [
        parameter for parameter in bridge.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    current_step = 0
    initialization: dict[str, object] | None = None
    if args.init_bridge_checkpoint:
        initialization = initialize_bridge_from_checkpoint(
            bridge, args.init_bridge_checkpoint
        )
        rank0_print(
            context,
            "Initialized DreamLite bridge weights with fresh optimizer/step: "
            + json.dumps(initialization),
        )
    resume_path = (
        output_dir / "dreamlite_image_bridge_latest.pt"
        if args.resume == "auto"
        else Path(args.resume)
    )
    if args.resume not in {"none", "auto"} and not resume_path.is_file():
        raise FileNotFoundError(f"Missing explicit resume checkpoint: {resume_path}")
    if args.resume != "none" and resume_path.is_file():
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        bridge.load_trainable_state_dict(payload["bridge"])
        optimizer.load_state_dict(payload["optimizer"])
        current_step = int(payload["step"])
        initialization = payload.get("initialization")
        rank0_print(
            context,
            f"Resumed DreamLite bridge from {resume_path} at step={current_step}",
        )
    if context.is_distributed:
        ddp_devices = [context.local_rank] if context.device.type == "cuda" else None
        bridge = DDP(bridge, device_ids=ddp_devices, broadcast_buffers=False)

    trainable_count = sum(parameter.numel() for parameter in trainable)
    rank0_print(
        context,
        f"DreamLite bridge distillation: world_size={context.world_size} "
        f"batch_per_gpu={args.batch_size} global_batch={args.batch_size * context.world_size} "
        f"trainable_params={trainable_count:,} generation_rows={len(generation_data)} "
        f"semantic_rows={len(semantic_data)} semantic_probability={args.semantic_prompt_probability:g} "
        f"edit_rows={len(edit_loader.dataset) if edit_loader else 0} target_step={args.target_step} "
        f"shared_smolvlm2={bool(shared_video_bridge)} "
        f"representation_objective={args.resolved_representation_objective}",
    )
    rank0_print(
        context,
        f"LR schedule: base={args.lr:g} warmup={args.lr_warmup_steps} "
        f"decay_end={lr_decay_end_step} final={args.lr * args.lr_final_scale:g}",
    )
    rank0_print(
        context,
        "Prompt sources: " + json.dumps(generation_data.source_summary),
    )
    rank0_print(
        context,
        f"Grounded functional: probability={args.grounded_functional_probability:g} "
        f"dedicated_batch_probability={args.grounded_batch_probability:g} "
        f"weight={args.grounded_functional_weight:g} start_step={grounded_start_step} "
        f"sources={sorted(grounded_source_names) or ['all-image-sources']}",
    )
    if grounded_data is not None:
        rank0_print(
            context,
            "Dedicated grounded prompt sources: "
            + json.dumps(grounded_data.source_summary),
        )
    rank0_print(
        context,
        "Resolution buckets: "
        + ", ".join(
            f"{bucket.label}:{bucket.weight:g}" for bucket in resolution_buckets
        ),
    )
    generation_epoch = 0
    grounded_epoch = 0
    edit_epoch = 0
    generation_iter = iter(generation_loader)
    grounded_iter = iter(grounded_loader) if grounded_loader is not None else None
    semantic_epoch = 0
    semantic_iter = iter(semantic_loader)
    edit_iter = iter(edit_loader) if edit_loader else None
    history_path = output_dir / "history.jsonl"
    progress = tqdm(
        total=max(args.target_step - current_step, 0),
        desc="Train Mobile-OV DreamLite image bridge",
        disable=not context.is_main,
    )

    def next_generation():
        nonlocal generation_iter, generation_epoch
        try:
            return next(generation_iter)
        except StopIteration:
            generation_epoch += 1
            if generation_sampler is not None:
                generation_sampler.set_epoch(generation_epoch)
            generation_iter = iter(generation_loader)
            return next(generation_iter)

    def next_grounded():
        nonlocal grounded_iter, grounded_epoch
        if grounded_loader is None or grounded_iter is None:
            raise RuntimeError("Dedicated grounded loader is unavailable")
        try:
            return next(grounded_iter)
        except StopIteration:
            grounded_epoch += 1
            if grounded_sampler is not None:
                grounded_sampler.set_epoch(grounded_epoch)
            grounded_iter = iter(grounded_loader)
            return next(grounded_iter)

    def next_edit():
        nonlocal edit_iter, edit_epoch
        if edit_loader is None or edit_iter is None:
            raise RuntimeError("Edit loader is unavailable")
        try:
            return next(edit_iter)
        except StopIteration:
            edit_epoch += 1
            if edit_sampler is not None:
                edit_sampler.set_epoch(edit_epoch)
            edit_iter = iter(edit_loader)
            return next(edit_iter)

    def next_semantic():
        nonlocal semantic_iter, semantic_epoch
        try:
            return next(semantic_iter)
        except StopIteration:
            semantic_epoch += 1
            if semantic_sampler is not None:
                semantic_sampler.set_epoch(semantic_epoch)
            semantic_iter = iter(semantic_loader)
            return next(semantic_iter)

    try:
        while current_step < args.target_step:
            current_step += 1
            scale = lr_scale(
                current_step,
                total_steps=lr_decay_end_step,
                warmup_steps=args.lr_warmup_steps,
                final_scale=args.lr_final_scale,
            )
            for group in optimizer.param_groups:
                group["lr"] = args.lr * scale
            # Every DDP rank must execute the same objective branch. Samples
            # remain rank-sharded, but mode selection is deterministic globally.
            mode_rng = random.Random(args.seed + current_step)
            resolution = choose_resolution_bucket(
                resolution_buckets,
                seed=args.seed,
                step=current_step,
            )
            functional_rng = random.Random(args.seed + 130363 * current_step)
            functional_call_index = functional_rng.choices(
                range(controller.num_steps),
                weights=functional_call_weights,
                k=1,
            )[0]
            student_state_scale = linear_ramp(
                current_step,
                start_step=args.student_state_start_step,
                ramp_steps=args.student_state_ramp_steps,
            )
            student_state_probability = (
                args.student_state_probability * student_state_scale
            )
            functional_state_source = (
                "student"
                if functional_rng.random() < student_state_probability
                else "teacher"
            )
            use_dedicated_grounded_batch = (
                grounded_loader is not None
                and current_step
                >= max(grounded_start_step, args.functional_start_step)
                and functional_rng.random() < args.grounded_batch_probability
            )
            use_edit = (
                not use_dedicated_grounded_batch
                and edit_loader is not None
                and mode_rng.random() < args.edit_probability
            )
            if use_edit:
                image_paths, prompts = next_edit()
                images = []
                for path in image_paths:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB").copy())
                mode = "edit"
                prompt_source = "edit"
                generation_image_paths: list[str] = []
                generation_source_names: list[str] = []
                use_semantic_prompts = False
            else:
                use_semantic_prompts = (
                    not use_dedicated_grounded_batch
                    and mode_rng.random() < args.semantic_prompt_probability
                )
                if use_dedicated_grounded_batch:
                    (
                        prompts,
                        generation_image_paths,
                        generation_source_names,
                    ) = next_grounded()
                elif use_semantic_prompts:
                    prompts = list(next_semantic())
                    generation_image_paths = [""] * len(prompts)
                    generation_source_names = ["semantic"] * len(prompts)
                else:
                    (
                        prompts,
                        generation_image_paths,
                        generation_source_names,
                    ) = next_generation()
                images = None
                mode = "generate"
                prompt_source = "+".join(sorted(set(generation_source_names)))
            grounded_images: list[Image.Image] = []
            grounded_indices: list[int] = []
            attempt_grounded = (
                mode == "generate"
                and not use_semantic_prompts
                and current_step
                >= max(grounded_start_step, args.functional_start_step)
                and (
                    use_dedicated_grounded_batch
                    or (
                        args.grounded_functional_probability > 0
                        and functional_rng.random()
                        < args.grounded_functional_probability
                    )
                )
            )
            if attempt_grounded:
                grounded_images, grounded_indices = load_grounding_images(
                    generation_image_paths,
                    generation_source_names,
                    allowed_sources=grounded_source_names,
                    max_images=args.functional_batch_size,
                )
                if use_dedicated_grounded_batch:
                    expected = min(args.functional_batch_size, len(prompts))
                    if len(grounded_images) != expected:
                        raise RuntimeError(
                            "Dedicated grounded batch lost an image path: "
                            f"expected={expected}, loaded={len(grounded_images)}."
                        )
            optimizer.zero_grad(set_to_none=True)
            autocast_enabled = (
                context.device.type == "cuda" and inference_dtype != torch.float32
            )
            with torch.autocast(
                device_type=context.device.type,
                dtype=inference_dtype,
                enabled=autocast_enabled,
            ):
                if shared_video_bridge is not None:
                    if mode != "generate":
                        raise RuntimeError("Shared SmolVLM2 training only supports generation mode.")
                    shared_prompts = [
                        str(prompt) + args.shared_prompt_suffix for prompt in prompts
                    ]
                    _, shared_source_mask, shared_hidden_layers = (
                        shared_video_bridge.encode_smolvlm2_features(shared_prompts)
                    )
                    if shared_hidden_layers is None:
                        raise RuntimeError("Shared NeoDragon bridge did not return hidden layers.")
                    student = bridge(
                        prompts,
                        mode="generate",
                        shared_hidden_layers=shared_hidden_layers,
                        shared_source_mask=shared_source_mask,
                    )
                else:
                    student = bridge(prompts, mode=mode, images=images)
                teacher_condition = teacher.encode(prompts, mode=mode, images=images)
                use_content_alignment = (
                    args.resolved_representation_objective == "content_aware"
                    and mode == "generate"
                )
                use_direct_alignment = (
                    args.resolved_representation_objective == "direct"
                    and mode == "generate"
                )
                if use_content_alignment:
                    repr_losses = dreamlite_content_aware_representation_losses(
                        student,
                        teacher_condition,
                        prefix_tokens=args.content_prefix_tokens,
                        suffix_tokens=args.content_suffix_tokens,
                        contrastive_temperature=args.contrastive_temperature,
                        global_contrastive=args.global_contrastive,
                    )
                    projected_student = controller.project_condition(student)
                    projected_teacher = controller.project_condition(teacher_condition)
                    projected_losses = dreamlite_content_aware_representation_losses(
                        projected_student,
                        projected_teacher,
                        prefix_tokens=args.content_prefix_tokens,
                        suffix_tokens=args.content_suffix_tokens,
                        contrastive_temperature=args.contrastive_temperature,
                        global_contrastive=args.global_contrastive,
                    )
                    projected_value = content_aware_representation_total(
                        projected_losses,
                        args,
                        projected=True,
                    )
                    repr_value = (
                        content_aware_representation_total(
                            repr_losses,
                            args,
                            projected=False,
                        )
                        + args.projected_weight * projected_value
                    )
                elif use_direct_alignment:
                    repr_losses = dreamlite_direct_representation_losses(
                        student, teacher_condition
                    )
                    projected_student = controller.project_condition(student)
                    projected_teacher = controller.project_condition(teacher_condition)
                    projected_losses = dreamlite_direct_representation_losses(
                        projected_student,
                        projected_teacher,
                    )
                    projected_value = projected_representation_total(
                        projected_losses, args
                    )
                    repr_value = representation_total(repr_losses, args)
                    repr_value = repr_value + args.projected_weight * projected_value
                else:
                    repr_losses = dreamlite_representation_losses(
                        student, teacher_condition
                    )
                    projected_losses = None
                    projected_value = student.prompt_embeds.new_zeros(
                        (), dtype=torch.float32
                    )
                    repr_value = representation_total(repr_losses, args)
                    repr_value = repr_value + args.projected_weight * projected_value
                if (
                    args.closed_loop_weight <= 0
                    or current_step < args.closed_loop_start_step
                ):
                    repr_scale = 1.0
                else:
                    repr_scale = args.representation_final_scale
                functional = DreamLiteFunctionalResult(
                    relative_mse=repr_value.new_zeros(()),
                    cosine=repr_value.new_zeros(()),
                    transition_relative_mse=repr_value.new_zeros(()),
                    transition_cosine=repr_value.new_zeros(()),
                    call_index=-1,
                    state_source="none",
                )
                closed = DreamLiteClosedLoopResult(
                    prediction_relative_mse=repr_value.new_zeros(()),
                    prediction_cosine=repr_value.new_zeros(()),
                    transition_relative_mse=repr_value.new_zeros(()),
                    transition_cosine=repr_value.new_zeros(()),
                    terminal_relative_mse=repr_value.new_zeros(()),
                    calls=0,
                )
                functional_loss_multiplier = 1.0
                functional_scale = linear_ramp(
                    current_step,
                    start_step=args.functional_start_step,
                    ramp_steps=args.functional_ramp_steps,
                )
                closed_scale = linear_ramp(
                    current_step,
                    start_step=args.closed_loop_start_step,
                    ramp_steps=args.closed_loop_ramp_steps,
                )
                run_closed = (
                    args.closed_loop_weight > 0
                    and current_step >= args.closed_loop_start_step
                    and (current_step - args.closed_loop_start_step)
                    % args.closed_loop_every
                    == 0
                )
                if run_closed:
                    phase = "closed_loop"
                    closed = controller.closed_loop_loss(
                        student,
                        teacher_condition,
                        source_images=images,
                        height=resolution.height,
                        width=resolution.width,
                        batch_size=args.closed_loop_batch_size,
                    )
                elif current_step >= args.functional_start_step:
                    if grounded_images:
                        phase = "grounded_functional"
                        functional_loss_multiplier = args.grounded_functional_weight
                        functional = controller.grounded_functional_loss(
                            select_condition(student, grounded_indices),
                            select_condition(teacher_condition, grounded_indices),
                            target_images=grounded_images,
                            height=resolution.height,
                            width=resolution.width,
                            time_id_height=resolution.time_id_height,
                            time_id_width=resolution.time_id_width,
                            batch_size=len(grounded_images),
                            call_index=functional_call_index,
                        )
                    else:
                        phase = "functional"
                        functional = controller.functional_loss(
                            student,
                            teacher_condition,
                            source_images=images,
                            height=resolution.height,
                            width=resolution.width,
                            time_id_height=resolution.time_id_height,
                            time_id_width=resolution.time_id_width,
                            batch_size=args.functional_batch_size,
                            call_index=functional_call_index,
                            state_source=functional_state_source,
                        )
                else:
                    phase = "representation"
                loss = repr_scale * repr_value
                loss = loss + functional_scale * functional_loss_multiplier * (
                    args.functional_weight * functional.relative_mse
                    + args.functional_cos_weight * functional.cosine
                    + args.transition_weight * functional.transition_relative_mse
                    + args.transition_cos_weight * functional.transition_cosine
                )
                loss = loss + closed_scale * args.closed_loop_weight * (
                    args.closed_loop_prediction_weight * closed.prediction_relative_mse
                    + args.closed_loop_cos_weight
                    * (closed.prediction_cosine + closed.transition_cosine)
                    + args.closed_loop_transition_weight
                    * closed.transition_relative_mse
                    + args.closed_loop_terminal_weight * closed.terminal_relative_mse
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad_norm)
            optimizer.step()

            should_log = current_step == 1 or current_step % args.log_every == 0
            if should_log:
                item = {
                    "step": current_step,
                    "mode": mode,
                    "prompt_source": prompt_source if mode == "generate" else "edit",
                    "phase": phase,
                    "representation_objective": args.resolved_representation_objective,
                    "resolution_bucket": resolution.label,
                    "actual_width": resolution.width,
                    "actual_height": resolution.height,
                    "time_id_width": resolution.time_id_width,
                    "time_id_height": resolution.time_id_height,
                    "loss": scalar_mean(loss.detach(), context),
                    "representation": scalar_mean(repr_value.detach(), context),
                    "projected_representation": scalar_mean(
                        projected_value.detach(), context
                    ),
                    "token_cosine": scalar_mean(
                        repr_losses.get(
                            "token_cosine", repr_losses.get("content_token_cosine")
                        ).detach(),
                        context,
                    ),
                    "pooled_cosine": scalar_mean(
                        repr_losses.get(
                            "pooled_cosine", repr_losses.get("content_pooled_cosine")
                        ).detach(),
                        context,
                    ),
                    "content_token_cosine": scalar_mean(
                        repr_losses.get(
                            "content_token_cosine", repr_value.new_zeros(())
                        ).detach(),
                        context,
                    ),
                    "wrapper_token_cosine": scalar_mean(
                        repr_losses.get(
                            "wrapper_token_cosine", repr_value.new_zeros(())
                        ).detach(),
                        context,
                    ),
                    "semantic_contrastive": scalar_mean(
                        repr_losses.get(
                            "semantic_contrastive", repr_value.new_zeros(())
                        ).detach(),
                        context,
                    ),
                    "semantic_batch_size": scalar_mean(
                        repr_losses.get(
                            "semantic_batch_size", repr_value.new_zeros(())
                        ).detach(),
                        context,
                    ),
                    "content_pooled_normalized_mse": scalar_mean(
                        repr_losses.get(
                            "content_pooled_normalized_mse",
                            repr_value.new_zeros(()),
                        ).detach(),
                        context,
                    ),
                    "content_fraction": scalar_mean(
                        repr_losses.get(
                            "content_fraction", repr_value.new_zeros(())
                        ).detach(),
                        context,
                    ),
                    "token_mean": scalar_mean(
                        repr_losses.get(
                            "token_mean",
                            repr_losses.get(
                                "content_token_mean", repr_value.new_zeros(())
                            ),
                        ).detach(),
                        context,
                    ),
                    "token_std": scalar_mean(
                        repr_losses.get(
                            "token_std",
                            repr_losses.get(
                                "content_token_std", repr_value.new_zeros(())
                            ),
                        ).detach(),
                        context,
                    ),
                    "mask_agreement": scalar_mean(
                        repr_losses.get(
                            "mask_agreement", repr_value.new_ones(())
                        ).detach(),
                        context,
                    ),
                    "functional_relative_mse": scalar_mean(
                        functional.relative_mse.detach(), context
                    ),
                    "functional_call_index": functional.call_index,
                    "functional_state_source": functional.state_source,
                    "grounded_images": len(grounded_images),
                    "dedicated_grounded_batch": use_dedicated_grounded_batch,
                    "functional_transition_relative_mse": scalar_mean(
                        functional.transition_relative_mse.detach(),
                        context,
                    ),
                    "closed_terminal_relative_mse": scalar_mean(
                        closed.terminal_relative_mse.detach(), context
                    ),
                    "grad_norm": scalar_mean(
                        torch.as_tensor(grad_norm, device=context.device), context
                    ),
                    "lr": optimizer.param_groups[0]["lr"],
                }
                if context.device.type == "cuda":
                    item["cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated(
                        context.device
                    ) / (1024**3)
                    item["cuda_peak_reserved_gib"] = torch.cuda.max_memory_reserved(
                        context.device
                    ) / (1024**3)
                if context.is_main:
                    with history_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(item) + "\n")
                    postfix = {
                        "mode": mode,
                        "phase": phase,
                        "loss": f"{item['loss']:.4f}",
                        "func": f"{item['functional_relative_mse']:.4f}",
                        "res": resolution.label,
                    }
                    if args.resolved_representation_objective == "content_aware":
                        postfix["trans"] = (
                            f"{item['functional_transition_relative_mse']:.4f}"
                        )
                        postfix["state"] = item["functional_state_source"]
                    else:
                        postfix["roll"] = f"{item['closed_terminal_relative_mse']:.4f}"
                    progress.set_postfix(**postfix)
            progress.update(1)

            save_latest = (
                current_step % args.save_latest_every == 0
                or current_step == args.target_step
            )
            save_archive = (
                current_step % args.save_archive_every == 0
                or current_step == args.target_step
            )
            if save_latest or save_archive:
                barrier()
                if context.is_main:
                    module = unwrap(bridge)
                    payload = {
                        "step": current_step,
                        "bridge": {
                            key: value.detach().cpu()
                            for key, value in module.trainable_state_dict().items()
                        },
                        "optimizer": optimizer.state_dict(),
                        "config": vars(args),
                        "initialization": initialization,
                        "architecture": (
                            "MobileOVDreamLiteSharedSmolVLM2ImageBridgeV1"
                            if shared_video_bridge is not None
                            else
                            "MobileOVDreamLiteCompactBridgeV12"
                            if args.training_version.lower().startswith("v12")
                            else
                            "MobileOVDreamLiteCompactBridgeV11"
                            if args.training_version.lower().startswith("v11")
                            else "MobileOVDreamLiteCompactBridgeV7"
                            if args.training_version.lower()
                            in {"v7", "v8", "v9", "v10", "shared_v1"}
                            else "MobileOVDreamLiteCompactBridgeV6"
                            if args.training_version.lower() == "v6"
                            else "MobileOVDreamLiteCompactBridgeV5"
                            if args.training_version.lower() == "v5"
                            else "MobileOVDreamLiteCompactBridgeV4"
                            if args.training_version.lower() == "v4"
                            else "MobileOVDreamLiteCompactBridgeV3"
                            if args.representation_mode == "direct"
                            else "MobileOVDreamLiteImageBridge"
                        ),
                        "teacher": "Qwen3-VL BF16 from DreamLite-mobile",
                        "shared_smolvlm2": {
                            "enabled": shared_video_bridge is not None,
                            "video_bridge_checkpoint": shared_video_bridge_ckpt or None,
                            "video_bridge_config": (
                                args.shared_video_config
                                if shared_video_bridge is not None
                                else None
                            ),
                            "contract": (
                                "Exp1 prompt plus shared suffix, 512-token window, "
                                "strict 128-token selection"
                                if shared_video_bridge is not None
                                else None
                            ),
                            "prompt_suffix": (
                                args.shared_prompt_suffix
                                if shared_video_bridge is not None
                                else None
                            ),
                        },
                        "prompt_sources": generation_data.source_summary,
                        "functional_teacher": (
                            "frozen DreamLite-mobile UNet, native 4-call schedule, "
                            "mixed generated and real-image-derived same-state response distillation"
                            if args.training_version.lower().startswith("v11")
                            or args.training_version.lower().startswith("v12")
                            or args.training_version.lower()
                            in {"v7", "v8", "v9", "v10", "shared_v1"}
                            else "frozen DreamLite-mobile UNet, native 4-call schedule, "
                            "mixed teacher/student-prefix same-state prediction and transition distillation"
                            if args.training_version.lower() in {"v5", "v6"}
                            else "same-state teacher-forced prefixes"
                            if args.training_version.lower() == "v4"
                            else "frozen DreamLite-mobile UNet, native 4-call schedule"
                        ),
                    }
                    if save_latest:
                        torch.save(
                            payload, output_dir / "dreamlite_image_bridge_latest.pt"
                        )
                    if save_archive:
                        torch.save(
                            payload,
                            output_dir
                            / f"dreamlite_image_bridge_step{current_step:06d}.pt",
                        )
                    rank0_print(
                        context,
                        f"Saved DreamLite bridge step={current_step} latest={save_latest} archive={save_archive}",
                    )
                barrier()
    finally:
        progress.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
