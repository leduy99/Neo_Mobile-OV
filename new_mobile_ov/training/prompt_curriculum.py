from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return " ".join(str(value).strip().split())


@dataclass(frozen=True)
class PromptExample:
    prompt: str
    image_path: str
    source: str


class CaptionManifestDataset(Dataset):
    """Caption rows with optional raw-image paths for grounded supervision."""

    def __init__(
        self,
        path: str | Path,
        *,
        source_name: str,
        variant_columns: Sequence[str],
        variant_weights: Sequence[float],
        fallback_column: str,
        image_columns: Sequence[str],
        max_samples: int,
        require_existing_image: bool = False,
    ) -> None:
        self.path = Path(path).expanduser()
        sep = "\t" if self.path.suffix.lower() == ".tsv" else ","
        wanted_columns = {
            fallback_column,
            "caption",
            "prompt",
            "text",
            *variant_columns,
            *image_columns,
        }
        frame = pd.read_csv(
            self.path,
            sep=sep,
            low_memory=False,
            usecols=lambda name: name in wanted_columns,
        )
        fallback = fallback_column if fallback_column in frame.columns else None
        if fallback is None:
            fallback = next(
                (
                    name
                    for name in ("caption", "prompt", "text")
                    if name in frame.columns
                ),
                None,
            )
        available_variants = [
            name for name in variant_columns if name in frame.columns
        ]
        if fallback is None and not available_variants:
            raise ValueError(f"{self.path} has no caption, prompt, or text column.")

        available_image_columns = [
            name for name in image_columns if name in frame.columns
        ]
        self.source_name = str(source_name).strip() or self.path.stem
        self.variants = [
            (name, float(weight))
            for name, weight in zip(variant_columns, variant_weights)
            if name in frame.columns
        ]
        self.fallback = fallback
        self.image_columns = available_image_columns
        self.require_existing_image = bool(require_existing_image)
        caption_columns = sorted(
            {name for name, _ in self.variants}
            | ({fallback} if fallback is not None else set())
        )
        for name in caption_columns:
            frame[name] = (
                frame[name]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )
        keep_columns = list(dict.fromkeys(caption_columns + available_image_columns))
        valid = frame[caption_columns].ne("").any(axis=1)
        frame = frame.loc[valid, keep_columns]
        if max_samples > 0:
            frame = frame.head(max_samples)
        self.frame = frame.reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No valid generation prompts found in {self.path}.")
        image_candidates = pd.Series(False, index=self.frame.index)
        image_suffixes = tuple(IMAGE_EXTENSIONS)
        for name in self.image_columns:
            values = self.frame[name].fillna("").astype(str).str.strip().str.lower()
            image_candidates |= values.str.endswith(image_suffixes)
        self.image_candidate_rows = int(image_candidates.sum())
        if self.require_existing_image and self.image_candidate_rows == 0:
            raise ValueError(
                f"{self.path} has no rows with an image path supported for grounded training."
            )

    def _image_path(self, row: pd.Series, columns: Sequence[str]) -> str:
        for column in columns:
            value = clean_text(row.get(column))
            if not value:
                continue
            path = Path(value).expanduser()
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if not path.is_absolute():
                # Source manifests normally store absolute paths. These fallbacks
                # cover repo-relative and output-root-relative manifests.
                repo_relative = Path.cwd() / path
                manifest_relative = self.path.parent / path
                output_relative = self.path.parent.parent / path
                path = next(
                    (
                        candidate
                        for candidate in (
                            repo_relative,
                            manifest_relative,
                            output_relative,
                        )
                        if candidate.is_file()
                    ),
                    repo_relative,
                )
            return str(path)
        return ""

    def __len__(self) -> int:
        return len(self.frame)

    def _sample_once(self, index: int, *, rng: random.Random) -> PromptExample:
        row = self.frame.iloc[int(index) % len(self.frame)]
        choices = []
        for name, weight in self.variants:
            value = clean_text(row.get(name))
            if value:
                choices.append((value, weight))
        if not choices and self.fallback is not None:
            choices = [(clean_text(row.get(self.fallback)), 1.0)]
        prompt = rng.choices(
            [value for value, _ in choices],
            weights=[weight for _, weight in choices],
            k=1,
        )[0]
        image_path = self._image_path(row, self.image_columns)
        return PromptExample(prompt, image_path, self.source_name)

    def sample(self, index: int, *, rng: random.Random) -> PromptExample:
        example = self._sample_once(index, rng=rng)
        if not self.require_existing_image:
            return example
        if example.image_path and Path(example.image_path).is_file():
            return example

        # A manifest may contain stale paths. Retry deterministically so a
        # grounded batch never silently falls back to ungrounded supervision.
        attempts = min(max(len(self.frame), 1), 128)
        for offset in range(1, attempts + 1):
            candidate = self._sample_once(int(index) + offset, rng=rng)
            if candidate.image_path and Path(candidate.image_path).is_file():
                return candidate
        raise FileNotFoundError(
            f"Could not find a readable image after {attempts} rows in {self.path}."
        )

    def __getitem__(self, index: int) -> PromptExample:
        rng = random.Random(int(index))
        return self.sample(index, rng=rng)


class MixedPromptDataset(Dataset):
    """Deterministically sample prompt manifests with source-level weights."""

    def __init__(
        self,
        datasets: Sequence[CaptionManifestDataset],
        weights: Sequence[float],
        *,
        seed: int,
    ) -> None:
        if not datasets:
            raise ValueError("At least one caption manifest is required.")
        if len(datasets) != len(weights):
            raise ValueError("Prompt datasets and source weights must have equal length.")
        if any(float(weight) < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("Prompt source weights must be non-negative with positive sum.")
        self.datasets = list(datasets)
        self.weights = [float(weight) for weight in weights]
        self.seed = int(seed)
        self.length = sum(len(dataset) for dataset in self.datasets)

    @property
    def source_summary(self) -> list[dict[str, object]]:
        total_weight = sum(self.weights)
        return [
            {
                "name": dataset.source_name,
                "path": str(dataset.path),
                "rows": len(dataset),
                "weight": weight / total_weight,
                "image_columns": dataset.image_columns,
                "image_candidate_rows": dataset.image_candidate_rows,
            }
            for dataset, weight in zip(self.datasets, self.weights)
        ]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> PromptExample:
        rng = random.Random(self.seed + 1000003 * int(index))
        if len(self.datasets) == 1:
            dataset_index = 0
            sample_index = int(index) % len(self.datasets[0])
        else:
            dataset_index = rng.choices(
                range(len(self.datasets)), weights=self.weights, k=1
            )[0]
            sample_index = rng.randrange(len(self.datasets[dataset_index]))
        return self.datasets[dataset_index].sample(sample_index, rng=rng)


def prompt_example_collate(
    items: Sequence[PromptExample],
) -> tuple[list[str], list[str], list[str]]:
    return (
        [item.prompt for item in items],
        [item.image_path for item in items],
        [item.source for item in items],
    )
