from pathlib import Path

import torch

from tools.train_neodragon_bridge_rollout_distill import (
    make_validation_dataset,
    newest_checkpoint,
)
from tools.train_neodragon_dit_bridge import VideoPromptDataset


def _dummy_dataset() -> VideoPromptDataset:
    dataset = VideoPromptDataset.__new__(VideoPromptDataset)
    dataset.rows = [
        {
            "caption": f"long {index}",
            "caption_short": f"short {index}",
            "caption_medium": f"medium {index}",
            "caption_long": f"long {index}",
            "latent_path": f"latent-{index}.pt",
        }
        for index in range(6)
    ]
    dataset.prompt_col = "caption"
    dataset.caption_fallback_column = "caption"
    dataset.caption_variant_columns = [
        "caption_short",
        "caption_medium",
        "caption_long",
    ]
    dataset.caption_aug = True
    dataset.caption_variant_weights = [1.0, 1.0, 1.0]
    dataset.latent_col = "latent_path"
    dataset.video_col = None
    dataset.has_latents = True
    return dataset


def test_validation_split_is_reserved_and_caption_balanced() -> None:
    dataset = _dummy_dataset()
    validation = make_validation_dataset(dataset, 3)

    assert validation is not None
    assert len(dataset.rows) == 3
    assert len(validation.rows) == 3
    assert validation.caption_aug is False
    assert [row["caption"] for row in validation.rows] == [
        "short 3",
        "medium 4",
        "long 5",
    ]


def test_newest_checkpoint_prefers_highest_global_step(tmp_path: Path) -> None:
    step_70k = tmp_path / "neodragon_rollout_bridge_step070000.pt"
    step_80k = tmp_path / "neodragon_rollout_bridge_step080000.pt"
    step_70k.touch()
    step_80k.touch()
    torch.save({"step": 82000}, tmp_path / "neodragon_rollout_bridge_best.pt")

    assert newest_checkpoint(tmp_path) == tmp_path / "neodragon_rollout_bridge_best.pt"


def test_newest_checkpoint_uses_archive_when_best_is_older(tmp_path: Path) -> None:
    archive = tmp_path / "neodragon_rollout_bridge_step090000.pt"
    archive.touch()
    torch.save({"step": 88000}, tmp_path / "neodragon_rollout_bridge_best.pt")

    assert newest_checkpoint(tmp_path) == archive
