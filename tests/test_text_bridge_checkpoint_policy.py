from pathlib import Path
from types import SimpleNamespace

import torch

from tools.train_neodragon_text_bridge import (
    PromptDataset,
    functional_unit_for_step,
    learning_rate_scale,
    newest_text_checkpoint,
    promote_trainable_parameters_to_fp32,
    reserve_validation_prompts,
)


def _prompt_dataset() -> PromptDataset:
    dataset = PromptDataset.__new__(PromptDataset)
    dataset.items = [
        {
            "fallback": f"long {index}",
            "variants": [
                (f"short {index}", "caption_short", 1.0),
                (f"medium {index}", "caption_medium", 1.0),
                (f"long {index}", "caption_long", 1.0),
            ],
        }
        for index in range(6)
    ]
    return dataset


def test_reserve_validation_prompts_is_balanced() -> None:
    dataset = _prompt_dataset()

    prompts = reserve_validation_prompts(dataset, 3)

    assert len(dataset.items) == 3
    assert prompts == ["short 3", "medium 4", "long 5"]


def test_newest_text_checkpoint_uses_highest_step(tmp_path: Path) -> None:
    archive = tmp_path / "neodragon_text_bridge_step080000.pt"
    archive.touch()
    torch.save({"step": 82000}, tmp_path / "neodragon_text_bridge_best.pt")

    assert newest_text_checkpoint(tmp_path) == tmp_path / "neodragon_text_bridge_best.pt"


def test_lr_schedule_reaches_requested_final_scale() -> None:
    start = learning_rate_scale(
        64001,
        initial_step=64000,
        target_step=200000,
        warmup_steps=500,
        final_scale=0.1,
    )
    end = learning_rate_scale(
        200000,
        initial_step=64000,
        target_step=200000,
        warmup_steps=500,
        final_scale=0.1,
    )

    assert 0.2 < start < 0.21
    assert end == 0.1


def test_fp32_master_conversion_only_changes_trainable_parameters() -> None:
    module = torch.nn.Sequential(
        torch.nn.Linear(4, 4, dtype=torch.bfloat16),
        torch.nn.Linear(4, 4, dtype=torch.bfloat16),
    )
    module[0].requires_grad_(False)

    promote_trainable_parameters_to_fp32(module)

    assert module[0].weight.dtype == torch.bfloat16
    assert module[1].weight.dtype == torch.float32


def test_exp1_200k_script_uses_single_call_trainer_not_rollout() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "exp1_distill_64k_to200k_v2_1node8gpu.sbatch"
    ).read_text(encoding="utf-8")

    assert "tools/train_neodragon_text_bridge.py" in script
    assert "train_neodragon_bridge_rollout_distill.py" not in script
    assert "--functional-every 1" in script
    assert "--trainable-fp32" in script
    assert "--no-save-latest" in script


def test_monolithic_functional_cycle_covers_all_native_units() -> None:
    cfg = SimpleNamespace(data=SimpleNamespace(frame_num=49))
    args = SimpleNamespace(
        functional_unit_policy="cycle",
        functional_include_first_unit=True,
        functional_start_step=1,
        functional_every=1,
    )

    sampled = [functional_unit_for_step(cfg, args, step) for step in range(1, 15)]

    assert sampled == [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]


def test_original_exp1_functional_policy_remains_random_and_skips_unit_zero() -> None:
    cfg = SimpleNamespace(data=SimpleNamespace(frame_num=49))
    args = SimpleNamespace(
        functional_unit_policy="random",
        functional_include_first_unit=False,
        functional_start_step=1,
        functional_every=1,
    )

    assert functional_unit_for_step(cfg, args, step=1) is None


def test_monolithic_submit_keeps_fp32_weights_with_fsdp() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train_neodragon_monolithic_text_bridge_1node8gpu.sbatch"
    ).read_text(encoding="utf-8")

    assert "--parallel fsdp" in script
    assert "--trainable-fp32" in script


def test_anchor_bridge_targets_monolithic_stack_and_skips_unit_zero() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train_neodragon_monolithic_video_units_text_bridge_1node8gpu.sbatch"
    ).read_text(encoding="utf-8")

    assert "--target-stack multistep" in script
    assert "--functional-dit-checkpoint" not in script
    assert "--no-functional-include-first-unit" in script
    assert "--functional-unit-policy cycle" in script


def test_single_rank_fsdp_is_not_downgraded_to_no_parallel() -> None:
    trainer = Path(__file__).resolve().parents[1] / "tools" / "train_neodragon_text_bridge.py"
    source = trainer.read_text(encoding="utf-8")

    assert 'if args.parallel == "ddp" and not ctx.is_distributed:' in source
    assert 'elif args.parallel == "fsdp" and not ctx.is_distributed:' in source
    assert "using NO_SHARD for a faithful FSDP smoke test" in source
