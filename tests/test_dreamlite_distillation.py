from __future__ import annotations

import torch
import torch.nn as nn
from types import SimpleNamespace

from new_mobile_ov.bridge import dreamlite_image_bridge as bridge_module
from new_mobile_ov.bridge.dreamlite_image_bridge import (
    DreamLiteCondition,
    MobileOVDreamLiteImageBridge,
    dreamlite_generation_lengths,
)
from new_mobile_ov.config import BridgeConfig, DreamLiteBridgeConfig
from new_mobile_ov.training.dreamlite_distillation import (
    dreamlite_direct_representation_losses,
    dreamlite_representation_losses,
)


def make_conditions():
    student_tokens = torch.randn(3, 16, 32, requires_grad=True)
    teacher_tokens = torch.randn(3, 11, 32)
    teacher_mask = torch.tensor(
        [
            [1] * 11,
            [1] * 7 + [0] * 4,
            [1] * 4 + [0] * 7,
        ]
    )
    student = DreamLiteCondition(
        student_tokens,
        torch.ones(3, 16, dtype=torch.long),
    )
    teacher = DreamLiteCondition(teacher_tokens, teacher_mask)
    return student, teacher


def test_dreamlite_representation_losses_are_finite_and_reach_all_queries() -> None:
    student, teacher = make_conditions()
    losses = dreamlite_representation_losses(student, teacher)
    total = sum(losses.values())
    total.backward()

    assert torch.isfinite(total)
    assert all(torch.isfinite(value) for value in losses.values())
    assert student.prompt_embeds.grad is not None
    assert torch.isfinite(student.prompt_embeds.grad).all()
    assert student.prompt_embeds.grad.norm() > 0


def test_teacher_padding_does_not_change_dreamlite_representation_targets() -> None:
    student, teacher = make_conditions()
    baseline = dreamlite_representation_losses(student, teacher)
    changed_tokens = teacher.prompt_embeds.clone()
    changed_tokens[~teacher.attention_mask.bool()] = 1000.0
    changed = dreamlite_representation_losses(
        student,
        DreamLiteCondition(changed_tokens, teacher.attention_mask),
    )

    for name in baseline:
        assert torch.allclose(baseline[name], changed[name])


class FakeNativeTokenizer:
    def __init__(self, lengths: list[int], prefix_tokens: int = 34) -> None:
        self.lengths = lengths
        self.prefix_tokens = prefix_tokens

    def __call__(self, *, text, padding, return_tensors):
        del padding, return_tensors
        assert len(text) == len(self.lengths)
        total = [length + self.prefix_tokens for length in self.lengths]
        mask = torch.zeros(len(total), max(total), dtype=torch.long)
        for index, length in enumerate(total):
            mask[index, :length] = 1
        return SimpleNamespace(attention_mask=mask)


class FakeFeatureProvider(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        del args, kwargs
        self.hidden_size = 960
        self.smolvlm2_model = nn.Linear(1, 1, bias=False)
        self.smolvlm2_model.requires_grad_(False)

    def forward(self, prompts, *, mode, images=None):
        del mode, images
        batch = len(prompts)
        source = torch.randn(batch, 12, self.hidden_size)
        mask = torch.ones(batch, 12, dtype=torch.long)
        return [source.clone() for _ in range(4)], mask


def compact_bridge(monkeypatch) -> MobileOVDreamLiteImageBridge:
    monkeypatch.setattr(
        bridge_module,
        "SmolVLM2MultimodalFeatureProvider",
        FakeFeatureProvider,
    )
    cfg = DreamLiteBridgeConfig(
        attention_dim=384,
        num_heads=6,
        num_layers=2,
        ff_mult=3,
        variable_length_generation=True,
    )
    return MobileOVDreamLiteImageBridge(
        BridgeConfig(),
        cfg,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_native_generation_lengths_are_variable_and_capped() -> None:
    tokenizer = FakeNativeTokenizer([7, 19, 200])
    lengths = dreamlite_generation_lengths(
        tokenizer,
        ["short", "medium", "long"],
        prefix_tokens=34,
        max_length=128,
    )
    assert lengths.tolist() == [7, 19, 128]


def test_compact_bridge_has_variable_mask_and_target_parameter_count(monkeypatch) -> None:
    bridge = compact_bridge(monkeypatch)
    bridge._native_tokenizer = FakeNativeTokenizer([7, 19, 11])
    condition = bridge(["a", "b", "c"], mode="generate")

    trainable = sum(parameter.numel() for parameter in bridge.parameters() if parameter.requires_grad)
    assert 5_000_000 <= trainable <= 6_000_000
    assert condition.prompt_embeds.shape == (3, 19, 2048)
    assert condition.attention_mask.sum(dim=1).tolist() == [7, 19, 11]
    assert torch.count_nonzero(condition.prompt_embeds[0, 7:]) == 0
    assert torch.count_nonzero(condition.prompt_embeds[2, 11:]) == 0


def test_direct_losses_ignore_padding_and_are_zero_for_identical_tokens() -> None:
    tokens = torch.randn(3, 11, 32)
    mask = torch.tensor(
        [
            [1] * 11,
            [1] * 7 + [0] * 4,
            [1] * 4 + [0] * 7,
        ]
    )
    student_tokens = tokens.clone().requires_grad_(True)
    student = DreamLiteCondition(student_tokens, mask)
    teacher_tokens = tokens.clone()
    teacher_tokens[~mask.bool()] = 1000.0
    teacher = DreamLiteCondition(teacher_tokens, mask)
    losses = dreamlite_direct_representation_losses(student, teacher)

    for name, value in losses.items():
        expected = 1.0 if name == "mask_agreement" else 0.0
        assert torch.allclose(value, value.new_tensor(expected), atol=1e-6), name
    sum(value for name, value in losses.items() if name != "mask_agreement").backward()
    assert student_tokens.grad is not None
    assert torch.isfinite(student_tokens.grad).all()
