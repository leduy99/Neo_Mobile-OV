from __future__ import annotations

from argparse import Namespace

import torch

from new_mobile_ov.bridge.ssd1b_image_bridge import SSD1BImageCondition
from new_mobile_ov.training.ssd1b_distillation import (
    SSD1BTeacherCondition,
    ssd1b_representation_losses,
)
from tools.train_ssd1b_image_bridge import representation_loss, representation_scale


def conditions(batch: int = 4, sequence: int = 12):
    mask = torch.zeros(batch, sequence, dtype=torch.long)
    for index, length in enumerate((3, 5, 8, 12)):
        mask[index, :length] = 1
    student = SSD1BImageCondition(
        clip_l_tokens=torch.randn(batch, sequence, 16, requires_grad=True),
        clip_big_g_tokens=torch.randn(batch, sequence, 24, requires_grad=True),
        pooled=torch.randn(batch, 24, requires_grad=True),
    )
    teacher = SSD1BTeacherCondition(
        clip_l_tokens=torch.randn(batch, sequence, 16),
        clip_big_g_tokens=torch.randn(batch, sequence, 24),
        pooled=torch.randn(batch, 24),
        clip_l_mask=mask,
        clip_big_g_mask=mask.clone(),
    )
    return student, teacher


def v2_args() -> Namespace:
    return Namespace(
        objective_version="v2",
        clip_l_weight=0.35,
        clip_big_g_weight=0.35,
        pooled_weight=0.5,
        norm_weight=0.05,
        geometry_weight=0.25,
        retrieval_weight=0.1,
        variance_weight=0.05,
        eos_weight=0.5,
        padding_weight=0.15,
        closed_loop_start_step=25,
        representation_decay_steps=15,
        representation_final_scale=0.35,
    )


def test_v2_representation_loss_is_finite_and_reaches_all_heads() -> None:
    student, teacher = conditions()
    losses = ssd1b_representation_losses(student, teacher)
    total = representation_loss(losses, v2_args())
    total.backward()

    assert torch.isfinite(total)
    assert all(torch.isfinite(value) for value in losses.values())
    assert student.clip_l_tokens.grad is not None
    assert student.clip_big_g_tokens.grad is not None
    assert student.pooled.grad is not None
    assert student.clip_l_tokens.grad.norm() > 0
    assert student.clip_big_g_tokens.grad.norm() > 0
    assert student.pooled.grad.norm() > 0


def test_content_loss_does_not_change_when_only_padding_target_changes() -> None:
    student, teacher = conditions()
    baseline = ssd1b_representation_losses(student, teacher)
    changed_l = teacher.clip_l_tokens.clone()
    changed_g = teacher.clip_big_g_tokens.clone()
    padding = ~teacher.clip_l_mask.bool()
    changed_l[padding] = torch.randn_like(changed_l[padding]) * 100.0
    changed_g[padding] = torch.randn_like(changed_g[padding]) * 100.0
    changed_teacher = SSD1BTeacherCondition(
        changed_l,
        changed_g,
        teacher.pooled,
        teacher.clip_l_mask,
        teacher.clip_big_g_mask,
    )
    changed = ssd1b_representation_losses(student, changed_teacher)

    assert torch.allclose(
        baseline["clip_l_content_normalized_mse"],
        changed["clip_l_content_normalized_mse"],
    )
    assert torch.allclose(
        baseline["clip_big_g_content_cosine"],
        changed["clip_big_g_content_cosine"],
    )
    assert not torch.allclose(
        baseline["clip_l_padding_normalized_mse"],
        changed["clip_l_padding_normalized_mse"],
    )


def test_representation_anchor_decays_but_never_disappears() -> None:
    args = v2_args()
    assert representation_scale(24, args) == 1.0
    assert 0.35 < representation_scale(30, args) < 1.0
    assert representation_scale(100, args) == 0.35
