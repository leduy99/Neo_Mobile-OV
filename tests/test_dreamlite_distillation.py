from __future__ import annotations

import torch

from new_mobile_ov.bridge.dreamlite_image_bridge import DreamLiteCondition
from new_mobile_ov.training.dreamlite_distillation import dreamlite_representation_losses


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
