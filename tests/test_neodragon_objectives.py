from __future__ import annotations

import unittest

import torch

from new_mobile_ov.training.neodragon_objectives import (
    bridge_representation_losses,
    flat_cosine_distance,
    linear_ramp,
    masked_token_mse,
    scheduled_weight,
    token_norm_alignment,
    weighted_loss_sum,
)


class NeoDragonObjectivesTest(unittest.TestCase):
    def test_objectives_have_finite_gradients(self) -> None:
        prediction = torch.randn(2, 5, 8, requires_grad=True)
        target = torch.randn(2, 5, 8)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        loss = (
            masked_token_mse(prediction, target, mask, normalize_tokens=True)
            + token_norm_alignment(prediction, target, mask)
            + flat_cosine_distance(prediction, target)
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_linear_ramp(self) -> None:
        self.assertEqual(linear_ramp(4, start_step=5, ramp_steps=10), 0.0)
        self.assertAlmostEqual(linear_ramp(5, start_step=5, ramp_steps=10), 0.1)
        self.assertEqual(linear_ramp(20, start_step=5, ramp_steps=10), 1.0)

    def test_scheduled_weight_warmup_and_cooldown(self) -> None:
        kwargs = {
            "total_steps": 100,
            "peak_weight": 1.0,
            "start_weight": 0.0,
            "warmup_steps": 10,
            "final_weight": 0.2,
            "cooldown_steps": 20,
        }
        self.assertAlmostEqual(scheduled_weight(1, **kwargs), 0.1)
        self.assertAlmostEqual(scheduled_weight(10, **kwargs), 1.0)
        self.assertAlmostEqual(scheduled_weight(50, **kwargs), 1.0)
        self.assertAlmostEqual(scheduled_weight(90, **kwargs), 0.6)
        self.assertAlmostEqual(scheduled_weight(100, **kwargs), 0.2)

    def test_complete_bridge_representation_objective_has_gradients(self) -> None:
        prediction = torch.randn(2, 5, 8, requires_grad=True)
        target = torch.randn(2, 5, 8)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        prediction_pooled = torch.randn(2, 6, requires_grad=True)
        target_pooled = torch.randn(2, 6)
        losses = bridge_representation_losses(
            prediction,
            target,
            mask,
            prediction_pooled,
            target_pooled,
        )
        self.assertEqual(
            set(losses),
            {
                "raw_token",
                "normalized_token",
                "token_cosine",
                "token_norm",
                "pooled_mse",
                "pooled_cosine",
                "relational",
            },
        )
        total = weighted_loss_sum(losses, {name: 1.0 for name in losses})
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue(torch.isfinite(prediction_pooled.grad).all())


if __name__ == "__main__":
    unittest.main()
