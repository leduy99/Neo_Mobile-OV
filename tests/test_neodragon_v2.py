from __future__ import annotations

import unittest

import torch

from new_mobile_ov.bridge.neodragon_text_bridge import (
    NeoDragonConditionHead,
    NeoDragonSequenceTranslator,
)
from new_mobile_ov.config import load_config
from new_mobile_ov.training.neodragon_objectives import (
    bridge_representation_losses,
    flat_cosine_distance,
    linear_ramp,
    mask_binary_cross_entropy,
    masked_token_mse,
    scheduled_weight,
    token_norm_alignment,
    weighted_loss_sum,
)


class NeoDragonV2Test(unittest.TestCase):
    def test_new_config_is_opt_in(self) -> None:
        legacy = load_config("configs/mobile_ov_neodragon.yaml")
        v2 = load_config("configs/mobile_ov_neodragon_bridge_v2.yaml")
        self.assertFalse(legacy.bridge.neodragon_v2_conditioning)
        self.assertTrue(v2.bridge.neodragon_v2_conditioning)

    def test_sequence_translator_preserves_shape_and_gradient(self) -> None:
        module = NeoDragonSequenceTranslator(dim=24, sequence_length=8, num_heads=4)
        tokens = torch.randn(2, 8, 24, requires_grad=True)
        mask = torch.tensor([[1] * 8, [1, 1, 1, 1, 0, 0, 0, 0]])
        output = module(tokens, mask)
        self.assertEqual(tuple(output.shape), (2, 8, 24))
        output.square().mean().backward()
        self.assertIsNotNone(tokens.grad)
        self.assertTrue(torch.isfinite(tokens.grad).all())

    def test_sequence_translator_rejects_wrong_length(self) -> None:
        module = NeoDragonSequenceTranslator(dim=24, sequence_length=8, num_heads=4)
        with self.assertRaisesRegex(ValueError, "fixed token sequence"):
            module(torch.randn(1, 7, 24), torch.ones(1, 7))

    def test_condition_head_controls_initial_output_scale(self) -> None:
        module = NeoDragonConditionHead(dim=32, bottleneck_dim=16, scale_init=0.5)
        output = module(torch.randn(4, 6, 32))
        expected_norm = 0.5 * (32**0.5)
        actual_norm = float(output.norm(dim=-1).mean().detach())
        self.assertAlmostEqual(actual_norm, expected_norm, delta=0.2)

    def test_objectives_have_finite_gradients(self) -> None:
        prediction = torch.randn(2, 5, 8, requires_grad=True)
        target = torch.randn(2, 5, 8)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        mask_logits = torch.randn(2, 5, requires_grad=True)
        loss = (
            masked_token_mse(prediction, target, mask, normalize_tokens=True)
            + token_norm_alignment(prediction, target, mask)
            + mask_binary_cross_entropy(mask_logits, mask)
            + flat_cosine_distance(prediction, target)
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue(torch.isfinite(mask_logits.grad).all())

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
        mask_logits = torch.randn(2, 5, requires_grad=True)
        losses = bridge_representation_losses(
            prediction,
            target,
            mask,
            prediction_pooled,
            target_pooled,
            mask_logits,
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
                "mask",
            },
        )
        total = weighted_loss_sum(losses, {name: 1.0 for name in losses})
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue(torch.isfinite(prediction_pooled.grad).all())
        self.assertTrue(torch.isfinite(mask_logits.grad).all())


if __name__ == "__main__":
    unittest.main()
