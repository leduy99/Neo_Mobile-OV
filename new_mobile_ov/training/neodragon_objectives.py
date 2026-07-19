from __future__ import annotations

import torch
import torch.nn.functional as F


def _token_weights(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return mask.to(device=reference.device, dtype=reference.dtype).unsqueeze(-1)


def masked_token_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    normalize_tokens: bool = False,
) -> torch.Tensor:
    prediction = prediction.float()
    target = target.float()
    if normalize_tokens:
        prediction = F.layer_norm(prediction, (prediction.shape[-1],))
        target = F.layer_norm(target, (target.shape[-1],))
    weights = _token_weights(mask, prediction)
    denom = weights.sum().clamp_min(1.0) * prediction.shape[-1]
    return ((prediction - target).pow(2) * weights).sum() / denom


def masked_token_cosine(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    cosine = F.cosine_similarity(prediction.float(), target.float(), dim=-1)
    weights = mask.to(device=prediction.device, dtype=cosine.dtype)
    return 1.0 - (cosine * weights).sum() / weights.sum().clamp_min(1.0)


def masked_mean_pool(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = _token_weights(mask, tokens)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / denom


def masked_mean_norm(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = _token_weights(mask, tokens)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (tokens.norm(dim=-1, keepdim=True) * weights).sum(dim=1) / denom


def token_norm_alignment(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    pred_norm = masked_mean_norm(prediction.float(), mask)
    target_norm = masked_mean_norm(target.float(), mask)
    return (((pred_norm - target_norm) / target_norm.clamp_min(1e-4)) ** 2).mean()


def pooled_cosine(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1).mean()


def relational_cosine(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape[0] < 2:
        return prediction.new_zeros(())
    pred_pool = F.normalize(masked_mean_pool(prediction.float(), mask), dim=-1)
    target_pool = F.normalize(masked_mean_pool(target.float(), mask), dim=-1)
    return F.mse_loss(pred_pool @ pred_pool.T, target_pool @ target_pool.T)


def mask_binary_cross_entropy(mask_logits: torch.Tensor | None, target_mask: torch.Tensor) -> torch.Tensor:
    if mask_logits is None:
        return target_mask.new_zeros((), dtype=torch.float32)
    return F.binary_cross_entropy_with_logits(mask_logits.float(), target_mask.float())


def flat_cosine_distance(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(
        prediction.float().reshape(prediction.shape[0], -1),
        target.float().reshape(target.shape[0], -1),
        dim=-1,
    ).mean()


def linear_ramp(step: int, *, start_step: int, ramp_steps: int) -> float:
    if step < start_step:
        return 0.0
    if ramp_steps <= 0:
        return 1.0
    return min(max((step - start_step + 1) / float(ramp_steps), 0.0), 1.0)


def scheduled_weight(
    step: int,
    *,
    total_steps: int,
    peak_weight: float,
    start_weight: float | None = None,
    warmup_steps: int = 0,
    final_weight: float | None = None,
    cooldown_steps: int = 0,
) -> float:
    """Interpolate one objective weight through warmup, plateau, and cooldown."""
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")
    if warmup_steps < 0 or cooldown_steps < 0:
        raise ValueError("warmup_steps and cooldown_steps must be >= 0")
    if warmup_steps + cooldown_steps > total_steps:
        raise ValueError("warmup_steps + cooldown_steps cannot exceed total_steps")

    start = peak_weight if start_weight is None else start_weight
    final = peak_weight if final_weight is None else final_weight
    value = float(peak_weight)
    if warmup_steps > 0 and step <= warmup_steps:
        progress = min(max(step / float(warmup_steps), 0.0), 1.0)
        value = float(start) + (float(peak_weight) - float(start)) * progress
    if cooldown_steps > 0 and step > total_steps - cooldown_steps:
        progress = (step - (total_steps - cooldown_steps)) / float(cooldown_steps)
        value = float(peak_weight) + (float(final) - float(peak_weight)) * progress
    return value


def bridge_representation_losses(
    prediction_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    target_mask: torch.Tensor,
    prediction_pooled: torch.Tensor,
    target_pooled: torch.Tensor,
    mask_logits: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Return the complete post-ContextAdapter bridge alignment objective."""
    return {
        "raw_token": masked_token_mse(prediction_tokens, target_tokens, target_mask),
        "normalized_token": masked_token_mse(
            prediction_tokens,
            target_tokens,
            target_mask,
            normalize_tokens=True,
        ),
        "token_cosine": masked_token_cosine(prediction_tokens, target_tokens, target_mask),
        "token_norm": token_norm_alignment(prediction_tokens, target_tokens, target_mask),
        "pooled_mse": F.mse_loss(prediction_pooled.float(), target_pooled.float()),
        "pooled_cosine": pooled_cosine(prediction_pooled, target_pooled),
        "relational": relational_cosine(prediction_tokens, target_tokens, target_mask),
        "mask": mask_binary_cross_entropy(mask_logits, target_mask),
    }


def weighted_loss_sum(
    losses: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> torch.Tensor:
    """Combine named losses while rejecting silent name mismatches."""
    unknown = set(weights) - set(losses)
    if unknown:
        raise KeyError(f"Unknown loss names: {sorted(unknown)}")
    if not losses:
        raise ValueError("losses cannot be empty")
    reference = next(iter(losses.values()))
    total = reference.new_zeros(())
    for name, weight in weights.items():
        total = total + float(weight) * losses[name]
    return total
