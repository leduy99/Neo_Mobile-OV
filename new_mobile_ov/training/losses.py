from __future__ import annotations

import torch
import torch.nn.functional as F


def latent_motion_weaver_loss(
    z_pred: torch.Tensor,
    z_gt: torch.Tensor,
    motion_weight: float = 0.5,
    accel_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if z_pred.shape != z_gt.shape:
        raise ValueError(f"z_pred/z_gt shape mismatch: {tuple(z_pred.shape)} vs {tuple(z_gt.shape)}")
    latent_loss = F.l1_loss(z_pred[:, :, 1:], z_gt[:, :, 1:])
    dz_pred = z_pred[:, :, 1:] - z_pred[:, :, :-1]
    dz_gt = z_gt[:, :, 1:] - z_gt[:, :, :-1]
    motion_loss = F.l1_loss(dz_pred, dz_gt)
    loss = latent_loss + float(motion_weight) * motion_loss
    metrics = {"latent_loss": latent_loss.detach(), "motion_loss": motion_loss.detach()}
    if accel_weight > 0 and z_gt.shape[2] >= 3:
        acc_pred = z_pred[:, :, 2:] - 2 * z_pred[:, :, 1:-1] + z_pred[:, :, :-2]
        acc_gt = z_gt[:, :, 2:] - 2 * z_gt[:, :, 1:-1] + z_gt[:, :, :-2]
        accel_loss = F.l1_loss(acc_pred, acc_gt)
        loss = loss + float(accel_weight) * accel_loss
        metrics["accel_loss"] = accel_loss.detach()
    return loss, metrics
