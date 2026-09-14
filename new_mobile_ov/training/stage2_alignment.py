"""Joint T2I/T2V adaptation after connector alignment, not DMD or joint VLM training."""
from __future__ import annotations

import torch
from torch import nn

from new_mobile_ov.training.stage1_alignment import flow_loss

FORMAT = "mobileov_stage2_mcp_dit_v1"
TASKS = ("t2i", "t2v")


def initialize_from_alignment(payload, connector, dit, *, expected_step, signatures,
                              config_sha256, processor_sha256, args):
    if payload.get("format") != "mobileov_stage1_mcp_v1" or payload.get("step") != expected_step:
        raise ValueError(f"Expected Stage-1 connector at step {expected_step}")
    contract = payload["contract"]
    expected = dict(config_sha256=config_sha256,
                    processor_sha256=processor_sha256, frozen_weights_sha256=signatures)
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"Stage-2 initialization mismatch: {key}")
    for key in ("short_side", "long_side", "max_tokens"):
        if contract.get(key) != getattr(args, key):
            raise ValueError(f"Stage-2 initialization mismatch: {key}")
    if payload["connector_spec"] != connector.spec:
        raise ValueError("Stage-2 connector architecture differs from Stage 1")
    if not all(bool(torch.isfinite(value).all()) for value in payload["connector"].values()):
        raise ValueError("Non-finite initialization weights")
    connector.load_state_dict(payload["connector"], strict=True)
    # FP32 master weights retain small updates; BF16 is used only for computation.
    connector.float().train().requires_grad_(True)
    dit.float().train().requires_grad_(True)
    dit.gradient_checkpointing = args.gradient_checkpointing
    dit.gradient_checkpointing_ratio = 0.0


class JointFlowModel(nn.Module):
    """One DDP graph owns both trainable components, including the flow objective."""
    def __init__(self, connector, dit):
        super().__init__()
        self.connector, self.dit = connector, dit

    def forward(self, layers, mask, latent, scheduler, *, stage, unit, generator):
        # DiT checkpoints each transformer block; do not also recompute the whole DiT.
        return flow_loss(self.dit, self.connector, layers, mask, latent, scheduler,
                         stage=stage, unit=unit, generator=generator, gradient_checkpointing=False)


def gradient_norm(module):
    norms = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
    return torch.stack(norms).norm() if norms else next(module.parameters()).new_zeros(())
