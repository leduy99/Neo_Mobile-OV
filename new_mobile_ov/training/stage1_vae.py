"""Bounded-memory encoding with the native causal VAE, without spatial tiling."""
from __future__ import annotations

import torch


def clear_temporal_cache(vae):
    # Native CausalConv3d keeps feature tensors even after a clip has finished.
    for module in vae.modules():
        clear = getattr(module, "_clear_context_cache", None)
        if clear is not None:
            clear()


@torch.no_grad()
def encode_posterior(vae, video, *, window_size=16):
    if window_size < 8 or window_size % 8:
        raise ValueError("VAE window must be a positive multiple of its 8-frame temporal stride")
    if vae.training or any(p.requires_grad for p in vae.parameters()):
        raise ValueError("Streaming target encoding requires a frozen evaluation VAE")
    clear_temporal_cache(vae)
    try:
        return vae.encode(video, temporal_chunk=video.shape[2] > 1,
                          window_size=window_size).latent_dist
    finally:
        # Context is needed between chunks of THIS clip, not between training samples.
        clear_temporal_cache(vae)
