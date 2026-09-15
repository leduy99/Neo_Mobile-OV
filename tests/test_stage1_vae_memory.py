from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from new_mobile_ov.training.stage1_vae import encode_posterior
from tools.train_mobileov_stage1 import parse_args


class CachedVAE(nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.cache_front_feat = deque([torch.ones(1)])
        self.fail = fail
        self.eval()

    def _clear_context_cache(self):
        self.cache_front_feat.clear()

    def encode(self, video, *, temporal_chunk, window_size):
        assert not self.cache_front_feat
        assert not torch.is_grad_enabled()
        self.cache_front_feat.append(video.clone())
        self.kwargs = dict(temporal_chunk=temporal_chunk, window_size=window_size)
        if self.fail:
            raise RuntimeError("encode failed")
        return SimpleNamespace(latent_dist=video + 1)


@pytest.mark.parametrize("frames", [1, 49])
def test_cache_freed_without_dropping_frames(frames):
    vae = CachedVAE()
    video = torch.zeros(1, 3, frames, 8, 8)
    posterior = encode_posterior(vae, video, window_size=8)
    assert posterior.shape == video.shape and torch.all(posterior == 1)
    assert vae.kwargs == dict(temporal_chunk=frames > 1, window_size=8)
    assert not vae.cache_front_feat


def test_cache_freed_on_exception():
    vae = CachedVAE(fail=True)
    with pytest.raises(RuntimeError, match="encode failed"):
        encode_posterior(vae, torch.zeros(1, 3, 49, 8, 8), window_size=8)
    assert not vae.cache_front_feat
    with pytest.raises(ValueError, match="frozen"):
        encode_posterior(vae.train(), torch.zeros(1, 3, 49, 8, 8))


@pytest.mark.parametrize("extra", [["--vae-window-size", "7"], ["--vae-window-size", "0"],
                                    ["--cuda-memory-limit-gib", "-1"], ["--cuda-memory-limit-gib", "nan"]])
def test_reject_invalid_memory_options(extra):
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", "unused", *extra])


def test_native_causal_vae_chunks_match_on_cpu(monkeypatch):
    repo = Path(__file__).resolve().parents[1] / "checkpoints" / "neodragon_repo"
    if not repo.exists():
        pytest.skip("Optional native VAE source checkout is not installed")
    monkeypatch.syspath_prepend(str(repo))
    from neodragon.asymmetric_causal_video_vae import AsymmetricCausalVideoVAE
    torch.manual_seed(52)
    vae = AsymmetricCausalVideoVAE(encoder_out_channels=4, encoder_layers_per_block=(1, 1, 1, 1),
                                  encoder_block_out_channels=(8, 8, 8, 8), encoder_norm_num_groups=4,
                                  decoder_num_features=(8, 8, 8, 8)).eval().requires_grad_(False)
    clip = torch.randn(1, 3, 49, 32, 32)
    old = encode_posterior(vae, clip, window_size=16).parameters
    new = encode_posterior(vae, clip, window_size=8).parameters
    assert old.shape[2] == new.shape[2] == 7
    torch.testing.assert_close(old, new, atol=2e-6, rtol=1e-5)
    # A different video and image must not inherit context from this clip.
    for video in (clip * .2, clip[:, :, :1]):
        a = encode_posterior(vae, video, window_size=8).parameters
        b = encode_posterior(vae, video, window_size=16).parameters
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)
