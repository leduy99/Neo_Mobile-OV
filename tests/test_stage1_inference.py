from types import SimpleNamespace
from contextlib import nullcontext
import sys

import pytest
import torch

from tools.infer_mobileov_stage1 import verify_inference_stack
from tools import infer_mobileov_stage1 as infer
from new_mobile_ov.training.stage1_alignment import frozen_signatures
from tools.data_prepare.download_alignment_images import sha256_file


def test_repacked_file_requires_identical_tensors_and_processor(tmp_path):
    path = tmp_path / "model.pt"
    modules = {key: torch.nn.Linear(2, 2) for key in ("smolvlm2", "dit", "vae")}
    encoder, dit, vae = (modules[key] for key in ("smolvlm2", "dit", "vae"))
    encoder.processor_sha256 = "processor"
    torch.save(encoder.state_dict(), path)
    cfg = SimpleNamespace(bridge=SimpleNamespace(smolvlm2_ckpt_path=str(path)))
    contract = dict(processor_sha256="processor", frozen_weights_sha256=frozen_signatures(modules),
                    smolvlm2_sha256=sha256_file(path))
    assert not verify_inference_stack(cfg, encoder, dit, vae, contract)["smolvlm2_file_repacked"]
    torch.save({"model": encoder.state_dict(), "training_metadata": "different packaging"}, path)
    assert verify_inference_stack(cfg, encoder, dit, vae, contract)["smolvlm2_file_repacked"]
    encoder.processor_sha256 = "different"
    with pytest.raises(ValueError, match="Processor"):
        verify_inference_stack(cfg, encoder, dit, vae, contract)
    encoder.processor_sha256 = "processor"
    with torch.no_grad():
        encoder.weight.add_(.01)
    with pytest.raises(ValueError, match="weights differ"):
        verify_inference_stack(cfg, encoder, dit, vae, contract)


@pytest.mark.parametrize("frames", [1, 49])
def test_paired_noise_is_independent_of_model_loading_rng(frames, monkeypatch):
    histories, steps = [], []

    def generate(scheduler, dit, stages, latent, history, *condition, **kwargs):
        histories.append(len(history))
        steps.append(kwargs["num_inference_steps"])
        return latent + torch.randn_like(latent) * .1

    utils = SimpleNamespace(
        _prepare_latent_noise=lambda batch, channels, units, *args: torch.randn(batch, channels, units, 4, 4),
        _downsample_noise_2x=lambda latent, times: latent,
        _prepare_past_condition_latents=lambda generated, stages, cfg: generated.copy(),
        _generate_one_unit=generate,
        _decode_latent=lambda vae, latents: [None] * (1 + (latents.shape[2] - 1) * 8),
    )
    monkeypatch.setitem(sys.modules, "neodragon.utils.generation_utils", utils)
    monkeypatch.setattr(infer, "install_neodragon_generation_patches", lambda **kwargs: None)
    monkeypatch.setattr(infer, "autocast", lambda device: nullcontext())
    dit = SimpleNamespace(config=SimpleNamespace(in_channels=2))
    kwargs = dict(device=torch.device("cpu"), seed=42, num_frames=frames)
    _, first = infer.generate_frames(dit, None, None, (), **kwargs)
    torch.randn(1000)  # Model or conditioner initialization must not change generation noise.
    _, second = infer.generate_frames(dit, None, None, (), **kwargs)
    assert first == second
    units = 1 + (frames - 1) // 8
    assert histories == list(range(units)) * 2
    assert steps == ([[20] * 3] + [[10] * 3] * (units - 1)) * 2
