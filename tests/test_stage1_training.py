import json
from contextlib import nullcontext
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from new_mobile_ov.bridge.stage1_conditioner import Stage1Connector, FrozenStage1Encoder
from new_mobile_ov.training import stage1_alignment as stage1
from tools.train_mobileov_stage1 import parse_args, atomic_torch_save, FORMAT
from tools.infer_mobileov_stage1 import pad_condition


@pytest.mark.parametrize("count", [1, 2, 7, 100, 101])
def test_stateless_permutation_is_bijective_per_epoch(count):
    for epoch in (0, 1):
        values = [stage1.permuted_index(epoch * count + i, count, 42, "t2v") for i in range(count)]
        assert sorted(values) == list(range(count))
    assert stage1.permuted_index(12, count, 42, "t2v") == stage1.permuted_index(12, count, 42, "t2v")


def test_rank_partition_resume_and_task_coverage(tmp_path, monkeypatch):
    (tmp_path / "train").mkdir()
    (tmp_path / "sources.json").write_text("{}")
    for task in stage1.TASKS:
        (tmp_path / "train" / f"{task}.jsonl").write_text("".join(
            json.dumps(dict(task=task, sample_id=f"{task}:{i}")) + "\n" for i in range(100)))
    monkeypatch.setattr(stage1, "prepare_sample", lambda record, *args: record.copy())
    common = dict(steps=10, accumulation=3, world_size=2, seed=42)
    a = stage1.Stage1Dataset(tmp_path, rank=0, **common)
    b = stage1.Stage1Dataset(tmp_path, rank=1, **common)
    resumed = stage1.Stage1Dataset(tmp_path, rank=0, start_step=3, **common)
    assert resumed[0] == a[9]
    for i in range(len(resumed)):
        assert resumed[i] == a[i + 9]
    assert [a[i]["task"] for i in range(3)] == list(stage1.TASKS)
    assert {a[i]["sample_id"] for i in range(30)}.isdisjoint({b[i]["sample_id"] for i in range(30)})


def test_full_tokens_preserved_and_pooled_head_trainable():
    connector = Stage1Connector(input_dim=8, token_dim=16, pooled_dim=12)
    layers = [torch.randn(1, 193, 8) for _ in range(3)]
    mask = torch.ones(1, 193, dtype=torch.long)
    tokens, returned_mask, pooled = connector(layers, mask)
    assert tokens.shape == (1, 193, 16) and pooled.shape == (1, 12)
    assert torch.equal(returned_mask, mask)
    (tokens.square().mean() + pooled.square().mean()).backward()
    assert all(p.grad is not None for p in connector.parameters())
    with pytest.raises(ValueError, match="Invalid feature"):
        connector(layers, torch.zeros_like(mask))


@pytest.mark.parametrize("stage", [0, 1, 2])
@pytest.mark.parametrize("unit", [0, 1, 6])
def test_flow_reaches_connector_but_not_frozen_dit(stage, unit, monkeypatch):
    histories = []

    def prepare_history(past, stages, cfg):
        histories.append(past)
        return [past] * stages

    monkeypatch.setitem(sys.modules, "neodragon.utils.generation_utils",
                        SimpleNamespace(_prepare_past_condition_latents=prepare_history))

    class FrozenDiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()), requires_grad=False)

        def forward(self, sample, encoder_hidden_states, encoder_attention_mask, pooled_projections, timestep_ratio):
            signal = encoder_hidden_states[..., 0].mean() + pooled_projections.mean()
            return [sample[0][-1] * self.weight + signal]

    torch.manual_seed(1)
    dit = FrozenDiT()
    connector = Stage1Connector(input_dim=8, token_dim=16, pooled_dim=12)
    old = {key: value.clone() for key, value in connector.state_dict().items()}
    optimizer = torch.optim.AdamW(connector.parameters(), lr=1e-4)
    scheduler = SimpleNamespace(config=SimpleNamespace(stages=3, num_train_timesteps=10),
                                sigmas_per_stage={s: torch.linspace(1, 0.1, 10) for s in range(3)},
                                timesteps_per_stage={s: torch.linspace(1000, 100, 10) for s in range(3)},
                                start_sigmas={0: 1., 1: .8, 2: .5}, end_sigmas={0: .67, 1: .33, 2: 0.})
    latents = torch.randn(1, 4, 7, 16, 24)
    layers = [torch.randn(1, 16, 8) for _ in range(3)]
    loss = stage1.flow_loss(dit, connector, layers, torch.ones(1, 16), latents, scheduler,
                            stage=stage, unit=unit, generator=torch.Generator().manual_seed(2))
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert dit.weight.grad is None and dit.weight.item() == 1
    assert any(not torch.equal(value, old[key]) for key, value in connector.state_dict().items())
    assert len(histories[0]) == unit
    for index, value in enumerate(histories[0]):
        assert torch.equal(value, latents[:, :, index:index + 1])


class FakeProcessor:
    def __init__(self):
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return "chat"

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        result = dict(input_ids=torch.ones(1, 5, dtype=torch.long), attention_mask=torch.ones(1, 5))
        if "images" in kwargs:
            result["pixel_values"] = torch.ones(1, 1, 3, 16, 16)
        return result


class FakeVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1), requires_grad=False)
        self.inputs = None

    def forward(self, **kwargs):
        self.inputs = kwargs
        return SimpleNamespace(hidden_states=[torch.ones(1, 5, 8) for _ in range(4)])


def fake_encoder():
    encoder = FrozenStage1Encoder.__new__(FrozenStage1Encoder)
    nn.Module.__init__(encoder)
    encoder.backbone, encoder.processor, encoder.max_tokens = FakeVLM(), FakeProcessor(), 20
    return encoder


def test_reconstruction_is_image_only_and_cfg_drops_the_image():
    encoder = fake_encoder()
    image = object()
    encoder("", image)
    assert encoder.processor.messages[0]["content"] == [{"type": "image"}]
    assert encoder.processor.kwargs["images"] == [[image]]
    assert "pixel_values" in encoder.backbone.inputs
    with pytest.raises(ValueError, match="caption"):
        encoder("target caption", image)
    encoder("", image, drop_condition=True)
    assert encoder.processor.messages[0]["content"] == [{"type": "text", "text": ""}]
    assert "images" not in encoder.processor.kwargs and "pixel_values" not in encoder.backbone.inputs
    encoder.train()
    assert encoder.backbone.training is False


def test_t2v_is_text_only_and_does_not_silently_truncate():
    encoder = fake_encoder()
    encoder("A dog runs.")
    assert encoder.processor.messages[0]["content"] == [{"type": "text", "text": "A dog runs."}]
    assert encoder.processor.kwargs["truncation"] is False
    encoder.max_tokens = 4
    with pytest.raises(ValueError, match="truncation"):
        encoder("too many tokens")


def test_checkpoint_is_connector_only_and_cfg_padding_is_masked(tmp_path):
    connector = Stage1Connector(input_dim=8, token_dim=16, pooled_dim=12)
    checkpoint = tmp_path / "latest.pt"
    atomic_torch_save(dict(format=FORMAT, connector=connector.state_dict()), checkpoint)
    loaded = torch.load(checkpoint, weights_only=True)
    assert set(loaded) == {"format", "connector"}
    assert not any("smolvlm2" in key or "dit" in key for key in loaded["connector"])
    condition = (torch.ones(1, 5, 8), torch.ones(1, 5), torch.ones(1, 4))
    tokens, mask, pooled = pad_condition(condition, 8)
    assert tokens.shape == (1, 8, 8) and mask[:, 5:].sum() == 0
    assert torch.equal(pooled, condition[2])
    assert not checkpoint.with_suffix(".pt.tmp").exists()


@pytest.mark.parametrize("extra", [["--accumulation", "2"], ["--short-side", "250"],
                                  ["--steps", "0"], ["--condition-dropout", "1"]])
def test_invalid_training_contract_rejected(extra):
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", "unused", *extra])


def test_shell_scripts_are_valid():
    root = Path(__file__).resolve().parents[1]
    for name in ("train_mobileov_stage1_alignment_1node8gpu.sbatch", "smoke_mobileov_stage1_local.sbatch"):
        subprocess.run(["bash", "-n", str(root / "scripts" / name)], check=True)


def ddp_features(rank, micro):
    generator = torch.Generator().manual_seed(100 + rank * 3 + micro)
    length = 9 + rank + micro
    return [torch.randn(1, length, 8, generator=generator) for _ in range(3)], torch.ones(1, length)


def ddp_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=2)
    try:
        torch.manual_seed(123)
        connector = Stage1Connector(input_dim=8, token_dim=16, pooled_dim=12)
        model = torch.nn.parallel.DistributedDataParallel(connector, broadcast_buffers=False)
        for micro in range(3):
            with model.no_sync() if micro < 2 else nullcontext():
                tokens, _, pooled = model(*ddp_features(rank, micro))
                ((tokens[..., 0].square().mean() + pooled.square().mean()) / 3).backward()
        torch.save({key: p.grad for key, p in connector.named_parameters()}, Path(directory) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_two_rank_ddp_accumulation_matches_global_batch(tmp_path):
    mp.spawn(ddp_worker, args=(str(tmp_path),), nprocs=2, join=True)
    torch.manual_seed(123)
    reference = Stage1Connector(input_dim=8, token_dim=16, pooled_dim=12)
    for rank in range(2):
        for micro in range(3):
            tokens, _, pooled = reference(*ddp_features(rank, micro))
            ((tokens[..., 0].square().mean() + pooled.square().mean()) / 6).backward()
    a = torch.load(tmp_path / "rank0.pt", weights_only=True)
    b = torch.load(tmp_path / "rank1.pt", weights_only=True)
    for key, parameter in reference.named_parameters():
        assert torch.equal(a[key], b[key]), key
        assert torch.allclose(a[key], parameter.grad, atol=1e-6, rtol=1e-4), key


def test_frozen_hash_detects_change_including_scalar_parameters():
    module = nn.ParameterDict({"scalar": nn.Parameter(torch.tensor(1.))})
    before = stage1.frozen_signatures({"module": module})
    with torch.no_grad():
        module["scalar"].add_(1)
    assert before != stage1.frozen_signatures({"module": module})
