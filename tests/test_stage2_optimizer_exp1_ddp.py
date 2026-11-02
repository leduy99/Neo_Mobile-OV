import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from new_mobile_ov.bridge.stage1_conditioner import Stage1Connector
from new_mobile_ov.training.distributed import DistributedContext
from new_mobile_ov.training import stage2_media_epochs as media
from tools import train_mobileov_stage1 as legacy
from tools import train_mobileov_stage2_optimizer_exp1 as trainer
from test_stage2_training import TinyDiT, scheduler, install_history
from test_stage2_epoch_extension_resume import training_environment, same


class Slots(torch.utils.data.Dataset):
    sources = {}
    def __init__(self, root, *, plan, rank, start_step, **kw):
        self.plan, self.rank, self.start = plan, rank, start_step
        self.records = {"t2i": range(plan.images), "t2v": range(plan.videos)}
    def __getitem__(self, index):
        step, micro = divmod(index, self.plan.accumulation)
        slot = self.plan.slot(step + self.start, micro, self.rank)
        return dict(task=slot["source"], sample_id=f"{slot['source']}:{slot['index']}", **slot)


def stack(args, device):
    cfg = SimpleNamespace(bridge=SimpleNamespace(smolvlm2_ckpt_path=str(Path(args.config).parent / "smol")))
    encoder, vae, dit = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2), TinyDiT().bfloat16()
    for module in (encoder, vae, dit):
        module.eval().requires_grad_(False)
    encoder.processor_sha256 = "processor"
    return cfg, encoder, Stage1Connector(8, 16, 12), dit, vae, scheduler()


def encode(encoder, vae, sample, device, rng, dropout, **kw):
    layers = [torch.randn(1, 9, 8, generator=rng) for _ in range(3)]
    latent = torch.randn(1, 4, 7 if sample["task"] == "t2v" else 1, 16, 24, generator=rng)
    return layers, torch.ones(1, 9), latent, False


def reconstruction(encoder, sample):
    rng = torch.Generator().manual_seed(700 + sample["micro"])
    return [torch.randn(1, 13, 8, generator=rng) for _ in range(3)], torch.ones(1, 13)


def worker(rank, root, name, action):
    torch.set_num_threads(1)
    root = Path(root)
    install_history()
    dist.init_process_group("gloo", init_method=f"file://{root}/init_{name}_{action}", rank=rank, world_size=2)
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(legacy, "load_stack", stack)
            patch.setattr(legacy, "encode_sample", encode)
            patch.setattr(legacy, "verify_release", lambda _: None)
            patch.setattr(legacy, "validate", lambda *a: {"loss": float(torch.rand(()))})
            patch.setattr(media, "MediaEpochDataset", Slots)
            patch.setattr(legacy.stage2, "reconstruction_condition", reconstruction)
            extra = ["--stop-after", "2"] if action == "pause" else (["--resume", str(root / name / "exp1_resume.pt")] if action == "resume" else [])
            args = trainer.parse_args(["--parent-resume", str(root / "parent/stage2_resume.pt"),
                "--output-dir", str(root / name), "--config", str(root / "config"), "--data-root", str(root / "data"),
                "--integration-smoke", "--expected-parent-step", "57", "--steps", "6", "--decay-steps", "6",
                "--workers", "0", "--save-every", "2", "--evaluate-every", "3", "--validate-every", "2", *extra])
            trainer.train(args, DistributedContext(rank, rank, 2, torch.device("cpu")))
    finally:
        dist.destroy_process_group()


def test_actual_two_rank_accumulation_mixed_tasks_ema_broadcast_and_resume(training_environment):
    env = training_environment
    env.run("parent")
    (env.root / "full").mkdir()
    (env.root / "paused").mkdir()
    mp.spawn(worker, args=(str(env.root), "full", "full"), nprocs=2, join=True)
    mp.spawn(worker, args=(str(env.root), "paused", "pause"), nprocs=2, join=True)
    mp.spawn(worker, args=(str(env.root), "paused", "resume"), nprocs=2, join=True)
    a, b = [torch.load(env.root / name / "exp1_resume.pt", weights_only=True) for name in ("full", "paused")]
    same(a, b)
    assert a["exp1_update"] == a["ema"]["updates"] == 6
    assert len(a["rng_states"]) == 2 and a["optimizer"]["param_groups"][0]["lr"] == 0
    rows = [json.loads(line) for line in (env.root / "full/history.jsonl").read_text().splitlines()]
    assert any(row["valid_samples_in_update"]["t2i"] > 0 and
               sum(row["valid_samples_in_update"][k] for k in media.VIDEO_MODES) > 0 for row in rows)
