"""Stateful optimizer-only continuation; leave all media/flow targets unchanged."""
from contextlib import contextmanager
import copy
import math

import torch

from new_mobile_ov.training import stage2_alignment as stage2
from new_mobile_ov.training import stage2_epoch_extension as extension
from new_mobile_ov.training import stage2_media_epochs as media

FORMAT = "mobileov_stage2_optimizer_exp1_resume_v1"
POLICY = "epoch3_adam_resume_batch64_linear_decay_ema_v1"
PARENT_STEP = 283977
UPDATES = 20000


def learning_rate(update, *, total=UPDATES, decay_steps=UPDATES, base=2e-5):
    if not 0 <= update <= total or not 1 <= decay_steps <= total or not math.isfinite(base) or base <= 0:
        raise ValueError("Invalid linear-decay schedule position/settings")
    return base * min(1., (total - update) / decay_steps)


def parent_plan(contract, *, total=UPDATES, global_batch=64):
    old = contract["media_epochs"]
    old_batch = old["global_batch"]
    if global_batch % old_batch:
        raise ValueError("New batch must group complete original global batches")
    end = contract["steps"] + total * (global_batch // old_batch)
    return media.MediaEpochPlan(old["images"], old["videos"], old["world_size"], old["accumulation"],
        epochs=math.ceil(end / old["steps_per_epoch"]), seed=old["seed"], allow_partial_cycle=True,
        rng_world_size=old.get("rng_layout", {}).get("world_size"))


def validate_parent(saved, *, expected_step=PARENT_STEP, parameters=None):
    contract = saved.get("contract", {})
    if saved.get("format") != stage2.FORMAT or saved.get("step") != expected_step:
        raise ValueError(f"Exp 1 requires the original epoch-3 optimizer at step {expected_step}")
    plan = contract.get("media_epochs", {})
    if contract.get("steps") != expected_step or plan.get("epochs") != 3 or 3 * plan.get("steps_per_epoch", 0) != expected_step:
        raise ValueError("Parent must be the completed original three-media-epoch run")
    # Reuse the original-recipe gate without weakening any legacy resume guards.
    target_plan = media.MediaEpochPlan(plan["images"], plan["videos"], plan["world_size"],
        plan["accumulation"], epochs=5, seed=plan["seed"], allow_partial_cycle=True,
        rng_world_size=plan.get("rng_layout", {}).get("world_size"))
    target = dict(contract, steps=target_plan.steps, media_epochs=target_plan.contract())
    extension.validate_extension(contract, target, expected_step)
    if not saved.get("connector") or not saved.get("dit") or not saved.get("connector_spec"):
        raise ValueError("Missing original connector/DiT weights")
    return extension.optimizer_audit(saved.get("optimizer"), expected_step, 2e-5, parameters)


class GroupedMediaDataset(torch.utils.data.Dataset):
    """Each new update consumes consecutive old batches, retaining sample/noise IDs."""
    def __init__(self, root, *, parent, rank, world_size, accumulation=4, total=UPDATES,
                 start_update=0, short_side=320, long_side=512):
        old = parent["media_epochs"]
        if (world_size not in (old["world_size"], old["global_batch"])
                or world_size * accumulation != 4 * old["global_batch"]
                or not 0 <= rank < world_size or not 0 <= start_update < total):
            raise ValueError("Group four original batches per update using the original or doubled GPU count")
        self.total, self.start, self.accumulation = total, start_update, accumulation
        self.parent_accumulation = old["accumulation"]
        self.virtual_rank = rank % old["world_size"]
        self.world_size, self.parent_world_size = world_size, old["world_size"]
        self.microsteps_per_old_batch = old["global_batch"] // world_size
        self.rank = rank
        self.plan = parent_plan(parent, total=total, global_batch=world_size * accumulation)
        self.original = media.MediaEpochDataset(root, plan=self.plan, rank=self.virtual_rank,
            start_step=parent["steps"], short_side=short_side, long_side=long_side)
        self.sources, self.records = self.original.sources, self.original.records

    def __len__(self):
        return (self.total - self.start) * self.accumulation

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        old_offset, part = divmod(self.start * self.accumulation + index, self.microsteps_per_old_batch)
        virtual_micro = (part * self.world_size + self.rank) // self.parent_world_size
        sample = self.original[old_offset * self.parent_accumulation + virtual_micro]
        return dict(sample, rng_rank=sample.get("rng_rank", self.virtual_rank))


class FP32EMA:
    """Rank-zero CPU shadow; no second DiT or optimizer allocation on a GPU."""
    def __init__(self, modules, decay=.9995):
        if not math.isfinite(decay) or not 0 < decay < 1:
            raise ValueError("EMA decay must lie in (0,1)")
        self.decay, self.updates = decay, 0
        self.shadow = {name: {key: value.detach().to(device="cpu", dtype=torch.float32).clone()
                             if value.is_floating_point() else value.detach().cpu().clone()
                             for key, value in module.state_dict().items()}
                       for name, module in modules.items()}

    @torch.no_grad()
    def update(self, modules):
        for name, module in modules.items():
            state = module.state_dict()
            if state.keys() != self.shadow[name].keys():
                raise ValueError("EMA architecture changed")
            for key, value in state.items():
                target = self.shadow[name][key]
                value = value.detach().to(device="cpu", dtype=target.dtype)
                if target.is_floating_point():
                    target.lerp_(value, 1 - self.decay)
                else:
                    target.copy_(value)
        self.updates += 1

    def state_dict(self):
        return dict(decay=self.decay, updates=self.updates, shadow=self.shadow)

    def load_state_dict(self, saved, *, expected_updates):
        if saved.get("decay") != self.decay or saved.get("updates") != expected_updates:
            raise ValueError("EMA decay/update counter differs; do not restart EMA on resume")
        state = saved.get("shadow", {})
        if state.keys() != self.shadow.keys():
            raise ValueError("EMA components differ")
        for name, values in self.shadow.items():
            if state[name].keys() != values.keys():
                raise ValueError("EMA architecture differs")
            for key, target in values.items():
                value = state[name][key]
                if value.shape != target.shape or value.dtype != target.dtype or not bool(torch.isfinite(value).all()):
                    raise ValueError("Invalid EMA tensor")
                target.copy_(value)
        self.updates = expected_updates


def cpu_weights(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


@contextmanager
def ema_weights(modules, ema, ctx):
    """Swap only for validation; restore raw trainable weights even on failure."""
    backup = {name: cpu_weights(module) for name, module in modules.items()}
    try:
        for name, module in modules.items():
            if ctx.is_main:
                module.load_state_dict(ema.shadow[name], strict=True)
            if ctx.is_distributed:
                for tensor in module.state_dict().values():
                    torch.distributed.broadcast(tensor, src=0)
        yield
    finally:
        for name, module in modules.items():
            module.load_state_dict(backup[name], strict=True)


def continuation_contract(parent, *, parent_sha256, world_size, accumulation, total, decay_steps, ema_decay):
    old = parent["media_epochs"]
    if (world_size not in (old["world_size"], old["global_batch"])
            or world_size * accumulation != 4 * old["global_batch"]):
        raise ValueError("Exp 1 requires four original batches per update; use 16 GPUs x4 or 8 GPUs x8")
    consumed_batches = total * 4
    contract = copy.deepcopy(parent)
    contract.update(scope=POLICY, parent_contract=copy.deepcopy(parent), parent_resume_sha256=parent_sha256,
        steps=parent["steps"] + total, world_size=world_size, accumulation=accumulation, warmup=0,
        initialization="epoch3_weights_and_original_Adam_moments", ema=dict(decay=ema_decay,
            dtype="float32", device="cpu_rank0", initialized_from="epoch3_raw_weights"),
        lr_schedule=dict(kind="constant_then_linear", base_lr=2e-5, warmup_steps=0,
                         total_updates=total, decay_steps=decay_steps, final_lr=0.),
        optimizer_exp1=dict(policy=POLICY, parent_step=parent["steps"], total_updates=total,
            global_batch=world_size * accumulation, original_global_batch=old["global_batch"],
            updates_are="additional_optimizer_updates; not_original_media_steps",
            sampler="four_consecutive_original_global_batches_per_update; source_and_loss_weights_unchanged",
            original_media_batches_consumed=consumed_batches,
            media_presentations_including_zero_weight_padding=total * world_size * accumulation,
            bitwise_parent_continuation=False, default_inference_weights="ema"))
    # Keep the learned I2V routes discoverable without labelling 20K updates a full epoch.
    contract["media_epochs"] = dict(recipe="media_epochs_i2v_v1", continuation_policy=POLICY,
        global_batch=world_size * accumulation, video_modes=list(media.VIDEO_MODES),
        original_steps_per_epoch=old["steps_per_epoch"], start_media_cursor=parent["steps"],
        end_media_cursor=parent["steps"] + consumed_batches, complete_media_passes=False)
    return contract


def media_cursor(contract, update):
    spec = contract["optimizer_exp1"]
    return spec["parent_step"] + update * (spec["global_batch"] // spec["original_global_batch"])


def validate_own_resume(saved, contract, *, parameters=None):
    if saved.get("format") != FORMAT or saved.get("contract") != contract:
        raise ValueError("Not an exact own-run Exp-1 resume; recipe or source changed")
    count = saved.get("exp1_update")
    total = contract["optimizer_exp1"]["total_updates"]
    origin = contract["optimizer_exp1"]["parent_step"]
    if type(count) is not int or not 0 < count < total or saved.get("step") != origin + count:
        raise ValueError("Invalid or already-completed Exp-1 resume position")
    schedule = contract["lr_schedule"]
    if saved.get("scheduler") != dict(schedule, completed_updates=count):
        raise ValueError("Missing or inconsistent LR scheduler state")
    if saved.get("sampler_cursor") != media_cursor(contract, count):
        raise ValueError("Missing or inconsistent dataset cursor")
    if saved.get("ema", {}).get("updates") != count or saved["ema"].get("decay") != contract["ema"]["decay"]:
        raise ValueError("Missing or inconsistent EMA state")
    if len(saved.get("rng_states", [])) != contract["world_size"]:
        raise ValueError("Own-run resume requires every rank's RNG state")
    return extension.optimizer_audit(saved.get("optimizer"), origin + count,
        learning_rate(count, total=total, decay_steps=schedule["decay_steps"]), parameters)
