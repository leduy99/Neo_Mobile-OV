"""Extend the original media-epoch recipe without restarting Adam or warmup."""
import math

import torch

from new_mobile_ov.training.stage2_alignment import FORMAT, reconstruction_contract
from new_mobile_ov.training.stage2_media_epochs import MediaEpochPlan, METRIC_TASKS


def validate_extension(previous, target, step, *, repartition=False):
    old, new = previous.get("media_epochs", {}), target.get("media_epochs", {})
    if (previous.get("format") != FORMAT or target.get("format") != FORMAT
            or old.get("epochs") not in (3, 4, 5) or new.get("epochs") not in (4, 5)
            or old["epochs"] > new["epochs"]):
        raise ValueError("Extend only the original three-epoch Stage 2 to four or five total epochs")
    if (previous.get("initialization") != "stage1_connector_and_released_multistep_dit"
            or previous.get("scope") != "fresh_multimodal_stage2; public_data; not_exact_UniVideo_recipe"
            or previous.get("trainable_components") != ["connector", "dit"]
            or previous.get("frozen_components") != ["smolvlm2", "vae"]
            or previous.get("auxiliary_reconstruction") != reconstruction_contract(.1)
            or previous.get("video_units") != "T2V/reference:0_to6; anchor:1_to6; teacher_forced_causal_history"
            or previous.get("tasks") != list(METRIC_TASKS)
            or previous.get("lr") != 2e-5 or previous.get("warmup") != 50
            or previous.get("ema") is not False
            or any(key in previous for key in ("first_frame_focus", "history_corruption", "parent_contract", "lora_spec"))):
        raise ValueError("Not the unchanged original recipe; never resume Job A/B/history as an epoch extension")
    keys = ("images", "videos", "seed")
    if any(old.get(key) != new.get(key) for key in keys):
        raise ValueError("Epoch extension must keep the media counts and seed")
    layout_changed = any(old.get(key) != new.get(key) for key in ("world_size", "accumulation"))
    if layout_changed:
        if (not repartition or old["epochs"] != 3 or new["world_size"] != 2 * old["world_size"]
                or old["accumulation"] != 2 * new["accumulation"]
                or old["global_batch"] != new["global_batch"]
                or new.get("rng_layout", {}).get("world_size") != old.get("rng_layout", {}).get("world_size", old["world_size"])):
            raise ValueError("DDP repartition requires explicit opt-in, doubled GPUs and the same global batch/sample RNG")
    elif old.get("rng_layout") != new.get("rng_layout"):
        raise ValueError("Own-run resume must preserve the sample RNG layout")
    keys = (*keys, "world_size", "accumulation")
    for actual in (old, new):
        expected = MediaEpochPlan(**{key: actual[key] for key in keys}, epochs=actual["epochs"],
                                  allow_partial_cycle=True,
                                  rng_world_size=actual.get("rng_layout", {}).get("world_size"))
        if actual != expected.contract():
            raise ValueError("Epoch plan changed beyond a total-epoch extension")
    per_epoch = old["steps_per_epoch"]
    if (previous.get("steps") != old["steps"] or target.get("steps") != new["steps"]
            or type(step) is not int or not 3 * per_epoch <= step <= previous["steps"]
            or step >= target["steps"]):
        raise ValueError("Resume must start at completed epoch 3 or an unfinished epoch-4/5 extension")
    changed = {key for key in set(previous) | set(target) if previous.get(key) != target.get(key)}
    allowed = {"steps", "media_epochs"} | ({"world_size", "accumulation"} if repartition else set())
    if changed - allowed:
        raise ValueError(f"Epoch extension changed the training recipe: {sorted(changed)}")
    for contract, plan in ((previous, old), (target, new)):
        if contract.get("world_size") != plan["world_size"] or contract.get("accumulation") != plan["accumulation"]:
            raise ValueError("Runtime DDP layout differs from the epoch plan")


def optimizer_audit(state, step, lr, parameters=None):
    if not isinstance(state, dict):
        raise ValueError("Missing original Adam state; use stage2_resume.pt, not inference weights")
    groups, moments = state.get("param_groups", []), state.get("state", {})
    if len(groups) != 1 or not moments:
        raise ValueError("Missing original Adam state; use stage2_resume.pt, not inference weights")
    group = groups[0]
    if (group.get("lr") != lr or tuple(group.get("betas", ())) != (.9, .95)
            or group.get("eps") != 1e-8 or group.get("weight_decay") != 0):
        raise ValueError("Saved Adam learning rate/settings differ from the original recipe")
    ids = group.get("params", [])
    if len(ids) != len(set(ids)) or not set(moments).issubset(ids):
        raise ValueError("Invalid Adam parameter mapping")
    parameters = list(parameters) if parameters is not None else None
    if parameters is not None and len(parameters) != len(ids):
        raise ValueError("Adam parameters differ from the trainable bridge and DiT")
    shapes = dict(zip(ids, (parameter.shape for parameter in parameters))) if parameters is not None else {}
    updates = []
    for key, value in moments.items():
        if not isinstance(value, dict):
            raise ValueError("Invalid Adam state entry")
        counter = value.get("step")
        counter = counter.item() if isinstance(counter, torch.Tensor) and counter.numel() == 1 else counter
        if not isinstance(counter, (int, float)) or not math.isfinite(counter) or not 0 < counter <= step or int(counter) != counter:
            raise ValueError("Invalid Adam update counter")
        first, second = value.get("exp_avg"), value.get("exp_avg_sq")
        if (not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor)
                or first.dtype != torch.float32 or second.dtype != torch.float32
                or first.shape != second.shape or (key in shapes and first.shape != shapes[key])):
            raise ValueError("Missing or incompatible FP32 Adam moments")
        if not bool(torch.isfinite(first).all()) or not bool(torch.isfinite(second).all()) or bool((second < 0).any()):
            raise ValueError("Non-finite or invalid Adam moments")
        updates.append(int(counter))
    if max(updates) != step:
        raise ValueError("Adam counters do not reach the checkpoint step; refuse a restarted optimizer")
    return dict(parameters_with_state=len(moments), min_adam_step=min(updates),
                max_adam_step=max(updates), lr=group["lr"], optimizer_restored=True, warmup_restarted=False)


def contract_from_resume(saved, current, plan, signatures, *, repartition=False):
    previous = saved.get("contract", {})
    expected = dict(current, initial_weights_sha256=signatures)
    keys = ("format", "data_summary_sha256", "config_sha256", "processor", "processor_sha256",
            "frozen_weights_sha256", "initial_weights_sha256", "accumulation", "world_size", "lr", "warmup",
            "condition_dropout", "short_side", "long_side", "max_tokens", "seed", "geometry", "temporal",
            "vae_window_size", "optimizer", "objective", "reconstruction")
    for key in keys:
        if repartition and key in ("world_size", "accumulation"):
            continue
        if previous.get(key) != expected.get(key):
            raise ValueError(f"Original recipe/runtime mismatch: {key}")
    target = dict(previous, steps=plan.steps, media_epochs=plan.contract(),
                  world_size=current["world_size"], accumulation=current["accumulation"])
    validate_extension(previous, target, saved.get("step"), repartition=repartition)
    optimizer_audit(saved.get("optimizer", {}), saved["step"], previous["lr"])
    return target
