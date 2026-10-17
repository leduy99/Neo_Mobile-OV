"""Stage-resolution history corruption and runtime evidence for the opt-in fork."""
import math

import torch
import torch.distributed as dist

from new_mobile_ov.training.neodragon_pyramid_flow import corrupt_history

HISTORY_SEED_OFFSET = 777777
MAX_HISTORY_SIGMA = 1 / 3
SOURCE = "https://github.com/jy0205/Pyramid-Flow/blob/a012faa1dc4d71301a7a153c7f9554c081947ea2/pyramid_dit/pyramid_dit_for_video_gen_pipeline.py"


def corrupt_stage_history(history, *, maximum, generator, preserve_first_anchor=False, stats=None):
    if not math.isfinite(maximum) or not 0 <= maximum <= MAX_HISTORY_SIGMA:
        raise ValueError("History noise must be within the published [0, 1/3] range")
    protected = int(preserve_first_anchor and bool(history))
    eligible = [(i, value[:, :, protected if i == 0 else 0:]) for i, value in enumerate(history)]
    eligible = [(i, value) for i, value in eligible if value.shape[2]]
    batch = history[0].shape[0] if history else 0
    count = sum(value.shape[0] * value.shape[2] for _, value in eligible)
    if stats is not None:
        stats.update(eligible_units=count, protected_units=protected * batch, applied_units=0,
                     sigma_sum=0., sigma_samples=0, change_squared_sum=0., changed_elements=0)
    if not maximum or not eligible:
        return history
    if generator is None:
        raise ValueError("History corruption requires an independent reproducible RNG")
    amount = torch.rand(batch, device=history[0].device, generator=generator) * maximum
    clean = [value for _, value in eligible]
    noisy = corrupt_history(clean, sigma=amount, generator=generator)
    result = list(history)
    for (index, _), value in zip(eligible, noisy):
        result[index] = torch.cat((history[0][:, :, :1], value), dim=2) if protected and index == 0 else value
    if stats is not None:
        stats.update(applied_units=count, sigma_sum=amount.sum().detach(), sigma_samples=batch,
                     change_squared_sum=sum((a.float() - b.float()).square().sum().detach() for a, b in zip(clean, noisy)),
                     changed_elements=sum(value.numel() for value in clean))
    return result


def history_contract(maximum):
    return dict(max_sigma=float(maximum), distribution="uniform_per_sample_sigma; independent_Gaussian_per_history_chunk",
                location="after_native_temporal_spatial_pyramid; before_DiT",
                mixture="sigma*Gaussian + (1-sigma)*clean_history", source=SOURCE,
                rng_seed_offset=HISTORY_SEED_OFFSET, current_unit_target="unchanged",
                observed_anchor="keep_unit0_clean_only_in_i2v_anchor; task_specific_adaptation",
                validation="clean_history; matched_to_parent", diffusion_forcing=False)


def summarize_history(records, ctx, maximum):
    keys = ("eligible_units", "applied_units", "protected_units", "sigma_sum", "sigma_samples",
            "change_squared_sum", "changed_elements")
    totals = torch.zeros(len(keys), device=ctx.device, dtype=torch.float64)
    for record in records:
        if any(key not in record for key in keys):
            raise RuntimeError("History corruption audit missing: the actual forward path did not report its inputs")
        for index, key in enumerate(keys):
            totals[index] += torch.as_tensor(record[key], device=ctx.device, dtype=torch.float64)
    if ctx.is_distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    eligible, applied, protected, sigma, samples, square, elements = totals.tolist()
    if applied != eligible or not all(math.isfinite(value) for value in totals.tolist()):
        raise RuntimeError("History corruption was not applied to every eligible past unit, or is non-finite")
    return dict(enabled=True, max_sigma=float(maximum), eligible_units=int(eligible), applied_units=int(applied),
                preserved_anchor_units=int(protected), sigma_mean=sigma / samples if samples else None,
                changed_rms=math.sqrt(square / elements) if elements else None)
