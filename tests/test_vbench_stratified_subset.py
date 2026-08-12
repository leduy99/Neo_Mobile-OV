from __future__ import annotations

from tools.prepare_vbench_stratified_subset import select_prompts


def test_selection_is_deterministic_and_covers_each_dimension_group() -> None:
    prompts = {
        "prompt_a": ("object_class",),
        "prompt_b": ("object_class",),
        "prompt_c": ("scene",),
        "prompt_d": ("scene",),
        "prompt_e": ("temporal_style",),
        "prompt_f": ("temporal_style",),
    }

    first, first_groups = select_prompts(prompts, count=6, seed=20260812)
    second, second_groups = select_prompts(prompts, count=6, seed=20260812)

    assert first == second
    assert first_groups == second_groups
    assert len(first) == 6
    assert len(set(first)) == 6
    assert set(first_groups) == {"object_class", "scene", "temporal_style"}
    assert all(len(values) == 2 for values in first_groups.values())
