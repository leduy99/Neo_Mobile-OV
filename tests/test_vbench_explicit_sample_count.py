import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


@pytest.fixture
def evaluator(monkeypatch):
    fake = ModuleType("vbench")

    class Base:
        def __init__(self, device, info, output):
            self.full_info_dir, self.output_path = info, output

        def build_full_info_json(self, *args, **kwargs):
            return "official-five-sample-method"

    fake.VBench = Base
    monkeypatch.setitem(sys.modules, "vbench", fake)
    path = Path(__file__).resolve().parents[1] / "tools/evaluate_vbench_resume.py"
    spec = importlib.util.spec_from_file_location("explicit_sample_count", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_one_sample_retains_auxiliary_info_and_excludes_extra_seeds(tmp_path, evaluator):
    info = tmp_path / "info.json"
    info.write_text(json.dumps([
        dict(prompt_en="panda", dimension=["object_class"], auxiliary_info={"object": "panda"}),
        dict(prompt_en="car", dimension=["color"]),
    ]))
    for index in range(5):
        (tmp_path / f"panda-{index}.mp4").write_bytes(b"video")
    model = evaluator.SampleCountVBench("cpu", str(info), str(tmp_path), samples_per_prompt=1)
    target = model.build_full_info_json(str(tmp_path), "object_class", ["object_class"])
    rows = json.loads(Path(target).read_text())
    assert len(rows) == 1 and rows[0]["auxiliary_info"] == {"object": "panda"}
    assert rows[0]["video_list"] == [str(tmp_path / "panda-0.mp4")]
    assert not (tmp_path / "car-0.mp4").exists()


def test_five_samples_delegate_to_official_vbench(tmp_path, evaluator):
    model = evaluator.SampleCountVBench("cpu", "unused.json", str(tmp_path), samples_per_prompt=5)
    assert model.build_full_info_json(str(tmp_path), "scene", ["scene"]) == "official-five-sample-method"


def test_missing_requested_sample_is_not_silently_ignored(tmp_path, evaluator):
    info = tmp_path / "info.json"
    info.write_text(json.dumps([dict(prompt_en="panda", dimension=["scene"])]))
    model = evaluator.SampleCountVBench("cpu", str(info), str(tmp_path), samples_per_prompt=1)
    with pytest.raises(RuntimeError, match="Missing requested samples"):
        model.build_full_info_json(str(tmp_path), "scene", ["scene"])
