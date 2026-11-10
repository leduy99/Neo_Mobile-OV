from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from tools.data_prepare import run_mobileov_exp2_data as launcher


def test_existing_pyav_needs_no_install_and_paths_are_role_specific(tmp_path, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="cpython-310\n" if "cache_tag" in command[-1]
                               else "17.0.0\n", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    original = {"PYTHONPATH": "/scene:/repo", "PYTHONNOUSERSITE": "1"}
    quality = launcher.prepare_runtime_env("/quality/python", original, tmp_path, "quality")
    caption = launcher.prepare_runtime_env("/caption/python", original, tmp_path, "caption")
    assert quality["PYTHONPATH"] == f"{tmp_path}/runtime/quality_cpython-310:/scene:/repo"
    assert caption["PYTHONPATH"] == f"{tmp_path}/runtime/caption_cpython-310:/scene:/repo"
    assert original["PYTHONPATH"] == "/scene:/repo"
    assert not any("pip" in command for command, _ in calls)


def test_missing_pyav_installs_wheel_only_in_isolated_runtime(tmp_path, monkeypatch):
    calls, probes = [], []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if "cache_tag" in command[-1]:
            return SimpleNamespace(stdout="cpython-311\n")
        if "pip" in command:
            return SimpleNamespace(returncode=0)
        probes.append(command)
        return SimpleNamespace(returncode=1 if len(probes) == 1 else 0,
                               stdout="16.1.0\n" if len(probes) > 1 else "",
                               stderr="ModuleNotFoundError: No module named 'av'" if len(probes) == 1 else "")

    monkeypatch.setattr(subprocess, "run", run)
    env = launcher.prepare_runtime_env("/quality/python", {"PYTHONPATH": "/repo"}, tmp_path, "quality")
    installs = [(command, kwargs) for command, kwargs in calls if "pip" in command]
    assert len(installs) == 1 and len(probes) == 2
    command, kwargs = installs[0]
    assert command[:4] == ["/quality/python", "-m", "pip", "install"]
    assert "--no-deps" in command and "--only-binary=:all:" in command
    assert command[command.index("--target") + 1] == str(tmp_path / "runtime/quality_cpython-311")
    assert command[-1] == "av==16.1.0"
    assert kwargs["env"]["PYTHONPATH"] == "/repo"
    assert env["PYTHONPATH"].endswith(":/repo")


def test_failed_install_stops_before_workers(tmp_path, monkeypatch):
    def run(command, **kwargs):
        if "cache_tag" in command[-1]:
            return SimpleNamespace(stdout="cpython-310\n")
        if "pip" in command:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(returncode=1, stderr="No module named 'av'")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        launcher.prepare_runtime_env("/quality/python", {}, tmp_path, "quality")


def test_reader_imports_are_checked_in_both_worker_environments(tmp_path, monkeypatch):
    args = SimpleNamespace(quality_python="/quality/python", caption_python="/caption/python")
    calls = []
    monkeypatch.setattr(launcher, "prepare_runtime_env", lambda python, env, output, role: {"ROLE": role})
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    environments = launcher.preflight_workers(args, {}, tmp_path, ("quality", "caption"))
    assert set(environments) == {"quality", "caption"}
    assert [kwargs["env"]["ROLE"] for _, kwargs in calls] == ["quality", "caption"]
    assert all("import tools.data_prepare.prepare_mobileov_exp2" in command[-1] for command, _ in calls)
    assert all("Exp2 imported an unrelated tools package" in command[-1] for command, _ in calls)


def test_quality_cannot_start_when_caption_preflight_fails(tmp_path, monkeypatch):
    args = SimpleNamespace(quality_python="/quality/python", caption_python="/caption/python")

    def prepare(python, env, output, role):
        if role == "caption":
            raise RuntimeError("missing caption dependency")
        return {}

    monkeypatch.setattr(launcher, "prepare_runtime_env", prepare)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="caption"):
        launcher.preflight_workers(args, {}, tmp_path, ("quality", "caption"))


def test_worker_launches_use_role_specific_runtime_and_preflight_comes_first():
    source = Path(launcher.__file__).read_text()
    assert "dict(environments[phase], CUDA_VISIBLE_DEVICES=" in source
    assert source.index("environments = preflight_workers(") < source.index("for phase in phases:", source.index("def main():"))
