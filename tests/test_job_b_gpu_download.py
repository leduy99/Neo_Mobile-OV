import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.utils import gpu_status_heartbeat as heartbeat

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/download_mobileov_job_b_data_1node1gpu.sbatch"


class FakeCUDA:
    def __init__(self, count=1, available=True):
        self.count, self.available = count, available
        self.initialized, self.synchronized = False, False

    def is_available(self):
        return self.available

    def device_count(self):
        return self.count

    def init(self):
        self.initialized = True

    def synchronize(self, device):
        assert device == 0
        self.synchronized = True

    def mem_get_info(self, device):
        return 3 * 1024**3, 4 * 1024**3

    def get_device_name(self, device):
        return "fixture-GPU"


def test_status_heartbeat_no_dummy_gpu_work(tmp_path, capsys):
    cuda = FakeCUDA()
    stop = SimpleNamespace(is_set=lambda: False, wait=lambda _: True)
    args = heartbeat.parse_args(["--ready-file", str(tmp_path / "ready")])
    heartbeat.run(args, cuda=cuda, stop=stop)
    report = json.loads(capsys.readouterr().out)
    assert cuda.initialized and cuda.synchronized
    assert report["device"] == "fixture-GPU" and not report["artificial_gpu_work"]
    assert report["free_gib"] == 3 and not args.ready_file.exists()
    assert "matmul" not in Path(heartbeat.__file__).read_text()


@pytest.mark.parametrize("cuda", [FakeCUDA(count=0), FakeCUDA(count=2), FakeCUDA(available=False)])
def test_heartbeat_requires_one_cuda_device(tmp_path, cuda):
    args = heartbeat.parse_args(["--ready-file", str(tmp_path / "ready")])
    with pytest.raises(RuntimeError, match="one visible"):
        heartbeat.run(args, cuda=cuda)
    assert not args.ready_file.exists()


@pytest.mark.parametrize("interval", ["nan", "inf", "0", "-1"])
def test_invalid_interval(tmp_path, interval):
    with pytest.raises(SystemExit):
        heartbeat.parse_args(["--ready-file", str(tmp_path / "ready"), "--interval", interval])


@pytest.fixture
def launch_environment(tmp_path):
    (tmp_path / "scripts").mkdir()
    shutil.copy(ROOT / "scripts/download_mobileov_job_b_data_1nodecpu.sbatch", tmp_path / "scripts")
    binary = tmp_path / "bin"
    binary.mkdir()
    mock_python = binary / "mock-python"
    mock_python.write_text(f"#!{sys.executable}\n" + '''
import json, os, signal, sys, time
from pathlib import Path
root = Path(os.environ["PROJECT_ROOT"])
if "-c" in sys.argv:
    sys.exit(0)
if "tools/utils/gpu_status_heartbeat.py" in sys.argv:
    (root / "monitor.pid").write_text(str(os.getpid()))
    if os.environ.get("MONITOR_ACTION") == "fail-start":
        sys.exit(6)
    ready = Path(sys.argv[sys.argv.index("--ready-file") + 1])
    ready.write_text("ready")
    def stop(*_):
        ready.unlink(missing_ok=True)
        (root / "monitor_stopped").touch()
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    while True:
        if os.environ.get("MONITOR_ACTION") == "fail-mid" and (root / "download_started").exists():
            sys.exit(9)
        time.sleep(.02)
else:
    (root / "download.pid").write_text(str(os.getpid()))
    (root / "download_started").touch()
    (root / "download_args.json").write_text(json.dumps(sys.argv[1:]))
    time.sleep(float(os.environ.get("DOWNLOAD_DELAY", ".05")))
    sys.exit(int(os.environ.get("DOWNLOAD_EXIT", "0")))
''')
    mock_python.chmod(0o755)
    srun = binary / "srun"
    srun.write_text('#!/usr/bin/env bash\nshift\nexec "$@"\n')
    srun.chmod(0o755)
    return dict(os.environ, PROJECT_ROOT=str(tmp_path), TMPDIR=str(tmp_path), PYTHON=str(mock_python),
                PATH=f"{binary}:{os.environ['PATH']}", SLURM_JOB_ID="123", SLURM_JOB_RESERVATION="safe",
                DOWNLOAD_WORKERS="8")


@pytest.mark.parametrize("exit_code", [0, 7])
def test_gpu_launcher_cleanup_and_exit_status(tmp_path, launch_environment, exit_code):
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    script = LAUNCHER.read_text()
    assert "#SBATCH --gpus=1" in script and "#SBATCH --reservation=safe" in script
    assert "tools/utils/gpu_heartbeat.py" not in script
    env = dict(launch_environment, DOWNLOAD_EXIT=str(exit_code))
    result = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == exit_code, result.stderr
    assert (tmp_path / "monitor_stopped").exists()
    assert not list(tmp_path.glob("mov-jobb-heartbeat.*"))
    argv = json.loads((tmp_path / "download_args.json").read_text())
    assert argv[argv.index("--workers") + 1] == "8"


@pytest.mark.parametrize("action", ["fail-start", "fail-mid"])
def test_failed_monitor_stops_download(tmp_path, launch_environment, action):
    result = subprocess.run(["bash", str(LAUNCHER)],
                            env=dict(launch_environment, MONITOR_ACTION=action, DOWNLOAD_DELAY="5"),
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    if action == "fail-start":
        assert not (tmp_path / "download_started").exists()
    else:
        assert (tmp_path / "download_started").exists()
        with pytest.raises(ProcessLookupError):
            os.kill(int((tmp_path / "download.pid").read_text()), 0)
    assert not list(tmp_path.glob("mov-jobb-heartbeat.*"))


def test_gpu_launcher_rejects_missing_safe_reservation(tmp_path, launch_environment):
    env = dict(launch_environment, SLURM_JOB_RESERVATION="")
    result = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 1 and "reservation=safe" in result.stderr
    assert not (tmp_path / "download_started").exists()
