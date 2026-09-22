"""Execute the production shell wrappers with CPU-only stand-ins for GPU programs."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools.generate_vbench_mobileov_stage2 import parse_args

ROOT = Path(__file__).resolve().parents[1]
BASE = "vbench_mobileov_stage2_100k_1node1gpu.sbatch"
WRAPPER = "vbench_mobileov_stage2_reconstruction150k_1node1gpu.sbatch"
HF_FILE = "mobileov_stage2_reconstruction150k/17558065/stage2_dit_connector_latest.pt"
SHA256 = "2d53da4aa25285eeb2e4c05fd019398396f8434121248bbff028254abe7fd263"


@pytest.fixture
def launcher(tmp_path):
    project = tmp_path / "project"
    for directory in ("tools", "scripts", "logs"):
        (project / directory).mkdir(parents=True)
    for script in (BASE, WRAPPER):
        (project / "scripts" / script).write_text((ROOT / "scripts" / script).read_text())
    (project / "tools/generate_vbench_mobileov_stage2.py").touch()
    python = tmp_path / "bin/python"
    python.parent.mkdir()
    python.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
from pathlib import Path
with Path(os.environ["CALLS"]).open("a") as stream:
    stream.write(json.dumps({"args": sys.argv[1:], "cwd": os.getcwd()}) + "\\n")
if Path(sys.argv[1]).name == "evaluate_vbench_resume.py":
    directory = Path(sys.argv[sys.argv.index("--output") + 1]) / "vbench"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text('{"total_score": 0.5}')
''')
    python.chmod(0o755)
    vbench = tmp_path / "vbench"
    (vbench / "vbench").mkdir(parents=True)
    (vbench / "vbench/VBench_full_info.json").write_text("[]")
    env = {k: v for k, v in os.environ.items() if k not in (
        "HF_FILE", "HF_REPO", "HF_REVISION", "EXPECTED_STEP", "EXPECTED_SHA256", "CHECKPOINT",
        "RUN_ROOT", "SEED", "SAMPLES_PER_PROMPT", "MAX_PROMPTS", "VBENCH_INFO", "CONFIG")}
    env.update(PROJECT_ROOT=str(project), SLURM_JOB_ID="test999", SKIP_MODULE_LOAD="1", SKIP_VBENCH_INSTALL="1",
               PYTHON_BIN=str(python), VBENCH_ENV=str(tmp_path), VBENCH_REPO=str(vbench),
               HF_HOME=str(tmp_path / "hf"), TMPDIR=str(tmp_path / "tmp"), CALLS=str(tmp_path / "calls"))
    return project, env


def launch(project, env, script=WRAPPER):
    result = subprocess.run(["bash", str(project / "scripts" / script)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return [json.loads(line) for line in Path(env["CALLS"]).read_text().splitlines()]


def generation_args(calls):
    argv = next(c["args"] for c in calls if Path(c["args"][0]).name == "generate_vbench_mobileov_stage2.py")
    return parse_args(argv[1:])


def test_reconstruction_wrapper_pins_model_and_keeps_100k_protocol(launcher):
    project, env = launcher
    # Stale model settings must not turn this into the text-only 150k experiment.
    env.update(HF_FILE="wrong.pt", HF_REPO="wrong/repo", HF_REVISION="main", EXPECTED_STEP="100000",
               EXPECTED_SHA256="0" * 64)
    calls = launch(project, env)
    args = generation_args(calls)
    assert args.hf_repo == "Amshaker/Mobile-OV" and args.hf_file == HF_FILE
    assert args.hf_revision == "dde3f1b64e3ca35f066b100469407d146ca44c99"
    assert args.expected_step == 150000 and args.expected_sha256 == SHA256
    assert str(args.checkpoint) == f"checkpoints/hf_mobile_ov/{HF_FILE}"
    assert (args.samples_per_prompt, args.max_prompts, args.seed) == (1, 0, 20260911)
    assert (args.frames, args.height, args.width, args.fps) == (49, 320, 512, 24)
    assert (args.first_steps, args.video_steps, args.guidance, args.video_guidance) == (20, 10, 7, 5)
    assert args.output_dir == project / "output/vbench_mobileov_stage2_reconstruction150k_1x"
    evaluator = next(c for c in calls if Path(c["args"][0]).name == "evaluate_vbench_resume.py")
    assert evaluator["cwd"] == env["VBENCH_REPO"]
    assert "--expected-samples-per-prompt" in evaluator["args"]
    reporter = next(c for c in calls if Path(c["args"][0]).name == "report_vbench_low_prompts.py")
    assert "--failure-dir" in reporter["args"] and "--top-k" in reporter["args"]


def test_local_checkpoint_is_preferred_over_download(launcher):
    project, env = launcher
    path = project / "output" / HF_FILE
    path.parent.mkdir(parents=True)
    path.write_text("weights")
    args = generation_args(launch(project, env))
    assert str(args.checkpoint) == f"output/{HF_FILE}"
    assert args.expected_sha256 == SHA256


def test_smoke_subset_never_claims_full_vbench(launcher):
    project, env = launcher
    calls = launch(project, dict(env, MAX_PROMPTS="2"))
    assert generation_args(calls).max_prompts == 2
    assert not any(Path(c["args"][0]).name == "evaluate_vbench_resume.py" for c in calls)


def test_original_100k_defaults_are_unchanged(launcher):
    project, env = launcher
    args = generation_args(launch(project, env, BASE))
    assert args.expected_step == 100000 and args.expected_sha256 is None
    assert "17519025" in args.hf_file
    assert args.output_dir == project / "output/vbench_mobileov_stage2_100k_1x"


def test_scripts_have_valid_syntax_and_one_gpu():
    for name in (WRAPPER, "smoke_vbench_mobileov_stage2_reconstruction_local.sbatch"):
        path = ROOT / "scripts" / name
        subprocess.run(["bash", "-n", str(path)], check=True)
        assert "#SBATCH --gres=gpu:1" in path.read_text()
