import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/vbench_joint_v2_vs_native_quicksr_1node8gpu.sbatch"
JOINT_FILE = "mobileov_monolithic_pyramid_cfg_dit/mobileov_monolithic_pyramid_cfg_dit_17346642/neodragon_dit_bridge_step010000.pt"


@pytest.fixture
def launch(tmp_path):
    root = tmp_path / "project with spaces"
    (root / "scripts").mkdir(parents=True)
    for name in ("install_dreamlite_dependencies.sh", "install_vbench_env.sh"):
        (root / "scripts" / name).write_text('printf "%s\\n" "$0" >> "$SETUP_LOG"\n')
    metadata = root / "checkpoints/vbench_repo/vbench/VBench_full_info.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("[]")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in {
        "module": "exit 0\n",
        "srun": 'while [[ "$1" == --* ]]; do shift; done\nexec "$@"\n',
    }.items():
        path = bin_dir / name
        path.write_text("#!/usr/bin/env bash\n" + body)
        path.chmod(0o755)
    python = tmp_path / "env/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(f"#!{sys.executable}\n" + '''import json, os, pathlib, sys
argv = sys.argv[1:]
pathlib.Path(os.environ["ARGV_LOG"]).write_text(json.dumps(argv))
output = pathlib.Path(argv[argv.index("--output-dir") + 1])
systems = argv[argv.index("--systems") + 1:argv.index("--protocol")] if "--systems" in argv else []
report = "paper_reference_comparison" if "tools/reproduce_native_neodragon_vbench.py" in argv else ("quicksr_ablation" if ("tools/benchmark_joint_v2_quicksr_ablation.py" in argv or "tools/benchmark_mobileov_stage2_media3epochs_dual.py" in argv) else ("comparison" if len(systems) == 2 else "result"))
output.mkdir(parents=True, exist_ok=True)
(output / (report + ".json")).write_text("{}")
(output / (report + ".md")).write_text("mock report")
if report == "paper_reference_comparison":
    (output / "quicksr_ablation.json").write_text("{}")
''')
    python.chmod(0o755)
    names = ("PROJECT_ROOT", "SAMPLES_PER_PROMPT", "MAX_PROMPTS", "RUN_ROOT", "BENCHMARK_SYSTEMS",
             "JOINT_CKPT", "IMAGE_CKPT", "JOINT_1X", "NATIVE_1X", "NATIVE_MODEL_DIR", "NATIVE_REPO",
             "NATIVE_CACHE", "SEED", "VBENCH_REPO", "EXPORT_VERSIONS", "CHECKPOINT", "EXPECTED_SHA256")
    env = {name: value for name, value in os.environ.items() if name not in names}
    env.update(PATH=f"{bin_dir}:{env['PATH']}", SLURM_SUBMIT_DIR=str(root), CONDA_ENV=str(python.parents[1]),
               VBENCH_ENV=str(python.parents[1]), HF_HOME=str(tmp_path / "cache"), TMPDIR=str(tmp_path / "tmp"),
               ARGV_LOG=str(tmp_path / "argv.json"), SETUP_LOG=str(tmp_path / "setup.log"))

    def run(**overrides):
        result = subprocess.run(["bash", str(SCRIPT)], env={**env, **overrides},
                                capture_output=True, text=True)
        log = Path(env["ARGV_LOG"])
        argv = json.loads(log.read_text()) if log.exists() else []
        return result, argv

    return root, env, run


def option(argv, name):
    return argv[argv.index(name) + 1]


def test_full_job_uses_restored_weight_and_matches_local_five_seed_protocol(launch):
    root, _, run = launch
    weight = root / "output" / JOINT_FILE
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"restored checkpoint")
    result, argv = run()
    assert result.returncode == 0, result.stderr
    assert option(argv, "--joint-checkpoint") == str(weight)
    assert option(argv, "--samples-per-prompt") == "5"
    assert option(argv, "--workers") == "8"
    assert option(argv, "--protocol") == "standard5x"
    assert option(argv, "--seed-mapping") == "extendable"
    assert option(argv, "--max-prompts") == "0"
    assert option(argv, "--seed") == "2026"
    assert "tools/benchmark_joint_v2_quicksr_ablation.py" in argv
    assert option(argv, "--output-dir").endswith("_5x_dual")
    assert argv[argv.index("--systems") + 1:argv.index("--protocol")] == ["joint_v2", "native_hybrid"]
    assert "mock report" in result.stdout
    assert not (root / "checkpoints/hf_mobile_ov" / JOINT_FILE).exists()


def test_cache_fallback_and_explicit_paths_are_forwarded(launch):
    root, _, run = launch
    result, argv = run()
    assert result.returncode == 0, result.stderr
    assert option(argv, "--joint-checkpoint") == str(root / "checkpoints/hf_mobile_ov" / JOINT_FILE)
    result, argv = run(JOINT_CKPT="/custom/joint weight.pt", IMAGE_CKPT="/custom/image.pt",
                       NATIVE_MODEL_DIR="/custom/native snapshot")
    assert result.returncode == 0, result.stderr
    assert option(argv, "--joint-checkpoint") == "/custom/joint weight.pt"
    assert option(argv, "--image-checkpoint") == "/custom/image.pt"
    assert option(argv, "--native-model-dir") == "/custom/native snapshot"


def test_trained_only_job_never_requests_native_generation(launch):
    _, _, run = launch
    result, argv = run(BENCHMARK_SYSTEMS="joint_v2")
    assert result.returncode == 0, result.stderr
    assert argv[argv.index("--systems") + 1:argv.index("--protocol")] == ["joint_v2"]
    assert "vbench_joint_v2_quicksr_berzelius_5x" in option(argv, "--output-dir")
    assert "mock report" in result.stdout


def test_importing_a_completed_one_seed_run_is_opt_in(launch):
    root, _, run = launch
    source = root / "old one seed"
    source.mkdir()
    (source / "generation_summary.json").write_text("{}")
    result, argv = run(JOINT_1X=str(source), EXPORT_VERSIONS="quicksr")
    assert result.returncode == 0, result.stderr
    assert option(argv, "--reuse-1x-run") == f"joint_v2={source}"


@pytest.mark.parametrize("overrides", [
    {"SAMPLES_PER_PROMPT": "1"}, {"SAMPLES_PER_PROMPT": "6"}, {"MAX_PROMPTS": "-1"},
    {"BENCHMARK_SYSTEMS": "wrong"}, {"BENCHMARK_SYSTEMS": "joint_v2 joint_v2"},
    {"BENCHMARK_SYSTEMS": "joint_v2 native_hybrid joint_v2"}, {"JOINT_1X": "/missing/source"},
    {"EXPORT_VERSIONS": "bad"},
])
def test_invalid_settings_fail_before_environment_setup(launch, overrides):
    _, env, run = launch
    result, argv = run(**overrides)
    assert result.returncode != 0
    assert not argv and not Path(env["SETUP_LOG"]).exists()


def test_small_smoke_keeps_a_separate_output_and_does_not_require_scores(launch):
    _, _, run = launch
    result, argv = run(MAX_PROMPTS="2", SAMPLES_PER_PROMPT="1")
    assert result.returncode == 0, result.stderr
    assert option(argv, "--max-prompts") == "2"
    assert "_1x_smoke2" in option(argv, "--output-dir")


def test_shell_syntax_for_launcher_and_safe_environment_setup():
    for path in (SCRIPT, ROOT / "scripts/install_vbench_env.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)
    launcher = SCRIPT.read_text()
    assert "#SBATCH --partition=berzelius" in launcher
    assert "#SBATCH --gres=gpu:8" in launcher
    setup = (ROOT / "scripts/install_vbench_env.sh").read_text()
    assert 'rm -rf "${VBENCH_REPO}"' not in setup
    assert 'importlib.import_module(f"vbench.{dimension}")' in setup
