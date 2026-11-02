import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "scripts/train_mobileov_stage2_optimizer_exp1_2node16gpu.sbatch"
RUNNER = ROOT / "scripts/run_mobileov_stage2_optimizer_exp1_16gpu.sh"


def test_two_node_launch_pinned_recipe_and_own_state_continuation(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    calls = tmp_path / "calls"
    source = tmp_path / "parent_resume.pt"
    source.write_text("epoch3 optimizer")
    out = tmp_path / "out"
    out.mkdir()
    commands = {
        "module": "exit 0",
        "scontrol": 'printf "node01\\nnode02\\n"',
        "srun": '''printf "SRUN %s\\n" "$*" >> "$CALLS"
if [[ "${TRAIN_RC:-0}" != 0 ]]; then exit "$TRAIN_RC"; fi
SLURM_NODEID=0 bash scripts/run_mobileov_stage2_optimizer_exp1_16gpu.sh --node
SLURM_NODEID=1 bash scripts/run_mobileov_stage2_optimizer_exp1_16gpu.sh --node''',
        "torchrun": 'printf "TRAIN %s\\n" "$*" >> "$CALLS"',
        "sbatch": 'printf "CONTINUE %s RESUME=%s OUT=%s\\n" "$*" "$RESUME" "$OUT" >> "$CALLS"; echo 999',
    }
    for name, body in commands.items():
        path = binary / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)
    env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}", CONDA_ENV=str(tmp_path),
               PROJECT_ROOT=str(ROOT), SLURM_NNODES="2", SLURM_JOB_NODELIST="node[01-02]",
               SLURM_JOB_ID="123", TMPDIR=str(tmp_path / "tmp"), OUT=str(out), CALLS=str(calls),
               PARENT_RESUME=str(source))
    for key in ("RESUME", "STEPS", "LR", "ACCUMULATION", "INIT_STAGE2", "AUTO_CONTINUE"):
        env.pop(key, None)
    def run(**updates):
        calls.unlink(missing_ok=True)
        return subprocess.run(["bash", str(JOB)], env={**env, **updates}, capture_output=True, text=True)
    for script in (JOB, RUNNER, ROOT / "scripts/vbench_mobileov_stage2_optimizer_exp1_1node8gpu.sbatch"):
        subprocess.run(["bash", "-n", str(script)], check=True)
    result = run()
    assert result.returncode == 0, result.stderr
    text = calls.read_text()
    assert text.count("TRAIN ") == 2 and "--node_rank=0" in text and "--node_rank=1" in text
    assert "--nnodes=2 --nproc_per_node=8" in text
    assert "--steps 20000 --decay-steps 20000 --ema-decay .9995 --accumulation 4" in text
    assert "--expected-parent-step 283977" in text and "--resume" not in text
    assert "--nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1" in text
    assert "--evaluate-every 10000" in text
    assert run(SLURM_NNODES="1").returncode != 0
    assert run(LR="1e-4").returncode != 0
    assert run(PARENT_RESUME="missing.pt").returncode != 0
    assert run(TRAIN_RC="1").returncode != 0
    assert "CONTINUE" not in calls.read_text()
    (out / "training_paused.json").write_text("{}")
    (out / "exp1_resume.pt").write_text("own Adam+EMA")
    result = run()
    assert result.returncode == 0, result.stderr
    assert f"RESUME={out}/exp1_resume.pt OUT={out}" in calls.read_text()
    assert "--dependency=afterok:123" in calls.read_text()
    result = run(RESUME=str(out / "exp1_resume.pt"))
    assert result.returncode == 0, result.stderr
    assert f"--resume {out}/exp1_resume.pt" in calls.read_text()
    assert run(RESUME=str(source)).returncode != 0
