import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("nested", [False, True])
def test_repository_tools_wins_over_installed_tools_package(tmp_path, nested):
    site = tmp_path / "foreign_site"
    package = site / "tools"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('ORIGIN = "unrelated installed package"\n')
    if nested:
        (package / "data_prepare").mkdir()
        (package / "data_prepare/__init__.py").write_text('ORIGIN = "unrelated data tools"\n')
    code = f"""import tools.data_prepare.prepare_mobileov_exp2 as pipeline
import tools.data_prepare.exp2_models as models
import tools.data_prepare.run_mobileov_exp2_data as launcher
from pathlib import Path
assert Path(pipeline.__file__).resolve() == Path({str(ROOT / 'tools/data_prepare/prepare_mobileov_exp2.py')!r})
assert Path(models.__file__).resolve() == Path({str(ROOT / 'tools/data_prepare/exp2_models.py')!r})
assert Path(launcher.__file__).resolve() == Path({str(ROOT / 'tools/data_prepare/run_mobileov_exp2_data.py')!r})
print('repository-tools-import-ok')
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT), str(site))), PYTHONNOUSERSITE="1")
    # Not running from the repo also exercises the launcher's PYTHONPATH setup.
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path,
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "repository-tools-import-ok" in result.stdout


def test_tools_initializers_do_not_import_models(tmp_path):
    code = f"""import sys
sys.modules.update(torch=None, av=None, transformers=None)
import tools, tools.data_prepare
from pathlib import Path
assert Path(tools.__file__).resolve() == Path({str(ROOT / 'tools/__init__.py')!r})
assert Path(tools.data_prepare.__file__).resolve() == Path({str(ROOT / 'tools/data_prepare/__init__.py')!r})
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path,
                            env=dict(os.environ, PYTHONPATH=str(ROOT)), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
