import os
import subprocess
import sys


def test_installed_package_imports_from_outside_repo(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import clinical_world_model; "
                "import clinical_world_model.planner; "
                "print(clinical_world_model.__file__)"
            ),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "clinical_world_model" in result.stdout
