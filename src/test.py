import os
from pathlib import Path
from utils import _run
from venv import VENV_NAME


def run_pytest(target: str | None = None) -> bool:
    """Run pytest inside the venv. Returns True if tests succeed."""
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = VENV_NAME
    env["PATH"] = str(Path(VENV_NAME) / "bin") + os.pathsep + env["PATH"]

    cmd = ["uv", "run", "pytest"]
    if target:
        cmd.append(target)

    result = _run(cmd, env=env)
    return result.returncode == 0
