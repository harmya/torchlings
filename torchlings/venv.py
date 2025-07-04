import shutil
from pathlib import Path
from typing import List
from torchlings.utils import _run
import os
import click

VENV_NAME = ".venv"
REQUIREMENTS: List[str] = ["torch", "pytest", "numpy"]


def is_uv_installed() -> bool:
    return shutil.which("uv") is not None


def install_uv() -> None:
    click.echo("Installing uv via official installer…")
    _run(["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"], check=True)
    if not is_uv_installed():
        raise click.ClickException("`uv` still not found after installation; aborting.")


def venv_exists() -> bool:
    return Path(VENV_NAME).exists()


def create_venv() -> None:
    _run(["uv", "venv", VENV_NAME], check=True)


def _uv_pip(args: List[str]):
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = VENV_NAME
    env["PATH"] = str(Path(VENV_NAME) / "bin") + os.pathsep + env["PATH"]
    return _run(["uv", "pip", *args], env=env, check=True)


def is_package_installed(pkg: str) -> bool:
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = VENV_NAME
    cp = _run(["uv", "pip", "list", "--format=freeze"], env=env)
    return any(line.startswith(pkg + "==") for line in cp.stdout.splitlines())


def are_requirements_installed() -> bool:
    return all(is_package_installed(req) for req in REQUIREMENTS)


def install_requirements() -> None:
    click.echo(
        f"Installing requirements into venv {VENV_NAME}: {', '.join(REQUIREMENTS)}"
    )
    _uv_pip(["install", *REQUIREMENTS])


def setup_python_environment(exercises_path: Path) -> None:
    """Ensure `uv`, the venv and required packages are ready."""

    os.chdir(exercises_path)

    if not is_uv_installed():
        install_uv()

    if not venv_exists():
        click.echo(f"Creating virtual environment {VENV_NAME}…")
        create_venv()

    if not are_requirements_installed():
        install_requirements()

    click.secho("✅ Python environment ready!", fg="green")


def initialise_exercises_directory(exercises_path: Path) -> None:
    """Initialise the exercises directory & Python environment."""
    if not exercises_path.exists():
        exercises_path.mkdir(parents=True, exist_ok=True)
        click.echo(f"📁 Created exercises directory: {exercises_path}")

    click.echo(click.style("Setting up Python environment…", fg="cyan"))
    setup_python_environment(exercises_path)

