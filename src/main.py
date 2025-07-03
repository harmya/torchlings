#!/usr/bin/env python3
"""Torchlings Command‑Line Interface (Click version)

Mirrors the original Rust CLI:
 • Creates/initialises an exercises folder
 • Manages a dedicated Python virtual env via `uv`
 • Installs the required packages (torch, pytest, numpy)
 • Runs pytest across the exercises, pretty‑printing a summary

Usage:
    torchlings init [--exercises-path/-e PATH]
    torchlings test [--exercises-path/-e PATH] [--verbose/-v]
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import click

# ────────────────────────────────────────────────────────────────────────────────
# Configuration constants
# ────────────────────────────────────────────────────────────────────────────────
VENV_NAME = ".torchlings"
REQUIREMENTS: List[str] = ["torch", "pytest", "numpy"]

# ────────────────────────────────────────────────────────────────────────────────
# Helper functions for environment management
# ────────────────────────────────────────────────────────────────────────────────


def _run(cmd: List[str], *, env: dict | None = None, check: bool = False, **popen_kwargs):
    """Light wrapper around subprocess.run that prints & returns CompletedProcess."""
    click.echo(click.style("$ " + " ".join(cmd), fg="blue"), err=True)
    result = subprocess.run(cmd, env=env, text=True, capture_output=True, **popen_kwargs)
    if result.stdout:
        click.echo(result.stdout.rstrip())
    if result.stderr:
        click.echo(result.stderr.rstrip(), err=True)
    if check and result.returncode != 0:
        raise click.ClickException(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
    return result


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
    # uv expects PATH modifications; simply prepending the venv bin is enough
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
    click.echo(f"Installing requirements into venv {VENV_NAME}: {', '.join(REQUIREMENTS)}")
    _uv_pip(["install", *REQUIREMENTS])


def setup_python_environment() -> None:
    """Ensure `uv`, the venv and required packages are ready."""
    if not is_uv_installed():
        install_uv()

    if not venv_exists():
        click.echo(f"Creating virtual environment {VENV_NAME}…")
        create_venv()

    if not are_requirements_installed():
        install_requirements()

    click.secho("✅ Python environment ready!", fg="green")


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

# ────────────────────────────────────────────────────────────────────────────────
# Utility functions for exercises discovery
# ────────────────────────────────────────────────────────────────────────────────

def is_python_file(path: Path) -> bool:
    return path.suffix == ".py"


def is_ignored(path: Path) -> bool:
    return any(
        part.startswith(".")
        or part in {"venv", "__pycache__"}
        or part.endswith("-venv")
        for part in path.parts
    )


def find_python_files(exercises_path: Path) -> List[Path]:
    return [
        p
        for p in exercises_path.rglob("*.py")
        if p.is_file() and not is_ignored(p)
    ]

# ────────────────────────────────────────────────────────────────────────────────
# Pretty banner (ANSI true‑colour)
# ────────────────────────────────────────────────────────────────────────────────

BANNER = (
    " _                     _      _  _                    \n"
    "| |                   | |    | |(_)                   \n"
    "| |_  ___   _ __  ___ | |__  | | _  _ __    __ _  ___ \n"
    "| __|/ _ \\ | '__|/ __|| '_ \\ | || || '_ \\  / _` |/ __|\n"
    "| |_| (_) || |  | (__ | | | ||t|| || | | || (_| |\\__ \\ \n"
    " \\__|\\___/ |_|   \\___||_| |_| |_||_||_| |_| \\__, ||___/\n"
    "                                            __/ |     \n"
    "                                           |___/      "
)


def print_banner():
    click.echo(click.style(BANNER, fg="bright_yellow"))

# ────────────────────────────────────────────────────────────────────────────────
# Click CLI definition
# ────────────────────────────────────────────────────────────────────────────────

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option("0.1.0", prog_name="torchlings")
def cli():
    """Exercises to get you used to reading and writing basic PyTorch code."""
    pass


# INIT -------------------------------------------------------------------------


@cli.command("init")
@click.option(
    "--exercises-path",
    "-e",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
    default=Path("exercises"),
    show_default=True,
    help="Path to exercises directory",
)

def init_cmd(exercises_path: Path):
    """Initialise the exercises directory & Python environment."""
    print_banner()

    # Ensure exercises dir exists
    if not exercises_path.exists():
        exercises_path.mkdir(parents=True, exist_ok=True)
        click.echo(f"📁 Created exercises directory: {exercises_path}")

    # Setup Python env
    click.echo(click.style("Setting up Python environment…", fg="cyan"))
    setup_python_environment()

    click.secho("\n🚀 Torchlings project initialised successfully!", fg="green", bold=True)
    click.echo(f"Run {click.style('torchlings test', fg='cyan')} to start testing your exercises.")


# TEST -------------------------------------------------------------------------


@cli.command("test")
@click.option(
    "--exercises-path",
    "-e",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("exercises"),
    show_default=True,
    help="Path to exercises directory",
)
@click.option("--verbose", "-v", is_flag=True, help="Show verbose output")

def test_cmd(exercises_path: Path, verbose: bool):
    """Run all Python exercise files and report results."""
    click.secho("Torchlings Test Runner", fg="yellow", bold=True)
    click.secho("=======================", fg="yellow")
    click.echo()

    if not exercises_path.exists():
        click.secho("Exercises directory does not exist!", fg="red")
        click.echo(f"Run {click.style('torchlings init', fg='cyan')} to initialise the project.")
        sys.exit(0)

    # Prepare env first
    setup_python_environment()

    py_files = find_python_files(exercises_path)
    if not py_files:
        click.secho("No Python files found in exercises directory!", fg="red")
        click.echo(f"Run {click.style('torchlings init', fg='cyan')} to add starter exercises.")
        sys.exit(0)

    # Run pytest
    success = run_pytest(str(exercises_path))

    # Summary
    click.echo()
    click.secho("=== Test Summary ===", fg="yellow", bold=True)
    if success:
        click.secho(f"✅ All tests passed across {len(py_files)} Python files! 🎉", fg="green")
    else:
        click.secho(f"❌ Some tests failed across {len(py_files)} Python files", fg="red")
        click.echo(f"Run again with {click.style('--verbose', fg='cyan')} for detailed output.")
    sys.exit(0 if success else 1)

# ────────────────────────────────────────────────────────────────────────────────
# Entry point for `python -m torchlings_cli`
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
