from pathlib import Path
from typing import List
import click
import subprocess


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
        p for p in exercises_path.rglob("*.py") if p.is_file() and not is_ignored(p)
    ]


def _run(
    cmd: List[str], *, env: dict | None = None, check: bool = False, display_name: str | None = None, **popen_kwargs
):
    """Wrapper around subprocess.run that prints & returns CompletedProcess."""
    display_name = display_name or "$ running..."
    click.echo(click.style(f"{display_name} ", fg="blue"), err=True)
    result = subprocess.run(
        cmd, env=env, text=True, capture_output=True, **popen_kwargs
    )
    if result.stdout:
        click.echo(result.stdout.rstrip())
    if result.stderr:
        click.echo(result.stderr.rstrip(), err=True)
    if check and result.returncode != 0:
        raise click.ClickException(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )
    return result
