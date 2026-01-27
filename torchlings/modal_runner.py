"""Run GPU exercises on Modal when CUDA is not available locally."""

import subprocess
import os
import click

MODAL_SIGNUP_URL = "https://modal.com"
GPU_SECTIONS = {"07_gpu", "09_compile", "10_advanced"}


def is_gpu_exercise(exercise_path) -> bool:
    """Check if this exercise belongs to a GPU section."""
    parts = str(exercise_path).split("/")
    return any(section in parts for section in GPU_SECTIONS)


def _check_modal_available() -> tuple[bool, str]:
    """Check if Modal is installed and authenticated. Returns (ok, message)."""
    try:
        import modal  # noqa: F401
    except ImportError:
        return False, "not_installed"

    result = subprocess.run(
        ["modal", "profile", "current"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False, "not_authenticated"

    return True, "ok"


def print_modal_setup_guide():
    click.echo()
    click.echo(click.style("─" * 50, fg="yellow"))
    click.echo(
        click.style(
            "This exercise requires a GPU. No CUDA device found locally.",
            fg="yellow",
            bold=True,
        )
    )
    click.echo()
    click.echo(
        "torchlings can run GPU exercises on "
        + click.style("Modal", fg="cyan", bold=True)
        + " (free $30 credit for new accounts)."
    )
    click.echo()
    click.echo("  1. Sign up at " + click.style(MODAL_SIGNUP_URL, fg="cyan"))
    click.echo("  2. " + click.style("pip install modal", fg="green"))
    click.echo("  3. " + click.style("modal setup", fg="green"))
    click.echo()
    click.echo("Then re-run this exercise.")
    click.echo(click.style("─" * 50, fg="yellow"))
    click.echo()


def run_on_modal(exercise_path) -> bool:
    """Run an exercise on Modal GPU. Returns True if tests pass."""
    available, status = _check_modal_available()

    if not available:
        if status == "not_installed":
            print_modal_setup_guide()
        else:
            click.echo(
                click.style("Modal is installed but not authenticated. Run: ", fg="yellow")
                + click.style("modal setup", fg="cyan", bold=True)
            )
        return False

    import modal

    click.echo(click.style("Running on Modal GPU...", fg="cyan", bold=True))

    with open(exercise_path) as f:
        exercise_content = f.read()

    app = modal.App("torchlings")
    image = modal.Image.debian_slim(python_version="3.12").pip_install(
        "torch", "pytest", "numpy", "triton"
    )

    @app.function(gpu="T4", image=image, timeout=180)
    def _run_exercise(content: str) -> tuple[str, int]:
        import tempfile
        import subprocess as sp

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir="/tmp"
        ) as f:
            f.write(content)
            path = f.name

        result = sp.run(
            [
                "python", "-m", "pytest", path,
                "-vv", "--tb=long", "--no-header", "--color=yes",
            ],
            capture_output=True,
            text=True,
        )
        os.unlink(path)
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        return output, result.returncode

    with app.run():
        output, returncode = _run_exercise.remote(exercise_content)

    if output:
        click.echo(output.rstrip())

    return returncode == 0
