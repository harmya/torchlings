"""Run GPU exercises on Modal when CUDA is not available locally."""

import click

MODAL_SIGNUP_URL = "https://modal.com"
GPU_SECTIONS = {"07_gpu", "09_compile", "10_advanced"}


def is_gpu_exercise(exercise_path) -> bool:
    """Check if this exercise belongs to a GPU section."""
    parts = str(exercise_path).split("/")
    return any(section in parts for section in GPU_SECTIONS)


def check_modal_available() -> tuple[bool, str]:
    """Check if Modal is installed and authenticated. Returns (ok, reason)."""
    try:
        import modal  # noqa: F401
    except ImportError:
        return False, "not_installed"

    try:
        modal.config._profile
    except Exception:
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
