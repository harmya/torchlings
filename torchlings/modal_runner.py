"""Run GPU exercises on Modal when CUDA is not available locally."""

import subprocess
import shutil
import tempfile
import base64
import os
import click

MODAL_SIGNUP_URL = "https://modal.com"

GPU_SECTIONS = {"07_gpu", "09_compile", "10_advanced"}


def is_gpu_exercise(exercise_path) -> bool:
    """Check if this exercise belongs to a GPU section."""
    parts = str(exercise_path).split("/")
    return any(section in parts for section in GPU_SECTIONS)


def is_modal_installed() -> bool:
    return shutil.which("modal") is not None


def is_modal_authenticated() -> bool:
    result = subprocess.run(
        ["modal", "profile", "current"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def check_cuda_available() -> bool:
    """Check if CUDA is available without importing torch in the main process."""
    result = subprocess.run(
        ["python3", "-c", "import torch; print(torch.cuda.is_available())"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() == "True"


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


def _make_modal_script(exercise_b64: str) -> str:
    return f'''import modal
import base64
import tempfile
import subprocess
import os

app = modal.App("torchlings")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch", "pytest", "numpy", "triton"
)

EXERCISE_B64 = "{exercise_b64}"

@app.function(gpu="T4", image=image, timeout=180)
def run_exercise():
    content = base64.b64decode(EXERCISE_B64).decode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    ) as f:
        f.write(content)
        path = f.name

    result = subprocess.run(
        ["python", "-m", "pytest", path, "-vv", "--tb=long", "--no-header", "--color=yes"],
        capture_output=True,
        text=True,
    )
    os.unlink(path)
    output = result.stdout
    if result.stderr:
        output += "\\n" + result.stderr
    return output, result.returncode

@app.local_entrypoint()
def main():
    output, code = run_exercise.remote()
    print(output)
    raise SystemExit(code)
'''


def run_on_modal(exercise_path) -> bool:
    """Run an exercise on Modal GPU. Returns True if tests pass."""
    if not is_modal_installed():
        print_modal_setup_guide()
        return False

    if not is_modal_authenticated():
        click.echo(
            click.style(
                "Modal is installed but not authenticated. Run: ",
                fg="yellow",
            )
            + click.style("modal setup", fg="cyan", bold=True)
        )
        return False

    click.echo(
        click.style("Running on Modal GPU...", fg="cyan", bold=True)
    )

    with open(exercise_path) as f:
        exercise_content = f.read()

    exercise_b64 = base64.b64encode(exercise_content.encode()).decode()
    script = _make_modal_script(exercise_b64)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp", prefix="torchlings_modal_"
    ) as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            ["modal", "run", script_path],
            text=True,
        )
        return result.returncode == 0
    finally:
        os.unlink(script_path)
