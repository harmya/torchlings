import os
from pathlib import Path
from torchlings.utils import _run
from torchlings.venv import VENV_NAME
from torchlings.modal_runner import (
    is_gpu_exercise,
    check_modal_available,
    print_modal_setup_guide,
)
from watchfiles import watch
import click

CONTROLS_DESCRIPTION = {
    "n": "Go to next exercise",
    "q": "Quit torchlings",
    "h": "Show help message",
    "t": "Run the current exercise",
    "l": "List all exercises",
}

EXERCISE_ORDER = [
    "01_tensors",
    "02_autograd",
    "03_nn",
    "04_loss",
    "05_data",
    "06_train",
    "07_gpu",
    "08_cv",
    "09_compile",
    "10_advanced",
]


class Runner:
    def __init__(self, exercises_path: Path, start_from: str | None = None):
        self.current_index = 0
        self.exercises_path = exercises_path
        self.exercises = self._discover_exercises()
        self.total_exercises = len(self.exercises)
        self.progress_file = exercises_path / ".torchlings_progress"
        if start_from:
            self._start_from(start_from)
        else:
            self._load_progress()

    def _start_from(self, folder: str) -> None:
        """Set progress to the first exercise in the given folder."""
        for i, ex in enumerate(self.exercises):
            if folder in str(ex):
                self.current_index = i
                self._save_progress()
                return
        raise click.ClickException(
            f"No exercises found matching '{folder}'. "
            f"Available: {', '.join(EXERCISE_ORDER)}"
        )

    def _load_progress(self) -> None:
        """Load the progress from the progress file."""
        if not self.progress_file.exists():
            self._save_progress()
        with open(self.progress_file, "r") as f:
            self.current_index = int(f.read())

    def _save_progress(self) -> None:
        """Save the progress to the progress file."""
        with open(self.progress_file, "w") as f:
            f.write(str(self.current_index))

    def go_to_next_exercise(self):
        self.current_index += 1
        if self.current_index >= self.total_exercises:
            self.current_index = -1
        self._save_progress()

    def _discover_exercises(self) -> list[Path]:
        """Discover all exercises in the exercises path."""
        exercises = []

        for dir in self.exercises_path.iterdir():
            if dir.is_dir():
                exercise_in_topic = []
                for exercise in dir.iterdir():
                    if exercise.is_file() and exercise.suffix == ".py":
                        exercise_in_topic.append(exercise)

                exercises.extend(exercise_in_topic)

        def exercise_order_key(x):
            group_idx = len(EXERCISE_ORDER)
            for i, name in enumerate(EXERCISE_ORDER):
                if name in str(x):
                    group_idx = i
                    break
            try:
                file_num = int(x.stem)
            except Exception:
                file_num = 0
            return (group_idx, file_num)

        exercises.sort(key=exercise_order_key)

        return exercises

    def run(self):
        with click.progressbar(
            range(self.total_exercises),
            label=click.style("Progress", fg="yellow", bold=True),
            fill_char=click.style("█", fg="green"),
            empty_char=click.style("░", fg="red"),
            bar_template="%(label)s  %(bar)s  %(info)s",
            show_percent=True,
            show_pos=True,
        ) as bar:
            bar.update(self.current_index)
            click.echo()
            click.echo(click.style("─" * 50, fg="white"))
            for _ in bar:
                click.echo()
                click.echo(
                    click.style(
                        f"Working on {self.exercises[self.current_index]}",
                        fg="yellow",
                        bold=True,
                    )
                )
                result = self.run_pytest(str(self.exercises[self.current_index]))
                if not result:
                    self.watch_file(self.exercises[self.current_index])
                self.go_to_next_exercise()

    def watch_file(self, exercise_path: Path):
        TARGET = exercise_path.resolve()
        for _ in watch(TARGET, debounce=1):
            result = self.run_pytest(str(TARGET))
            if result:
                break

    def run_pytest(self, target: str | None = None) -> bool:
        """Run pytest inside the venv. Returns True if tests succeed."""
        if target and is_gpu_exercise(target) and not self._has_cuda():
            ok, reason = check_modal_available()
            if not ok:
                if reason == "not_installed":
                    print_modal_setup_guide()
                else:
                    click.echo(
                        click.style(
                            "Modal is installed but not authenticated. Run: ",
                            fg="yellow",
                        )
                        + click.style("modal setup", fg="cyan", bold=True)
                    )
                return False

            return self._run_pytest_on_modal(target)

        env = os.environ.copy()
        env["VIRTUAL_ENV"] = VENV_NAME
        env["PATH"] = str(Path(VENV_NAME) / "bin") + os.pathsep + env["PATH"]

        cmd = ["pytest", "-vv", "--color=yes", "--tb=long", "--no-header"]
        if target:
            cmd.append(target)

        result = _run(cmd, env=env, display_name=f"Testing {target}")
        return result.returncode == 0

    def _run_pytest_on_modal(self, target: str) -> bool:
        """Run a GPU exercise on Modal."""
        import modal

        click.echo(click.style("Running on Modal GPU...", fg="cyan", bold=True))

        with open(target) as f:
            exercise_content = f.read()

        app = modal.App("torchlings")
        image = modal.Image.debian_slim(python_version="3.12").pip_install(
            "torch", "pytest", "numpy", "triton"
        )

        @app.function(gpu="T4", image=image, timeout=180)
        def run_exercise(content: str) -> tuple[str, int]:
            import tempfile
            import subprocess
            import os as _os

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, dir="/tmp"
            ) as f:
                f.write(content)
                path = f.name

            result = subprocess.run(
                [
                    "python", "-m", "pytest", path,
                    "-vv", "--tb=long", "--no-header", "--color=yes",
                ],
                capture_output=True,
                text=True,
            )
            _os.unlink(path)
            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr
            return output, result.returncode

        with app.run():
            output, returncode = run_exercise.remote(exercise_content)

        if output:
            click.echo(output.rstrip())

        return returncode == 0

    def _has_cuda(self) -> bool:
        """Check if CUDA is available in the exercise venv."""
        if hasattr(self, "_cuda_available"):
            return self._cuda_available
        import subprocess

        env = os.environ.copy()
        env["VIRTUAL_ENV"] = VENV_NAME
        env["PATH"] = str(Path(VENV_NAME) / "bin") + os.pathsep + env["PATH"]
        result = subprocess.run(
            ["python", "-c", "import torch; print(torch.cuda.is_available())"],
            env=env,
            capture_output=True,
            text=True,
        )
        self._cuda_available = result.stdout.strip() == "True"
        return self._cuda_available
