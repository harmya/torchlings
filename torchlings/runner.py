import os
from pathlib import Path
from torchlings.utils import _run
from torchlings.venv import VENV_NAME
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
    "09_text",
    "10_advanced",
]

class Runner:
    def __init__(self, exercises_path: Path):
        self.current_index = 0
        self.exercises_path = exercises_path
        self.exercises = self._discover_exercises()
        self.total_exercises = len(self.exercises)
        self.progress_file = exercises_path / ".torchlings_progress"
        self._load_progress()

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
        with click.progressbar(range(self.total_exercises), 
                      label=click.style("Progress", fg="yellow", bold=True),
                      fill_char=click.style('█', fg="green"),
                      empty_char=click.style('░', fg="red"),
                      bar_template='%(label)s  %(bar)s  %(info)s',
                      show_percent=True,
                      show_pos=True,
                      ) as bar:
            
            bar.update(self.current_index)
            click.echo()
            click.echo(click.style("─" * 50, fg="white"))
            for _ in bar:
                click.echo()
                click.echo(click.style(f"Working on {self.exercises[self.current_index]}", fg="yellow", bold=True))
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
        env = os.environ.copy()
        env["VIRTUAL_ENV"] = VENV_NAME
        env["PATH"] = str(Path(VENV_NAME) / "bin") + os.pathsep + env["PATH"]

        cmd = ["pytest", "-vv", "--color=yes", "--tb=long", "--no-header"]
        if target:
            cmd.append(target)

        result = _run(cmd, env=env, display_name=f"Testing {target}")
        return result.returncode == 0
