import os
from pathlib import Path
from torchlings.utils import _run
from torchlings.venv import VENV_NAME
from watchfiles import watch

class Runner:
    def __init__(self, exercises_path: Path):
        self.current_index = 0
        self.exercises_path = exercises_path
        self.exercises = self._discover_exercises()
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
    
    def _discover_exercises(self) -> list[Path]:
        """Discover all exercises in the exercises path."""
        exercises = []

        for dir in self.exercises_path.iterdir():
            if dir.is_dir():
                exercise_in_topic = []
                for exercise in dir.iterdir():
                    if exercise.is_file() and exercise.suffix == ".py":
                        exercise_in_topic.append(exercise)

                exercise_in_topic.sort()
                exercises.extend(exercise_in_topic)
        print(exercises)
        return exercises
    
    def run(self):
        self.watch_file(self.exercises[self.current_index])
    
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

        result = _run(cmd, env=env)
        return result.returncode == 0
