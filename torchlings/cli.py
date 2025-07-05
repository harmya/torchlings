from pathlib import Path
from torchlings.pretty import print_banner, print_welcome_message
from torchlings.venv import setup_python_environment
import click
from importlib import resources
import shutil
from torchlings.runner import Runner

@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    name="torchlings",
    invoke_without_command=True,
)
def cli():
    pass

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
    if not exercises_path.exists():
        exercises_path.mkdir(parents=True, exist_ok=True)
    
    try:
        import torchlings.exercises
        exercises_package = resources.files(torchlings.exercises)
        
        for directory in exercises_package.iterdir():
            if not directory.is_dir():
                continue
            
            topic_dir = exercises_path / directory.name
            topic_dir.mkdir(parents=True, exist_ok=True)

            for file in directory.iterdir():
                if file.name.endswith('.py') and file.name != "__init__.py":
                    shutil.copy2(file, topic_dir / file.name)
                
    except Exception as e:
        click.echo(f"⚠️  Warning: Could not copy exercise files: {e}")

    click.echo(f"📁 Created exercises directory: {exercises_path}")
    click.echo(click.style("Setting up Python environment…", fg="cyan"))
    setup_python_environment(exercises_path)

    click.secho("\n🚀 Torchlings initialised successfully!", fg="green", bold=True)
    click.echo(
        f"Run {click.style('torchlings run', fg='cyan')} to start testing your exercises."
    )

@cli.command("run")
@click.option(
    "--exercises-path",
    "-e",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=Path("sol_init"),
    show_default=True,
    help="Path to exercises directory",
)
def run_cmd(exercises_path: Path):
    """Launch the interactive testing interface."""
    runner = Runner(exercises_path=Path("sol_init"))
    runner.run()

def main():
    print_banner()
    print_welcome_message()
    cli()

if __name__ == "__main__":
    main()
