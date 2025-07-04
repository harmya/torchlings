from pathlib import Path
from pretty import print_banner
from venv import setup_python_environment
import click


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
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
        click.echo(f"📁 Created exercises directory: {exercises_path}")

    click.echo(click.style("Setting up Python environment…", fg="cyan"))
    setup_python_environment(exercises_path)

    click.secho("\n🚀 Torchlings initialised successfully!", fg="green", bold=True)
    click.echo(
        f"Run {click.style('torchlings test', fg='cyan')} to start testing your exercises."
    )


if __name__ == "__main__":
    print_banner()
    cli()
