import click

BANNER = (
    " _                     _      _  _                    \n"
    "| |                   | |    | |(_)                   \n"
    "| |_  ___   _ __  ___ | |__  | | _  _ __    __ _  ___ \n"
    "| __|/ _ \\ | '__|/ __|| '_ \\ | || || '_ \\  / _` |/ __|\n"
    "| |_| (_) || |  | (__ | | | || || || | | || (_| |\\__ \\ \n"
    " \\__|\\___/ |_|   \\___| |_||_||_||_||_| |_| \\__, ||___/\n"
    "                                            __/ |     \n"
    "                                           |___/      "
)



def print_banner():
    click.echo(click.style(BANNER, fg="bright_yellow"), nl=True)
    click.echo()


def print_welcome_message():
    click.echo(click.style("Welcome to the torchlings! These exercises are designed to help get used to writing and running PyTorch code.", fg="bright_yellow"))
    click.echo(click.style("torchlings takes care of everything! Just write your code and save it :)", fg="bright_yellow"))
    click.echo(click.style(f"To get started, initilize the exercises with {click.style('torchlings init', fg='blue', bold=True)}", fg="bright_yellow"))
