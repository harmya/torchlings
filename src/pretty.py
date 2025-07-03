import click

BANNER = (
    " _                     _      _  _                    \n"
    "| |                   | |    | |(_)                   \n"
    "| |_  ___   _ __  ___ | |__  | | _  _ __    __ _  ___ \n"
    "| __|/ _ \\ | '__|/ __|| '_ \\ | || || '_ \\  / _` |/ __|\n"
    "| |_| (_) || |  | (__ | | | ||t|| || | | || (_| |\\__ \\ \n"
    " \\__|\\___/ |_|   \\___||_| |_| |_||_||_| |_| \\__, ||___/\n"
    "                                            __/ |     \n"
    "                                           |___/      "
)

WELCOME_MESSAGE = "Welcome to the exercises!"


def print_banner():
    click.echo(click.style(BANNER, fg="bright_yellow"))


def print_welcome_message():
    click.echo(click.style(WELCOME_MESSAGE, fg="bright_yellow"))
