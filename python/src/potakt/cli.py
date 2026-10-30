"""Command-line interface for Potakt."""

from typing import Annotated

import typer

from potakt import __version__

app = typer.Typer(no_args_is_help=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"potakt {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Potakt: the LLM understands, the solver guarantees, the human decides."""
