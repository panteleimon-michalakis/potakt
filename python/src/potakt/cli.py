"""Command-line interface for Potakt."""

import io
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from potakt import __version__
from potakt.model import (
    DecisionModelError,
    ValidationIssue,
    load_decision_model,
    load_instance,
    render_json_schema,
    validate_data,
)

app = typer.Typer(no_args_is_help=True)


def use_utf8(stream: object) -> None:
    """Make a text stream emit UTF-8 (Windows defaults to the ANSI code page when piped)."""
    if isinstance(stream, io.TextIOWrapper) and stream.encoding.lower() not in ("utf-8", "utf8"):
        stream.reconfigure(encoding="utf-8")


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
    use_utf8(sys.stdout)
    use_utf8(sys.stderr)


@app.command()
def schema(
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write the schema to this file instead of printing it."),
    ] = None,
) -> None:
    """Print or write the JSON Schema of the Decision Model."""
    text = render_json_schema()
    if out is None:
        typer.echo(text, nl=False)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")


@app.command()
def validate(
    model: Annotated[Path, typer.Argument(help="Decision Model JSON file.", show_default=False)],
) -> None:
    """Validate a Decision Model and the CSV data it binds (paths relative to the model file)."""
    try:
        dm = load_decision_model(model)
        instance = load_instance(dm, model.parent)
    except DecisionModelError as error:
        _fail(error.issues)
    issues = validate_data(dm, instance)
    if issues:
        _fail(issues)
    typer.echo(
        f"OK: {dm.title} — {len(instance.resources)} resources, "
        f"{len(instance.tasks)} tasks, {len(dm.constraints)} constraints"
    )


def _fail(issues: Iterable[ValidationIssue]) -> NoReturn:
    for issue in issues:
        typer.echo(str(issue))
    raise typer.Exit(code=1)
