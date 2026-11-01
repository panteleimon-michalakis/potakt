"""Command-line interface for Potakt."""

import io
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from potakt import __version__
from potakt.checker import check, render_check_result
from potakt.evaluation import (
    CORPUS_PATH,
    INSTRUCTION_PATH,
    STRONG_MODEL,
    LoadedScenario,
    ScenarioError,
    UnknownModelError,
    evaluate_candidate,
    load_scenario,
    render_payload,
    render_test_a,
    request_payload,
    summarize,
)
from potakt.model import (
    DecisionModel,
    DecisionModelError,
    ProblemInstance,
    Solution,
    ValidationIssue,
    load_decision_model,
    load_instance,
    load_solution,
    render_json_schema,
    render_solution_json_schema,
    validate_data,
    validate_solution,
)

app = typer.Typer(no_args_is_help=True)


def use_utf8(stream: object) -> None:
    """Make a text stream write UTF-8 with ``\\n`` line endings on every platform (ADR-0006 §1).

    Applied every time, also when the stream is already UTF-8: Windows translates ``\\n`` to
    ``\\r\\n`` by default. The stream's error handler is kept.
    """
    if isinstance(stream, io.TextIOWrapper):
        stream.reconfigure(encoding="utf-8", errors=stream.errors, newline="\n")


def entry_point() -> None:
    """The ``potakt`` console script: fix the output streams before any argument is parsed, so
    help, ``--version`` and usage errors are UTF-8 with ``\\n`` too."""
    use_utf8(sys.stdout)
    use_utf8(sys.stderr)
    app()


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
    # Also here for callers of ``app()`` that bypass the console script (``entry_point``).
    use_utf8(sys.stdout)
    use_utf8(sys.stderr)


@app.command()
def schema(
    solution: Annotated[
        bool,
        typer.Option("--solution", help="The schema of a solution instead of a Decision Model."),
    ] = False,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write the schema to this file instead of printing it."),
    ] = None,
) -> None:
    """Print or write the JSON Schema of the Decision Model (or of a solution)."""
    text = render_solution_json_schema() if solution else render_json_schema()
    if out is None:
        typer.echo(text, nl=False)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")


@app.command()
def validate(
    model: Annotated[Path, typer.Argument(help="Decision Model JSON file.", show_default=False)],
) -> None:
    """Validate a Decision Model and the CSV data it binds (paths relative to the model file).

    Exit code 0: valid ("OK" on stdout); 1: invalid (one issue per line on stderr); 2: usage error.
    """
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
        typer.echo(str(issue), err=True)
    raise typer.Exit(code=1)


# 2 is the framework's usage error (Click/Typer), so "infeasible" gets its own code (ADR-0006 §1).
EXIT_INFEASIBLE = 3


@app.command("check")
def check_command(
    model: Annotated[Path, typer.Argument(help="Decision Model JSON file.", show_default=False)],
    solution: Annotated[Path, typer.Argument(help="Solution JSON file.", show_default=False)],
    as_json: Annotated[bool, typer.Option("--json", help="Print the result as JSON.")] = False,
) -> None:
    """Score a solution against a Decision Model and its data.

    Exit code 0: feasible; 1: invalid input; 3: infeasible (hard violations); 2: usage error.
    Input issues go to stderr (solution issues prefixed with the solution file), the report to
    stdout.
    """
    dm, instance, parsed = _load_for_check(model, solution)
    result = check(dm, instance, parsed)
    typer.echo(result.to_json() if as_json else render_check_result(result, dm), nl=False)
    if not result.score.feasible:
        raise typer.Exit(code=EXIT_INFEASIBLE)


def _load_for_check(model: Path, solution: Path) -> tuple[DecisionModel, ProblemInstance, Solution]:
    """The validated inputs of ``check``; otherwise print every issue to stderr and exit 1."""
    issues: list[str] = []
    dm = instance = parsed = None
    try:
        dm = load_decision_model(model)
        instance = load_instance(dm, model.parent)
        issues += [str(issue) for issue in validate_data(dm, instance)]
    except DecisionModelError as error:
        issues += [str(issue) for issue in error.issues]
    try:
        parsed = load_solution(solution)
    except DecisionModelError as error:
        issues += [f"{solution}#{issue}" for issue in error.issues]
    if not issues and dm is not None and instance is not None and parsed is not None:
        # Solution ids are only meaningful against a valid model and data.
        issues += [f"{solution}#{issue}" for issue in validate_solution(dm, instance, parsed)]
    if issues or dm is None or instance is None or parsed is None:
        for line in issues:
            typer.echo(line, err=True)
        raise typer.Exit(code=1)
    return dm, instance, parsed


# --- eval: the Phase 0 experiment, offline (ADR-0007 §7) -------------------------------------

eval_app = typer.Typer(no_args_is_help=True)
app.add_typer(
    eval_app,
    name="eval",
    help="Phase 0 evaluation (ADR-0007), offline: no model calls, no credentials.",
)
EXIT_USAGE = 2  # the framework's usage error code (Click/Typer)


def _stop(code: int, *lines: str) -> NoReturn:
    for line in lines:
        typer.echo(line, err=True)
    raise typer.Exit(code=code)


def _scenario(directory: Path) -> LoadedScenario:
    try:
        return load_scenario(directory)
    except ScenarioError as error:
        _stop(1, *(str(issue) for issue in error.issues))


@eval_app.command("prompt")
def eval_prompt(
    scenario: Annotated[Path, typer.Argument(help="Scenario directory.", show_default=False)],
    model: Annotated[str, typer.Option("--model", help="Model id.")] = STRONG_MODEL,
) -> None:
    """Print the exact JSON request payload of the Test A request for one scenario (ADR-0007 §7),
    with no credentials; nothing is sent.

    Claude ids get the Anthropic Messages API body, every other id the Gemini generateContent
    body. Exit code 0: payload printed; 1: invalid scenario; 2: usage error (also a Claude model
    without pre-registered settings).
    """
    loaded = _scenario(scenario)
    try:
        prompt = render_test_a(loaded)
    except OSError as error:  # outside a source checkout the prompt file is absent
        path = error.filename or INSTRUCTION_PATH
        _stop(1, f"cannot read the instruction file {path}: {error.strerror or error}")
    except ValueError as error:
        _stop(1, str(error))
    try:
        payload = request_payload(prompt, model)
    except UnknownModelError as error:
        _stop(EXIT_USAGE, str(error))
    typer.echo(render_payload(payload), nl=False)


@eval_app.command("compare")
def eval_compare(
    scenario: Annotated[Path, typer.Argument(help="Scenario directory.", show_default=False)],
    candidate: Annotated[
        Path, typer.Argument(help="Candidate DM text file (UTF-8).", show_default=False)
    ],
) -> None:
    """Validate a candidate against the scenario's CSV files only, compare it with the truth and
    print the comparison JSON (ADR-0007 §5, §7).

    Exit code 0: comparison printed, for valid and invalid candidates alike (both are results);
    1: invalid scenario or unreadable candidate file; 2: usage error.
    """
    loaded = _scenario(scenario)
    try:
        raw = candidate.read_bytes()
    except OSError as error:
        _stop(1, f"{candidate}: cannot read: {error.strerror or error}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        _stop(1, f"{candidate}: not valid UTF-8")
    typer.echo(evaluate_candidate(loaded, text).to_json(), nl=False)


@eval_app.command("summary")
def eval_summary(
    runs: Annotated[
        Path,
        typer.Argument(
            help="One model's runs directory (<model dir>).",
            metavar="RUNS_DIR",
            show_default=False,
        ),
    ],
    corpus: Annotated[
        Path,
        typer.Option(
            "--corpus",
            help="Scenario corpus directory (default: the repository's docs/validation/scenarios).",
            show_default=False,
        ),
    ] = CORPUS_PATH,
    as_json: Annotated[bool, typer.Option("--json", help="Print the summary as JSON.")] = False,
) -> None:
    """Summarize one model's recorded Test A runs: recall, merged recall, precision, validity,
    breakdowns, diagnostics, usage and cost (ADR-0007 §7 «Records»).

    Exit code 0: summary printed; 1: missing directory; 2: usage error.
    """
    if not corpus.is_dir():
        _stop(1, f"{corpus} is not a directory")
    try:
        summary = summarize(runs, corpus)
    except ValueError as error:
        _stop(1, str(error))
    typer.echo(summary.to_json() if as_json else summary.to_text(), nl=False)
