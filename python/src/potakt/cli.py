"""Command-line interface for Potakt."""

import io
import os
import sys
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from potakt import __version__
from potakt.checker import check, render_check_result
from potakt.evaluation import (
    CORPUS_PATH,
    DEV_CORPUS_PATH,
    INSTRUCTION_PATH,
    REGISTERED_MODELS,
    STRONG_MODEL,
    LoadedScenario,
    ScenarioError,
    UnknownModelError,
    evaluate_candidate,
    load_scenario,
    render_payload,
    render_test_a,
    request_payload,
    rescore_run,
    summarize,
)
from potakt.evaluation.draws import draws_report
from potakt.evaluation.records import RESPONSE_FILE
from potakt.evaluation.repairs_report import repairs_report
from potakt.evaluation.runner import (
    DEV_RUNS_PATH,
    REPAIR_DIR,
    RUNS_PATH,
    audit_batches,
    check_frozen,
    check_order,
    check_pins,
    check_set,
    corpus_scenarios,
    preflight,
    run_anthropic,
    run_directory,
    run_gemini,
    run_plan,
    take_lock,
)
from potakt.evaluation.test_a import ANTHROPIC_PREFIX, check_registered
from potakt.evaluation.transport import (
    HttpResponse,
    RunStopped,
    Transport,
    UrllibTransport,
    server_text,
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

# Never print local variables in a traceback: one of them could hold an API key (TASK-005c3 F5).
app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


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


# --- eval: the Phase 0 experiment (ADR-0007 §7) ----------------------------------------------

eval_app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
app.add_typer(
    eval_app,
    name="eval",
    help=(
        "Phase 0 evaluation (ADR-0007). eval run calls the model APIs (paid, with a key from "
        "the environment); eval preflight makes free, read-only calls with the same keys (no "
        "prompt); the other commands are offline: no model calls, no credentials."
    ),
)
EXIT_USAGE = 2  # the framework's usage error code (Click/Typer)


def _stop(code: int, *lines: str) -> NoReturn:
    for line in lines:
        typer.echo(line, err=True)
    raise typer.Exit(code=code)


# TASK-005e6 decision 5 (ADR-0008 D1): --dev of the offline commands only changes their defaults.
DEV_HELP = (
    "Read the development set (ADR-0008 D1): the defaults become docs/validation/dev and "
    "docs/validation/runs/dev/round-1."
)
CORPUS_HELP = (
    "Scenario corpus directory (default: the repository's docs/validation/scenarios, or "
    "docs/validation/dev with --dev)."
)
RUNS_HELP = (
    "Test A runs directory of all models (default: the repository's docs/validation/runs/test-a, "
    "or docs/validation/runs/dev/round-1 with --dev)."
)


def _corpus(corpus: Path | None, dev: bool) -> Path:
    """An explicit ``--corpus``, else the default of the set (``--dev``), read when called."""
    if corpus is not None:
        return corpus
    return DEV_CORPUS_PATH if dev else CORPUS_PATH


def _runs(runs: Path | None, dev: bool) -> Path:
    """An explicit ``--runs``, else the default of the set (``--dev``), read when called."""
    if runs is not None:
        return runs
    return DEV_RUNS_PATH if dev else RUNS_PATH


def _scenario(directory: Path) -> LoadedScenario:
    try:
        return load_scenario(directory)
    except ScenarioError as error:
        _stop(1, *(str(issue) for issue in error.issues))


@eval_app.command("prompt")
def eval_prompt(
    scenario: Annotated[Path, typer.Argument(help="Scenario directory.", show_default=False)],
    model: Annotated[
        str,
        typer.Option("--model", help=f"Model id: {', '.join(REGISTERED_MODELS)}."),
    ] = STRONG_MODEL,
) -> None:
    """Print the exact JSON payload of one scenario's Test A request; nothing is sent.

    The payload of the Test A request for one scenario (ADR-0007 §7), with no credentials.

    The registered models are claude-haiku-5-5, gemini-3.8-flash, claude-opus-5-5 (in run
    order); the Claude ones get the Anthropic Messages API body, gemini-3.8-flash the Gemini
    generateContent body. Exit code 0: payload printed; 1: invalid scenario; 2: usage error
    (also an unregistered model id).
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
    """Compare a candidate with a scenario's truth and print the comparison JSON.

    Validate a candidate against the scenario's CSV files only, compare it with the truth and
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
        Path | None, typer.Option("--corpus", help=CORPUS_HELP, show_default=False)
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the summary as JSON.")] = False,
    dev: Annotated[bool, typer.Option("--dev", help=DEV_HELP)] = False,
) -> None:
    """Summarize one model's recorded Test A runs.

    Summarize one model's recorded Test A runs: recall, merged recall, precision, validity,
    completeness (the corpus scenarios still missing), breakdowns, diagnostics, how the answers
    stopped, usage and cost at the batch prices (ADR-0007 §7 «Records» and its amendment), and,
    without changing the gate, one row per scenario, the recall over valid candidates, a bootstrap
    interval of the recall, the recall after look-alike ids are replaced and the assumptions next
    to unsupported requests (ADR-0008 D3, D9, D12).

    Exit code 0: summary printed; 1: missing directory; 2: usage error.
    """
    corpus = _corpus(corpus, dev)
    if not corpus.is_dir():
        _stop(1, f"{corpus} is not a directory")
    try:
        summary = summarize(runs, corpus)
    except ValueError as error:
        _stop(1, str(error))
    typer.echo(summary.to_json() if as_json else summary.to_text(), nl=False)


@eval_app.command("rescore")
def eval_rescore(
    runs: Annotated[
        Path,
        typer.Argument(
            help="One model's runs directory, named after a registered model.",
            metavar="RUNS_DIR",
            show_default=False,
        ),
    ],
    corpus: Annotated[
        Path | None, typer.Option("--corpus", help=CORPUS_HELP, show_default=False)
    ] = None,
    dev: Annotated[bool, typer.Option("--dev", help=DEV_HELP)] = False,
) -> None:
    """Recompute one model's Test A scores from response.json alone; nothing is sent.

    Recompute candidate.txt and comparison.json of one model's recorded Test A runs from
    response.json alone (ADR-0007 §7 amendment «Records»); no request is sent.

    One line per scenario directory, in byte order of its name: unchanged, updated (<files>) or
    skipped (no response.json). Only missing or different files are written; response.json and
    request.json are never touched. Exit code 0: all directories processed; 1: missing runs or
    corpus directory, a runs directory not named after a registered model, or a scenario not in
    the corpus (its line is printed and processing continues); 2: usage error.
    """
    corpus = _corpus(corpus, dev)
    for directory in (runs, corpus):
        if not directory.is_dir():
            _stop(1, f"{directory} is not a directory")
    model = runs.resolve().name  # so "." from inside the runs directory works too
    try:
        check_registered(model)
    except UnknownModelError as error:
        _stop(1, str(error))
    in_corpus = True
    for run in sorted(
        (path for path in runs.iterdir() if path.is_dir()),
        key=lambda path: path.name.encode("utf-8"),
    ):
        outcome = _rescored(run, corpus, model)
        in_corpus = in_corpus and outcome is not None
        typer.echo(f"{run.name}: {outcome or 'not a scenario of the corpus'}")
    raise typer.Exit(code=0 if in_corpus else 1)


@eval_app.command("repairs")
def eval_repairs(
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help=f"Model id: {', '.join(REGISTERED_MODELS)}.",
            show_default=False,
        ),
    ],
    runs: Annotated[Path | None, typer.Option("--runs", help=RUNS_HELP, show_default=False)] = None,
    corpus: Annotated[
        Path | None, typer.Option("--corpus", help=CORPUS_HELP, show_default=False)
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
    dev: Annotated[bool, typer.Option("--dev", help=DEV_HELP)] = False,
) -> None:
    """Report the recall after one repair of a model's invalid candidates; nothing is sent.

    Read the model's first run (<runs>/<model dir>) and its repair run (<runs>/repair/<model
    dir>, written by eval run --repair) and print, for each invalid candidate of the first run,
    what its one repair turn gave (valid, invalid, without an answer, not sent, missing), and the
    recall after one repair (ADR-0008 D7): the matched count of each repair that is valid
    replaces the 0 of its first candidate. It is never the gate. Offline: no model call, no
    credentials.

    Exit code 0: report printed; 1: missing runs or corpus directory, or the model's first run
    not complete; 2: usage error (also an unregistered model id).
    """
    try:
        check_registered(model)
    except UnknownModelError as error:
        _stop(EXIT_USAGE, str(error))
    runs, corpus = _runs(runs, dev), _corpus(corpus, dev)
    try:
        report = repairs_report(model, runs, corpus)
    except (ValueError, RunStopped) as error:
        _stop(1, str(error))
    typer.echo(report.to_json() if as_json else report.to_text(), nl=False)


@eval_app.command("draws")
def eval_draws(
    runs: Annotated[Path | None, typer.Option("--runs", help=RUNS_HELP, show_default=False)] = None,
    corpus: Annotated[
        Path | None, typer.Option("--corpus", help=CORPUS_HELP, show_default=False)
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
    dev: Annotated[bool, typer.Option("--dev", help=DEV_HELP)] = False,
) -> None:
    """Report the draw-to-draw variation of claude-opus-5-5's three draws; nothing is sent.

    Read the three draws of claude-opus-5-5 (<runs>/claude-opus-5-5, then
    <runs>/repeat-2/claude-opus-5-5 and <runs>/repeat-3/claude-opus-5-5, written by eval run
    --repeat 2 and --repeat 3) and print each draw's micro recall and, per scenario, its recall
    in each draw with their arithmetic mean, minimum and maximum (ADR-0008 D8). It is never the
    gate. Offline: no model call, no credentials.

    Exit code 0: report printed; 1: missing runs or corpus directory, or a draw not complete;
    2: usage error.
    """
    runs, corpus = _runs(runs, dev), _corpus(corpus, dev)
    try:
        report = draws_report(runs, corpus)
    except ValueError as error:
        _stop(1, str(error))
    typer.echo(report.to_json() if as_json else report.to_text(), nl=False)


def _rescored(run: Path, corpus: Path, model: str) -> str | None:
    """What rescoring did to one run; ``None`` if it is not a (loadable) corpus scenario."""
    try:
        scenario = load_scenario(corpus / run.name)
    except ScenarioError:
        return None
    if not (run / RESPONSE_FILE).is_file():
        return f"skipped (no {RESPONSE_FILE})"
    written = rescore_run(run, scenario, model)
    return f"updated ({', '.join(written)})" if written else "unchanged"


# --- eval run: the only command that sends prompts to the model APIs (TASK-005c3 F5) ---------

ANTHROPIC_KEY_VARIABLE = "POTAKT_ANTHROPIC_API_KEY"
GEMINI_KEY_VARIABLE = "POTAKT_GEMINI_API_KEY"
MIN_POLL_SECONDS = 10
INTERRUPTED = "interrupted: run the same command again to resume (nothing is lost)"


def read_key(variable: str) -> str:
    """The API key in ``variable``: the only reader of the process environment in ``potakt``
    (CLAUDE.md golden rule 3). Exit 1 if it is unset or empty, or not printable ASCII without
    whitespace; the value is never printed."""
    value = os.environ.get(variable, "")
    if not value:
        _stop(1, f"{variable} is not set in this shell")
    if not (value.isascii() and value.isprintable()) or any(char.isspace() for char in value):
        _stop(1, f"{variable} is not a valid key")
    return value


def make_transport() -> Transport:
    """The HTTPS transport of a real run (tests replace this factory)."""
    return UrllibTransport()


def utc_now() -> datetime:
    """The clock of a real run (tests replace it, together with ``time.sleep``)."""
    return datetime.now(UTC)


class _Counting:
    """A transport that counts the calls it is given (TASK-005c3d H13: no audit without one).
    A call counts when it is attempted, whatever its outcome: it may have reached the API."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport
        self.calls = 0

    def send(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
    ) -> HttpResponse:
        self.calls += 1
        return self._transport.send(method, url, headers, body, timeout)


@eval_app.command("run")
def eval_run(
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help=f"Model id: {', '.join(REGISTERED_MODELS)}.",
            show_default=False,
        ),
    ],
    runs: Annotated[
        Path | None,
        typer.Option(
            "--runs",
            help=(
                "Test A runs directory of all models, created if missing (default: the "
                "repository's docs/validation/runs/test-a, or docs/validation/runs/dev/round-1 "
                "with --dev)."
            ),
            show_default=False,
        ),
    ] = None,
    corpus: Annotated[
        Path | None,
        typer.Option(
            "--corpus",
            help=(
                "Scenario corpus directory (default: the repository's docs/validation/scenarios, "
                "or docs/validation/dev with --dev)."
            ),
            show_default=False,
        ),
    ] = None,
    poll_seconds: Annotated[
        int,
        typer.Option(
            "--poll-seconds",
            min=MIN_POLL_SECONDS,
            help="Seconds between two polls of a batch (Claude models), at least 10.",
        ),
    ] = 60,
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Send the requests (paid calls); without it, nothing is sent."),
    ] = False,
    release: Annotated[
        str | None,
        typer.Option(
            "--release",
            metavar="BATCH_ID",
            help=(
                "Mark a held batch done after the reviewer's decision (only with --yes), then "
                "continue the run."
            ),
            show_default=False,
        ),
    ] = None,
    repair: Annotated[
        bool,
        typer.Option(
            "--repair",
            help=(
                "Send one repair request per invalid candidate of the model's complete first "
                "run (ADR-0008 D7), recorded under <runs>/repair/<model dir>; never the gate."
            ),
        ),
    ] = False,
    repeat: Annotated[
        int | None,
        typer.Option(
            "--repeat",
            min=2,
            max=3,
            metavar="K",
            help=(
                "claude-opus-5-5 only: send the pinned first requests again as draw K (2 or 3, "
                "ADR-0008 D8), recorded under <runs>/repeat-<K>/claude-opus-5-5; never the gate."
            ),
            show_default=False,
        ),
    ] = None,
    dev: Annotated[
        bool,
        typer.Option(
            "--dev",
            help=(
                "Run the development set (ADR-0008 D1): docs/validation/dev, recorded under "
                "docs/validation/runs/dev/round-1; never the gate."
            ),
        ),
    ] = False,
) -> None:
    """Run Test A for one model; without --yes, print the plan and send nothing.

    Send every corpus scenario without a recorded answer and record the answers (ADR-0007 §7 and its
    amendments). Every run first checks the prompt file and the schema section against the SHA256
    values pinned in ADR-0008 Appendix A, which revises ADR-0007 §7, and stops, sending nothing, if
    one differs. Each request is checked too, when it is built and before anything is sent, against
    the SHA256 that the harness pins for its scenario and model (the dry run too).

    Without --yes, print the plan and the cost upper bound; nothing is sent and no key is read.
    With --yes, read the key from POTAKT_ANTHROPIC_API_KEY (Claude models, Message Batches API)
    or POTAKT_GEMINI_API_KEY (gemini-3.8-flash, generateContent), send, record, then print the
    model's summary. A Claude run first compares the laptop clock with the API's (the Date header
    of one free call) and stops, sending nothing, if they differ by more than 2 minutes.
    claude-opus-5-5 runs only after claude-haiku-5-5 has a complete run. The same command again
    resumes; a recorded scenario is never sent again. A batch whose results need the reviewer is
    held: every run stops until --yes --release <id> after the reviewer's decision, which first
    records what the batch's kept results already hold. Before the first batch of a Claude
    model, a run waits until 2 minutes after it started and stops, sending nothing, if the
    workspace has a batch that is not cancelled, is in no batches.json and is not in
    <runs>/acknowledged-batches.txt. Only one --yes run at a time: it holds the lock file
    <runs>/.eval-run.lock while it runs. After a Claude run that made an API call other than the
    clock check and was not interrupted, it runs a check for unrecorded batches after the run (up
    to 5 minutes later) and names every batch that no run recorded, so the reviewer can decide
    about a paid duplicate (only a batch still in progress can be cancelled); this check never
    changes the exit code.

    A Claude scenario without an answer after four batches is sent once more at least 24 hours
    after the fourth, and after a fifth it counts as an invalid candidate (unanswered.json,
    ADR-0008 D6). With --repair, only after the model's first run is complete, it sends one repair
    request per invalid candidate whose answer holds text (ADR-0008 D7), records under
    <runs>/repair/<model dir> with the files, lock, batch rules and checks of a first run, and
    prints the eval repairs report; with --repeat K (claude-opus-5-5 only, K = 2 after its
    complete first run, then K = 3 after repeat 2) it sends the pinned first requests again as
    draw K under <runs>/repeat-<K>/claude-opus-5-5 (ADR-0008 D8); none of these changes the gate.

    With --dev, run the development set instead (ADR-0008 D1): --corpus defaults to the
    repository's docs/validation/dev and --runs to docs/validation/runs/dev/round-1, and each
    request is checked against the development pins. A run with --dev refuses a corpus scenario
    and a runs directory under docs/validation/runs/test-a; a run without it refuses a
    development scenario and a runs directory under docs/validation/runs/dev. Before the commit
    tagged phase0-freeze, --yes without --dev stops before anything is sent: no corpus scenario
    is sent before the freeze; the dry run still prints the plan.

    Files: <runs>/<model dir>/<scenario id>/ with request.json, response.json, candidate.txt,
    comparison.json; for Claude models also <runs>/<model dir>/ batches.json and
    batch-<id>.results.jsonl (and unanswered.json for a scenario given up after five batches);
    the same under <runs>/repair/<model dir>/ and <runs>/repeat-<K>/claude-opus-5-5/.

    Exit code 0: everything pending is recorded, or the plan was printed; 1: the run stopped (the
    reason on stderr), a held batch, a missing or invalid key, claude-opus-5-5 before a complete
    claude-haiku-5-5 run, a repair or repeat run before the run it needs is complete, another run
    active (the lock file), a --release of a batch that is not held, an unknown batch in the
    workspace, a request that differs from its pin, a corpus --yes run before the freeze, a
    scenario or runs directory of the other set, a missing directory or a corpus without a
    scenario, an unexpected error (its type only, never its text), or an interruption; 2: usage
    error (also an unregistered model id, --release without --yes, --repeat for another model than
    claude-opus-5-5, or --repair with --repeat).
    """
    # TASK-005e6 decision 3: the defaults are resolved when the command runs.
    runs, corpus = _runs(runs, dev), _corpus(corpus, dev)
    try:
        kind = _run_kind(model, repair, repeat)
        _eval_run(model, runs, corpus, poll_seconds, yes, release, kind, dev)
    except (typer.Exit, typer.Abort):  # RuntimeError subclasses in Typer 0.27.3: not unexpected
        raise
    except KeyboardInterrupt:  # TASK-005c3h H8: also outside the run (Typer: exit 130, silent)
        _stop(1, INTERRUPTED)
    except Exception as error:  # TASK-005c3d H10: never a traceback, never the error's text
        _stop(1, UNEXPECTED.format(kind=type(error).__name__))


UNEXPECTED = (
    "eval run stopped by an unexpected error ({kind}): run the dry run and send its output and "
    "this message to the reviewer"
)
# TASK-005e5 decision 8 (ADR-0008 D8): the usage errors of --repeat.
REPEAT_WITH_REPAIR = "--repair and --repeat cannot be used together"
REPEAT_MODEL_ONLY = f"--repeat is only for {STRONG_MODEL} (ADR-0008 D8)"


def _run_kind(model: str, repair: bool, repeat: int | None) -> str | None:
    """The run kind of ``eval run``'s options (TASK-005e5 decisions 6 and 8): ``repair``,
    ``repeat-<K>`` or ``None`` (the first run); a usage error (exit 2) for ``--repair`` with
    ``--repeat`` and for ``--repeat`` with another model than the strong one (an unregistered
    model id is the usage error of ``_eval_run``)."""
    if repeat is None:
        return REPAIR_DIR if repair else None
    if repair:
        _stop(EXIT_USAGE, REPEAT_WITH_REPAIR)
    if model in REGISTERED_MODELS and model != STRONG_MODEL:
        _stop(EXIT_USAGE, REPEAT_MODEL_ONLY)
    return f"repeat-{repeat}"


def _eval_run(
    model: str,
    runs: Path,
    corpus: Path,
    poll_seconds: int,
    yes: bool,
    release: str | None,
    kind: str | None,
    dev: bool,
) -> None:
    try:
        check_registered(model)
    except UnknownModelError as error:
        _stop(EXIT_USAGE, str(error))
    if release is not None and not yes:
        _stop(EXIT_USAGE, "--release needs --yes: a dry run never changes anything")
    if not corpus.is_dir():
        _stop(1, f"{corpus} is not a directory")
    if runs.exists() and not runs.is_dir():
        _stop(1, f"{runs} is not a directory")
    if not corpus_scenarios(corpus):
        _stop(1, f"no scenario in {corpus}")
    anthropic = model.startswith(ANTHROPIC_PREFIX)
    try:
        check_set(corpus, runs, dev)  # TASK-005e6 decision 3 (ADR-0008 D1): the other set
        check_pins()  # TASK-005c3g G8: before anything is rendered, read or sent
        check_order(model, runs, corpus, kind)
        if not yes:
            typer.echo(run_plan(model, corpus, runs, kind, dev=dev), nl=False)
            return
        check_frozen(dev)  # TASK-005e6 decision 4: before any key, lock or request
        if release is not None and not anthropic:
            raise RunStopped(f"{server_text(release)} is not a held batch of {model}")
        key = read_key(ANTHROPIC_KEY_VARIABLE if anthropic else GEMINI_KEY_VARIABLE)
        # TASK-005c3g G2: the clock call goes through the raw transport, so it is not counted:
        # a run that made no other call (stopped by the clock, or nothing to run) has no audit.
        raw = make_transport()
        transport = _Counting(raw)
        lock = take_lock(runs, model, utc_now())  # E5: before anything else that writes
        try:
            if anthropic:
                stopped: Exception | None = None
                try:
                    run_anthropic(
                        model,
                        corpus,
                        runs,
                        key,
                        transport,
                        sleep=time.sleep,
                        now=utc_now,
                        poll_seconds=poll_seconds,
                        echo=typer.echo,
                        release=release,
                        clock_transport=raw,
                        kind=kind,
                    )
                except Exception as error:  # a stop, or an unexpected error (H16 review)
                    stopped = error
                # TASK-005c3d H13: after a run that made a counted call (any call but the clock
                # check) and was not interrupted (an interrupt skips this), with the lock still
                # held; it never changes the outcome: the stop or the error is raised again after
                # it.
                if transport.calls:
                    audit_batches(
                        model,
                        runs,
                        key,
                        transport,
                        sleep=time.sleep,
                        now=utc_now,
                        echo=typer.echo,
                        kind=kind,
                    )
                if stopped is not None:
                    raise stopped
            else:
                run_gemini(
                    model,
                    corpus,
                    runs,
                    key,
                    transport,
                    sleep=time.sleep,
                    now=utc_now,
                    echo=typer.echo,
                    kind=kind,
                )
        finally:
            warning = lock.release()  # however the run ends; only a crash leaves it
            if warning is not None:
                typer.echo(warning, err=True)
    except RunStopped as error:
        _stop(1, str(error))
    except KeyboardInterrupt:
        _stop(1, INTERRUPTED)
    model_dir = run_directory(runs, model, kind)
    if not model_dir.is_dir():
        return
    try:
        # TASK-005e5 decision 7: a repair run closes with eval repairs's text, not the summary.
        if kind == REPAIR_DIR:
            text = repairs_report(model, runs, corpus).to_text()
        else:
            text = summarize(model_dir, corpus).to_text()
    except (ValueError, RunStopped) as error:  # the records are kept; only the report is not
        typer.echo(f"cannot summarize {model_dir}: {error}", err=True)
        return
    except OSError as error:  # TASK-005c3e J5: likewise; never the error's text
        reason = error.strerror or type(error).__name__
        typer.echo(f"cannot summarize {model_dir}: {reason}", err=True)
        return
    typer.echo(text, nl=False)


# --- eval preflight: free checks before a paid run (TASK-005c3e J7) --------------------------

PREFLIGHT_UNEXPECTED = (
    "eval preflight stopped by an unexpected error ({kind}): send this message to the reviewer"
)
PREFLIGHT_INTERRUPTED = (
    "interrupted: the preflight sent no prompt and recorded nothing; run it again"
)


@eval_app.command("preflight")
def eval_preflight(
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help=f"Model id: {', '.join(REGISTERED_MODELS)}.",
            show_default=False,
        ),
    ],
) -> None:
    """Check the key and the model with free, read-only calls before a paid run.

    Free, read-only calls with the key of the run (ADR-0007 §7,
    amendment of TASK-005c3e): for a Claude model GET /v1/models/<id> and one page of the batch
    list, for gemini-3.8-flash GET /v1beta/models/<id>; no prompt is sent, nothing is written
    and nothing is charged.

    Reads the key from POTAKT_ANTHROPIC_API_KEY (Claude models) or POTAKT_GEMINI_API_KEY
    (gemini-3.8-flash) and prints one line per check. A check fails only on a definite
    contradiction (an answer for another model, batches or effort high not supported, an output
    limit below the run's, an unreadable batch list, a laptop clock more than 2 minutes from the
    API's (the Date header) for a Claude model); a field the API does not report is shown as not
    reported, except the model id, which must match.

    Exit code 0: every check passed; 1: a failed check (preflight failed: <check>: <reason> on
    stderr), a missing or invalid key, a request that failed after its retries, an unexpected
    error (its type only, never its text) or an interruption; 2: usage error (also an
    unregistered model id).
    """
    try:
        _eval_preflight(model)
    except (typer.Exit, typer.Abort):  # RuntimeError subclasses in Typer 0.27.3: not unexpected
        raise
    except KeyboardInterrupt:  # TASK-005c3h H8: also outside the calls (Typer: exit 130, silent)
        _stop(1, PREFLIGHT_INTERRUPTED)
    except Exception as error:  # H10: never a traceback, never the error's text
        _stop(1, PREFLIGHT_UNEXPECTED.format(kind=type(error).__name__))


def _eval_preflight(model: str) -> None:
    try:
        check_registered(model)
    except UnknownModelError as error:
        _stop(EXIT_USAGE, str(error))
    anthropic = model.startswith(ANTHROPIC_PREFIX)
    key = read_key(ANTHROPIC_KEY_VARIABLE if anthropic else GEMINI_KEY_VARIABLE)
    try:
        preflight(model, key, make_transport(), sleep=time.sleep, echo=typer.echo, now=utc_now)
    except RunStopped as error:
        _stop(1, str(error))
    except KeyboardInterrupt:
        _stop(1, PREFLIGHT_INTERRUPTED)
