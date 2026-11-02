"""``potakt eval run`` (TASK-005c3 F5): the dry run, the order rule, the key reader and the runs.

No test reads a real key or opens a connection: ``--yes`` runs get the fake key through
``monkeypatch.setenv`` and a fake transport through the ``make_transport`` factory.
"""

import ast
import json
from pathlib import Path

import pytest
import typer
from runner_helpers import ALL_SCENARIOS, KEY, Sleeps, anthropic_message, corpus_copy
from test_evaluation_runner_anthropic import FakeBatches, succeeded
from test_evaluation_runner_gemini import FakeGemini
from typer.testing import CliRunner, Result

import potakt.cli as cli
import potakt.evaluation.runner as runner_module
from potakt.cli import app, eval_app
from potakt.evaluation import CORPUS_PATH, load_scenario, render_test_a, summarize, write_answer
from potakt.evaluation.runner import BATCH_STATE_FILE, RUNS_PATH, cost_upper_bound
from potakt.evaluation.transport import HttpResponse, UrllibTransport

GOLDEN = Path(__file__).resolve().parent / "golden"
SOURCE = Path(__file__).resolve().parents[1] / "src" / "potakt"
HAIKU = "claude-haiku-5-5"
OPUS = "claude-opus-5-5"
GEMINI = "gemini-3.8-flash"
ANTHROPIC_VARIABLE = "POTAKT_ANTHROPIC_API_KEY"
GEMINI_VARIABLE = "POTAKT_GEMINI_API_KEY"

runner = CliRunner()


def eval_run(runs: Path, model: str, *extra: str, corpus: Path = CORPUS_PATH) -> Result:
    args = ["eval", "run", "--model", model, "--runs", str(runs), "--corpus", str(corpus)]
    return runner.invoke(app, [*args, *extra])


def shown(text: str, runs: Path) -> str:
    """Output with the machine-specific runs path replaced, separators as ``/``."""
    return text.replace(str(runs), "<runs>").replace("\\", "/")


def complete_haiku_run(runs: Path, corpus: Path = CORPUS_PATH) -> None:
    for name in ALL_SCENARIOS:
        scenario = load_scenario(corpus / name)
        write_answer(runs / HAIKU / name, scenario, HAIKU, anthropic_message(HAIKU))


class NoTransport:
    """A transport that fails the test if anything is sent."""

    def send(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
    ) -> HttpResponse:
        raise AssertionError(f"nothing may be sent: {method} {url}")


@pytest.fixture(autouse=True)
def no_real_transport(monkeypatch: pytest.MonkeyPatch) -> Sleeps:
    """Every test gets a transport that refuses to send and a recording sleep."""
    monkeypatch.setattr(cli, "make_transport", NoTransport)
    sleeps = Sleeps()
    monkeypatch.setattr(cli.time, "sleep", sleeps)
    return sleeps


# --- the dry run -----------------------------------------------------------------------------


def test_the_haiku_dry_run_matches_the_golden(tmp_path: Path) -> None:
    result = eval_run(tmp_path, HAIKU)

    assert result.exit_code == 0, result.stderr
    assert result.stderr == ""
    expected = (GOLDEN / "eval-run-plan-claude-haiku-5-5.txt").read_text(encoding="utf-8")
    assert shown(result.stdout, tmp_path) == expected
    assert not (tmp_path / HAIKU).exists()  # a dry run writes nothing


def test_the_opus_dry_run_after_a_complete_haiku_run_matches_the_golden(tmp_path: Path) -> None:
    complete_haiku_run(tmp_path)

    result = eval_run(tmp_path, OPUS)

    assert result.exit_code == 0, result.stderr
    expected = (GOLDEN / "eval-run-plan-claude-opus-5-5.txt").read_text(encoding="utf-8")
    assert shown(result.stdout, tmp_path) == expected


def test_the_gemini_dry_run_is_free(tmp_path: Path) -> None:
    result = eval_run(tmp_path, GEMINI)

    assert result.exit_code == 0
    assert "Cost upper bound: $0 (free tier)\n" in result.stdout
    assert result.stdout.endswith("Run again with --yes to send.\n")


def test_the_dry_run_reads_no_key_even_when_one_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    read: list[str] = []
    monkeypatch.setattr(cli, "read_key", read.append)

    result = eval_run(tmp_path, HAIKU)

    assert result.exit_code == 0
    assert read == []
    assert KEY not in result.stdout


def test_the_dry_run_lists_recorded_scenarios_and_an_open_batch(tmp_path: Path) -> None:
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:3])
    runs = tmp_path / "runs"
    write_answer(
        runs / HAIKU / "01-pharmacy",
        load_scenario(corpus / "01-pharmacy"),
        HAIKU,
        anthropic_message(HAIKU),
    )
    entry = {
        "id": "msgbatch_07",
        "scenarios": ["02-hotel-reception", "03-call-centre"],
        "created_at": "2026-11-02T10:00:05Z",
        "state": "open",
    }
    state = json.dumps({"batches": [entry]})
    (runs / HAIKU / BATCH_STATE_FILE).write_text(state, encoding="utf-8")

    result = eval_run(runs, HAIKU, corpus=corpus)

    assert result.exit_code == 0, result.stderr
    lines = result.stdout.splitlines()
    assert "Pending (2): 02-hotel-reception, 03-call-centre" in lines
    assert "Recorded (1): 01-pharmacy" in lines
    assert "Open batch: msgbatch_07 (2 scenarios), resumed by --yes" in lines


def prompt_bytes(name: str) -> int:
    prompt = render_test_a(load_scenario(CORPUS_PATH / name))
    return len(prompt.system.encode("utf-8")) + len(prompt.user.encode("utf-8"))


def test_the_cost_bound_uses_bytes_and_max_tokens_at_the_batch_prices() -> None:
    size = prompt_bytes("01-pharmacy")

    assert cost_upper_bound(HAIKU, CORPUS_PATH, ["01-pharmacy"]) == pytest.approx(
        (size * 0.05 + 32000 * 0.25) / 1e6
    )
    assert cost_upper_bound(OPUS, CORPUS_PATH, ["01-pharmacy"]) == pytest.approx(
        (size * 2.0 + 64000 * 10.0) / 1e6
    )


def test_a_prompt_over_the_long_prompt_threshold_is_priced_at_that_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    size = prompt_bytes("01-pharmacy")
    tiers = {HAIKU: (size - 1, (0.25, 1.25))}
    monkeypatch.setattr(runner_module, "LONG_PROMPT_PRICES_USD_PER_MILLION", tiers)

    assert cost_upper_bound(HAIKU, CORPUS_PATH, ["01-pharmacy"]) == pytest.approx(
        (size * 0.25 + 32000 * 1.25) / 1e6
    )
    monkeypatch.setattr(
        runner_module, "LONG_PROMPT_PRICES_USD_PER_MILLION", {HAIKU: (size, tiers[HAIKU][1])}
    )
    assert cost_upper_bound(HAIKU, CORPUS_PATH, ["01-pharmacy"]) == pytest.approx(
        (size * 0.05 + 32000 * 0.25) / 1e6
    )


def test_the_dry_run_of_a_complete_model_has_nothing_pending(tmp_path: Path) -> None:
    complete_haiku_run(tmp_path)

    result = eval_run(tmp_path, HAIKU)

    assert result.exit_code == 0
    assert "Pending (0): none\n" in result.stdout
    assert "Cost upper bound: $0.0000 " in result.stdout


# --- refusals ---------------------------------------------------------------------------------


def test_opus_before_a_complete_haiku_run_is_refused(tmp_path: Path) -> None:
    result = eval_run(tmp_path, OPUS, "--yes")

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == (
        "claude-opus-5-5 runs only after claude-haiku-5-5 has a complete run "
        f"(missing: {', '.join(ALL_SCENARIOS)})\n"
    )


def test_opus_after_an_incomplete_haiku_run_is_refused_even_as_a_dry_run(tmp_path: Path) -> None:
    complete_haiku_run(tmp_path)
    (tmp_path / HAIKU / "04-restaurant" / "comparison.json").unlink()

    result = eval_run(tmp_path, OPUS)

    assert result.exit_code == 1
    assert result.stderr == (
        "claude-opus-5-5 runs only after claude-haiku-5-5 has a complete run "
        "(missing: 04-restaurant)\n"
    )


@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "gpt-6"])
def test_an_unregistered_model_is_a_usage_error(tmp_path: Path, model: str) -> None:
    result = eval_run(tmp_path, model)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr == (
        f"no pre-registered Test A settings for {model!r}; "
        "use claude-haiku-5-5, gemini-3.8-flash or claude-opus-5-5\n"
    )


def test_the_model_is_required(tmp_path: Path) -> None:
    result = runner.invoke(app, ["eval", "run", "--runs", str(tmp_path)])

    assert result.exit_code == 2


def test_poll_seconds_below_10_is_a_usage_error(tmp_path: Path) -> None:
    result = eval_run(tmp_path, HAIKU, "--poll-seconds", "9")

    assert result.exit_code == 2


def test_a_missing_corpus_exits_1(tmp_path: Path) -> None:
    result = eval_run(tmp_path, HAIKU, corpus=tmp_path / "nope")

    assert result.exit_code == 1
    assert result.stderr == f"{tmp_path / 'nope'} is not a directory\n"


def test_a_runs_path_that_is_a_file_exits_1(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.write_text("", encoding="utf-8")

    result = eval_run(runs, HAIKU)

    assert result.exit_code == 1
    assert result.stderr == f"{runs} is not a directory\n"


@pytest.mark.parametrize(
    ("model", "variable"), [(HAIKU, ANTHROPIC_VARIABLE), (GEMINI, GEMINI_VARIABLE)]
)
def test_a_missing_key_exits_1_naming_the_variable(
    tmp_path: Path, model: str, variable: str
) -> None:
    result = eval_run(tmp_path, model, "--yes")

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == f"{variable} is not set in this shell\n"
    assert not (tmp_path / model).exists()


def test_an_empty_key_is_not_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, "")

    result = eval_run(tmp_path, HAIKU, "--yes")

    assert result.exit_code == 1
    assert result.stderr == f"{ANTHROPIC_VARIABLE} is not set in this shell\n"


@pytest.mark.parametrize("value", [f"{KEY} ", f" {KEY}", "potakt test", "potakt-tést", "key\x07"])
def test_a_key_with_whitespace_or_odd_characters_is_not_valid_and_never_printed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, value)

    result = eval_run(tmp_path, HAIKU, "--yes")

    assert result.exit_code == 1
    assert result.stderr == f"{ANTHROPIC_VARIABLE} is not a valid key\n"
    assert value.strip() not in result.stdout + result.stderr


# --- --yes with a fake transport --------------------------------------------------------------


def test_yes_runs_haiku_and_prints_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_real_transport: Sleeps
) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    server = FakeBatches()
    monkeypatch.setattr(cli, "make_transport", lambda: server)
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:3])
    runs = tmp_path / "runs"

    result = eval_run(runs, HAIKU, "--yes", "--poll-seconds", "10", corpus=corpus)

    assert result.exit_code == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout.startswith("batch msgbatch_01 created for 3 scenarios\n")
    summary = summarize(runs / HAIKU, corpus).to_text()
    assert result.stdout.endswith(summary)
    assert "Complete: yes" in summary
    assert no_real_transport == [10]
    assert all(call.headers["x-api-key"] == KEY for call in server.calls)
    assert KEY not in result.stdout


def test_yes_runs_gemini(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GEMINI_VARIABLE, KEY)
    server = FakeGemini()
    monkeypatch.setattr(cli, "make_transport", lambda: server)
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:2])

    result = eval_run(tmp_path / "runs", GEMINI, "--yes", corpus=corpus)

    assert result.exit_code == 0, result.stderr
    assert result.stdout.startswith("01-pharmacy: recorded (STOP, invalid, recall 0.000)\n")
    assert "Test A summary: gemini-3.8-flash\n" in result.stdout
    assert [call.headers["x-goog-api-key"] for call in server.calls] == [KEY, KEY]


def test_yes_with_opus_after_a_complete_haiku_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    complete_haiku_run(tmp_path)
    server = FakeBatches(polls_in_progress=0, outcome=lambda n, s: succeeded(model=OPUS))
    monkeypatch.setattr(cli, "make_transport", lambda: server)

    result = eval_run(tmp_path, OPUS, "--yes")

    assert result.exit_code == 0, result.stderr
    assert summarize(tmp_path / OPUS, CORPUS_PATH).complete


def test_a_stopped_run_exits_1_with_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": KEY}}
    rejected = HttpResponse(400, {}, json.dumps(body).encode())
    monkeypatch.setattr(cli, "make_transport", lambda: FakeBatches(create_script=[rejected]))
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:1])

    result = eval_run(tmp_path / "runs", HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == (
        "creating a batch failed: HTTP 400 (invalid_request_error: [redacted])\n"
    )


def test_an_interruption_exits_1_and_says_how_to_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)

    class Interrupted:
        def send(
            self,
            method: str,
            url: str,
            headers: dict[str, str],
            body: bytes | None,
            timeout: float,
        ) -> HttpResponse:
            raise KeyboardInterrupt

    monkeypatch.setattr(cli, "make_transport", Interrupted)
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:1])

    result = eval_run(tmp_path / "runs", HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 1
    assert result.stderr == "interrupted: run the same command again to resume (nothing is lost)\n"


# --- the app, the key reader and the help ---------------------------------------------------


def test_the_real_factory_builds_the_hardened_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()  # the autouse fixture replaced the factory

    assert isinstance(cli.make_transport(), UrllibTransport)


@pytest.mark.parametrize("typer_app", [app, eval_app], ids=["app", "eval_app"])
def test_tracebacks_never_show_local_variables(typer_app: typer.Typer) -> None:
    assert typer_app.pretty_exceptions_show_locals is False


def test_the_default_runs_directory_is_the_repository_one() -> None:
    expected = Path(__file__).resolve().parents[2] / "docs" / "validation" / "runs" / "test-a"

    assert expected == RUNS_PATH


def environment_readers(path: Path) -> list[str]:
    """The functions of a module that read ``os.environ`` or call ``getenv``."""
    tree = ast.parse(path.read_bytes())
    found: list[str] = []
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for node in ast.walk(function):
            if (isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv")) or (
                isinstance(node, ast.Name) and node.id in ("environ", "getenv")
            ):
                found.append(function.name)
                break
    return found


def test_read_key_is_the_only_reader_of_the_environment_in_src() -> None:
    readers = {
        path.relative_to(SOURCE).as_posix(): names
        for path in sorted(SOURCE.rglob("*.py"))
        if (names := environment_readers(path))
    }
    text = "\n".join(path.read_text(encoding="utf-8") for path in SOURCE.rglob("*.py"))

    assert readers == {"cli.py": ["read_key"]}
    assert text.count("os.environ") == 1
    assert "getenv" not in text


def test_no_key_variable_without_the_potakt_prefix_in_src() -> None:
    for path in SOURCE.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for variable in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
            assert text.count(variable) == text.count(f"POTAKT_{variable}"), path


def help_text(*command: str) -> str:
    result = runner.invoke(app, [*command, "--help"], terminal_width=200)
    assert result.exit_code == 0
    return " ".join(result.stdout.replace("\N{BOX DRAWINGS LIGHT VERTICAL}", " ").split())


def test_the_eval_group_says_which_command_calls_the_apis() -> None:
    text = help_text("eval")

    assert "eval run calls the model APIs" in text
    assert "the other commands are offline: no model calls, no credentials" in text


def test_the_run_help_states_the_exit_codes_and_files() -> None:
    text = help_text("eval", "run")

    assert "Without --yes, print the plan and the cost upper bound" in text
    assert "POTAKT_ANTHROPIC_API_KEY" in text
    assert "POTAKT_GEMINI_API_KEY" in text
    assert "request.json, response.json, candidate.txt, comparison.json" in text
    assert "batches.json and batch-<id>.results.jsonl" in text
    assert (
        "Exit code 0: everything pending is recorded, or the plan was printed; 1: the run "
        "stopped (the reason on stderr), a missing or invalid key, claude-opus-5-5 before a "
        "complete claude-haiku-5-5 run, a missing directory, or an interruption; 2: usage error "
        "(also an unregistered model id)"
    ) in text
    assert "default: the repository's docs/validation/runs/test-a" in text
    assert str(RUNS_PATH) not in text


# --- the runbook (F6) -------------------------------------------------------------------------

RUNBOOK = Path(__file__).resolve().parents[2] / "docs" / "validation" / "test-a-runbook.md"
SESSION_BLOCK = (
    "```powershell\n"
    "cd <repo>\\python\n"
    '$secure = Read-Host "Anthropic API key" -AsSecureString\n'
    "$env:POTAKT_ANTHROPIC_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR("
    "[Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))\n"
    "Remove-Variable secure\n"
    "```\n"
)


def test_the_runbook_has_the_verbatim_session_block_for_both_keys() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")

    assert SESSION_BLOCK in text
    gemini = SESSION_BLOCK.replace("cd <repo>\\python\n", "").replace("Anthropic", "Gemini")
    assert gemini.replace("ANTHROPIC", "GEMINI") in text
    assert "Remove-Item Env:POTAKT_ANTHROPIC_API_KEY, Env:POTAKT_GEMINI_API_KEY" in text


def test_the_runbook_steps_are_in_order_and_hold_no_personal_path() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    headings = [line for line in text.splitlines() if line.startswith("## ")]

    assert headings == [
        "## 1. Keys",
        "## 2. Session-only variables",
        "## 3. Haiku",
        "## 4. Check the Haiku records",
        "## 5. Gemini",
        "## 6. Opus",
        "## 7. Commit the records",
        "## 8. If something goes wrong",
    ]
    assert "potakt-phase0" in text
    assert "$10" in text
    assert 'git commit -m "docs(validation): record Test A runs (<model>)"' in text
    assert "C:\\Users" not in text
    assert KEY not in text
