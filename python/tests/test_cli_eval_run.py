"""``potakt eval run`` (TASK-005c3 F5): the dry run, the order rule, the key reader and the runs.

No test reads a real key or opens a connection: ``--yes`` runs get the fake key through
``monkeypatch.setenv`` and a fake transport through the ``make_transport`` factory.
"""

import ast
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import typer
from runner_helpers import (
    ALL_SCENARIOS,
    KEY,
    Sleeps,
    anthropic_message,
    batch_entry,
    corpus_copy,
    write_state,
)
from test_evaluation_runner_anthropic import FakeBatches, succeeded
from test_evaluation_runner_gemini import FakeGemini
from typer.testing import CliRunner, Result

import potakt.cli as cli
import potakt.evaluation.runner as runner_module
from potakt.cli import app, eval_app
from potakt.evaluation import CORPUS_PATH, load_scenario, render_test_a, summarize, write_answer
from potakt.evaluation.runner import RUNS_PATH, cost_upper_bound
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
    """Every test gets a transport that refuses to send, a recording sleep and its clock."""
    monkeypatch.setattr(cli, "make_transport", NoTransport)
    sleeps = Sleeps()
    monkeypatch.setattr(cli.time, "sleep", sleeps)
    monkeypatch.setattr(cli, "utc_now", sleeps.now)
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
    pending = ["02-hotel-reception", "03-call-centre"]
    write_state(runs / HAIKU, batch_entry("msgbatch_07", pending, "open"))

    result = eval_run(runs, HAIKU, corpus=corpus)

    assert result.exit_code == 0, result.stderr
    lines = result.stdout.splitlines()
    assert "Pending (2): 02-hotel-reception, 03-call-centre" in lines
    assert "Recorded (1): 01-pharmacy" in lines
    assert "Open batch: msgbatch_07 (2 scenarios), resumed by --yes" in lines
    assert not any(line.startswith("Unconfirmed") for line in lines)


def test_the_dry_run_shows_an_unconfirmed_create_request_and_lists_nothing(
    tmp_path: Path,
) -> None:
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:3])
    runs = tmp_path / "runs"
    failed = batch_entry(None, ["01-pharmacy"], "failed", started_at="2026-11-02T09:00:00Z")
    creating = batch_entry(None, ALL_SCENARIOS[:3], "creating", started_at="2026-11-02T10:15:03Z")
    write_state(runs / HAIKU, failed, creating)

    result = eval_run(runs, HAIKU, corpus=corpus)  # NoTransport: any call fails the test

    assert result.exit_code == 0, result.stderr
    assert (
        "Unconfirmed create request: started 2026-11-02T10:15:03Z (3 scenarios); --yes checks "
        "for its batch before sending anything"
    ) in result.stdout.splitlines()
    assert "Open batch" not in result.stdout


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
    server = FakeBatches()

    class InterruptedPoll:
        """Creates the batch, then is interrupted at the first poll."""

        def send(
            self,
            method: str,
            url: str,
            headers: dict[str, str],
            body: bytes | None,
            timeout: float,
        ) -> HttpResponse:
            if method == "GET":
                raise KeyboardInterrupt
            return server.send(method, url, headers, body, timeout)

    monkeypatch.setattr(cli, "make_transport", InterruptedPoll)
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:1])

    result = eval_run(tmp_path / "runs", HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 1
    assert result.stderr == "interrupted: run the same command again to resume (nothing is lost)\n"


def test_an_interruption_during_the_create_call_is_resumed_by_the_same_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # TASK-005c3b D2 step 5: "nothing is lost" is now true while the create call is in flight.
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    server = FakeBatches(create_script=["interrupt-created"])
    monkeypatch.setattr(cli, "make_transport", lambda: server)
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:1])

    interrupted = eval_run(tmp_path / "runs", HAIKU, "--yes", corpus=corpus)
    resumed = eval_run(tmp_path / "runs", HAIKU, "--yes", corpus=corpus)

    assert interrupted.exit_code == 1
    assert interrupted.stderr == (
        "interrupted: run the same command again to resume (nothing is lost)\n"
    )
    assert resumed.exit_code == 0, resumed.stderr
    assert resumed.stdout.startswith(
        "adopted batch msgbatch_01 created by an earlier create request\n"
    )
    assert len(server.creates()) == 1
    assert "Complete: yes" in resumed.stdout


# --- TASK-005c3c E4: --release ----------------------------------------------------------------

HELD_PROBLEM = "batch msgbatch_07: no result for 02-hotel-reception"


def held_state(runs: Path) -> None:
    held = batch_entry("msgbatch_07", ALL_SCENARIOS[:2], "held", problems=[HELD_PROBLEM])
    write_state(runs / HAIKU, held)


def test_release_without_yes_is_a_usage_error(tmp_path: Path) -> None:
    held_state(tmp_path)
    before = (tmp_path / HAIKU / "batches.json").read_bytes()

    result = eval_run(tmp_path, HAIKU, "--release", "msgbatch_07")

    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr == "--release needs --yes: a dry run never changes anything\n"
    assert (tmp_path / HAIKU / "batches.json").read_bytes() == before


@pytest.mark.parametrize("state", ["open", "done", "unknown"])
def test_release_of_a_batch_that_is_not_held_exits_1_and_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:2])
    runs = tmp_path / "runs"
    entries = [] if state == "unknown" else [batch_entry("msgbatch_07", ALL_SCENARIOS[:2], state)]
    write_state(runs / HAIKU, *entries)
    before = (runs / HAIKU / "batches.json").read_bytes()

    result = eval_run(runs, HAIKU, "--yes", "--release", "msgbatch_07", corpus=corpus)

    assert result.exit_code == 1
    assert result.stderr == "msgbatch_07 is not a held batch of claude-haiku-5-5\n"
    assert (runs / HAIKU / "batches.json").read_bytes() == before


def test_release_for_gemini_exits_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GEMINI_VARIABLE, KEY)

    result = eval_run(tmp_path, GEMINI, "--yes", "--release", "msgbatch_07")

    assert result.exit_code == 1
    assert result.stderr == "msgbatch_07 is not a held batch of gemini-3.8-flash\n"
    assert not (tmp_path / ".eval-run.lock").exists()


def test_a_held_batch_stops_yes_and_release_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    server = FakeBatches()
    monkeypatch.setattr(cli, "make_transport", lambda: server)
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:2])
    runs = tmp_path / "runs"
    write_answer(
        runs / HAIKU / "01-pharmacy",
        load_scenario(corpus / "01-pharmacy"),
        HAIKU,
        anthropic_message(HAIKU),
    )
    held_state(runs)

    stopped = eval_run(runs, HAIKU, "--yes", corpus=corpus)
    released = eval_run(runs, HAIKU, "--yes", "--release", "msgbatch_07", corpus=corpus)

    assert stopped.exit_code == 1
    assert stopped.stderr == (
        f"batch msgbatch_07 is held for the reviewer: {HELD_PROBLEM}; after the reviewer's "
        "decision, run the same command with --release msgbatch_07\n"
    )
    assert released.exit_code == 0, released.stderr
    assert released.stdout.startswith(
        "batch msgbatch_07 released after review\nbatch msgbatch_01 created for 1 scenarios\n"
    )
    assert server.creates() == [["02-hotel-reception"]]
    assert "Complete: yes" in released.stdout


def test_the_dry_run_shows_a_held_batch(tmp_path: Path) -> None:
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:2])
    runs = tmp_path / "runs"
    held_state(runs)

    result = eval_run(runs, HAIKU, corpus=corpus)

    assert result.exit_code == 0, result.stderr
    assert (
        f"Held batch: msgbatch_07 (2 scenarios): {HELD_PROBLEM}; --yes --release msgbatch_07 "
        "after the reviewer's decision"
    ) in result.stdout.splitlines()


# --- TASK-005c3c E5: one run at a time ----------------------------------------------------------

LOCK = ".eval-run.lock"


def test_a_run_takes_the_lock_and_removes_it_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    runs = tmp_path / "runs"
    seen: list[str] = []
    server = FakeBatches(on_first_poll=lambda: seen.append(lock_text(runs)))
    monkeypatch.setattr(cli, "make_transport", lambda: server)
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:1])

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 0, result.stderr
    assert seen == ["claude-haiku-5-5 2026-11-02T10:00:00Z\n"]
    assert not (runs / LOCK).exists()


def lock_text(runs: Path) -> str:
    return (runs / LOCK).read_bytes().decode("utf-8")


def test_the_lock_is_removed_after_a_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    rejected = HttpResponse(400, {}, b"")
    monkeypatch.setattr(cli, "make_transport", lambda: FakeBatches(create_script=[rejected]))
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:1])

    result = eval_run(tmp_path / "runs", HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 1
    assert result.stderr == "creating a batch failed: HTTP 400\n"
    assert not (tmp_path / "runs" / LOCK).exists()


def test_the_lock_is_removed_after_an_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    server = FakeBatches(create_script=["interrupt-created"])
    monkeypatch.setattr(cli, "make_transport", lambda: server)
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:1])

    result = eval_run(tmp_path / "runs", HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 1
    assert result.stderr == cli.INTERRUPTED + "\n"
    assert not (tmp_path / "runs" / LOCK).exists()


@pytest.mark.parametrize("model", [HAIKU, GEMINI])
def test_a_second_run_while_the_lock_exists_exits_1_and_sends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    monkeypatch.setenv(GEMINI_VARIABLE, KEY)
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:1])
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / LOCK).write_bytes(b"claude-opus-5-5 2026-11-02T09:00:00Z\n")

    result = eval_run(runs, model, "--yes", corpus=corpus)  # NoTransport: any call fails

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == (
        "another eval run is active (claude-opus-5-5 2026-11-02T09:00:00Z), or one was cut off: "
        f"if no other window runs eval run, delete {runs / LOCK} and run again\n"
    )
    assert lock_text(runs) == "claude-opus-5-5 2026-11-02T09:00:00Z\n"  # left as it was
    assert not (runs / model).exists()


def test_an_odd_lock_is_shown_safely(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / LOCK).write_bytes(b"\x1b[2J\xff" + b"x" * 500)

    result = eval_run(runs, HAIKU, "--yes")

    assert result.exit_code == 1
    assert "\x1b" not in result.stderr
    assert result.stderr.startswith("another eval run is active (?[2J")
    assert len(result.stderr) < 400
    assert (runs / LOCK).exists()  # never removed by the run it stopped


def test_the_dry_run_reports_a_lock_and_takes_none(tmp_path: Path) -> None:
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:1])
    runs = tmp_path / "runs"
    unlocked = eval_run(runs, HAIKU, corpus=corpus)
    runs.mkdir()
    (runs / LOCK).write_bytes(b"claude-haiku-5-5 2026-11-02T09:00:00Z\n")

    locked = eval_run(runs, HAIKU, corpus=corpus)

    assert unlocked.exit_code == 0
    assert not any(line.startswith("Lock:") for line in unlocked.stdout.splitlines())
    assert locked.exit_code == 0
    assert (
        "Lock: claude-haiku-5-5 2026-11-02T09:00:00Z (another run may be active)"
        in locked.stdout.splitlines()
    )
    assert sorted(path.name for path in runs.iterdir()) == [LOCK]  # nothing else written


def test_the_repository_lock_is_ignored_by_git() -> None:
    gitignore = Path(__file__).resolve().parents[2] / ".gitignore"
    lines = gitignore.read_text(encoding="utf-8").splitlines()

    assert "docs/validation/runs/test-a/.eval-run.lock" in lines
    repository = Path(__file__).resolve().parents[2]
    assert repository / "docs/validation/runs/test-a/.eval-run.lock" == RUNS_PATH / LOCK


def test_a_lock_that_cannot_be_read_still_blocks_with_the_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A directory in the lock's place: every OS refuses to read or create it; our code decides.
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    runs = tmp_path / "runs"
    (runs / LOCK).mkdir(parents=True)

    result = eval_run(runs, HAIKU, "--yes")

    assert result.exit_code == 1
    assert result.stderr.startswith("another eval run is active (unreadable), or one was cut off")
    assert (runs / LOCK).is_dir()


def test_a_lock_that_cannot_be_created_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    real_open = Path.open

    def refusing_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if (args[0] if args else kwargs.get("mode")) == "x":
            raise PermissionError(13, "Permission denied")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", refusing_open)
    runs = tmp_path / "runs"

    result = eval_run(runs, HAIKU, "--yes")

    assert result.exit_code == 1
    assert result.stderr == f"cannot create {runs / LOCK}: Permission denied\n"


# --- the app, the key reader and the help ---------------------------------------------------


def test_the_real_factory_builds_the_hardened_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()  # the autouse fixture replaced the factory

    assert isinstance(cli.make_transport(), UrllibTransport)


def test_the_real_clock_is_aware_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()  # the autouse fixture replaced the clock

    assert cli.utc_now().utcoffset() == timedelta(0)


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
        "stopped (the reason on stderr), a held batch, a missing or invalid key, claude-opus-5-5 "
        "before a complete claude-haiku-5-5 run, another run active (the lock file), a --release "
        "of a batch that is not held, a missing directory, or an interruption; 2: usage error "
        "(also an unregistered model id, or --release without --yes)"
    ) in text
    assert "--release" in text
    assert (
        "Mark a held batch done after the reviewer's decision (only with --yes), then continue"
        in text
    )
    assert "<runs>/.eval-run.lock" in text
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


def test_the_runbook_quotes_the_reconciliation_messages_and_the_new_rules() -> None:
    # TASK-005c3b D9: the runbook names the runner's messages as they are printed.
    text = " ".join(RUNBOOK.read_text(encoding="utf-8").split())
    lines = runner_module.SEVERAL_CANDIDATES.format(count=2, ids="<ids>")

    for quoted in (
        "`adopted batch <id> created by an earlier create request`",
        "`no batch was created by the earlier create request (started <time>)`",
        "`Unconfirmed create request: started <time> (<n> scenarios); --yes checks for its batch "
        "before sending anything`",
        "`<scenario>: invalid_request_error: …`",
        "`" + runner_module.HELD.format(id="<id>")[2:] + "`",
        "`batch <id> is held for the reviewer: <problems>; after the reviewer's decision, run the "
        "same command with --release <id>`",
        "`batch <id> released after review`",
        "`batch <id>: no result for <scenario>`",
        "cancel all of them but the oldest, then run the same command again; it adopts the one "
        "left",
        "One run at a time: never run `potakt eval run` in two windows at once",
        "Keep the laptop awake and plugged in while a batch runs",
        "again after every 10 polls without a change (every 10 minutes)",
    ):
        assert quoted in text, quoted
    assert "cancel all but the oldest, then run the same command again" in lines
    # The old exception (an interrupted create needed the Console) is gone.
    assert "interrupted while creating a batch:" not in text
    assert "cancel any batch created in the last minutes" not in text
    assert "a crash" in text
