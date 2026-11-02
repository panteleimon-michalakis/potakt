"""``potakt eval run`` (TASK-005c3 F5): the dry run, the order rule, the key reader and the runs.

No test reads a real key or opens a connection: ``--yes`` runs get the fake key through
``monkeypatch.setenv`` and a fake transport through the ``make_transport`` factory.
"""

import ast
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import typer
from runner_helpers import (
    ALL_SCENARIOS,
    KEY,
    NOW,
    Sleeps,
    anthropic_message,
    batch_entry,
    corpus_copy,
    write_state,
)
from test_evaluation_runner_anthropic import CreateStep, FakeBatches, other_batch, succeeded
from test_evaluation_runner_gemini import FakeGemini
from typer.testing import CliRunner, Result

import potakt.cli as cli
import potakt.evaluation.runner as runner_module
import potakt.evaluation.test_a as test_a_module
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
    # The poll, then the audit's wait until started_at + 300 s (TASK-005c3d H13).
    assert no_real_transport == [10, 290]
    assert "waiting 290 s before checking for batches that no run recorded\n" in result.stdout
    assert all(call.headers["authorization"] == f"Bearer {KEY}" for call in server.calls)
    assert not any("x-api-key" in call.headers for call in server.calls)
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
    # The create was a call, so the audit runs before the stop message (TASK-005c3d H13).
    assert result.stdout == "waiting 300 s before checking for batches that no run recorded\n"
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
        "waiting 180 s before looking for a batch the create request may have made\n"
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


def test_the_lock_line_is_synced_before_the_lock_is_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # TASK-005c3d H11: take_lock fsyncs its line, like every write of the runner.
    runs = tmp_path / "runs"
    synced: list[str] = []

    def fsync(fd: int) -> None:
        synced.append(lock_text(runs))  # the line is flushed and the file still open

    monkeypatch.setattr(os, "fsync", fsync)

    lock = runner_module.take_lock(runs, HAIKU, datetime(2026, 11, 2, 10, tzinfo=UTC))

    assert synced == ["claude-haiku-5-5 2026-11-02T10:00:00Z\n"]
    assert lock.release() is None


# -- E9 review findings (Minor 4): the lock's edge cases --


def test_a_lock_that_cannot_be_written_is_removed_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # It used to escape as a traceback and leave an empty lock behind ("active ()").
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    real_open = Path.open

    class FullDisk:
        def __init__(self, handle: Any) -> None:
            self.handle = handle

        def __enter__(self) -> "FullDisk":
            return self

        def __exit__(self, *exc: object) -> None:
            self.handle.close()

        def write(self, text: str) -> int:
            raise OSError(28, "No space left on device")

    def full_disk_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        handle = real_open(self, *args, **kwargs)
        return FullDisk(handle) if (args[0] if args else kwargs.get("mode")) == "x" else handle

    monkeypatch.setattr(Path, "open", full_disk_open)
    runs = tmp_path / "runs"

    result = eval_run(runs, HAIKU, "--yes")

    assert result.exit_code == 1
    assert result.stderr == f"cannot write {runs / LOCK}: No space left on device\n"
    assert not (runs / LOCK).exists()


def test_a_lock_gone_before_it_could_be_read_is_not_shown_as_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    real_open = Path.open

    def racing_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if (args[0] if args else kwargs.get("mode")) == "x":
            raise FileExistsError(17, "File exists")  # and it is gone again at once
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", racing_open)
    runs = tmp_path / "runs"

    result = eval_run(runs, HAIKU, "--yes")

    assert result.exit_code == 1
    assert result.stderr.startswith("another eval run is active (gone), or one was cut off")


def test_a_lock_that_cannot_be_removed_keeps_the_result_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A PermissionError in the finally used to replace the summary with a traceback.
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    monkeypatch.setattr(cli, "make_transport", FakeBatches)
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:1])
    runs = tmp_path / "runs"
    real_unlink = Path.unlink

    def held_by_antivirus(self: Path, missing_ok: bool = False) -> None:
        if self.name == LOCK:
            raise PermissionError(13, "Permission denied")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", held_by_antivirus)

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 0, result.stderr
    assert "Complete: yes" in result.stdout
    assert result.stderr == (
        f"could not remove the lock {runs / LOCK} (Permission denied): delete it before the "
        "next run\n"
    )


def test_a_lock_deleted_during_the_run_ends_quietly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    runs = tmp_path / "runs"

    def owner_deletes_it() -> None:
        (runs / LOCK).unlink()

    server = FakeBatches(on_first_poll=owner_deletes_it)
    monkeypatch.setattr(cli, "make_transport", lambda: server)
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:1])

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 0, result.stderr
    assert result.stderr == ""
    assert not (runs / LOCK).exists()


def test_a_lock_that_cannot_be_read_at_the_end_is_still_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    monkeypatch.setattr(cli, "make_transport", FakeBatches)
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:1])
    runs = tmp_path / "runs"
    real_read_bytes = Path.read_bytes

    def locked_for_reading(self: Path) -> bytes:
        if self.name == LOCK:
            raise PermissionError(13, "Permission denied")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", locked_for_reading)

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 0, result.stderr
    assert result.stderr == ""
    assert not (runs / LOCK).exists()


def test_a_lock_taken_over_by_another_run_is_left_to_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The owner deleted a "stale" lock while run A was still running and started run B: A's
    # finally used to remove B's lock.
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    runs = tmp_path / "runs"
    other = b"claude-opus-5-5 2026-11-02T10:00:30Z\n"

    def another_run_takes_over() -> None:
        (runs / LOCK).write_bytes(other)

    server = FakeBatches(on_first_poll=another_run_takes_over)
    monkeypatch.setattr(cli, "make_transport", lambda: server)
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:1])

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 0, result.stderr
    assert (runs / LOCK).read_bytes() == other
    assert result.stderr == (
        f"the lock {runs / LOCK} now belongs to another run (claude-opus-5-5 "
        "2026-11-02T10:00:30Z); it was left in place\n"
    )


# --- TASK-005c3d H10: eval run never ends with a traceback --------------------------------------


def assert_clean_exit_1(result: Result, message: str) -> None:
    """Exit 1 with ``message`` on stderr, no traceback, and the fake key in no output."""
    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit)
    assert message in result.stderr
    assert "Traceback" not in result.output
    assert KEY not in result.output


# Gemini's dry run renders no prompt (its cost is $0), so it has no case here.
WITH_PROMPT = [(HAIKU, False), (HAIKU, True), (GEMINI, True)]
WITH_PROMPT_IDS = ["haiku-dry-run", "haiku-yes", "gemini-yes"]


@pytest.mark.parametrize(("model", "yes"), WITH_PROMPT, ids=WITH_PROMPT_IDS)
def test_a_missing_instruction_file_exits_1_with_the_eval_prompt_wording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, yes: bool, model: str
) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    monkeypatch.setenv(GEMINI_VARIABLE, KEY)
    missing = tmp_path / "test-a.md"
    monkeypatch.setattr(test_a_module, "INSTRUCTION_PATH", missing)
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:1])

    result = eval_run(tmp_path / "runs", model, *(["--yes"] if yes else []), corpus=corpus)

    assert_clean_exit_1(result, f"cannot read the instruction file {missing}: ")
    assert not (tmp_path / "runs" / ".eval-run.lock").exists()


def undecodable_unbound_csv(tmp_path: Path) -> Path:
    """A corpus whose scenario lists a CSV that the truth does not bind and that is not UTF-8:
    the scenario loads, the bundle cannot decode the file."""
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:1])
    directory = corpus / "01-pharmacy"
    (directory / "extra.csv").write_bytes(b"name\n\xff\xfe\n")
    metadata = json.loads((directory / "scenario.json").read_bytes())
    metadata["csv_files"].append("extra.csv")
    (directory / "scenario.json").write_text(json.dumps(metadata), encoding="utf-8")
    return corpus


@pytest.mark.parametrize(("model", "yes"), WITH_PROMPT, ids=WITH_PROMPT_IDS)
def test_an_undecodable_csv_exits_1_naming_the_scenario(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, yes: bool, model: str
) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    monkeypatch.setenv(GEMINI_VARIABLE, KEY)
    corpus = undecodable_unbound_csv(tmp_path)

    result = eval_run(tmp_path / "runs", model, *(["--yes"] if yes else []), corpus=corpus)

    assert_clean_exit_1(result, "01-pharmacy: extra.csv is not valid utf-8\n")


def test_a_request_that_cannot_be_written_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:1])
    runs = tmp_path / "runs"

    def full_disk(run_dir: Path, payload: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(runner_module, "write_request", full_disk)

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    path = runs / HAIKU / "01-pharmacy" / "request.json"
    assert_clean_exit_1(result, f"cannot write {path}: No space left on device\n")


@pytest.mark.parametrize("yes", [False, True], ids=["dry-run", "yes"])
@pytest.mark.parametrize("model", [HAIKU, GEMINI, OPUS])
def test_an_empty_corpus_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, yes: bool, model: str
) -> None:
    # U4: summarize raised ValueError for an empty corpus, after a run that sent nothing.
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    monkeypatch.setenv(GEMINI_VARIABLE, KEY)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "not-a-scenario").mkdir()

    result = eval_run(tmp_path / "runs", model, *(["--yes"] if yes else []), corpus=corpus)

    assert_clean_exit_1(result, f"no scenario in {corpus}\n")
    assert result.stderr == f"no scenario in {corpus}\n"
    assert not (tmp_path / "runs").exists()


def test_an_unexpected_error_exits_1_without_its_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:1])
    runs = tmp_path / "runs"

    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError(f"x-api-key: {KEY}")

    monkeypatch.setattr(cli, "run_anthropic", broken)

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert_clean_exit_1(result, "")
    assert result.stderr == (
        "eval run stopped by an unexpected error (RuntimeError): run the dry run and send its "
        "output and this message to the reviewer\n"
    )
    assert not (runs / LOCK).exists()


def test_a_missing_key_keeps_its_own_message(tmp_path: Path) -> None:
    # typer.Exit subclasses RuntimeError in Typer 0.27.3: the last resort must not catch it.
    result = eval_run(tmp_path, HAIKU, "--yes")

    assert result.exit_code == 1
    assert result.stderr == f"{ANTHROPIC_VARIABLE} is not set in this shell\n"
    assert "unexpected error" not in result.output


def test_a_summary_that_fails_is_reported_and_the_run_still_exits_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(GEMINI_VARIABLE, KEY)
    monkeypatch.setattr(cli, "make_transport", FakeGemini)
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:1])
    runs = tmp_path / "runs"

    def broken(runs_dir: Path, corpus_dir: Path) -> Any:
        raise ValueError(f"{runs_dir} is broken")

    monkeypatch.setattr(cli, "summarize", broken)

    result = eval_run(runs, GEMINI, "--yes", corpus=corpus)

    assert result.exit_code == 0, result.output
    assert result.stdout == "01-pharmacy: recorded (STOP, invalid, recall 0.000)\n"
    assert result.stderr == f"cannot summarize {runs / GEMINI}: {runs / GEMINI} is broken\n"


def test_no_summary_without_a_model_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(GEMINI_VARIABLE, KEY)
    monkeypatch.setattr(cli, "run_gemini", lambda *args, **kwargs: None)
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:1])

    result = eval_run(tmp_path / "runs", GEMINI, "--yes", corpus=corpus)

    assert result.exit_code == 0, result.output
    assert result.output == ""


# --- TASK-005c3d H13: the audit of batches that no run recorded ---------------------------------

AUDIT_WAIT = "waiting {} s before checking for batches that no run recorded"
UNRECORDED = (
    "warning: batch {id} ({count} requests, created {created}) is recorded by no run: it may be "
    "a paid duplicate; send this warning to the reviewer, who decides whether to cancel it in the "
    "Claude Console (Batches)"
)


Setup = Callable[[FakeBatches], tuple[Path, Path]]


@pytest.fixture
def haiku_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_real_transport: Sleeps) -> Setup:
    """Set up a Haiku ``--yes`` run on one scenario against ``server``, which shares the run's
    clock (it sees the seconds slept)."""

    def setup(server: FakeBatches) -> tuple[Path, Path]:
        monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
        server.clock = no_real_transport
        monkeypatch.setattr(cli, "make_transport", lambda: server)
        return corpus_copy(tmp_path, ALL_SCENARIOS[:1]), tmp_path / "runs"

    return setup


def test_the_audit_names_a_late_duplicate_of_an_adopted_batch(haiku_run: Setup) -> None:
    # R4: a proxy-duplicated create whose second batch is listed only 130 s after the answer
    # is still paid twice; the audit names it so that the owner can cancel it.
    server = FakeBatches(create_script=["duplicated-staggered"], lag=90)
    corpus, runs = haiku_run(server)

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert "adopted batch msgbatch_01 created by an earlier create request" in lines
    assert AUDIT_WAIT.format(120) in lines
    warning = UNRECORDED.format(id="msgbatch_02", count=1, created="2026-11-02T10:00:40Z")
    assert warning in lines
    assert lines.index(warning) < lines.index("Test A summary: claude-haiku-5-5")
    assert result.stderr == ""
    state = json.loads((runs / HAIKU / "batches.json").read_bytes())
    assert [(entry["id"], entry["state"]) for entry in state["batches"]] == [
        ("msgbatch_01", "done")
    ]
    assert server.list_times[-1] == NOW + timedelta(seconds=300)


def test_a_clean_run_is_audited_once_without_a_warning(
    haiku_run: Setup, no_real_transport: Sleeps
) -> None:
    server = FakeBatches()
    corpus, runs = haiku_run(server)

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 0, result.output
    assert AUDIT_WAIT.format(240) in result.stdout.splitlines()
    assert "warning" not in result.output
    assert server.lists == 1
    assert no_real_transport == [60, 240]  # the poll, then the audit's wait to started_at + 300 s


def test_a_resume_after_an_interrupt_is_audited(
    monkeypatch: pytest.MonkeyPatch, haiku_run: Setup
) -> None:
    server = FakeBatches()
    corpus, runs = haiku_run(server)

    class InterruptedPoll:
        def send(
            self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
        ) -> HttpResponse:
            if method == "GET":
                raise KeyboardInterrupt
            return server.send(method, url, headers, body, timeout)

    monkeypatch.setattr(cli, "make_transport", InterruptedPoll)
    interrupted = eval_run(runs, HAIKU, "--yes", corpus=corpus)
    monkeypatch.setattr(cli, "make_transport", lambda: server)
    resumed = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert interrupted.exit_code == 1
    assert "checking for batches" not in interrupted.output  # no audit after an interrupt
    assert resumed.exit_code == 0, resumed.output
    assert resumed.stdout.startswith("resuming batch msgbatch_01\n")
    assert server.lists == 1


def test_nothing_to_run_makes_no_audit_call(
    monkeypatch: pytest.MonkeyPatch, haiku_run: Setup
) -> None:
    server = FakeBatches()
    corpus, runs = haiku_run(server)
    assert eval_run(runs, HAIKU, "--yes", corpus=corpus).exit_code == 0
    monkeypatch.setattr(cli, "make_transport", NoTransport)  # any call fails the test

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("nothing to run\nTest A summary:")


def test_a_held_batch_at_start_makes_no_call_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)  # NoTransport: any call fails the test
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:2])
    runs = tmp_path / "runs"
    held_state(runs)

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr.startswith("batch msgbatch_07 is held for the reviewer: ")


def test_a_failed_audit_listing_warns_and_keeps_the_outcome(haiku_run: Setup) -> None:
    # The clean run makes no list call itself, so the only listing is the audit's.
    server = FakeBatches(list_answer=HttpResponse(403, {}, b""))
    corpus, runs = haiku_run(server)

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 0, result.output
    assert (
        "warning: could not check for batches that no run recorded (listing batches failed: "
        "HTTP 403): look in the Claude Console (Batches)"
    ) in result.stdout.splitlines()
    assert "Complete: yes\n" in result.stdout


def test_a_stopped_run_is_audited_before_its_stop_message(haiku_run: Setup) -> None:
    server = FakeBatches(create_script=[HttpResponse(400, {}, b"")])
    corpus, runs = haiku_run(server)

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 1
    assert result.stdout == AUDIT_WAIT.format(300) + "\n"
    assert result.stderr == "creating a batch failed: HTTP 400\n"
    assert server.lists == 1


def test_another_models_recorded_batch_is_never_named(haiku_run: Setup) -> None:
    server = FakeBatches()
    corpus, runs = haiku_run(server)
    other = server.register(["01-pharmacy"])
    write_state(runs / OPUS, batch_entry(other, ["01-pharmacy"], "done"))

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 0, result.output
    assert "warning" not in result.output
    assert server.lists == 1


def test_the_candidates_of_a_pending_create_are_not_named(haiku_run: Setup) -> None:
    # The run stops naming two candidates; the next run adopts the one the owner keeps.
    server = FakeBatches(create_script=["duplicated"])
    corpus, runs = haiku_run(server)

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 1
    assert result.stderr.startswith("2 recent batches could be the one created by an earlier")
    assert "warning" not in result.output
    assert result.stdout.splitlines()[-1] == AUDIT_WAIT.format(180)
    assert server.lists == 2  # the reconciliation's and the audit's


@pytest.mark.parametrize("rejected", [False, True], ids=["success", "stop"])
def test_an_interrupt_in_the_audit_ends_only_the_audit(
    monkeypatch: pytest.MonkeyPatch, haiku_run: Setup, no_real_transport: Sleeps, rejected: bool
) -> None:
    script: list[CreateStep] = [HttpResponse(400, {}, b"")] if rejected else []
    server = FakeBatches(create_script=script)
    corpus, runs = haiku_run(server)

    def sleep(seconds: float) -> None:
        if seconds >= 240:  # the audit's wait (the poll sleeps 60 s)
            raise KeyboardInterrupt
        no_real_transport(seconds)

    monkeypatch.setattr(cli.time, "sleep", sleep)

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert "audit interrupted: look in the Claude Console (Batches)" in result.stdout.splitlines()
    assert server.lists == 0
    assert not (runs / LOCK).exists()
    if rejected:
        assert result.exit_code == 1
        assert result.stderr == "creating a batch failed: HTTP 400\n"
    else:
        assert result.exit_code == 0, result.output
        assert "Complete: yes\n" in result.stdout


def test_a_run_ended_by_an_unexpected_error_after_a_call_is_audited(
    monkeypatch: pytest.MonkeyPatch, haiku_run: Setup
) -> None:
    # H16 review, Minor 1: only RunStopped was audited; a full disk after a create call went to
    # the last resort, and a batch that call may have made was named nowhere.
    server = FakeBatches(listing=[other_batch("msgbatch_orphan", 1, "2026-11-02T10:00:05Z")])
    corpus, runs = haiku_run(server)
    write_state(runs / HAIKU, batch_entry("msgbatch_91", ["01-pharmacy"], "done"))

    def create_then_full_disk(
        model: str, corpus_dir: Path, runs_dir: Path, key: str, transport: Any, **kwargs: Any
    ) -> None:
        body = json.dumps({"requests": [{"custom_id": "01-pharmacy", "params": {}}]}).encode()
        transport.send("POST", runner_module.BATCHES_URL, {}, body, 60.0)  # makes msgbatch_01
        raise OSError(28, f"No space left on device {KEY}")

    monkeypatch.setattr(cli, "run_anthropic", create_then_full_disk)

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert_clean_exit_1(result, "")
    created = "2026-11-02T10:00:05.123456Z"  # the fake's created_at of msgbatch_01
    assert [line for line in result.stdout.splitlines() if line.startswith("warning")] == [
        UNRECORDED.format(id="msgbatch_01", count=1, created=created),
        UNRECORDED.format(id="msgbatch_orphan", count=1, created="2026-11-02T10:00:05Z"),
    ]
    assert result.stderr == (
        "eval run stopped by an unexpected error (OSError): run the dry run and send its output "
        "and this message to the reviewer\n"
    )
    assert not (runs / LOCK).exists()


def test_gemini_runs_are_not_audited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GEMINI_VARIABLE, KEY)
    server = FakeGemini()
    monkeypatch.setattr(cli, "make_transport", lambda: server)
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:1])

    result = eval_run(tmp_path / "runs", GEMINI, "--yes", corpus=corpus)

    assert result.exit_code == 0, result.output
    assert "checking for batches" not in result.output
    assert len(server.calls) == 1


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
        "of a batch that is not held, a missing directory or a corpus without a scenario, an "
        "unexpected error (its type only, never its text), or an interruption; 2: usage error "
        "(also an unregistered model id, or --release without --yes)"
    ) in text
    assert "--release" in text
    assert (
        "Mark a held batch done after the reviewer's decision (only with --yes), then continue"
        in text
    )
    assert "<runs>/.eval-run.lock" in text
    assert "runs a check for unrecorded batches after the run" in text
    assert "this check never changes the exit code" in text
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
        # TASK-005c3c E8: the 120 s wait, the lock and the line of the dry run.
        "the run may pause about two minutes; this is normal",
        "`" + runner_module.LOCKED.format(contents="<model> <time>", path="<path>") + "`",
        "`Lock: <model> <time> (another run may be active)`",
        "docs/validation/runs/test-a/.eval-run.lock",
        "--yes --release <id>",
        "cancel all of them but the oldest, then run the same command again; it adopts the one "
        "left",
        "One run at a time: never run `potakt eval run` in two windows at once",
        "Keep the laptop awake and plugged in while a batch runs",
        "again after every 10 polls without a change (every 10 minutes)",
        # TASK-005c3d H3-H6: the announced wait, the future started_at, the page cap.
        "`" + runner_module.WAITING.format(seconds="<n>") + "` (at most 180 s)",
        "three minutes after the request, for a request of an interrupted run",
        "`" + runner_module.FUTURE_STOP.format(started_at="<time>") + "`",
        "`" + runner_module.TOO_MANY_BATCHES + "`",
        "page by page back to the request",
        # TASK-005c3d H8-H10: unrecordable results, the empty corpus, the last resort.
        "`" + runner_module.UNEXPECTED_RECORD.format(name="<scenario>", kind="<type>") + "`",
        "`"
        + runner_module.UNEXPECTED_RECORD.format(name="<scenario>", kind="<type>")
        + runner_module.TO_REVIEWER
        + "`",
        "`" + runner_module.UNEXPECTED_COMPLETION.format(name="<scenario>", kind="<type>") + "`",
        "`no scenario in <corpus>`",
        "`" + cli.UNEXPECTED.format(kind="<type>") + "`",
        "`malformed` in the summary's `Stops` line",
        # TASK-005c3d H12: the bounded problems.
        "At most 20 problems are kept and shown",
        "`" + runner_module.MORE_PROBLEMS.format(count="<n>", id="<id>") + "`",
        # TASK-005c3d H13: the audit.
        "`" + runner_module.AUDIT_WAITING.format(seconds="<n>") + "` (at most five minutes",
        "`" + runner_module.UNRECORDED.format(id="<id>", count="<n>", created="<time>") + "`",
        "`" + runner_module.AUDIT_FAILED.format(reason="<reason>") + "`",
        "`" + runner_module.AUDIT_INTERRUPTED + "`",
        "This check never changes the records or the exit code",
        "Do not cancel it yourself and do not run `--yes` again",
        "also after an unexpected error",  # H16 review, Minor 1
        # TASK-005c3d H15: the current key types (R5).
        "Create a **personal** API key",
        "Linked account: yourself",
        "**restricted to the workspace `potakt-phase0`**",
        "expiry: **7 days**",
        "Workspace keys are deprecated",
        "makes every call fail with a 400 about `anthropic-workspace-id`, and the run stops with "
        "no batch",
        "`Authorization: Bearer <key>`, only to `api.anthropic.com`",
        "<https://platform.claude.com/docs/en/manage-claude/authentication>",
    ):
        assert quoted in text, quoted
    assert "cancel all but the oldest, then run the same command again" in lines
    # The old exception (an interrupted create needed the Console) is gone.
    assert "interrupted while creating a batch:" not in text
    assert "cancel any batch created in the last minutes" not in text
    assert "a crash" in text
    # The two-check rule of TASK-005c3b is gone (E3), and no line is wider than 100 (E8).
    assert "twice and a minute apart" not in text
    assert "check 2" not in text
    lines = RUNBOOK.read_text(encoding="utf-8").splitlines()
    long_lines = [line for line in lines if len(line) > 100]
    # The two verbatim PowerShell lines of §2 only (checked by content, not line number).
    assert len(long_lines) == 2
    assert all(line.startswith("$env:POTAKT_") and "PtrToStringBSTR" in line for line in long_lines)
