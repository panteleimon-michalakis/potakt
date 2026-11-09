"""``potakt eval run`` (TASK-005c3 F5): the dry run, the order rule, the key reader and the runs.

No test reads a real key or opens a connection: ``--yes`` runs get the fake key through
``monkeypatch.setenv`` and a fake transport through the ``make_transport`` factory.
"""

import ast
import json
import os
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

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
from test_evaluation_runner_anthropic import (
    CLOCK_URL,
    LIST_URL,
    CreateStep,
    FakeBatches,
    batch_object,
    other_batch,
    succeeded,
)
from test_evaluation_runner_gemini import FakeGemini
from test_evaluation_test_a import INSTRUCTION_PIN, SCHEMA_PIN, pin_in_force
from typer.testing import CliRunner, Result

import potakt.cli as cli
import potakt.evaluation.records as records_module
import potakt.evaluation.runner as runner_module
import potakt.evaluation.test_a as test_a_module
from potakt.cli import app, eval_app
from potakt.evaluation import CORPUS_PATH, load_scenario, render_test_a, summarize, write_answer
from potakt.evaluation.runner import RUNS_PATH, cost_upper_bound
from potakt.evaluation.transport import HttpResponse, TransportError, UrllibTransport

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
    # TASK-005c3g G9: $0 only if the key's project is on the free tier (runbook §1).
    assert (
        "Cost upper bound: $0 on the free tier (the key's project must be on the free tier: "
        "runbook §1)\n"
    ) in result.stdout
    assert "(free tier)" not in result.stdout
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


def leftover(runs: Path, model: str, name: str) -> Path:
    """A ``response.json.tmp`` left by an earlier run for scenario ``name``."""
    path = runs / model / name / "response.json.tmp"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"{}\n")
    return path


def unsaved_message(name: str, path: Path) -> str:
    return (
        f"{name}: an answer that could not be saved is kept in {path}; nothing was sent; send this "
        "message to the reviewer and do not delete or rename the file\n"
    )


@pytest.mark.parametrize("model", [HAIKU, GEMINI])
def test_the_dry_run_stops_on_a_leftover_answer_that_yes_would_stop_on(
    tmp_path: Path, model: str
) -> None:
    # TASK-005c3h H5 (the review's #16): the dry run priced a scenario that --yes stops on.
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:2])
    runs = tmp_path / "runs"
    path = leftover(runs, model, "02-hotel-reception")

    result = eval_run(runs, model, corpus=corpus)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == unsaved_message("02-hotel-reception", path)


def test_the_dry_run_says_that_a_leftover_of_the_open_batch_will_be_replaced(
    tmp_path: Path,
) -> None:
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:2])
    runs = tmp_path / "runs"
    write_state(runs / HAIKU, batch_entry("msgbatch_07", ALL_SCENARIOS[:2], "open"))
    leftover(runs, HAIKU, "02-hotel-reception")

    result = eval_run(runs, HAIKU, corpus=corpus)

    assert result.exit_code == 0, result.stderr
    lines = result.stdout.splitlines()
    assert (
        "02-hotel-reception: a leftover response.json.tmp will be replaced by the answer of batch "
        "msgbatch_07"
    ) in lines
    assert "Open batch: msgbatch_07 (2 scenarios), resumed by --yes" in lines


def test_the_dry_run_of_a_held_batch_shows_the_plan_whatever_its_leftovers(
    tmp_path: Path,
) -> None:
    # --yes stops on the held batch before it looks at leftovers, and --release first records
    # what the kept results hold: the dry run cannot know which leftovers will remain.
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:2])
    runs = tmp_path / "runs"
    held_state(runs)
    leftover(runs, HAIKU, "01-pharmacy")

    result = eval_run(runs, HAIKU, corpus=corpus)

    assert result.exit_code == 0, result.stderr
    assert any(line.startswith("Held batch: msgbatch_07") for line in result.stdout.splitlines())


def prompt_bytes(name: str) -> int:
    prompt = render_test_a(load_scenario(CORPUS_PATH / name))
    return len(prompt.system.encode("utf-8")) + len(prompt.user.encode("utf-8"))


def test_the_cost_bound_uses_bytes_and_max_tokens_at_the_batch_prices() -> None:
    size = prompt_bytes("01-pharmacy")

    assert cost_upper_bound(HAIKU, CORPUS_PATH, ["01-pharmacy"]) == pytest.approx(
        (size * 0.05 + 64000 * 0.25) / 1e6  # TASK-005c3g G8: Haiku's max_tokens is 64000
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
        (size * 0.25 + 64000 * 1.25) / 1e6
    )
    monkeypatch.setattr(
        runner_module, "LONG_PROMPT_PRICES_USD_PER_MILLION", {HAIKU: (size, tiers[HAIKU][1])}
    )
    assert cost_upper_bound(HAIKU, CORPUS_PATH, ["01-pharmacy"]) == pytest.approx(
        (size * 0.05 + 64000 * 0.25) / 1e6
    )


def test_the_whole_corpus_cost_bounds_of_the_two_claude_models() -> None:
    # TASK-005c3g G8: the Haiku bound with 64000 output tokens per request (was $0.1034).
    names = runner_module.corpus_scenarios(CORPUS_PATH)

    # TASK-005e1: the schema section states every rule, the instruction is longer (was $0.1834
    # and $7.3373).
    assert f"{cost_upper_bound(HAIKU, CORPUS_PATH, names):.4f}" == "0.1908"
    assert f"{cost_upper_bound(OPUS, CORPUS_PATH, names):.4f}" == "7.6301"


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
    assert result.stdout.startswith(
        "waiting 120 s before checking the workspace for batches that no run recorded\n"
        "batch msgbatch_01 created for 3 scenarios\n"
    )
    summary = summarize(runs / HAIKU, corpus).to_text()
    assert result.stdout.endswith(summary)
    assert "Complete: yes" in summary
    # TASK-005c3h H6: the workspace wait; the poll; then the audit's wait until started_at +
    # 300 s (TASK-005c3d H13).
    assert no_real_transport == [120, 10, 290]
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
    # The create was a call, so the audit runs before the stop message (TASK-005c3d H13), after
    # the workspace wait (TASK-005c3h H6).
    assert result.stdout == (
        "waiting 120 s before checking the workspace for batches that no run recorded\n"
        "waiting 300 s before checking for batches that no run recorded\n"
    )
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
            # TASK-005c3g G2, TASK-005c3h H6: the clock call and the workspace list pass.
            if method == "GET" and url.startswith(runner_module.BATCHES_URL + "/"):
                raise KeyboardInterrupt
            return server.send(method, url, headers, body, timeout)

    monkeypatch.setattr(cli, "make_transport", InterruptedPoll)
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:1])

    result = eval_run(tmp_path / "runs", HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 1
    assert result.stderr == "interrupted: run the same command again to resume (nothing is lost)\n"
    assert server.creates() == [["01-pharmacy"]]  # interrupted at the poll, after the create


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
    # TASK-005c3h H2: the reviewer emptied the kept results file ("send them all again").
    (runs / HAIKU / "batch-msgbatch_07.results.jsonl").write_bytes(b"")

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


TMP_RULE = "docs/validation/runs/test-a/**/*.tmp"


def gitignore_matches(rule: str, path: str) -> bool:
    """The gitignore rule (with a leading path, so anchored) matches ``path``: ``**/`` is zero or
    more directories, ``*`` any characters but ``/`` (gitignore(5), "Two consecutive
    asterisks")."""
    pattern = "".join(
        "(?:[^/]+/)*" if part == "**/" else "[^/]*" if part == "*" else re.escape(part)
        for part in re.split(r"(\*\*/|\*)", rule)
    )
    return re.fullmatch(pattern, path) is not None


@pytest.mark.parametrize(
    "path",
    [
        "docs/validation/runs/test-a/claude-haiku-5-5/01-pharmacy/response.json.tmp",
        "docs/validation/runs/test-a/claude-haiku-5-5/batches.json.tmp",
        "docs/validation/runs/test-a/.eval-run.lock.tmp",
    ],
    ids=["scenario", "model", "test-a"],
)
def test_a_leftover_temporary_file_is_ignored_by_git(path: str) -> None:
    # TASK-005c3g G11: a leftover *.tmp (G5 keeps an unsaved answer in one) is never committed.
    gitignore = Path(__file__).resolve().parents[2] / ".gitignore"

    assert TMP_RULE in gitignore.read_text(encoding="utf-8").splitlines()
    assert gitignore_matches(TMP_RULE, path)
    assert (Path(__file__).resolve().parents[2] / path).is_relative_to(RUNS_PATH)


@pytest.mark.parametrize(
    "path",
    [
        "docs/validation/runs/test-a/claude-haiku-5-5/01-pharmacy/response.json",
        "docs/validation/runs/test-a/claude-haiku-5-5/batches.json",
        "docs/validation/runs/test-a/claude-haiku-5-5/batch-msgbatch_01.results.jsonl",
        "docs/validation/scenarios/01-pharmacy/x.tmp",
    ],
)
def test_the_records_are_not_ignored_by_the_tmp_rule(path: str) -> None:
    assert not gitignore_matches(TMP_RULE, path)


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


# -- TASK-005c3g G3: the removal of the lock retries a refusal like write_atomic --

REFUSED_PAUSES = [0.05, 0.1, 0.2, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5]


def refused_unlink(monkeypatch: pytest.MonkeyPatch, path: Path, times: int) -> list[int]:
    """``Path.unlink`` of ``path`` refused with ``PermissionError`` ``times`` times."""
    real_unlink = Path.unlink
    refused: list[int] = []

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if self == path and len(refused) < times:
            refused.append(1)
            raise PermissionError(13, "Permission denied")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    return refused


@pytest.mark.parametrize("times", [1, 9])
def test_a_refused_removal_of_the_lock_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, times: int
) -> None:
    lock = runner_module.take_lock(tmp_path / "runs", HAIKU, datetime(2026, 11, 2, tzinfo=UTC))
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    refused = refused_unlink(monkeypatch, lock.path, times)

    assert lock.release() is None

    assert len(refused) == times
    assert slept == REFUSED_PAUSES[:times]
    assert not lock.path.exists()


def test_ten_refused_removals_of_the_lock_give_the_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = runner_module.take_lock(tmp_path / "runs", HAIKU, datetime(2026, 11, 2, tzinfo=UTC))
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    refused = refused_unlink(monkeypatch, lock.path, 10)

    warning = lock.release()

    assert warning == (
        f"could not remove the lock {lock.path} (Permission denied): delete it before the next run"
    )
    assert len(refused) == 10
    assert slept == REFUSED_PAUSES
    assert lock.path.exists()


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


# --- TASK-005c3g G8: the prompt and the schema are checked at run time -----------------------

PROMPT_DIFFERS = (
    "the prompt file differs from the one ADR-0007 §7 pins (SHA256 "
    "2dcb497384a35ae227ccc8df46349d6179162687093ac16e0cd48be9f406263c); nothing was sent"
)
SCHEMA_DIFFERS = (
    "the schema section differs from the one ADR-0007 §7 pins (SHA256 "
    "afd54559d52d492c8ce49914a886be49b2f377187d11098f8fff7395dec9c02c); nothing was sent"
)
EVERY_RUN = [(HAIKU, False), (HAIKU, True), (GEMINI, False), (GEMINI, True)]
EVERY_RUN_IDS = ["haiku-dry-run", "haiku-yes", "gemini-dry-run", "gemini-yes"]


def test_the_pins_in_the_code_are_the_adr_ones() -> None:
    # TASK-005e1: the prompt file and the schema section changed before any LLM run; ADR-0007
    # keeps its old pins (superseded) and ADR-0008 Appendix A pins the ones in force.
    assert pin_in_force(INSTRUCTION_PIN) == runner_module.INSTRUCTION_SHA256
    assert pin_in_force(SCHEMA_PIN) == runner_module.SCHEMA_SHA256


def changed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, what: str) -> None:
    """One changed byte in a copy of the prompt file, or in the schema section."""
    if what == "prompt":
        copy = tmp_path / "test-a.md"
        data = bytearray(test_a_module.INSTRUCTION_PATH.read_bytes())
        data[0] ^= 1
        copy.write_bytes(bytes(data))
        monkeypatch.setattr(test_a_module, "INSTRUCTION_PATH", copy)
    else:
        real = test_a_module.render_json_schema
        monkeypatch.setattr(test_a_module, "render_json_schema", lambda: real() + " ")


@pytest.mark.parametrize("what", ["prompt", "schema"])
@pytest.mark.parametrize(("model", "yes"), EVERY_RUN, ids=EVERY_RUN_IDS)
def test_a_changed_pin_stops_every_run_before_anything_is_sent_or_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str, yes: bool, what: str
) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    monkeypatch.setenv(GEMINI_VARIABLE, KEY)
    changed(monkeypatch, tmp_path, what)  # NoTransport: any call fails the test
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:1])
    runs = tmp_path / "runs"

    result = eval_run(runs, model, *(["--yes"] if yes else []), corpus=corpus)

    assert result.exit_code == 1, result.output
    assert result.stdout == ""
    assert result.stderr == (PROMPT_DIFFERS if what == "prompt" else SCHEMA_DIFFERS) + "\n"
    assert not runs.exists()


@pytest.mark.parametrize("model", [HAIKU, GEMINI])
def test_the_dry_run_stops_on_a_changed_request(tmp_path: Path, model: str) -> None:
    # TASK-005c3h H7: the dry run checks every pending request against its pin, as --yes does.
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:2])
    csv = sorted((corpus / "02-hotel-reception").glob("*.csv"))[0]
    data = bytearray(csv.read_bytes())
    data[-2] ^= 1
    csv.write_bytes(bytes(data))
    runs = tmp_path / "runs"

    result = eval_run(runs, model, corpus=corpus)

    pinned = runner_module.PAYLOAD_SHA256[(model, "02-hotel-reception")]
    assert result.exit_code == 1, result.output
    assert result.stdout == ""
    assert result.stderr == (
        f"02-hotel-reception: the request for {model} differs from the one the harness pins "
        f"(SHA256 {pinned}); nothing was sent\n"
    )
    assert not runs.exists()


def test_the_dry_run_stops_on_a_request_without_a_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pins = dict(runner_module.PAYLOAD_SHA256)
    del pins[(HAIKU, "01-pharmacy")]
    monkeypatch.setattr(runner_module, "PAYLOAD_SHA256", pins)

    result = eval_run(tmp_path, HAIKU, corpus=corpus_copy(tmp_path, ALL_SCENARIOS[:1]))

    assert result.exit_code == 1, result.output
    assert (
        result.stderr
        == "01-pharmacy: no request is pinned for claude-haiku-5-5; nothing was sent\n"
    )


def test_the_pins_are_checked_before_the_key_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changed(monkeypatch, tmp_path, "prompt")
    read: list[str] = []
    monkeypatch.setattr(cli, "read_key", read.append)

    result = eval_run(tmp_path / "runs", HAIKU, "--yes", corpus=corpus_copy(tmp_path))

    assert result.stderr == PROMPT_DIFFERS + "\n"
    assert read == []


# Gemini's dry run renders no prompt (its cost is $0), so it has no case here.
WITH_PROMPT = [(HAIKU, False), (HAIKU, True), (GEMINI, True)]
WITH_PROMPT_IDS = ["haiku-dry-run", "haiku-yes", "gemini-yes"]


@pytest.mark.parametrize(("model", "yes"), WITH_PROMPT, ids=WITH_PROMPT_IDS)
def test_a_missing_instruction_file_exits_1_with_the_eval_prompt_wording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, yes: bool, model: str
) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    monkeypatch.setenv(GEMINI_VARIABLE, KEY)
    monkeypatch.setattr(cli, "make_transport", FakeBatches)  # TASK-005c3g G2: the clock call
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
    monkeypatch.setattr(cli, "make_transport", FakeBatches)  # TASK-005c3g G2: the clock call
    corpus = undecodable_unbound_csv(tmp_path)

    result = eval_run(tmp_path / "runs", model, *(["--yes"] if yes else []), corpus=corpus)

    assert_clean_exit_1(result, "01-pharmacy: extra.csv is not valid utf-8\n")


def test_a_request_that_cannot_be_written_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    monkeypatch.setattr(cli, "make_transport", FakeBatches)  # TASK-005c3g G2: the clock call
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


INTERRUPTED_RUN = "interrupted: run the same command again to resume (nothing is lost)\n"


@pytest.mark.parametrize("yes", [False, True], ids=["dry-run", "yes"])
def test_ctrl_c_while_the_corpus_is_listed_exits_1_with_the_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, yes: bool
) -> None:
    # TASK-005c3h H8 (the review's #15): Typer turned it into exit 130 with no message.
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)

    def interrupted(corpus: Path) -> list[str]:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "corpus_scenarios", interrupted)

    result = eval_run(tmp_path, HAIKU, *(["--yes"] if yes else []))

    assert result.exit_code == 1
    assert result.stderr == INTERRUPTED_RUN


def test_ctrl_c_during_the_final_summary_exits_1_with_the_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_real_transport: Sleeps
) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)
    server = FakeBatches(clock=no_real_transport)  # its Date follows the run's clock
    monkeypatch.setattr(cli, "make_transport", lambda: server)
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:1])
    runs = tmp_path / "runs"

    def interrupted(model_dir: Path, corpus_dir: Path) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "summarize", interrupted)

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 1
    assert result.stderr == INTERRUPTED_RUN
    assert not (runs / LOCK).exists()
    # The same command again: nothing to run (the answer is recorded), then the summary.
    monkeypatch.setattr(cli, "summarize", summarize)
    again = eval_run(runs, HAIKU, "--yes", corpus=corpus)
    assert again.exit_code == 0, again.output
    assert "Complete: yes\n" in again.stdout


def test_a_recorded_answer_that_cannot_be_stat_ed_exits_1_with_no_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # TASK-005c3h H8 (TASK-005c3s S1-1): on 3.14 Path.exists() returned False for any OSError
    # (and on Windows never called os.stat), on 3.12 it raised: now our code decides, and an
    # unreadable record stops the run, the same on both.
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
    target = runs / HAIKU / "01-pharmacy" / "response.json"
    real = os.stat

    def stat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        if Path(path) == target:
            raise PermissionError(13, "Permission denied", str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat)

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert_clean_exit_1(result, "")
    assert result.stderr == (
        "eval run stopped by an unexpected error (PermissionError): run the dry run and send its "
        "output and this message to the reviewer\n"
    )
    assert server.creates() == []


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


def test_a_summary_that_cannot_be_read_is_reported_and_the_run_still_exits_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # TASK-005c3e F6: an OSError of the final summary went to the last resort (exit 1).
    monkeypatch.setenv(GEMINI_VARIABLE, KEY)
    monkeypatch.setattr(cli, "make_transport", FakeGemini)
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:1])
    runs = tmp_path / "runs"

    def broken(runs_dir: Path, corpus_dir: Path) -> Any:
        raise PermissionError(13, "Permission denied", "x")

    monkeypatch.setattr(cli, "summarize", broken)

    result = eval_run(runs, GEMINI, "--yes", corpus=corpus)

    assert result.exit_code == 0, result.output
    assert result.stdout == "01-pharmacy: recorded (STOP, invalid, recall 0.000)\n"
    assert result.stderr == f"cannot summarize {runs / GEMINI}: Permission denied\n"
    assert "unexpected error" not in result.output


def test_an_oserror_of_the_summary_without_strerror_shows_its_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(GEMINI_VARIABLE, KEY)
    monkeypatch.setattr(cli, "make_transport", FakeGemini)
    corpus = corpus_copy(tmp_path, ALL_SCENARIOS[:1])
    runs = tmp_path / "runs"

    def broken(runs_dir: Path, corpus_dir: Path) -> Any:
        raise FileNotFoundError(f"text with {KEY}")

    monkeypatch.setattr(cli, "summarize", broken)

    result = eval_run(runs, GEMINI, "--yes", corpus=corpus)

    assert result.exit_code == 0, result.output
    assert result.stderr == f"cannot summarize {runs / GEMINI}: FileNotFoundError\n"


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
        # TASK-005c3h H6: the first run waits 120 s for its workspace check; it starts that much
        # earlier, so the create is at NOW, as these tests state (Run.rewind in the runner tests).
        no_real_transport.advance(-runner_module.RECONCILE_DELAY_SECONDS)
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
    assert server.lists == 2  # TASK-005c3h H6: the workspace's, then the audit's
    # The workspace wait, the poll, then the audit's wait to started_at + 300 s.
    assert no_real_transport == [120, 60, 240]


def test_a_resume_after_an_interrupt_is_audited(
    monkeypatch: pytest.MonkeyPatch, haiku_run: Setup
) -> None:
    server = FakeBatches()
    corpus, runs = haiku_run(server)

    class InterruptedPoll:
        def send(
            self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
        ) -> HttpResponse:
            # TASK-005c3g G2, TASK-005c3h H6: the clock call and the workspace list pass.
            if method == "GET" and url.startswith(runner_module.BATCHES_URL + "/"):
                raise KeyboardInterrupt
            return server.send(method, url, headers, body, timeout)

    monkeypatch.setattr(cli, "make_transport", InterruptedPoll)
    interrupted = eval_run(runs, HAIKU, "--yes", corpus=corpus)
    monkeypatch.setattr(cli, "make_transport", lambda: server)
    resumed = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert interrupted.exit_code == 1
    assert "checking for batches that no run" not in interrupted.output  # no audit
    assert resumed.exit_code == 0, resumed.output
    assert resumed.stdout.startswith("resuming batch msgbatch_01\n")
    assert server.lists == 2  # the first run's workspace list, then the resumed run's audit


def test_nothing_to_run_makes_no_audit_call(
    monkeypatch: pytest.MonkeyPatch, haiku_run: Setup
) -> None:
    server = FakeBatches()
    corpus, runs = haiku_run(server)
    assert eval_run(runs, HAIKU, "--yes", corpus=corpus).exit_code == 0
    server.calls.clear()

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("nothing to run\nTest A summary:")
    # TASK-005c3g G2: the clock call alone is not counted for the audit.
    assert [(call.method, call.url) for call in server.calls] == [("GET", CLOCK_URL)]


def test_a_wrong_laptop_clock_stops_the_run_with_no_audit(
    monkeypatch: pytest.MonkeyPatch, haiku_run: Setup, no_real_transport: Sleeps
) -> None:
    # TASK-005c3g G2: no audit follows the clock stop: it is not counted as a call.
    server = FakeBatches()
    corpus, runs = haiku_run(server)
    monkeypatch.setattr(cli, "utc_now", lambda: no_real_transport.now() + timedelta(minutes=15))

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == (
        "the laptop clock differs from the API's by 900 s (more than 120 s): sync the Windows "
        "clock (Settings, Time & language, Date & time, Sync now) and run again; nothing was "
        "sent\n"
    )
    assert [(call.method, call.url) for call in server.calls] == [("GET", CLOCK_URL)]
    assert no_real_transport == []
    assert not (runs / HAIKU).exists()
    assert not (runs / ".eval-run.lock").exists()


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
    # The clean run's only listing of its own is the workspace check (TASK-005c3h H6), before
    # the create; the 403 answers the audit's, after it.
    server = FakeBatches(list_answer=HttpResponse(403, {}, b""), extras_after_create=True)
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
    assert result.stdout == (
        "waiting 120 s before checking the workspace for batches that no run recorded\n"
        + AUDIT_WAIT.format(300)
        + "\n"
    )
    assert result.stderr == "creating a batch failed: HTTP 400\n"
    assert server.lists == 2  # TASK-005c3h H6: the workspace's, then the audit's


def test_another_models_recorded_batch_is_never_named(haiku_run: Setup) -> None:
    server = FakeBatches()
    corpus, runs = haiku_run(server)
    other = server.register(["01-pharmacy"])
    write_state(runs / OPUS, batch_entry(other, ["01-pharmacy"], "done"))

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 0, result.output
    assert "warning" not in result.output
    assert server.lists == 2  # TASK-005c3h H6: the workspace's (it knows the batch), the audit's


def test_the_candidates_of_a_pending_create_are_not_named(haiku_run: Setup) -> None:
    # The run stops naming two candidates; the next run adopts the one the owner keeps.
    server = FakeBatches(create_script=["duplicated"])
    corpus, runs = haiku_run(server)

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 1
    assert result.stderr.startswith("2 recent batches could be the one created by an earlier")
    assert "warning" not in result.output
    assert result.stdout.splitlines()[-1] == AUDIT_WAIT.format(180)
    assert server.lists == 3  # the workspace's (H6), the reconciliation's and the audit's


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
    assert server.lists == 1  # TASK-005c3h H6: the workspace's only; the audit's never ran
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


def test_a_late_haiku_duplicate_stops_the_first_opus_run_an_hour_later(
    haiku_run: Setup, no_real_transport: Sleeps
) -> None:
    # TASK-005c3e F1 (probe P6): the twin is created 320 s after the adopted batch and listed
    # 90 s later (410 s), after Haiku's audit (300 s). Opus's audit used to name it after Opus
    # had paid its own batch; TASK-005c3h H6: Opus's first run now stops before it, naming it.
    server = FakeBatches(create_script=["duplicated-staggered"], stagger=320, lag=90)
    corpus, runs = haiku_run(server)

    haiku = eval_run(runs, HAIKU, "--yes", corpus=corpus)
    no_real_transport.advance(3600)
    server.outcome = lambda n, s: succeeded(model=OPUS)
    opus = eval_run(runs, OPUS, "--yes", corpus=corpus)

    assert haiku.exit_code == 0, haiku.output
    assert "adopted batch msgbatch_01 created by an earlier create request" in haiku.stdout
    assert "warning" not in haiku.output
    assert opus.exit_code == 1
    assert opus.stderr == (
        f"the workspace has batch msgbatch_02 that no batches.json in {runs} records and "
        "acknowledged-batches.txt does not name; nothing was sent: if the records of an earlier "
        "run were lost, restore them before running again; otherwise send this message to the "
        "reviewer\n"
    )
    assert len(server.creates()) == 1  # Haiku's only
    assert not (runs / OPUS).exists()


def test_an_unreadable_created_at_never_reaches_the_last_resort(haiku_run: Setup) -> None:
    # TASK-005c3e F2 (probe P5): a lone surrogate in created_at made _save_state raise.
    server = FakeBatches(
        create_script=[batch_object("\ud83d")],
        newest=[other_batch("msgbatch_01", 1, "\ud83d")],
        extras_after_create=True,  # the reconciliation's list, not the workspace's (H6)
    )
    corpus, runs = haiku_run(server)

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)
    dry_run = eval_run(runs, HAIKU, corpus=corpus)

    assert result.exit_code == 1
    assert result.stderr == "listing batches failed: unexpected batch object\n"
    assert "unexpected error" not in result.output
    state = json.loads((runs / HAIKU / "batches.json").read_bytes())
    assert [entry["state"] for entry in state["batches"]] == ["creating"]
    assert dry_run.exit_code == 0, dry_run.output
    assert "Unconfirmed create request: started " in dry_run.stdout


def test_an_invalid_state_of_another_model_stops_a_first_run_before_any_list(
    haiku_run: Setup,
) -> None:
    # TASK-005c3h H6: the workspace check cannot tell which batches are recorded, so it stops
    # (fail closed) before its wait, its list and the create. (The audit's own warning for an
    # invalid state: test_an_invalid_state_of_another_model_is_a_warning.)
    server = FakeBatches()
    corpus, runs = haiku_run(server)
    (runs / OPUS).mkdir(parents=True)
    (runs / OPUS / "batches.json").write_text("not json", encoding="utf-8")

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert (
        result.stderr
        == f"{runs / OPUS / 'batches.json'} is not a valid batch state; it was not changed\n"
    )
    assert [(call.method, call.url) for call in server.calls] == [("GET", CLOCK_URL)]


def test_a_run_whose_every_call_raised_is_still_audited(
    monkeypatch: pytest.MonkeyPatch, haiku_run: Setup
) -> None:
    # TASK-005c3e F3: a call that raised may have reached the API (a lost create), so it counts.
    server = FakeBatches()
    corpus, runs = haiku_run(server)

    class Unreachable:
        """Every call but the clock call (TASK-005c3g G2) and the first list call, the
        workspace check's (TASK-005c3h H6), raises: the create is lost, then every list."""

        listed = False

        def send(
            self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
        ) -> HttpResponse:
            if url == CLOCK_URL:
                return server.send(method, url, headers, body, timeout)
            if url.startswith(LIST_URL) and not Unreachable.listed:
                Unreachable.listed = True
                return server.send(method, url, headers, body, timeout)
            raise TransportError("TimeoutError: no response received")

    monkeypatch.setattr(cli, "make_transport", Unreachable)

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    stop = "the request failed 4 times; last: TimeoutError: no response received"
    assert result.exit_code == 1
    assert result.stdout.splitlines()[-1] == (
        f"warning: could not check for batches that no run recorded ({stop}): look in the "
        "Claude Console (Batches)"
    )
    assert result.stderr == stop + "\n"
    assert not (runs / LOCK).exists()


def test_a_clock_call_that_fails_four_times_stops_the_run_with_no_audit(
    monkeypatch: pytest.MonkeyPatch, haiku_run: Setup, no_real_transport: Sleeps
) -> None:
    # TASK-005c3g G2: the clock call is not counted, so a run that made no other call has no
    # audit, also when the network is down from the start.
    corpus, runs = haiku_run(FakeBatches())
    calls: list[str] = []

    class Unreachable:
        def send(
            self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
        ) -> HttpResponse:
            calls.append(url)
            raise TransportError("TimeoutError: no response received")

    monkeypatch.setattr(cli, "make_transport", Unreachable)

    result = eval_run(runs, HAIKU, "--yes", corpus=corpus)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == (
        "the request failed 4 times; last: TimeoutError: no response received\n"
    )
    assert calls == [CLOCK_URL] * 4
    assert no_real_transport == [15, 45, 135]
    assert not (runs / LOCK).exists()


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
    # TASK-005c3g G2
    assert (
        "A Claude run first compares the laptop clock with the API's (the Date header of one "
        "free call) and stops, sending nothing, if they differ by more than 2 minutes"
    ) in text
    # TASK-005c3g G12 (R1): an ended batch cannot be cancelled.
    assert (
        "so the reviewer can decide about a paid duplicate (only a batch still in progress can "
        "be cancelled)"
    ) in text
    assert "so a paid duplicate can be cancelled" not in text
    assert "POTAKT_ANTHROPIC_API_KEY" in text
    assert "POTAKT_GEMINI_API_KEY" in text
    assert "request.json, response.json, candidate.txt, comparison.json" in text
    assert "batches.json and batch-<id>.results.jsonl" in text
    assert (
        "Exit code 0: everything pending is recorded, or the plan was printed; 1: the run "
        "stopped (the reason on stderr), a held batch, a missing or invalid key, claude-opus-5-5 "
        "before a complete claude-haiku-5-5 run, another run active (the lock file), a --release "
        "of a batch that is not held, an unknown batch in the workspace, a request that differs "
        "from its pin, a missing directory or a corpus without a scenario, an unexpected error "
        "(its type only, never its text), or an interruption; 2: usage error (also an "
        "unregistered model id, or --release without --yes)"
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


@pytest.mark.parametrize("name", ["prompt", "compare", "summary", "rescore", "run", "preflight"])
def test_each_eval_command_has_a_one_line_summary(name: str) -> None:
    # TASK-005c3g G12 (n7): Typer lists the first paragraph of the docstring with its own line
    # breaks, so a two-line first paragraph broke the command list mid-sentence.
    command = cast(Any, typer.main.get_command(eval_app)).commands[name]
    summary = command.help.split("\n\n")[0]

    assert "\n" not in summary.strip()
    assert summary.strip().endswith(".")
    assert len(summary.strip()) <= 84  # fits the box of `potakt eval --help` at 100 columns


REPO = Path(__file__).resolve().parents[2]


def test_the_readmes_have_the_texts_of_the_pre_run_audit() -> None:
    # TASK-005c3g G12: n3 (the validation README) and n7 (the README).
    validation = " ".join((REPO / "docs/validation/README.md").read_text(encoding="utf-8").split())
    readme = " ".join((REPO / "README.md").read_text(encoding="utf-8").split())

    assert (
        "The summary's cost counts every answer's tokens at the pinned prices. A refusal outside "
        "the billed categories is not billed, so the summary may overstate the cost."
    ) in validation
    # TASK-005c3h H10 (#24): the clock call is not counted for the audit.
    assert (
        "after a Claude run that made an API call other than the clock check and was not "
        "interrupted it runs a check"
    ) in readme


# --- the runbook (F6) -------------------------------------------------------------------------

RUNBOOK = Path(__file__).resolve().parents[2] / "docs" / "validation" / "test-a-runbook.md"
SESSION_BLOCK = (  # TASK-005c3g G12 (n8): the BSTR is freed
    "```powershell\n"
    "cd <repo>\\python\n"
    '$secure = Read-Host "Anthropic API key" -AsSecureString\n'
    "$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)\n"
    "$env:POTAKT_ANTHROPIC_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)\n"
    "[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)\n"
    "Remove-Variable secure, bstr\n"
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
    # TASK-005e1 (ADR-0008 D13): the spend limit is raised in stages up to the US$60 cap.
    assert "spend limit to $15 for the development runs" in " ".join(text.split())
    assert "raise it to $45 before the corpus Test A runs and to $60 before Test B" in " ".join(
        text.split()
    )
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
        # TASK-005c3h H10 (#27): only an entry left by an earlier run prints that line.
        "the create request was made by an earlier run that was interrupted; the batch list was "
        "checked, page by page, at least three minutes after that request, and no batch was found",
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
        "It looks back to the first batch creation of any model",  # TASK-005c3e J3
        # TASK-005c3d H15: the current key types (R5).
        "Create a **personal** API key",
        "Linked account: yourself",
        "**restricted to the workspace `potakt-phase0`**",
        "expiry: **7 days**",
        "makes every call fail with a 400 about `anthropic-workspace-id`, and the run stops with "
        "no batch",
        "`Authorization: Bearer <key>`, only to `api.anthropic.com`",
        "<https://platform.claude.com/docs/en/manage-claude/authentication>",
        # TASK-005c3e J8: the documentation's own word, quoted from the English page.
        'Workspace keys are **legacy**: the page says "Workspace API keys work but are legacy."',
        '"You can also send the key in the legacy `x-api-key: YOUR_API_KEY` header in place of '
        '`Authorization`."',
        "(checked on 2026-11-06)",
        # TASK-005c3e J8: the preflight, the five-minute note and the Console check.
        "**Check the keys** (last step, once the variables of §2 are set)",
        "uv run potakt eval preflight --model claude-haiku-5-5 "
        "uv run potakt eval preflight --model claude-opus-5-5 "
        "uv run potakt eval preflight --model gemini-3.8-flash",
        "Send the three outputs to the reviewer before any `--yes`",
        "`key: accepted by api.anthropic.com`",
        "`" + runner_module.PREFLIGHT_PASSED + "`",
        "A field shown as `not reported` is not a failure",
        "`preflight failed: <check>: <reason>`",
        "`" + cli.PREFLIGHT_INTERRUPTED + "`",
        "`" + cli.PREFLIGHT_UNEXPECTED.format(kind="<type>") + "`",
        "If the key or the day changed since the check of §1, run the preflight again first",
        "**Check the Console after each model's run** (here and in §6)",
        "Open the Claude Console, Batches, for the workspace `potakt-phase0`",
        "Compare the batch ids listed there with the ids in "
        "`docs/validation/runs/test-a/<model dir>/batches.json`",
        "Send the reviewer that list.",
        "A batch listed there that no `batches.json` records is a paid duplicate, whatever the "
        "audit said.",
        "do the Console check of §3 for both `batches.json` files",
        # TASK-005c3e J5: the summary error after a complete run.
        "`cannot summarize <dir>: <reason>`",
        # TASK-005c3g G4: the release records from the kept results first.
        "`" + runner_module.RECORDED_FROM_KEPT.format(name="<scenario>", id="<id>") + "`",
        "`"
        + runner_module.KEPT_UNREADABLE.format(path="<path>", reason="<reason>", id="<id>")
        + "`",
        # TASK-005c3g G2: the preflight's clock line.
        "`clock: within <n> s of api.anthropic.com`",
        # TASK-005c3g G7: an error type that a repeat cannot fix holds the batch.
        "`<scenario>: errored (<type>); it would fail again`",
    ):
        assert quoted in text, quoted
    assert "deprecated" not in text  # F5: the English page says "legacy"
    assert "stops at once with no record" not in text  # the audit runs first (H13)
    assert "cancel all but the oldest, then run the same command again" in lines
    # The old exception (an interrupted create needed the Console) is gone.
    assert "interrupted while creating a batch:" not in text
    assert "cancel any batch created in the last minutes" not in text
    assert "a crash" in text
    # The two-check rule of TASK-005c3b is gone (E3), and no line is wider than 100 (E8).
    assert "twice and a minute apart" not in text
    assert "check 2" not in text
    lines = RUNBOOK.read_text(encoding="utf-8").splitlines()
    # TASK-005c3g n8: the PowerShell blocks of §2 are short now; no line is wider than 100.
    assert [line for line in lines if len(line) > 100] == []


def test_the_runbook_has_the_texts_of_the_pre_run_audit() -> None:
    # TASK-005c3g G12: R1-R8 and n8, as the task gives them (adapted only where G1-G9 changed a
    # fact), and the messages of §8 as the code prints them.
    text = " ".join(RUNBOOK.read_text(encoding="utf-8").split())
    gemini_block = SESSION_BLOCK.replace("cd <repo>\\python\n", "").replace("Anthropic", "Gemini")

    for quoted in (
        # R1, as TASK-005c3h H10 (#23) changes it.
        "It looks back to the first batch creation of any model, so a batch it once named is "
        "named again by every later Claude run that made an API call other than the clock check, "
        "unless it is cancelled or acknowledged. Only a batch still in progress can be cancelled "
        '("Batches may be canceled any time before processing ends."); a batch that has already '
        "ended stays in the batch list. Once the reviewer has decided about a named batch, the "
        "reviewer may acknowledge it in acknowledged-batches.txt (§8), and no run names it again; "
        "until then its warning on later runs is expected: send a warning again only if it names "
        "a new batch id.",
        # R2, §3 and §1 step 2
        "This first batch also confirms the `output_config` field. The API checks each request of "
        'a batch only while it processes the batch ("Validation of the `params` object for each '
        'message request is performed asynchronously"), so a refused field comes back as '
        "`<scenario>: invalid_request_error: …` for every scenario and the batch is held (§8); "
        "such results are not billed. A wrong API version or key is refused at once with a 400 or "
        "a 401, before any batch exists: the run stops with no charge and no record, and the "
        "reason is printed.",
        "If the limit is already reached when a batch is created, the API answers with a 400 "
        "error and the run stops with no record. If it is reached while a batch runs, the "
        "documentation does not say how the remaining requests end: send the run's output to the "
        "reviewer. The dry runs' cost upper bounds fit under the limit.",
        # R3
        "3. In Google AI Studio, Projects, check that the key's project shows **Free** in the "
        "Billing Tier column. A project linked to a Cloud Billing account is on a paid tier: every "
        "request is billed and a repeated one twice, although the dry run says $0. If it is not "
        "Free, stop and ask the reviewer.",
        "If the key or the day changed since the check of §1, run `uv run potakt eval preflight "
        "--model gemini-3.8-flash` again first. For Gemini the preflight proves only that the key "
        "exists: Google's documentation says that read-only metadata calls \"only validate that "
        "your API key exists and don't check project tier eligibility or billing status\".",
        "A 403 (for example `PERMISSION_DENIED` with `FREE_TIER_VERIFICATION_REQUIRED`), a 400 "
        "`FAILED_PRECONDITION` or a 402 stops the run at once with no record: the project cannot "
        "use the free tier as it is. Fix it in AI Studio (for example, verify the account) or send "
        "the message to the reviewer. Daily quotas reset at midnight Pacific time (10:00 in Athens "
        "in November and December).",
        # R4
        "`<scenario>: errored (<type>); pending` (only the four transient types), "
        "`<scenario>: expired; pending` and `<scenario>: canceled; pending`",
        "that request got no answer and is not billed",
        "the run sends it again in a new batch after a short pause",
        "`<scenario>: errored (<type>); it would fail again`: the batch is held; send the message "
        "to the reviewer.",
        "`<scenario> was sent in 4 batches without an answer; it is not sent again`: do not edit "
        "`batches.json`; send the message and `batches.json` to the reviewer.",
        "`the request failed 4 times; last: …`, `creating a batch failed 4 times; last: …` and "
        "`listing batches failed: …`",
        "the network or the API failed four times in a row",
        "nothing was lost and nothing is paid twice",
        "wait a few minutes and run the same command again. If it fails again, send the message "
        "to the reviewer.",
        "`creating a batch failed: HTTP <status> (<type>: …)`: the API refused the batch, and "
        "nothing was created or charged.",
        "A 400 or a 401 is the key or the spend limit (§1).",
        "A 402 `billing_error` is the organisation's credit balance.",
        "`POTAKT_ANTHROPIC_API_KEY is not set in this shell` (or `POTAKT_GEMINI_API_KEY`): this "
        "window has no key, so run the block of §2 in this window. `… is not a valid key`: the "
        "pasted value holds a space, a line break or a character that is not plain ASCII, so set "
        "it again (§2).",
        "`<path> is not a valid batch state; it was not changed`: do not edit it; send it with the "
        "message to the reviewer.",
        "`cannot create <path>: …` or `cannot write <path>: …`: the runs directory cannot be "
        "written (a full disk, or a folder the antivirus protects); fix that and run again.",
        "`" + runner_module.CLOCK_STOP.format(seconds="<n>", limit=120) + "`",
        "`" + runner_module.NO_DATE + "`",
        "`" + runner_module.UNSAVED.format(name="<scenario>", path="<path>") + "`",
        "`" + runner_module.PIN_DIFFERS.format(what="prompt file", pinned="<sha256>") + "`",
        "`" + runner_module.PIN_DIFFERS.format(what="schema section", pinned="<sha256>") + "`",
        "`"
        + runner_module.UNEXPECTED_RECORD.format(name="<scenario>", kind="<type>")
        + runner_module.STILL_HELD.format(id="<id>")
        + "`",
        # R5
        "On Windows, Ctrl+C takes effect only when the current network call returns: up to a "
        "minute while a batch is created, polled or listed, up to five minutes while results "
        "download, up to ten minutes during a Gemini request. Wait for `interrupted: run the same "
        "command again to resume (nothing is lost)`; do not close the window instead (a closed "
        "window leaves the lock). A Gemini answer that arrives during that wait is discarded, and "
        "that scenario is sent again by the next run.",
        # R6
        "The run compares the laptop clock with the API's and stops if they differ by more than "
        "two minutes. If it stops for that, sync the Windows clock (Settings, Time & language, "
        'Date & time, "Sync now") and run again.',
        "Never delete, move or edit anything under `docs/validation/runs/test-a/` (in particular "
        "`batches.json` and any `.tmp` file), never run `git clean` or `git stash -u` before the "
        "records are committed, and never run a model from a second clone or with another "
        "`--runs`: the run would send its pending scenarios again, and no check would see the "
        "duplicate.",
        # R7
        "Commit each model's records right after its run and its checks, before the next model",
        "from `python/`, go to the repository root (`cd ..`) and run `git status`",
        "only files under `docs/validation/runs/test-a/` may appear",
        "push the commits the same day to the review branch the reviewer names, for the check of "
        "a fresh clone; push to `main` only after that check",
        "until a commit is pushed, the records exist only on this laptop (Anthropic keeps batch "
        "results for 29 days after the batch was created)",
        # R8
        "In the Console's API keys list, check that the key's workspace is `potakt-phase0`. In "
        "Billing, check that the credit balance is at least $1 before Haiku and $8 before Opus.",
        # n8
        "Copy messages from the window itself; do not redirect the output with `>` (Windows "
        "PowerShell 5.1 garbles non-ASCII characters in files).",
        " ".join(gemini_block.replace("ANTHROPIC", "GEMINI").split()),
    ):
        assert quoted in text, quoted
    # Gone: the R1 and R2 sentences they replace, and the wait that G2 made untrue for a key.
    assert "until the reviewer has decided about it" not in text
    assert "This first batch also confirms the API version" not in text
    assert "When the limit is reached the API answers with a 400 error" not in text
    assert "the run waits up to five minutes for the check below before it prints" not in text
    assert "PtrToStringBSTR([Runtime" not in text
    # R3: midnight Pacific winter time (UTC-8) is 08:00 UTC, 10:00 in Athens winter time (UTC+2).
    pacific, athens = timezone(timedelta(hours=-8)), timezone(timedelta(hours=2))
    assert datetime(2026, 11, 8, 0, 0, tzinfo=pacific).astimezone(athens).hour == 10


def normalised(text: str) -> str:
    return " ".join(text.split())


def test_the_runbook_has_the_texts_of_the_max_effort_review() -> None:
    # TASK-005c3h H10: every new or changed message as the code prints it, and the texts the
    # task gives verbatim.
    text = normalised(RUNBOOK.read_text(encoding="utf-8"))
    r = runner_module
    workspace = r.UNKNOWN_BATCHES.format(noun="batch", ids="<ids>", runs="<runs>")
    messages = (
        # H2 and TASK-005c3s S3-3: the release keeps the batch held.
        r.KEPT_UNREADABLE.format(path="<path>", reason="<reason>", id="<id>"),
        r.KEPT_NOT_RESULTS.format(path="<path>", id="<id>"),
        # H3, H5, H6.
        r.ALREADY_RELEASED.format(id="<id>"),
        r.REPLACED.format(name="<scenario>", file="response.json.tmp", id="<id>"),
        r.WORKSPACE_WAITING.format(seconds="<n>"),
        workspace.split(" if the records")[0] + " …",
        r.WORKSPACE_TOO_MANY,
        r.LOST_RECORDS.format(name="<scenario>", file="batch-<id>.results.jsonl"),
        r.LOST_UNREADABLE.format(path="<path>", reason="<reason>"),
        r.NOT_UTF8.format(path="<path>"),
        r.NOT_A_BATCH_ID.format(path="<path>", number="<n>"),
        "cannot read <path>: <reason>",
        # H7.
        r.REQUEST_DIFFERS.format(name="<scenario>", model="<model>", pinned="<sha256>"),
        r.REQUEST_UNPINNED.format(name="<scenario>", model="<model>"),
        # H8 #3 (TASK-005c3s S3-6): Claude, then Gemini.
        r.RECORDED_INCOMPLETE.format(name="<scenario>", kind="<type>", when=r.COMPLETED_ON_RELEASE),
        r.RECORDED_INCOMPLETE.format(name="<scenario>", kind="<type>", when=r.COMPLETED_NEXT_RUN)
        + r.TO_REVIEWER,
        # TASK-005c3s S2-2: the lock taken over by another run.
        "the lock <path> now belongs to another run (…); it was left in place",
    )
    for message in messages:
        assert f"`{normalised(message)}`" in text, message
    verbatim = (
        # S2-2: the catch-all at the top of §8, and the lock entry.
        "Any stop or warning that is not listed below: do not run `--yes` again; send the whole "
        "output to the reviewer.",
        "`the lock <path> now belongs to another run (…); it was left in place`: two runs "
        "overlapped. Do the Console check of §3 for every model, and send the output of both "
        "windows to the reviewer.",
        # «Acknowledged batches» (TASK-005c3s S3-2).
        "Only on the reviewer's decision, from the repository root, once for each batch id the "
        "reviewer names: `Add-Content -Path docs/validation/runs/test-a/acknowledged-batches.txt "
        '-Value "<id>  # <reason>" -Encoding Ascii` Then commit the file with the records. A run '
        "never adopts or names an acknowledged batch, and the check before a model's first batch "
        "does not stop on it. Do not write the file in any other way: `Set-Content`, `Out-File` "
        "and `>` change its encoding.",
        # S2-1, after "…then run the same command again; it adopts the one left."
        "it adopts the one left. A batch that has already ended cannot be cancelled: then send "
        "the message to the reviewer, who decides which batches to acknowledge («Acknowledged "
        "batches»).",
        # The review's #4.
        "If saving a Gemini answer fails before its temporary file is complete, the answer is "
        "lost and the next run sends that scenario again (free tier: no charge).",
        # The review's #23, #24, #27.
        "After every Claude run that made an API call other than the clock check,",
        "the create request was made by an earlier run that was interrupted; the batch list was "
        "checked, page by page, at least three minutes after that request, and no batch was found;",
    )
    for quoted in verbatim:
        assert normalised(quoted) in text, quoted
    # H2's ways out, each with its exact command (the long ones as a short two-line block).
    assert (
        normalised(
            '$file = "docs/validation/runs/test-a/<model dir>/batch-<id>.results.jsonl" '
            "New-Item -ItemType File -Force -Path $file"
        )
        in text
    )
    assert (
        normalised(
            '$file = "docs/validation/runs/test-a/acknowledged-batches.txt" '
            'Add-Content -Path $file -Value "<id>  # <reason>" -Encoding Ascii'
        )
        in text
    )
    assert 'one `{"custom_id": …, "result": …}` object per line' in text
    # The old texts are gone.
    for old in (
        "its pending scenarios will be sent again",
        "the batch list was checked once, two minutes after the answer",
        "named again by every later run, of any model",
        "After every Claude run that called the API,",
    ):
        assert old not in text, old


def test_the_readmes_help_and_docstrings_have_the_texts_of_the_max_effort_review() -> None:
    # TASK-005c3h H10: #24-#29 old -> new, and the H6 and H7 sentences.
    readme = normalised((REPO / "README.md").read_text(encoding="utf-8"))
    validation = normalised((REPO / "docs/validation/README.md").read_text(encoding="utf-8"))
    run_help = normalised(cast(str, cli.eval_run.__doc__))
    preflight_help = normalised(cast(str, cli.eval_preflight.__doc__))
    assert (
        "After a Claude run that made an API call other than the clock check and was not "
        "interrupted"
    ) in validation
    assert (
        "After a Claude run that made an API call other than the clock check and was not "
        "interrupted"
    ) in run_help
    assert (
        "a field the API does not report is shown as not reported, except the model id, which "
        "must match."
    ) in preflight_help
    assert (
        "a field the API does not report being shown as `not reported` (the model id must be "
        "reported and must match)"
    ) in readme
    assert (
        "or ASCII with every non-ASCII character escaped if any string in the answer holds an "
        "unpaired surrogate, which has no UTF-8 form; the answer is malformed only when its text "
        "or its stop holds one)"
    ) in validation
    for doc in (readme, validation, run_help):
        assert "an unknown batch in the workspace, a request that differs from its pin" in doc
        assert "acknowledged-batches.txt" in doc
        assert "against the SHA256 that the harness pins for" in doc
    for doc in (readme, validation, run_help):
        assert "called the API" not in doc
    assert (
        "by listing every page back to the window (``_list_back_to``) at least 120 seconds after "
        "the later of the call and its answer, or, for an entry left by an earlier run, at least "
        "180 seconds after its ``started_at``, before anything else is sent;"
    ) in normalised(cast(str, runner_module.__doc__))
    records_doc = normalised(cast(str, records_module.__doc__))
    assert (
        "a leftover ``*.tmp`` is simply overwritten by the next write (but the runner first stops "
        "on a ``response.json.tmp`` of a pending scenario, unless that scenario belongs to the "
        "open batch)."
    ) in records_doc
    comment = normalised((SOURCE / "cli.py").read_text(encoding="utf-8").replace("#", " "))
    assert (
        "after a run that made a counted call (any call but the clock check) and was not "
        "interrupted"
    ) in comment
