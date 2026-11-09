"""TASK-005c3b D7, TASK-005c3c E8: a rehearsal of the Test A runbook, end to end, through the CLI.

One test follows the owner's runbook (``docs/validation/test-a-runbook.md``) in its order, over a
copy of the full ten-scenario corpus, with fake keys, fake transports and a fake clock: the dry
runs, Haiku (one answer is not valid Unicode: recorded as malformed, TASK-005c3d H8), the checks
of the Haiku records, Gemini in two invocations, then Opus with a lost create response whose
adopted batch misses one result: the batch is held, the plain re-run sends nothing, and
``--release`` after the reviewer's decision completes the run (TASK-005c3h H3: first stopped by a
wrong laptop clock after the release, then the same command again). Each Claude model's first run
waits and checks the workspace before its first batch (H6). Every invocation that called the
batch API is audited for unrecorded batches (H13), and every request carries only its own API's
key header (H7). At the end, the runs
directory holds exactly the expected files (no lock), the fake keys are in no file and no output,
and every ``request.json`` is the ``eval prompt`` payload.
"""

import email.utils
import json
import os
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from runner_helpers import (
    ALL_SCENARIOS,
    KEY,
    RefusedReplace,
    Sleeps,
    anthropic_message,
    corpus_copy,
    files_containing,
    truth_text,
)
from test_evaluation_runner_anthropic import FakeBatches
from test_evaluation_runner_gemini import FakeGemini, error
from typer.testing import CliRunner, Result

import potakt.cli as cli
from potakt.cli import app
from potakt.evaluation.runner import BATCHES_URL
from potakt.evaluation.transport import HttpResponse

HAIKU = "claude-haiku-5-5"
OPUS = "claude-opus-5-5"
GEMINI = "gemini-3.8-flash"
GEMINI_KEY = "potakt-test-gemini-key"  # a second fake key: only tests ever hold it
GOLDEN = Path(__file__).resolve().parent / "golden"
RUN_FILES = ("request.json", "response.json", "candidate.txt", "comparison.json")
# TASK-005c3h H6: the first run of each Claude model checks the workspace first.
WORKSPACE_WAIT = "waiting 120 s before checking the workspace for batches that no run recorded"


def fenced(name: str) -> str:
    """The truth of a scenario as a model would answer it: one ```json fence."""
    return "```json\n" + truth_text(name).rstrip("\n") + "\n```"


SURROGATE_SCENARIO = "05-supermarket-checkouts"  # TASK-005c3d H14: Haiku answers "\ud83d"


def claude_outcome(number: int, scenario: str) -> dict[str, Any]:
    """Batch 1 is Haiku's (04 hits the output limit, 05 is not valid Unicode), batch 2 Opus's."""
    model = HAIKU if number == 1 else OPUS
    stop = "max_tokens" if number == 1 and scenario == "04-restaurant" else "end_turn"
    text = "\ud83d" if number == 1 and scenario == SURROGATE_SCENARIO else fenced(scenario)
    return {"type": "succeeded", "message": anthropic_message(model, text, stop)}


class Router:
    """The one transport of every run: the Message Batches fake or the Gemini fake. Each request
    carries exactly its own API's key header (TASK-005c3d H7)."""

    def __init__(self, batches: FakeBatches, gemini: FakeGemini) -> None:
        self.batches = batches
        self.gemini = gemini

    def send(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
    ) -> HttpResponse:
        if url.startswith(BATCHES_URL):
            assert headers["authorization"] == f"Bearer {KEY}"
            assert "x-api-key" not in headers and "x-goog-api-key" not in headers
            response = self.batches.send(method, url, headers, body, timeout)
            assert "date" in response.headers  # TASK-005c3g G2: the API's clock, every answer
            return response
        assert headers["x-goog-api-key"] == GEMINI_KEY
        assert "authorization" not in headers and "x-api-key" not in headers
        response = self.gemini.send(method, url, headers, body, timeout)
        # TASK-005c3g G13: Google's answers carry a Date too (the Gemini run does not read it).
        moment = self.batches.now().replace(microsecond=0)
        date = email.utils.format_datetime(moment, usegmt=True)
        return replace(response, headers={**response.headers, "date": date})


def test_the_runbook_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = corpus_copy(tmp_path)
    runs = tmp_path / "runs"
    clock = Sleeps()
    batches = FakeBatches(outcome=claude_outcome, clock=clock, stamp_with_clock=True)
    gemini = FakeGemini(script={"03-call-centre": [error(503, "UNAVAILABLE", "overloaded")] * 4})
    monkeypatch.setattr(cli, "make_transport", lambda: Router(batches, gemini))
    monkeypatch.setattr(cli.time, "sleep", clock)
    monkeypatch.setattr(cli, "utc_now", clock.now)
    monkeypatch.setenv("POTAKT_ANTHROPIC_API_KEY", KEY)
    monkeypatch.setenv("POTAKT_GEMINI_API_KEY", GEMINI_KEY)
    results: list[Result] = []

    def potakt(*args: str | Path) -> Result:
        result = CliRunner().invoke(app, [str(arg) for arg in args])
        results.append(result)
        return result

    def eval_run(model: str, *extra: str) -> Result:
        return potakt("eval", "run", "--model", model, "--runs", runs, "--corpus", corpus, *extra)

    def audited(result: Result, lists_before: int, run_lists: int) -> None:
        """TASK-005c3d H13: an invocation that called the batch API lists the batches once more
        after its own ``run_lists`` listings, and names no batch (every one is recorded)."""
        assert batches.lists - lists_before == run_lists + 1
        assert "warning" not in result.output

    def shown(text: str) -> str:
        return text.replace(str(runs), "<runs>").replace("\\", "/")

    # 1. The dry runs: Haiku's plan and cost bound; Opus is refused.
    plan = eval_run(HAIKU)
    assert plan.exit_code == 0, plan.stderr
    golden = (GOLDEN / "eval-run-plan-claude-haiku-5-5.txt").read_text(encoding="utf-8")
    assert shown(plan.stdout) == golden
    # TASK-005c3g G8: max_tokens 64000; TASK-005e1: the schema section shows every rule.
    assert "Cost upper bound: $0.1908 " in plan.stdout
    refused = eval_run(OPUS)
    assert refused.exit_code == 1
    assert refused.stdout == ""
    assert refused.stderr.startswith(
        "claude-opus-5-5 runs only after claude-haiku-5-5 has a complete run"
    )

    # 2. Haiku --yes: one batch; 04 stops at max_tokens; 05's answer is not valid Unicode, so it
    # is recorded as malformed (an invalid candidate) and the run goes on (TASK-005c3d H8).
    # TASK-005c3g: the clock check passes first (G2); one replace of a response.json is refused
    # once, as Windows does while another program holds the file, and absorbed (G3); Haiku asks
    # for max_tokens 64000 (G8).
    lists = batches.lists
    refused = RefusedReplace(1, "response.json")
    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", refused)
        haiku = eval_run(HAIKU, "--yes")
    assert refused.refused == 1
    assert len(batches.clock_checks) == 1
    assert batches.calls[0].url == f"{BATCHES_URL}?limit=1"
    assert batches.calls[1].url == f"{BATCHES_URL}?limit=100"  # TASK-005c3h H6: the workspace
    sent = json.loads(batches.calls[2].body or b"{}")["requests"]
    assert {request["params"]["max_tokens"] for request in sent} == {64000}
    assert haiku.exit_code == 0, haiku.stderr
    assert haiku.stdout.startswith(
        f"{WORKSPACE_WAIT}\nbatch msgbatch_01 created for 10 scenarios\n"
    )
    assert f"{SURROGATE_SCENARIO}: recorded (malformed, invalid, recall 0.000)\n" in haiku.stdout
    assert "Complete: yes\n" in haiku.stdout
    assert "Stops: end_turn 8, malformed 1, max_tokens 1\n" in haiku.stdout
    assert "waiting 240 s before checking for batches that no run recorded\n" in haiku.stdout
    audited(haiku, lists, 1)  # the workspace's listing (H6), then the audit's
    raw = (runs / HAIKU / SURROGATE_SCENARIO / "response.json").read_bytes()
    assert raw.isascii()
    assert b"\\ud83d" in raw

    # 3. The checks of the Haiku records.
    rescored = potakt("eval", "rescore", runs / HAIKU, "--corpus", corpus)
    assert rescored.exit_code == 0, rescored.stderr
    assert rescored.stdout == "".join(f"{name}: unchanged\n" for name in ALL_SCENARIOS)
    summary = potakt("eval", "summary", runs / HAIKU, "--corpus", corpus)
    assert summary.exit_code == 0, summary.stderr
    assert "Complete: yes\n" in summary.stdout

    # 4. Gemini: 03 is overloaded four times, so the first invocation stops after 01-02.
    first = eval_run(GEMINI, "--yes")
    assert first.exit_code == 1
    assert first.stderr.startswith("03-call-centre: the request failed 4 times; last: HTTP 503")
    recorded = sorted(path.parent.name for path in (runs / GEMINI).glob("*/response.json"))
    assert recorded == ["01-pharmacy", "02-hotel-reception"]
    blocked = {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}, "usageMetadata": {}}
    gemini.script["06-ambulance-crews"] = [
        HttpResponse(200, {}, json.dumps(blocked).encode("utf-8"))
    ]
    second = eval_run(GEMINI, "--yes")
    assert second.exit_code == 0, second.stderr
    assert "06-ambulance-crews: recorded (blocked:PROHIBITED_CONTENT, invalid, recall" in (
        second.stdout
    )
    assert "Complete: yes\n" in second.stdout
    assert gemini.sent().count("01-pharmacy") == 1  # a recorded scenario is never sent again

    # 5. Opus: the dry run is allowed now; the create response is lost after the server created
    # the batch, and the run adopts it instead of paying for a second one. Its results miss the
    # line of 07-security-guards, so the batch is held for the reviewer (TASK-005c3c E4).
    opus_plan = eval_run(OPUS)
    assert opus_plan.exit_code == 0, opus_plan.stderr
    golden = (GOLDEN / "eval-run-plan-claude-opus-5-5.txt").read_text(encoding="utf-8")
    assert shown(opus_plan.stdout) == golden
    batches.create_script = ["lost"]
    missing = "07-security-guards"
    batches.raw_results["msgbatch_02"] = "".join(
        json.dumps({"custom_id": name, "result": claude_outcome(2, name)}) + "\n"
        for name in ALL_SCENARIOS
        if name != missing
    ).encode("utf-8")
    lists = batches.lists
    held = eval_run(OPUS, "--yes")
    audited(held, lists, 2)  # the workspace's (H6: Haiku's batch is known), the reconciliation's
    assert held.exit_code == 1
    assert held.stdout.startswith(
        f"{WORKSPACE_WAIT}\n"
        "waiting 120 s before looking for a batch the create request may have made\n"
        "adopted batch msgbatch_02 created by an earlier create request\n"
    )
    assert held.stderr == (
        f"batch msgbatch_02: no result for {missing}; batch msgbatch_02 is held: send "
        "batch-msgbatch_02.results.jsonl and this message to the reviewer\n"
    )
    # The plain re-run stops at once and sends nothing.
    calls = len(batches.calls)
    again = eval_run(OPUS, "--yes")
    assert again.exit_code == 1
    assert again.stderr == (
        f"batch msgbatch_02 is held for the reviewer: batch msgbatch_02: no result for {missing}; "
        "after the reviewer's decision, run the same command with --release msgbatch_02\n"
    )
    assert len(batches.calls) == calls  # no call, so no audit either
    assert "checking for batches" not in again.output
    # After the reviewer's decision: the release. TASK-005c3h H3: the laptop clock is 15 minutes
    # off, so the run stops after the release, at the clock check; the owner syncs the clock and
    # runs the same command again, which passes the release step and sends the missing scenario
    # in a new batch.
    with monkeypatch.context() as patch:
        patch.setattr(cli, "utc_now", lambda: clock.now() + timedelta(minutes=15))
        skewed = eval_run(OPUS, "--yes", "--release", "msgbatch_02")
    assert skewed.exit_code == 1
    assert skewed.stdout == "batch msgbatch_02 released after review\n"
    assert skewed.stderr.startswith("the laptop clock differs from the API's by 900 s")
    assert len(batches.calls) == calls + 1  # the free clock call only, so no audit either
    lists = batches.lists
    opus = eval_run(OPUS, "--yes", "--release", "msgbatch_02")
    audited(opus, lists, 0)
    assert opus.exit_code == 0, opus.stderr
    assert opus.stdout.startswith(
        "batch msgbatch_02 was already released\nbatch msgbatch_03 created for 1 scenarios\n"
    )
    assert batches.creates() == [list(ALL_SCENARIOS), list(ALL_SCENARIOS), [missing]]
    assert "Complete: yes\n" in opus.stdout
    rescored = potakt("eval", "rescore", runs / OPUS, "--corpus", corpus)
    assert rescored.stdout == "".join(f"{name}: unchanged\n" for name in ALL_SCENARIOS)
    summary = potakt("eval", "summary", runs / OPUS, "--corpus", corpus)
    assert "Complete: yes\n" in summary.stdout

    # 6. The final checks.
    files = sorted(path.relative_to(runs).as_posix() for path in runs.rglob("*") if path.is_file())
    expected = [
        f"{model}/{name}/{file}"
        for model in (HAIKU, GEMINI, OPUS)
        for name in ALL_SCENARIOS
        for file in RUN_FILES
    ] + [
        f"{HAIKU}/batch-msgbatch_01.results.jsonl",
        f"{HAIKU}/batches.json",
        f"{OPUS}/batch-msgbatch_02.results.jsonl",
        f"{OPUS}/batch-msgbatch_03.results.jsonl",
        f"{OPUS}/batches.json",
    ]
    assert files == sorted(expected)  # no lock file left either
    assert not list(runs.rglob("*.tmp"))
    haiku_state = json.loads((runs / HAIKU / "batches.json").read_bytes())
    assert [entry["state"] for entry in haiku_state["batches"]] == ["done"]
    opus_state = json.loads((runs / OPUS / "batches.json").read_bytes())
    assert [entry["state"] for entry in opus_state["batches"]] == ["done", "done"]
    assert opus_state["batches"][0]["problems"] == [f"batch msgbatch_02: no result for {missing}"]
    for key in (KEY, GEMINI_KEY):
        assert files_containing(tmp_path, key) == []
        assert not any(key in result.stdout + result.stderr for result in results)
    for model in (HAIKU, GEMINI, OPUS):
        for name in ALL_SCENARIOS:
            printed = CliRunner().invoke(
                app, ["eval", "prompt", str(corpus / name), "--model", model]
            )
            assert printed.exit_code == 0
            assert (runs / model / name / "request.json").read_bytes() == printed.stdout_bytes
    # TASK-005c3g G2: one clock check per Claude --yes run that got past the held check (Haiku,
    # the held Opus run, the skewed release, the release again); all but the skewed one passed.
    assert len(batches.clock_checks) == 4
