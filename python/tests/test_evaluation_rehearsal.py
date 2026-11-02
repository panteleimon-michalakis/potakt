"""TASK-005c3b D7: a rehearsal of the Test A runbook, end to end, through the CLI.

One test follows the owner's runbook (``docs/validation/test-a-runbook.md``) in its order, over a
copy of the full ten-scenario corpus, with fake keys, fake transports and a fake clock: the dry
runs, Haiku, the checks of the Haiku records, Gemini in two invocations, then Opus with a lost
create response. At the end, the runs directory holds exactly the expected files, the fake keys
are in no file and no output, and every ``request.json`` is the ``eval prompt`` payload.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from runner_helpers import (
    ALL_SCENARIOS,
    KEY,
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


def fenced(name: str) -> str:
    """The truth of a scenario as a model would answer it: one ```json fence."""
    return "```json\n" + truth_text(name).rstrip("\n") + "\n```"


def claude_outcome(number: int, scenario: str) -> dict[str, Any]:
    """Batch 1 is Haiku's (04 hits the output limit), batch 2 Opus's."""
    model = HAIKU if number == 1 else OPUS
    stop = "max_tokens" if number == 1 and scenario == "04-restaurant" else "end_turn"
    return {"type": "succeeded", "message": anthropic_message(model, fenced(scenario), stop)}


class Router:
    """The one transport of every run: the Message Batches fake or the Gemini fake."""

    def __init__(self, batches: FakeBatches, gemini: FakeGemini) -> None:
        self.batches = batches
        self.gemini = gemini

    def send(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
    ) -> HttpResponse:
        target = self.batches if url.startswith(BATCHES_URL) else self.gemini
        return target.send(method, url, headers, body, timeout)


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

    def shown(text: str) -> str:
        return text.replace(str(runs), "<runs>").replace("\\", "/")

    # 1. The dry runs: Haiku's plan and cost bound; Opus is refused.
    plan = eval_run(HAIKU)
    assert plan.exit_code == 0, plan.stderr
    golden = (GOLDEN / "eval-run-plan-claude-haiku-5-5.txt").read_text(encoding="utf-8")
    assert shown(plan.stdout) == golden
    assert "Cost upper bound: $0.1034 " in plan.stdout
    refused = eval_run(OPUS)
    assert refused.exit_code == 1
    assert refused.stdout == ""
    assert refused.stderr.startswith(
        "claude-opus-5-5 runs only after claude-haiku-5-5 has a complete run"
    )

    # 2. Haiku --yes: one batch; 04 stops at max_tokens.
    haiku = eval_run(HAIKU, "--yes")
    assert haiku.exit_code == 0, haiku.stderr
    assert haiku.stdout.startswith("batch msgbatch_01 created for 10 scenarios\n")
    assert "Complete: yes\n" in haiku.stdout
    assert "Stops: end_turn 9, max_tokens 1\n" in haiku.stdout

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
    # the batch, and the run adopts it instead of paying for a second one.
    opus_plan = eval_run(OPUS)
    assert opus_plan.exit_code == 0, opus_plan.stderr
    golden = (GOLDEN / "eval-run-plan-claude-opus-5-5.txt").read_text(encoding="utf-8")
    assert shown(opus_plan.stdout) == golden
    batches.create_script = ["lost"]
    opus = eval_run(OPUS, "--yes")
    assert opus.exit_code == 0, opus.stderr
    assert opus.stdout.startswith(
        "adopted batch msgbatch_02 created by an earlier create request\n"
    )
    assert len(batches.creates()) == 2  # one batch per Claude model
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
        f"{OPUS}/batches.json",
    ]
    assert files == sorted(expected)
    assert not list(runs.rglob("*.tmp"))
    for model in (HAIKU, OPUS):
        state = json.loads((runs / model / "batches.json").read_bytes())
        assert [entry["state"] for entry in state["batches"]] == ["done"]
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
