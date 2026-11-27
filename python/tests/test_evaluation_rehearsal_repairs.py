"""TASK-005e5: a rehearsal of the runbook's new sections, end to end, through the CLI.

One test follows ``docs/validation/test-a-runbook.md`` in its order, over a copy of the full
ten-scenario corpus, with fake keys, fake transports and a fake clock: Haiku's first run (one
data-level and one schema-invalid candidate, an answer of white space only, and 08, which every
batch marks expired: the run stops with ADR-0008 D6's wait after four batches and, once the fake
clock passes it, a fifth batch expires too, so 08 is unanswered); Haiku's repairs (§7: the dry
run, ``--yes``, ``eval repairs``); Gemini's first run and its repairs; Opus's first run, its two
repeats and ``eval draws`` (§8). Before the first batch of every new Claude run directory, the
workspace check lists the batches of every earlier kind and does not stop. At the end, the runs
directory holds exactly the expected files (no lock), the fake keys are in no file and no output,
every first-run and repeat ``request.json`` is the ``eval prompt`` payload and every repair
``request.json`` is ``repair_payload`` of it.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from runner_helpers import (
    ALL_SCENARIOS,
    KEY,
    Sleeps,
    anthropic_message,
    changed_truth,
    corpus_copy,
    files_containing,
    gemini_response,
)
from test_evaluation_rehearsal import GEMINI_KEY, Router, fenced
from test_evaluation_runner_anthropic import FakeBatches
from test_evaluation_runner_gemini import FakeGemini, ok
from typer.testing import CliRunner, Result

import potakt.cli as cli
from potakt.cli import app
from potakt.evaluation import answer_of, render_payload, repair_payload

HAIKU = "claude-haiku-5-5"
OPUS = "claude-opus-5-5"
GEMINI = "gemini-3.8-flash"
RUN_FILES = ("request.json", "response.json", "candidate.txt", "comparison.json")
WORKSPACE_WAIT = "waiting 120 s before checking the workspace for batches that no run recorded"
UNANSWERED = "08-warehouse"
SCHEMA = '{"schema_version": "0.2"}'
LOOK_ALIKE = "05-supermarket-checkouts"
EMPTY = "06-ambulance-crews"


def haiku_first(scenario: str) -> str:
    """Haiku's first answers: the truth, except 03 (schema), 05 (a look-alike id) and 06."""
    if scenario == "03-call-centre":
        return SCHEMA
    if scenario == LOOK_ALIKE:
        return changed_truth(LOOK_ALIKE, ("constraints", 6, "params", "resource_ids", 1), "T2")
    if scenario == EMPTY:
        return " \n\t "
    return fenced(scenario)


def outcome(number: int, scenario: str) -> dict[str, Any]:
    """Batches 1-5 are Haiku's first run (08 always expires), 6 Haiku's repairs (05 comes back
    as its truth, 03 as junk), 7 Opus's first run, 8 and 9 its repeats (02 is junk in draw 2)."""
    if number <= 5:
        if scenario == UNANSWERED:
            return {"type": "expired"}
        return {"type": "succeeded", "message": anthropic_message(HAIKU, haiku_first(scenario))}
    if number == 6:
        text = fenced(scenario) if scenario == LOOK_ALIKE else "Sorry."
        return {"type": "succeeded", "message": anthropic_message(HAIKU, text)}
    text = "Sorry." if number == 8 and scenario == "02-hotel-reception" else fenced(scenario)
    return {"type": "succeeded", "message": anthropic_message(OPUS, text)}


def started_plus_a_day(runs: Path, model_dir: str, index: int) -> str:
    """ADR-0008 D6's time: the ``started_at`` of entry ``index`` of a batches.json plus 24 h."""
    entry = json.loads((runs / model_dir / "batches.json").read_bytes())["batches"][index]
    moment = datetime.fromisoformat(entry["started_at"].replace("Z", "+00:00"))
    return (moment + timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")


def expected_repair(runs: Path, model: str, name: str, first_payload: bytes) -> bytes:
    """The repair request of ``name``: the ``eval prompt`` payload, the first answer's text (read
    from the first run's response.json) and the issues of its comparison.json."""
    first_dir = runs / model
    response = json.loads((first_dir / name / "response.json").read_bytes())
    issues = json.loads((first_dir / name / "comparison.json").read_bytes())["issues"]
    pairs = [(issue["path"], issue["message"]) for issue in issues]
    text = answer_of(model, response).text
    return render_payload(repair_payload(json.loads(first_payload), model, text, pairs)).encode()


def test_the_runbook_of_repairs_repeats_and_unanswered_scenarios_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = corpus_copy(tmp_path)
    runs = tmp_path / "runs"
    clock = Sleeps()
    batches = FakeBatches(outcome=outcome, clock=clock, stamp_with_clock=True)
    gemini = FakeGemini()
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

    def first_batch_checked(result: Result, created: str) -> None:
        """The run's first batch came after the workspace check, which did not stop."""
        assert result.stdout.startswith(f"{WORKSPACE_WAIT}\n{created}\n"), result.output

    # 1. Haiku's first run: 08 expires in four batches, then the run waits (ADR-0008 D6).
    stopped = eval_run(HAIKU, "--yes")
    assert stopped.exit_code == 1
    first_batch_checked(stopped, "batch msgbatch_01 created for 10 scenarios")
    assert batches.creates() == [list(ALL_SCENARIOS)] + [[UNANSWERED]] * 3
    time = started_plus_a_day(runs, HAIKU, 3)
    d6_stop = (
        f"{UNANSWERED} was sent in 4 batches without an answer; ADR-0008 D6 sends it once more "
        f"from {time}: run the same command again then; nothing was sent\n"
    )
    assert stopped.stderr == d6_stop
    again = eval_run(HAIKU, "--yes")  # the same command too early: the same stop, nothing sent
    assert (again.exit_code, again.stderr) == (1, d6_stop)
    assert len(batches.creates()) == 4
    plan = eval_run(HAIKU)
    assert f"Last batch (ADR-0008 D6) from {time}: {UNANSWERED}\n" in plan.stdout
    clock.advance(24 * 3600)  # the owner runs the same command the next day
    fifth = eval_run(HAIKU, "--yes")
    assert fifth.exit_code == 0, fifth.stderr
    assert batches.creates()[-1] == [UNANSWERED]
    assert (
        f"{UNANSWERED}: no answer after 5 batches; it counts as an invalid candidate (ADR-0008 D6)"
        "\n"
    ) in fifth.stdout
    assert "Complete: yes\n" in fifth.stdout
    assert "Stops: end_turn 9, no-answer 1\n" in fifth.stdout
    unanswered = json.loads((runs / HAIKU / UNANSWERED / "unanswered.json").read_bytes())
    assert unanswered == {"batches": [f"msgbatch_0{n}" for n in range(1, 6)]}
    rescored = potakt("eval", "rescore", runs / HAIKU, "--corpus", corpus)
    assert rescored.stdout == "".join(
        f"{name}: {'skipped (no response.json)' if name == UNANSWERED else 'unchanged'}\n"
        for name in ALL_SCENARIOS
    )
    assert f"Unanswered (ADR-0008 D6): {UNANSWERED}\n" in eval_run(HAIKU).stdout

    # 2. Haiku's repairs (§7): the dry run, then one batch of the two repairable candidates.
    repair_plan = eval_run(HAIKU, "--repair")
    assert repair_plan.exit_code == 0, repair_plan.stderr
    lines = shown(repair_plan.stdout).splitlines()
    assert lines[:6] == [
        "Test A run plan: claude-haiku-5-5 (repair)",
        "Runs directory: <runs>/repair/claude-haiku-5-5",
        f"Pending (2): 03-call-centre, {LOOK_ALIKE}",
        "Recorded (0): none",
        f"Not sent (empty answer): {EMPTY}",
        f"Not sent (no answer, ADR-0008 D6): {UNANSWERED}",
    ]
    payloads = [
        json.loads(expected_repair(runs, HAIKU, name, prompt_bytes(corpus, name, HAIKU)))
        for name in ("03-call-centre", LOOK_ALIKE)
    ]
    size = sum(
        len(payload["system"].encode("utf-8"))
        + sum(len(turn["content"].encode("utf-8")) for turn in payload["messages"])
        for payload in payloads
    )
    cost = (size * 0.05 + 2 * 64000 * 0.25) / 1e6
    assert lines[6] == (
        f"Cost upper bound: ${cost:.4f} (2 requests; input: the UTF-8 bytes of every message, "
        "output: max_tokens 64000; batch prices pinned on 2026-11-02)"
    )
    repaired = eval_run(HAIKU, "--repair", "--yes")
    assert repaired.exit_code == 0, repaired.stderr
    first_batch_checked(repaired, "batch msgbatch_06 created for 2 scenarios")
    assert batches.creates()[-1] == ["03-call-centre", LOOK_ALIKE]

    # 3. eval repairs: 46 of 79 matched first (01, 02, 04, 07, 09, 10), 05 adds its 8.
    report = potakt("eval", "repairs", "--model", HAIKU, "--runs", runs, "--corpus", corpus)
    expected = (
        "Repairs: claude-haiku-5-5 (ADR-0008 D7, not the gate)\n"
        "First run: 10 scored, 6 valid, 4 invalid\n"
        "Repair run: 2 recorded, 0 without an answer, 2 not sent, 0 missing\n"
        "Complete: yes\n"
        "Recall: 0.582 (46/79)\n"
        "Recall after one repair (not the gate): 0.684 (54/79); 1 of 4 invalid candidates valid "
        f"after one repair ({LOOK_ALIKE})\n"
        "Per invalid candidate:\n"
        "  03-call-centre: first invalid (schema), repair invalid (JSON or fence)\n"
        f"  {LOOK_ALIKE}: first invalid (data-level id or attribute), repair valid, "
        "matched 8 of 8\n"
        f"  {EMPTY}: first invalid (JSON or fence), not sent (empty answer)\n"
        f"  {UNANSWERED}: first invalid (stop reason), not sent (no answer, ADR-0008 D6)\n"
    )
    assert report.stdout == expected
    assert repaired.stdout.endswith(expected)  # the repair run's closing report
    rescored = potakt("eval", "rescore", runs / "repair" / HAIKU, "--corpus", corpus)
    assert rescored.stdout == f"03-call-centre: unchanged\n{LOOK_ALIKE}: unchanged\n"

    # 4. Gemini's first run (02 junk, 04 cut off by the limit) and its repairs.
    for name in ALL_SCENARIOS:
        gemini.script[name] = [ok(gemini_response(fenced(name)))]
    gemini.script["02-hotel-reception"] = [ok(gemini_response("Sorry."))]
    gemini.script["04-restaurant"] = [ok(gemini_response('{"title": ', "MAX_TOKENS"))]
    first_gemini = eval_run(GEMINI, "--yes")
    assert first_gemini.exit_code == 0, first_gemini.stderr
    assert "Complete: yes\n" in first_gemini.stdout
    gemini.script["02-hotel-reception"] = [ok(gemini_response(fenced("02-hotel-reception")))]
    calls = len(gemini.calls)
    gemini_repairs = eval_run(GEMINI, "--repair", "--yes")
    assert gemini_repairs.exit_code == 0, gemini_repairs.stderr
    assert gemini.sent()[calls:] == ["02-hotel-reception", "04-restaurant"]
    for call in gemini.calls[calls:]:
        roles = [turn["role"] for turn in json.loads(call.body)["contents"]]
        assert roles == ["user", "model", "user"]
    assert gemini_repairs.stdout.startswith("02-hotel-reception: recorded (STOP, valid, recall")
    assert "Repairs: gemini-3.8-flash (ADR-0008 D7, not the gate)\n" in gemini_repairs.stdout

    # 5. Opus's first run, then its repeats (§8) and eval draws.
    opus = eval_run(OPUS, "--yes")
    assert opus.exit_code == 0, opus.stderr
    first_batch_checked(opus, "batch msgbatch_07 created for 10 scenarios")
    too_early = eval_run(OPUS, "--repeat", "3", "--yes")
    assert too_early.exit_code == 1
    assert too_early.stderr == (
        "repeat 3 of claude-opus-5-5 runs only after repeat 2 is complete "
        f"(missing: {', '.join(ALL_SCENARIOS)})\n"
    )
    repeat_plan = eval_run(OPUS, "--repeat", "2")
    assert repeat_plan.stdout.startswith("Test A run plan: claude-opus-5-5 (repeat 2)\n")
    assert "Cost upper bound: $7.6301 (10 requests;" in repeat_plan.stdout  # = the first run's
    for k, number in (("2", 8), ("3", 9)):
        repeat = eval_run(OPUS, "--repeat", k, "--yes")
        assert repeat.exit_code == 0, repeat.stderr
        first_batch_checked(repeat, f"batch msgbatch_0{number} created for 10 scenarios")
        assert "Test A summary: claude-opus-5-5\n" in repeat.stdout
    draws = potakt("eval", "draws", "--runs", runs, "--corpus", corpus)
    assert draws.exit_code == 0, draws.stderr
    ones = "1.000, 1.000, 1.000; mean 1.000, min 1.000, max 1.000"
    assert draws.stdout == (
        "Draws: claude-opus-5-5 (ADR-0008 D8, not the gate)\n"
        "Draw 1 (claude-opus-5-5): recall 1.000 (79/79)\n"
        "Draw 2 (repeat-2/claude-opus-5-5): recall 0.899 (71/79)\n"
        "Draw 3 (repeat-3/claude-opus-5-5): recall 1.000 (79/79)\n"
        "Per scenario (recall in draws 1, 2, 3; mean, min, max):\n"
        + "".join(
            f"  {name}: "
            + (
                "1.000, 0.000, 1.000; mean 0.667, min 0.000, max 1.000"
                if name == "02-hotel-reception"
                else ones
            )
            + "\n"
            for name in ALL_SCENARIOS
        )
    )
    assert len(batches.creates()) == 9
    assert all("warning" not in result.output for result in results)  # no unrecorded batch

    # 6. The final checks.
    files = sorted(path.relative_to(runs).as_posix() for path in runs.rglob("*") if path.is_file())
    expected_files = [f"{HAIKU}/batches.json", f"repair/{HAIKU}/batches.json"]
    expected_files += [f"{HAIKU}/batch-msgbatch_0{n}.results.jsonl" for n in range(1, 6)]
    expected_files += [f"repair/{HAIKU}/batch-msgbatch_06.results.jsonl"]
    expected_files += [
        f"{HAIKU}/{name}/{file}"
        for name in ALL_SCENARIOS
        for file in (
            ("request.json", "comparison.json", "unanswered.json")
            if name == UNANSWERED
            else RUN_FILES
        )
    ]
    expected_files += [
        f"repair/{HAIKU}/{name}/{file}"
        for name in ("03-call-centre", LOOK_ALIKE)
        for file in RUN_FILES
    ]
    expected_files += [
        f"{GEMINI}/{name}/{file}"
        for name in ALL_SCENARIOS
        for file in (*RUN_FILES, "received.json")
    ]
    expected_files += [
        f"repair/{GEMINI}/{name}/{file}"
        for name in ("02-hotel-reception", "04-restaurant")
        for file in (*RUN_FILES, "received.json")
    ]
    for folder, number in ((OPUS, 7), (f"repeat-2/{OPUS}", 8), (f"repeat-3/{OPUS}", 9)):
        expected_files += [
            f"{folder}/batches.json",
            f"{folder}/batch-msgbatch_0{number}.results.jsonl",
            *(f"{folder}/{name}/{file}" for name in ALL_SCENARIOS for file in RUN_FILES),
        ]
    assert files == sorted(expected_files)  # no lock file left either
    assert not list(runs.rglob("*.tmp"))
    for key in (KEY, GEMINI_KEY):
        assert files_containing(tmp_path, key) == []
        assert not any(key in result.stdout + result.stderr for result in results)
    first_runs = {HAIKU: [HAIKU], GEMINI: [GEMINI], OPUS: [OPUS, f"repeat-2/{OPUS}"]}
    first_runs[OPUS].append(f"repeat-3/{OPUS}")
    for model, folders in first_runs.items():
        for name in ALL_SCENARIOS:
            first = prompt_bytes(corpus, name, model)
            for folder in folders:
                assert (runs / folder / name / "request.json").read_bytes() == first
    repairs = {
        HAIKU: ("03-call-centre", LOOK_ALIKE),
        GEMINI: ("02-hotel-reception", "04-restaurant"),
    }
    for model, names in repairs.items():
        for name in names:
            recorded = (runs / "repair" / model / name / "request.json").read_bytes()
            first = prompt_bytes(corpus, name, model)
            assert recorded == expected_repair(runs, model, name, first)


def prompt_bytes(corpus: Path, name: str, model: str) -> bytes:
    """The ``eval prompt`` payload of a scenario, as the command prints it."""
    printed = CliRunner().invoke(app, ["eval", "prompt", str(corpus / name), "--model", model])
    assert printed.exit_code == 0
    return printed.stdout_bytes
