"""The summary of recorded Test A runs (ADR-0007 §7 «Records» and its amendment; TASK-005c1 D4,
TASK-005c2 E4).

The fixtures in ``fixtures/test_a_runs/claude-opus-5-5`` are real answer shapes for three
scenarios: 01 with the truth in a ```` ```json ```` fence (``end_turn``), 03 with the imperfect
candidate of the compare tests as bare text (``end_turn``), and 04 with a refusal. Their
``candidate.txt`` and ``comparison.json`` are what ``rescore_run`` derives from ``response.json``,
so they cannot drift from the rules or from ``compare.py``.
"""

import dataclasses
import json
import random
import shutil
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from runner_helpers import anthropic_message, truth_text
from test_evaluation_lookalike_recall import written

from potakt.evaluation import (
    LONG_PROMPT_PRICES_USD_PER_MILLION,
    PRICES_USD_PER_MILLION,
    InvalidStage,
    bootstrap_interval,
    load_scenario,
    model_directory_name,
    rescore_run,
    summarize,
    write_answer,
)
from potakt.evaluation.summary import (
    ASSUMPTION_TEXT_LIMIT,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    GATE_RECALL,
    ISSUE_CLASSES,
    ScenarioRow,
)

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "docs" / "validation" / "scenarios"
RUNS = Path(__file__).resolve().parent / "fixtures" / "test_a_runs"
OPUS_RUNS = RUNS / "claude-opus-5-5"
GOLDEN = Path(__file__).resolve().parent / "golden"
FIXTURE_SCENARIOS = ["01-pharmacy", "03-call-centre", "04-restaurant"]
NOT_RUN = [
    "02-hotel-reception",
    "05-supermarket-checkouts",
    "06-ambulance-crews",
    "07-security-guards",
    "08-warehouse",
    "09-care-home",
    "10-municipal-nursery",
]


@pytest.mark.parametrize("name", FIXTURE_SCENARIOS)
def test_every_fixture_run_is_what_rescoring_derives_from_its_response(
    name: str, tmp_path: Path
) -> None:
    run = tmp_path / name
    shutil.copytree(OPUS_RUNS / name, run)  # rescoring may write: never into the fixtures

    assert rescore_run(run, load_scenario(CORPUS / name), "claude-opus-5-5") == ()


def test_the_fixture_answers_are_a_fence_bare_text_and_a_refusal() -> None:
    def response(name: str) -> dict[str, object]:
        return json.loads((OPUS_RUNS / name / "response.json").read_bytes())

    truth = (CORPUS / "01-pharmacy" / "truth.json").read_text(encoding="utf-8")
    bare = (OPUS_RUNS / "03-call-centre" / "candidate.txt").read_text(encoding="utf-8")
    assert response("01-pharmacy")["content"] == [
        {"type": "text", "text": "```json\n" + truth + "\n```"}
    ]
    assert response("03-call-centre")["content"] == [{"type": "text", "text": bare}]
    assert response("04-restaurant")["content"] == [
        {"type": "text", "text": "I am sorry, I cannot produce that model."}
    ]
    stops = [response(name)["stop_reason"] for name in FIXTURE_SCENARIOS]
    assert stops == ["end_turn", "end_turn", "refusal"]


def test_summary_json_matches_the_golden_byte_for_byte() -> None:
    summary = summarize(OPUS_RUNS, CORPUS)

    assert summary.to_json().encode("utf-8") == (GOLDEN / "test-a-summary.json").read_bytes()


def test_summary_text_matches_the_golden_byte_for_byte() -> None:
    summary = summarize(OPUS_RUNS, CORPUS)

    assert summary.to_text().encode("utf-8") == (GOLDEN / "test-a-summary.txt").read_bytes()


def test_summary_numbers_match_the_hand_computation() -> None:
    # 01: 5 of 5 (el). 03: 5 of 8 (en). 04: refused, invalid, 0 of 12 (en). Micro-averaged over 25.
    document = json.loads(summarize(OPUS_RUNS, CORPUS).to_json())

    assert (document["scenarios_run"], document["valid"]) == (3, 2)
    assert document["recall"] == 10 / 25
    assert document["merged_recall"] == 10 / 25
    assert document["precision"] == 10 / 13
    assert document["recall_by_language"] == {
        "el": {"matched": 5, "truth": 5, "recall": 1.0},
        "en": {"matched": 5, "truth": 20, "recall": 0.25},
    }
    assert document["recall_by_hardness"] == {
        "hard": {"matched": 7, "truth": 14, "recall": 0.5},
        "soft": {"matched": 3, "truth": 11, "recall": 3 / 11},
    }
    by_template = {k: (v["matched"], v["truth"]) for k, v in document["recall_by_template"].items()}
    assert by_template == {
        "apart": (0, 1),
        "balance": (0, 1),
        "max_consecutive_days": (0, 2),
        "max_per_period": (3, 4),
        "min_per_period": (1, 2),
        "min_rest": (0, 2),
        "no_overlap": (1, 2),
        "preference": (2, 4),
        "requires_attribute": (2, 3),
        "together": (0, 2),
        "unavailable": (1, 2),
    }
    assert document["diagnostics"] == {
        "missing": 13,
        "wrong_hardness": 1,
        "wrong_parameters": 1,
        "wrong_scope": 0,
        "other_template_same_effect": 0,
    }
    assert document["usage"] == {"input_tokens": 36000, "output_tokens": 18000}
    assert document["cost_usd"] == 0.252  # (36000 * 2 + 18000 * 10) / 1e6, batch prices
    assert (document["not_scored"], document["notes"]) == ([], [])
    assert (document["missing"], document["complete"]) == (NOT_RUN, False)
    assert document["stops"] == {"end_turn": 2, "refusal": 1}


def test_summary_text_states_completeness_stops_and_batch_prices() -> None:
    lines = summarize(OPUS_RUNS, CORPUS).to_text().splitlines()

    assert f"Complete: no (missing: {', '.join(NOT_RUN)})" in lines
    assert "Stops: end_turn 2, refusal 1" in lines
    assert "Usage: 36000 input tokens, 18000 output tokens" in lines
    assert "Cost: $0.2520 (batch prices pinned on 2026-11-02)" in lines


def test_summary_is_deterministic() -> None:
    assert summarize(OPUS_RUNS, CORPUS).to_json() == summarize(OPUS_RUNS, CORPUS).to_json()


# --- missing and unusable records ------------------------------------------------------------


def copy_runs(tmp_path: Path, model: str = "claude-opus-5-5") -> Path:
    target = tmp_path / model
    shutil.copytree(OPUS_RUNS, target)
    return target


def test_a_missing_comparison_is_reported_not_ignored(tmp_path: Path) -> None:
    runs = copy_runs(tmp_path)
    (runs / "03-call-centre" / "comparison.json").unlink()

    summary = summarize(runs, CORPUS)

    document = json.loads(summary.to_json())
    assert document["not_scored"] == ["03-call-centre"]
    assert document["notes"] == ["03-call-centre: comparison.json not found; not scored"]
    assert (document["scenarios_run"], document["valid"]) == (2, 1)
    assert document["recall"] == 5 / 17
    assert "Not scored: 03-call-centre" in summary.to_text()


def test_an_unreadable_comparison_is_reported(tmp_path: Path) -> None:
    runs = copy_runs(tmp_path)
    (runs / "01-pharmacy" / "comparison.json").write_text("{", encoding="utf-8")

    document = json.loads(summarize(runs, CORPUS).to_json())

    assert document["not_scored"] == ["01-pharmacy"]
    assert document["notes"] == [
        "01-pharmacy: comparison.json is not a valid comparison; not scored"
    ]


def test_a_comparison_whose_counts_disagree_is_reported(tmp_path: Path) -> None:
    runs = copy_runs(tmp_path)
    path = runs / "03-call-centre" / "comparison.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["truth_count"] = 9  # 5 matched + 3 unmatched truth constraints
    path.write_text(json.dumps(document), encoding="utf-8")

    summary = json.loads(summarize(runs, CORPUS).to_json())

    assert summary["not_scored"] == ["03-call-centre"]


def test_a_scenario_outside_the_corpus_is_reported(tmp_path: Path) -> None:
    runs = copy_runs(tmp_path)
    shutil.copytree(runs / "01-pharmacy", runs / "99-unknown")

    document = json.loads(summarize(runs, CORPUS).to_json())

    assert document["not_scored"] == ["99-unknown"]
    assert document["notes"] == ["99-unknown: not a scenario of the corpus; not scored"]
    assert document["scenarios_run"] == 3


def test_an_empty_directory_gives_null_scores(tmp_path: Path) -> None:
    runs = tmp_path / "claude-opus-5-5"
    runs.mkdir()

    document = json.loads(summarize(runs, CORPUS).to_json())

    assert (document["scenarios_run"], document["valid"]) == (0, 0)
    assert (document["recall"], document["merged_recall"], document["precision"]) == (
        None,
        None,
        None,
    )
    assert document["recall_by_language"] == {}
    assert document["cost_usd"] == 0.0
    assert document["missing"] == sorted([*FIXTURE_SCENARIOS, *NOT_RUN])
    assert (document["complete"], document["stops"]) == (False, {})


# --- completeness ----------------------------------------------------------------------------


def corpus_of(tmp_path: Path, names: list[str]) -> Path:
    """A corpus copy with only these scenarios (and a directory that is not a scenario)."""
    corpus = tmp_path / "corpus"
    for name in names:
        shutil.copytree(CORPUS / name, corpus / name)
    (corpus / "notes").mkdir()  # no scenario.json: not a corpus scenario
    return corpus


def test_a_complete_corpus_gives_complete_yes(tmp_path: Path) -> None:
    corpus = corpus_of(tmp_path, FIXTURE_SCENARIOS)

    summary = summarize(OPUS_RUNS, corpus)

    document = json.loads(summary.to_json())
    assert (document["missing"], document["complete"]) == ([], True)
    assert "Complete: yes" in summary.to_text().splitlines()


def test_a_run_that_is_not_scored_makes_the_summary_incomplete(tmp_path: Path) -> None:
    corpus = corpus_of(tmp_path, FIXTURE_SCENARIOS)
    runs = copy_runs(tmp_path)
    (runs / "04-restaurant" / "comparison.json").unlink()

    document = json.loads(summarize(runs, corpus).to_json())

    assert (document["missing"], document["complete"]) == (["04-restaurant"], False)


def test_a_run_outside_the_corpus_makes_the_summary_incomplete(tmp_path: Path) -> None:
    corpus = corpus_of(tmp_path, FIXTURE_SCENARIOS)
    runs = copy_runs(tmp_path)
    shutil.copytree(runs / "01-pharmacy", runs / "99-unknown")

    summary = summarize(runs, corpus)

    document = json.loads(summary.to_json())
    assert (document["missing"], document["complete"]) == ([], False)
    assert "Complete: no (not scored: 99-unknown)" in summary.to_text().splitlines()


def test_files_next_to_the_scenario_directories_are_ignored(tmp_path: Path) -> None:
    runs = copy_runs(tmp_path)
    (runs / "summary.json").write_text("{}", encoding="utf-8")

    assert summarize(runs, CORPUS).to_json() == summarize(OPUS_RUNS, CORPUS).to_json()


def test_a_missing_runs_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a directory"):
        summarize(tmp_path / "nope", CORPUS)


def test_a_missing_corpus_directory_is_the_same_error(tmp_path: Path) -> None:
    # Review finding: it raised FileNotFoundError from iterdir() instead.
    with pytest.raises(ValueError, match="not a directory"):
        summarize(OPUS_RUNS, tmp_path / "nope")


def test_the_model_is_the_name_of_the_resolved_runs_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Review finding: "." from inside the runs directory gave the model "" (no price, no stops).
    monkeypatch.chdir(OPUS_RUNS)

    assert summarize(Path("."), CORPUS).to_json() == summarize(OPUS_RUNS, CORPUS).to_json()


# --- usage and cost --------------------------------------------------------------------------


def one_run(tmp_path: Path, model: str, response: object | None) -> Path:
    """A runs directory with the 01 truth run and the given ``response.json`` (or none)."""
    runs = tmp_path / model
    run = runs / "01-pharmacy"
    shutil.copytree(OPUS_RUNS / "01-pharmacy", run)
    (run / "response.json").unlink()
    if response is not None:
        (run / "response.json").write_text(json.dumps(response), encoding="utf-8")
    return runs


def usage_and_notes(runs: Path) -> tuple[dict[str, int], list[str]]:
    document = json.loads(summarize(runs, CORPUS).to_json())
    return document["usage"], document["notes"]


def test_gemini_usage_counts_thoughts_as_output(tmp_path: Path) -> None:
    metadata = {"promptTokenCount": 900, "candidatesTokenCount": 70, "thoughtsTokenCount": 30}
    runs = one_run(tmp_path, "gemini-3.8-flash", {"usageMetadata": metadata})

    assert usage_and_notes(runs) == ({"input_tokens": 900, "output_tokens": 100}, [])


def test_gemini_usage_without_thoughts(tmp_path: Path) -> None:
    metadata = {"promptTokenCount": 900, "candidatesTokenCount": 70}
    runs = one_run(tmp_path, "gemini-3.8-flash", {"usageMetadata": metadata})

    assert usage_and_notes(runs) == ({"input_tokens": 900, "output_tokens": 70}, [])


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"usage": {"input_tokens": "many"}},
        {"usage": {"input_tokens": 5, "output_tokens": True}},
        {"usageMetadata": {"promptTokenCount": 5}},
        {
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 1,
                "thoughtsTokenCount": -1,
            }
        },
        [],
    ],
    ids=["no-usage", "not-a-number", "boolean", "gemini-no-output", "negative", "not-an-object"],
)
def test_unknown_usage_counts_zero_with_a_note(tmp_path: Path, response: object) -> None:
    runs = one_run(tmp_path, "claude-opus-5-5", response)

    assert usage_and_notes(runs) == (
        {"input_tokens": 0, "output_tokens": 0},
        ["01-pharmacy: usage unknown in response.json; usage counted as 0"],
    )


def test_a_missing_or_unreadable_response_counts_zero_with_a_note(tmp_path: Path) -> None:
    missing = one_run(tmp_path / "a", "claude-opus-5-5", None)
    broken = one_run(tmp_path / "b", "claude-opus-5-5", None)
    (broken / "01-pharmacy" / "response.json").write_text("[[", encoding="utf-8")

    assert usage_and_notes(missing) == (
        {"input_tokens": 0, "output_tokens": 0},
        ["01-pharmacy: response.json not found; usage counted as 0"],
    )
    assert usage_and_notes(broken) == (
        {"input_tokens": 0, "output_tokens": 0},
        ["01-pharmacy: response.json is not valid JSON; usage counted as 0"],
    )


# -- TASK-005e4 decision 5: an explicit nesting limit and unreadable files --


def nested(depth: int) -> str:
    return "[" * depth + "]" * depth


def raw_response(tmp_path: Path, data: bytes) -> Path:
    """The 01 truth run alone, with ``data`` as its ``response.json``."""
    runs = one_run(tmp_path, "claude-opus-5-5", None)
    (runs / "01-pharmacy" / "response.json").write_bytes(data)
    return runs


@pytest.mark.parametrize("depth", [65, 100_000])
def test_a_response_nested_too_deep_is_named_with_the_limit(tmp_path: Path, depth: int) -> None:
    # 100 000 levels: the base read it through RecursionError as "is not valid JSON".
    runs = raw_response(tmp_path, nested(depth).encode())

    assert usage_and_notes(runs) == (
        {"input_tokens": 0, "output_tokens": 0},
        ["01-pharmacy: response.json is nested more than 64 levels deep; usage counted as 0"],
    )


def test_a_response_nested_exactly_to_the_limit_is_read(tmp_path: Path) -> None:
    runs = raw_response(tmp_path, nested(64).encode())

    assert usage_and_notes(runs) == (
        {"input_tokens": 0, "output_tokens": 0},
        ["01-pharmacy: usage unknown in response.json; usage counted as 0"],
    )


def test_a_response_that_is_not_utf8_is_not_valid_json(tmp_path: Path) -> None:
    runs = raw_response(tmp_path, b'{"usage": "\xff"}')

    assert usage_and_notes(runs) == (
        {"input_tokens": 0, "output_tokens": 0},
        ["01-pharmacy: response.json is not valid JSON; usage counted as 0"],
    )


@pytest.mark.parametrize(("depth", "shown"), [(63, ["kept"]), (64, [])])
def test_a_candidate_nested_past_the_limit_shows_no_assumption(
    tmp_path: Path, depth: int, shown: list[str]
) -> None:
    # The object around "x" is one level more: 63 gives 64 levels in all, 64 gives 65.
    candidate = f'{{"assumptions": [{{"text": "kept"}}], "x": {nested(depth)}}}'
    runs = with_candidate(tmp_path, "03-call-centre", candidate)

    summary = summarize(runs, CORPUS)

    assert summary.rows[1].assumptions == tuple(shown)
    assert summary.notes == ()


@pytest.mark.parametrize(
    ("refused", "scored", "note"),
    [
        ("comparison.json", False, "03-call-centre: comparison.json cannot be read; not scored"),
        (
            "response.json",
            True,
            "03-call-centre: response.json cannot be read; usage counted as 0",
        ),
        (
            "candidate.txt",
            True,
            "03-call-centre: candidate.txt cannot be read; assumptions not shown",
        ),
    ],
)
def test_a_file_that_cannot_be_read_gives_one_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refused: str, scored: bool, note: str
) -> None:
    runs = tmp_path / "claude-opus-5-5"
    shutil.copytree(OPUS_RUNS / "03-call-centre", runs / "03-call-centre")
    real = Path.read_bytes

    def read_bytes(path: Path) -> bytes:
        if path.name == refused:  # as Windows answers while another program holds the file
            raise PermissionError(13, "Permission denied")
        return real(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    summary = summarize(runs, CORPUS)

    assert summary.notes == (note,)
    expected = (1, ()) if scored else (0, ("03-call-centre",))
    assert (summary.scenarios_run, summary.not_scored) == expected


def test_without_01s_candidate_the_only_note_is_its_own(tmp_path: Path) -> None:
    # TASK-005e3's traceability gap: 01 has no unsupported request, so no section line either.
    runs = copy_runs(tmp_path)
    (runs / "01-pharmacy" / "candidate.txt").unlink()

    summary = summarize(runs, CORPUS)

    assert summary.notes == ("01-pharmacy: candidate.txt not found; assumptions not shown",)
    assert summary.rows[0].assumptions == ()
    assert "  01-pharmacy" not in section_of(runs)


# --- thinking tokens (TASK-005e3 decision 1, ADR-0008 D3(a)) ---------------------------------


def thinking(runs: Path) -> tuple[int, list[str]]:
    document = json.loads(summarize(runs, CORPUS).to_json())
    return document["thinking_tokens"], document["thinking_unknown"]


def test_the_fixture_thinking_is_summed_and_the_unknown_run_named() -> None:
    # 01 and 03 report 6000 and 7000 in usage.output_tokens_details; 04 reports nothing.
    summary = summarize(OPUS_RUNS, CORPUS)

    assert thinking(OPUS_RUNS) == (13000, ["04-restaurant"])
    lines = summary.to_text().splitlines()
    usage = lines.index("Usage: 36000 input tokens, 18000 output tokens")
    assert lines[usage + 1] == "Thinking: 13000 of the output tokens (unknown for 04-restaurant)"


def test_the_thinking_line_has_no_unknown_part_when_every_run_reports_it(tmp_path: Path) -> None:
    usage = {
        "input_tokens": 10,
        "output_tokens": 100,
        "output_tokens_details": {"thinking_tokens": 40},
    }
    runs = one_run(tmp_path, "claude-opus-5-5", {"usage": usage})

    assert thinking(runs) == (40, [])
    assert "Thinking: 40 of the output tokens" in summarize(runs, CORPUS).to_text().splitlines()


def test_thinking_equal_to_the_output_is_a_count(tmp_path: Path) -> None:
    usage = {
        "input_tokens": 10,
        "output_tokens": 100,
        "output_tokens_details": {"thinking_tokens": 100},
    }
    runs = one_run(tmp_path, "claude-opus-5-5", {"usage": usage})

    assert thinking(runs) == (100, [])


@pytest.mark.parametrize(
    "details",
    [
        {},
        {"output_tokens_details": None},
        {"output_tokens_details": "many"},
        {"output_tokens_details": {}},
        {"output_tokens_details": {"thinking_tokens": None}},
        {"output_tokens_details": {"thinking_tokens": "6000"}},
        {"output_tokens_details": {"thinking_tokens": -1}},
        {"output_tokens_details": {"thinking_tokens": True}},
        {"output_tokens_details": {"thinking_tokens": 101}},
    ],
    ids=[
        "details-missing",
        "details-null",
        "details-string",
        "count-missing",
        "count-null",
        "count-string",
        "negative",
        "boolean",
        "larger-than-output",
    ],
)
def test_anthropic_thinking_that_is_not_a_count_is_unknown_and_named(
    tmp_path: Path, details: dict[str, object]
) -> None:
    usage = {"input_tokens": 10, "output_tokens": 100, **details}
    runs = one_run(tmp_path, "claude-opus-5-5", {"usage": usage})

    assert thinking(runs) == (0, ["01-pharmacy"])
    assert usage_and_notes(runs) == ({"input_tokens": 10, "output_tokens": 100}, [])
    assert (
        "Thinking: 0 of the output tokens (unknown for 01-pharmacy)"
        in summarize(runs, CORPUS).to_text().splitlines()
    )


def test_anthropic_thinking_is_read_from_the_details_only(tmp_path: Path) -> None:
    usage = {"input_tokens": 10, "output_tokens": 100, "thinking_tokens": 40}
    runs = one_run(tmp_path, "claude-opus-5-5", {"usage": usage})

    assert thinking(runs) == (0, ["01-pharmacy"])


def test_gemini_thinking_is_the_thoughts_count(tmp_path: Path) -> None:
    metadata = {"promptTokenCount": 900, "candidatesTokenCount": 70, "thoughtsTokenCount": 30}
    runs = one_run(tmp_path, "gemini-3.8-flash", {"usageMetadata": metadata})

    assert thinking(runs) == (30, [])


def test_gemini_without_thoughts_has_zero_thinking_and_is_not_named(tmp_path: Path) -> None:
    metadata = {"promptTokenCount": 900, "candidatesTokenCount": 70}
    runs = one_run(tmp_path, "gemini-3.8-flash", {"usageMetadata": metadata})

    assert thinking(runs) == (0, [])


def test_a_run_with_unknown_usage_is_not_named_again_for_thinking(tmp_path: Path) -> None:
    unknown = one_run(tmp_path / "a", "claude-opus-5-5", {"usage": {"input_tokens": "many"}})
    missing = one_run(tmp_path / "b", "claude-opus-5-5", None)

    assert thinking(unknown) == (0, [])
    assert thinking(missing) == (0, [])
    assert "Thinking: 0 of the output tokens" in summarize(unknown, CORPUS).to_text().splitlines()


def test_prices_are_the_batch_prices_per_million_tokens() -> None:
    # https://platform.claude.com/docs/en/about-claude/pricing «Batch processing», 2026-11-02.
    assert PRICES_USD_PER_MILLION == {
        "claude-opus-5-5": (2.0, 10.0),
        "claude-haiku-5-5": (0.05, 0.25),
        "gemini-3.8-flash": (0.0, 0.0),
    }
    assert LONG_PROMPT_PRICES_USD_PER_MILLION == {"claude-haiku-5-5": (100_000, (0.25, 1.25))}


def cost(runs: Path) -> float | None:
    return json.loads(summarize(runs, CORPUS).to_json())["cost_usd"]


def test_haiku_up_to_100000_input_tokens_uses_the_lower_tier(tmp_path: Path) -> None:
    runs = one_run(
        tmp_path, "claude-haiku-5-5", {"usage": {"input_tokens": 100_000, "output_tokens": 10_000}}
    )

    assert cost(runs) == 0.0075  # (100000 * 0.05 + 10000 * 0.25) / 1e6


def test_haiku_over_100000_input_tokens_uses_the_higher_tier(tmp_path: Path) -> None:
    runs = one_run(
        tmp_path, "claude-haiku-5-5", {"usage": {"input_tokens": 100_001, "output_tokens": 10_000}}
    )

    assert cost(runs) == 0.0375  # (100001 * 0.25 + 10000 * 1.25) / 1e6 = 0.03750025, rounded


def test_each_run_is_priced_on_its_own(tmp_path: Path) -> None:
    runs = one_run(
        tmp_path, "claude-haiku-5-5", {"usage": {"input_tokens": 1_000_000, "output_tokens": 0}}
    )
    shutil.copytree(OPUS_RUNS / "03-call-centre", runs / "03-call-centre")
    usage = {"usage": {"input_tokens": 50_000, "output_tokens": 0}}
    (runs / "03-call-centre" / "response.json").write_text(json.dumps(usage), encoding="utf-8")

    # 1,000,000 at the higher tier and 50,000 at the lower one; summed tokens would be one tier.
    assert cost(runs) == 0.2525  # (1000000 * 0.25 + 50000 * 0.05) / 1e6


def test_gemini_costs_nothing(tmp_path: Path) -> None:
    runs = one_run(
        tmp_path,
        "gemini-3.8-flash",
        {"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 10}},
    )

    assert cost(runs) == 0.0


def test_a_model_without_a_price_has_no_cost(tmp_path: Path) -> None:
    runs = one_run(tmp_path, "some-other-model", {"usage": {"input_tokens": 1, "output_tokens": 1}})

    summary = summarize(runs, CORPUS)

    assert json.loads(summary.to_json())["cost_usd"] is None
    assert "cost unknown" in summary.to_text()


# --- stops -----------------------------------------------------------------------------------


def test_a_gemini_run_counts_its_finish_reason(tmp_path: Path) -> None:
    candidate = {"content": {"parts": [{"text": "{}"}]}, "finishReason": "MAX_TOKENS"}
    runs = one_run(tmp_path, "gemini-3.8-flash", {"candidates": [candidate]})

    summary = summarize(runs, CORPUS)

    assert json.loads(summary.to_json())["stops"] == {"MAX_TOKENS": 1}
    assert "Stops: MAX_TOKENS 1" in summary.to_text().splitlines()


def test_the_stops_line_shows_a_server_stop_string_safely(tmp_path: Path) -> None:
    # TASK-005c3h H8 (the review's #9): an escape and a line break of a server's stop string
    # reached the terminal; the JSON keeps the stop exactly.
    stop = "MAX\x1b[2J\nTOKENS"
    candidate = {"content": {"parts": [{"text": "{}"}]}, "finishReason": stop}
    runs = one_run(tmp_path, "gemini-3.8-flash", {"candidates": [candidate]})

    summary = summarize(runs, CORPUS)

    assert json.loads(summary.to_json())["stops"] == {stop: 1}
    text = summary.to_text()
    assert "Stops: MAX?[2J?TOKENS 1" in text.splitlines()
    assert "\x1b" not in text


def test_a_run_without_a_readable_response_has_no_stop(tmp_path: Path) -> None:
    runs = copy_runs(tmp_path)
    (runs / "01-pharmacy" / "response.json").unlink()
    (runs / "03-call-centre" / "response.json").write_bytes(b"[[")

    document = json.loads(summarize(runs, CORPUS).to_json())

    assert document["stops"] == {"refusal": 1}


def test_an_unregistered_model_has_a_note_and_no_stops(tmp_path: Path) -> None:
    runs = one_run(tmp_path, "some-other-model", {"usage": {"input_tokens": 1, "output_tokens": 1}})

    summary = summarize(runs, CORPUS)

    document = json.loads(summary.to_json())
    assert document["stops"] == {}
    assert document["notes"] == ["some-other-model: not a registered model; stops not counted"]
    assert "Stops: none" in summary.to_text().splitlines()


@pytest.mark.parametrize(
    ("model", "directory"),
    [
        ("claude-opus-5-5", "claude-opus-5-5"),
        ("gemini-3.8-flash", "gemini-3.8-flash"),
        ("models/gemini 3:pro", "models-gemini-3-pro"),
        ("Ünï_cødé.v1", "-n-_c-d-.v1"),
    ],
)
def test_model_directory_name_replaces_every_other_character(model: str, directory: str) -> None:
    assert model_directory_name(model) == directory


# --- one row per scored scenario (TASK-005e3 decision 2, ADR-0008 D3(a)) --------------------

ROW_01 = (
    "  01-pharmacy: el easy, truth 5, valid, stop end_turn, output 8000 (thinking 6000), $0.1040,"
    " matched 5, recall 1.000, precision 1.000"
)
ROW_03 = (
    "  03-call-centre: en medium, truth 8, valid, stop end_turn, output 9500 (thinking 7000),"
    " $0.1180, matched 5, recall 0.625, precision 0.625"
)
ROW_04 = (
    "  04-restaurant: en hard, truth 12, invalid (stop reason), stop refusal, output 500"
    " (thinking n/a), $0.0300, matched 0, recall 0.000, precision n/a"
)
# The corpus's unsupported quotes of 03 and 04 (01 has none).
QUOTE_03 = "We also need at least one Spanish speaker on every weekday shift."
QUOTE_04 = "every evening we need at least one personal licence holder on the floor for the bar"


def rows_of(runs: Path) -> list[dict[str, object]]:
    return json.loads(summarize(runs, CORPUS).to_json())["scenarios"]


def row_lines(runs: Path) -> list[str]:
    """The lines of the ``Per scenario:`` block (the indented lines right after it)."""
    lines = summarize(runs, CORPUS).to_text().splitlines()
    start = lines.index("Per scenario:") + 1
    end = start
    while end < len(lines) and lines[end].startswith("  ") and ":" in lines[end]:
        end += 1
    return lines[start:end]


def test_issue_classes_are_the_words_of_adr_0008_d3a() -> None:
    assert ISSUE_CLASSES == {
        "json": "JSON or fence",
        "schema": "schema",
        "rule": "cross-field rule",
        "data_file": "data file",
        "data": "data-level id or attribute",
        None: "stop reason",
    }


def test_a_scenario_row_has_exactly_these_fields_in_this_order() -> None:
    names = [field.name for field in dataclasses.fields(ScenarioRow)]

    assert names == [
        "scenario",
        "language",
        "difficulty",
        "truth_count",
        "valid",
        "issue_class",
        "stop",
        "output_tokens",
        "thinking_tokens",
        "cost_usd",
        "matched_count",
        "candidate_count",
        "unsupported",
        "assumptions",
    ]
    row = summarize(OPUS_RUNS, CORPUS).rows[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.scenario = "other"  # type: ignore[misc]
    assert not hasattr(row, "__dict__")  # slotted


def test_the_fixture_rows_follow_the_per_scenario_line() -> None:
    lines = summarize(OPUS_RUNS, CORPUS).to_text().splitlines()

    cost = lines.index("Cost: $0.2520 (batch prices pinned on 2026-11-02)")
    assert lines[cost + 1 : cost + 5] == ["Per scenario:", ROW_01, ROW_03, ROW_04]


def test_the_fixture_rows_in_json() -> None:
    rows = rows_of(OPUS_RUNS)

    assert [row["scenario"] for row in rows] == FIXTURE_SCENARIOS
    candidate_01 = json.loads((OPUS_RUNS / "01-pharmacy" / "candidate.txt").read_bytes())
    assumptions_01 = [item["text"] for item in candidate_01["assumptions"]]
    assert len(assumptions_01) == 2
    assert rows[0] == {
        "scenario": "01-pharmacy",
        "language": "el",
        "difficulty": "easy",
        "truth_count": 5,
        "valid": True,
        "issue_class": None,
        "stop": "end_turn",
        "output_tokens": 8000,
        "thinking_tokens": 6000,
        "cost_usd": 0.104,  # (12000 * 2 + 8000 * 10) / 1e6
        "matched_count": 5,
        "candidate_count": 5,
        "recall": 1.0,
        "precision": 1.0,
        "unsupported": [],
        "assumptions": assumptions_01,
    }
    assert (rows[1]["cost_usd"], rows[1]["recall"], rows[1]["precision"]) == (0.118, 5 / 8, 5 / 8)
    assert (rows[1]["unsupported"], rows[1]["assumptions"]) == (
        [QUOTE_03],
        ["The Spanish speaker request cannot be expressed."],
    )
    assert (rows[2]["unsupported"], rows[2]["assumptions"]) == ([QUOTE_04], [])
    assert {key: rows[2][key] for key in ("valid", "issue_class", "stop", "thinking_tokens")} == {
        "valid": False,
        "issue_class": "stop reason",
        "stop": "refusal",
        "thinking_tokens": None,
    }
    assert (rows[2]["cost_usd"], rows[2]["recall"], rows[2]["precision"]) == (0.03, 0.0, None)


@pytest.mark.parametrize("stage", ["json", "schema", "rule", "data_file", "data", None])
def test_every_issue_class_reaches_the_row(tmp_path: Path, stage: InvalidStage | None) -> None:
    runs = copy_runs(tmp_path)
    path = runs / "04-restaurant" / "comparison.json"
    document = json.loads(path.read_bytes())
    document["invalid_stage"] = stage
    path.write_text(json.dumps(document), encoding="utf-8")

    issue_class = ISSUE_CLASSES[stage]
    assert rows_of(runs)[2]["issue_class"] == issue_class
    assert row_lines(runs)[2].startswith(
        f"  04-restaurant: en hard, truth 12, invalid ({issue_class}), stop refusal,"
    )


@pytest.mark.parametrize(
    ("name", "change"),
    [("01-pharmacy", {"invalid_stage": "schema"}), ("04-restaurant", None)],
    ids=["valid-with-a-stage", "no-stage-key"],
)
def test_a_comparison_with_a_wrong_or_missing_stage_is_not_scored(
    tmp_path: Path, name: str, change: dict[str, object] | None
) -> None:
    runs = copy_runs(tmp_path)
    path = runs / name / "comparison.json"
    document = json.loads(path.read_bytes())
    if change is None:
        del document["invalid_stage"]
    else:
        document.update(change)
    path.write_text(json.dumps(document), encoding="utf-8")

    summary = json.loads(summarize(runs, CORPUS).to_json())

    assert summary["not_scored"] == [name]
    assert summary["notes"] == [f"{name}: comparison.json is not a valid comparison; not scored"]
    assert name not in [row["scenario"] for row in summary["scenarios"]]


def test_a_run_that_is_not_scored_has_no_row(tmp_path: Path) -> None:
    runs = copy_runs(tmp_path)
    (runs / "03-call-centre" / "comparison.json").unlink()

    assert [row["scenario"] for row in rows_of(runs)] == ["01-pharmacy", "04-restaurant"]
    assert row_lines(runs) == [ROW_01, ROW_04]


def test_an_unregistered_model_without_a_price_has_stop_and_cost_n_a(tmp_path: Path) -> None:
    usage = {
        "input_tokens": 10,
        "output_tokens": 100,
        "output_tokens_details": {"thinking_tokens": 40},
    }
    runs = one_run(tmp_path, "some-other-model", {"usage": usage})

    row = rows_of(runs)[0]
    assert (row["stop"], row["output_tokens"], row["thinking_tokens"], row["cost_usd"]) == (
        None,
        100,
        40,
        None,
    )
    assert row_lines(runs) == [
        "  01-pharmacy: el easy, truth 5, valid, stop n/a, output 100 (thinking 40), cost n/a,"
        " matched 5, recall 1.000, precision 1.000"
    ]


def test_unknown_usage_gives_output_and_cost_n_a(tmp_path: Path) -> None:
    runs = one_run(tmp_path, "claude-opus-5-5", {"stop_reason": "end_turn", "content": []})

    row = rows_of(runs)[0]
    assert (row["stop"], row["output_tokens"], row["thinking_tokens"], row["cost_usd"]) == (
        "end_turn",
        None,
        None,
        None,
    )
    assert "stop end_turn, output n/a (thinking n/a), cost n/a," in row_lines(runs)[0]


def test_a_missing_response_gives_stop_output_and_cost_n_a(tmp_path: Path) -> None:
    runs = one_run(tmp_path, "claude-opus-5-5", None)

    assert ", stop n/a, output n/a (thinking n/a), cost n/a," in row_lines(runs)[0]


def test_a_row_cost_is_rounded_on_its_own(tmp_path: Path) -> None:
    runs = one_run(
        tmp_path, "claude-haiku-5-5", {"usage": {"input_tokens": 100_001, "output_tokens": 10_000}}
    )

    assert rows_of(runs)[0]["cost_usd"] == 0.0375  # 0.03750025 rounded to micro-dollars
    assert ", $0.0375, " in row_lines(runs)[0]


def test_the_row_keeps_a_server_stop_raw_in_json_and_safe_in_text(tmp_path: Path) -> None:
    stop = "MAX\x1b[2J\nTOKENS"
    candidate = {"content": {"parts": [{"text": "{}"}]}, "finishReason": stop}
    runs = one_run(tmp_path, "gemini-3.8-flash", {"candidates": [candidate]})

    assert rows_of(runs)[0]["stop"] == stop
    text = summarize(runs, CORPUS).to_text()
    assert ", stop MAX?[2J?TOKENS, " in row_lines(runs)[0]
    assert "\x1b" not in text


# --- recall over valid candidates and its bootstrap interval (TASK-005e3 decision 3) --------
# The pinned values come from the reviewer's scratch build (identical on CPython 3.12.16 and
# 3.14.10). Each is a ratio of two integers, so the floats compare exactly with ==.

COUNTS = [(5, 5), (5, 8), (0, 12), (3, 4), (7, 9), (2, 6), (4, 4), (1, 7), (6, 10), (3, 3)]
PRIMES = [
    (2, 3),
    (4, 7),
    (6, 11),
    (10, 13),
    (1, 17),
    (15, 19),
    (20, 23),
    (3, 29),
    (27, 31),
    (30, 37),
]
GATE_NOTE = (
    "Gate note: the interval contains 0.70; the point value decides, and it lies within the"
    " uncertainty of a {n}-scenario corpus"
)


def test_the_bootstrap_constants() -> None:
    assert (GATE_RECALL, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED) == (0.70, 10_000, 20261109)


def test_the_interval_of_counts_is_pinned() -> None:
    assert bootstrap_interval(COUNTS) == (23 / 76, 41 / 53)
    assert bootstrap_interval(COUNTS, seed=20261110) == (25 / 81, 47 / 61)
    assert bootstrap_interval(COUNTS, resamples=100) == (1 / 3, 40 / 53)
    assert bootstrap_interval(COUNTS, resamples=40) == (7 / 27, 19 / 27)


def test_the_interval_is_the_nearest_rank_percentile() -> None:
    # The positions 250 and 9 750 (off by one) would give (24 / 67, 30 / 37).
    assert bootstrap_interval(PRIMES) == (53 / 148, 167 / 206)


def test_the_interval_of_small_inputs() -> None:
    assert bootstrap_interval([(5, 5), (5, 8), (0, 12)]) == (0.0, 1.0)
    assert bootstrap_interval([(3, 4)]) == (0.75, 0.75)


@pytest.mark.parametrize("counts", [[], [(0, 0), (1, 2)]], ids=["empty", "a-zero-truth"])
def test_no_interval_without_scenarios_or_with_a_zero_truth(counts: list[tuple[int, int]]) -> None:
    assert bootstrap_interval(counts) is None


@st.composite
def scenario_counts(draw: st.DrawFn) -> list[tuple[int, int]]:
    truths = draw(st.lists(st.integers(1, 20), min_size=1, max_size=12))
    return [(draw(st.integers(0, truth)), truth) for truth in truths]


@settings(deadline=None, max_examples=200)
@given(scenario_counts())
def test_the_interval_lies_between_the_smallest_and_the_largest_ratio(
    counts: list[tuple[int, int]],
) -> None:
    # A ratio of sums lies between the smallest and the largest ratio, and correctly rounded
    # division keeps that order.
    interval = bootstrap_interval(counts, resamples=200)

    assert interval is not None
    ratios = [matched / truth for matched, truth in counts]
    assert min(ratios) <= interval[0] <= interval[1] <= max(ratios)


@settings(deadline=None, max_examples=200)
@given(
    st.integers(1, 20).flatmap(lambda b: st.tuples(st.integers(0, b), st.just(b))),
    st.lists(st.integers(1, 9), min_size=1, max_size=12),
)
def test_equal_ratios_give_exactly_that_ratio(
    ratio: tuple[int, int], multipliers: list[int]
) -> None:
    a, b = ratio
    counts = [(k * a, k * b) for k in multipliers]

    assert bootstrap_interval(counts, resamples=200) == (a / b, a / b)


def test_the_bootstrap_leaves_the_module_generator_untouched() -> None:
    random.seed(7)
    expected = random.random()
    random.seed(7)

    bootstrap_interval(COUNTS)

    assert random.random() == expected


def test_the_fixture_interval_and_its_gate_note() -> None:
    summary = summarize(OPUS_RUNS, CORPUS)

    assert summary.recall_interval == (0.0, 1.0)
    lines = summary.to_text().splitlines()
    recall = lines.index("Recall: 0.400 (10/25)")
    assert lines[recall + 1 : recall + 4] == [
        "Recall 95% interval: 0.000 to 1.000 (bootstrap over 3 scenarios, 10000 resamples,"
        " seed 20261109)",
        GATE_NOTE.format(n=3),
        "Recall over valid candidates (not the gate): 0.769 (10/13)",
    ]
    document = json.loads(summary.to_json())
    assert document["recall_interval"] == {
        "lower": 0.0,
        "upper": 1.0,
        "resamples": 10000,
        "seed": 20261109,
        "contains_gate": True,
    }
    assert document["recall_valid_only"] == {"matched": 10, "truth": 13, "recall": 10 / 13}
    assert summary.valid_counts == (10, 13)


def test_an_interval_without_the_gate_has_no_note(tmp_path: Path) -> None:
    runs = one_run(tmp_path, "claude-opus-5-5", {"usage": {"input_tokens": 1, "output_tokens": 1}})

    summary = summarize(runs, CORPUS)

    assert summary.recall_interval == (1.0, 1.0)
    text = summary.to_text()
    assert (
        "Recall 95% interval: 1.000 to 1.000 (bootstrap over 1 scenarios, 10000 resamples,"
        " seed 20261109)"
    ) in text.splitlines()
    assert "Gate note" not in text
    assert json.loads(summary.to_json())["recall_interval"]["contains_gate"] is False


def test_an_empty_runs_directory_has_no_interval(tmp_path: Path) -> None:
    runs = tmp_path / "claude-opus-5-5"
    runs.mkdir()

    summary = summarize(runs, CORPUS)

    lines = summary.to_text().splitlines()
    assert "Recall 95% interval: n/a" in lines
    assert "Recall over valid candidates (not the gate): n/a (0/0)" in lines
    assert not any(line.startswith("Gate note") for line in lines)
    document = json.loads(summary.to_json())
    assert document["recall_interval"] is None
    assert document["recall_valid_only"] == {"matched": 0, "truth": 0, "recall": None}


@pytest.mark.parametrize(
    ("interval", "contains"),
    [
        ((0.70, 0.70), True),
        ((0.5, 0.70), True),
        ((0.70000001, 0.9), False),
        ((0.5, 0.69999999), False),
    ],
)
def test_the_gate_note_boundary(interval: tuple[float, float], contains: bool) -> None:
    summary = dataclasses.replace(summarize(OPUS_RUNS, CORPUS), recall_interval=interval)

    assert (GATE_NOTE.format(n=3) in summary.to_text().splitlines()) is contains
    assert summary.to_dict()["recall_interval"]["contains_gate"] is contains


def test_without_a_valid_candidate_the_valid_only_recall_is_n_a(tmp_path: Path) -> None:
    runs = tmp_path / "claude-opus-5-5"
    shutil.copytree(OPUS_RUNS / "04-restaurant", runs / "04-restaurant")

    summary = summarize(runs, CORPUS)

    assert summary.valid_counts == (0, 0)
    assert (
        "Recall over valid candidates (not the gate): n/a (0/0)" in summary.to_text().splitlines()
    )
    assert summary.recall_interval == (0.0, 0.0)


def test_the_interval_resamples_the_rows_in_row_order() -> None:
    summary = summarize(OPUS_RUNS, CORPUS)

    counts = [(row.matched_count, row.truth_count) for row in summary.rows]
    assert counts == [(5, 5), (5, 8), (0, 12)]
    assert summary.recall_interval == bootstrap_interval(counts)


# --- assumptions next to the unsupported requests (TASK-005e3 decision 4, ADR-0008 D12) ------

SECTION = "Unsupported requests and assumptions:"
SECTION_03 = [
    "  03-call-centre",
    f"    unsupported: {QUOTE_03}",
    "    assumption: The Spanish speaker request cannot be expressed.",
]
SECTION_04 = ["  04-restaurant", f"    unsupported: {QUOTE_04}", "    assumption: none"]


def section_of(runs: Path) -> list[str]:
    """The section's lines: its heading and the indented lines right after it."""
    lines = summarize(runs, CORPUS).to_text().splitlines()
    start = lines.index(SECTION)
    end = start + 1
    while end < len(lines) and lines[end].startswith("  "):
        end += 1
    return lines[start:end]


def with_candidate(tmp_path: Path, name: str, candidate: str | bytes) -> Path:
    """The fixture runs with ``name``'s ``candidate.txt`` replaced (its comparison unchanged)."""
    runs = copy_runs(tmp_path)
    path = runs / name / "candidate.txt"
    if isinstance(candidate, bytes):
        path.write_bytes(candidate)
    else:
        path.write_text(candidate, encoding="utf-8")
    return runs


def assumptions(text: str) -> str:
    return json.dumps({"assumptions": [{"id": "a", "text": text, "status": "proposed"}]})


def test_the_assumption_text_limit() -> None:
    assert ASSUMPTION_TEXT_LIMIT == 1000


def test_the_fixture_section_follows_the_rows() -> None:
    lines = summarize(OPUS_RUNS, CORPUS).to_text().splitlines()

    start = lines.index(SECTION)
    assert lines[start - 3 : start] == [ROW_01, ROW_03, ROW_04]
    assert lines[start:] == [SECTION, *SECTION_03, *SECTION_04]


def test_a_scenario_without_unsupported_quotes_is_not_in_the_section() -> None:
    # 01 has assumptions but no unsupported request.
    assert "  01-pharmacy" not in section_of(OPUS_RUNS)
    assert summarize(OPUS_RUNS, CORPUS).rows[0].assumptions != ()


def test_without_a_row_with_unsupported_quotes_there_is_no_section(tmp_path: Path) -> None:
    runs = copy_runs(tmp_path)
    shutil.rmtree(runs / "03-call-centre")
    shutil.rmtree(runs / "04-restaurant")

    text = summarize(runs, CORPUS).to_text()

    assert "Unsupported requests" not in text
    assert "assumption:" not in text


def test_assumptions_are_read_whatever_the_validity(tmp_path: Path) -> None:
    runs = with_candidate(tmp_path, "04-restaurant", assumptions("Licence holders are known."))

    assert summarize(runs, CORPUS).rows[2].valid is False
    assert section_of(runs)[-1] == "    assumption: Licence holders are known."


def test_every_assumption_is_shown_in_order(tmp_path: Path) -> None:
    candidate = {"assumptions": [{"text": "First."}, {"text": "Second."}, {"text": "Third."}]}
    runs = with_candidate(tmp_path, "03-call-centre", json.dumps(candidate))

    assert section_of(runs)[3:6] == [
        "    assumption: First.",
        "    assumption: Second.",
        "    assumption: Third.",
    ]
    assert rows_of(runs)[1]["assumptions"] == ["First.", "Second.", "Third."]


def test_an_assumption_with_a_newline_and_a_control_character_is_printable(
    tmp_path: Path,
) -> None:
    runs = with_candidate(tmp_path, "03-call-centre", assumptions("one\ntwo\x1b[2Jthree"))

    assert section_of(runs)[3] == "    assumption: one?two?[2Jthree"
    assert rows_of(runs)[1]["assumptions"] == ["one?two?[2Jthree"]
    assert "\x1b" not in summarize(runs, CORPUS).to_text()


@pytest.mark.parametrize(("length", "shown"), [(1000, 1000), (1001, 1000)])
def test_an_assumption_is_cut_to_the_limit(tmp_path: Path, length: int, shown: int) -> None:
    runs = with_candidate(tmp_path, "03-call-centre", assumptions("x" * length))

    assert rows_of(runs)[1]["assumptions"] == ["x" * shown]
    assert section_of(runs)[3] == "    assumption: " + "x" * shown


def test_a_lone_surrogate_escape_is_shown_as_a_question_mark(tmp_path: Path) -> None:
    # The JSON escape \ud800 decodes to a lone surrogate, which UTF-8 cannot encode.
    candidate = '{"assumptions": [{"text": "before\\ud800after"}]}'
    runs = with_candidate(tmp_path, "03-call-centre", candidate)
    summary = summarize(runs, CORPUS)

    assert section_of(runs)[3] == "    assumption: before?after"
    summary.to_json().encode("utf-8")
    summary.to_text().encode("utf-8")


@pytest.mark.parametrize(
    "candidate",
    [
        "The Spanish speaker request cannot be expressed.",
        b"\xff\xfe not UTF-8",
        "[" * 100_000 + "]" * 100_000,
        json.dumps([{"text": "a list, not an object"}]),
        json.dumps("a string, not an object"),
        json.dumps({"constraints": []}),
        json.dumps({"assumptions": {"text": "an object, not a list"}}),
        json.dumps({"assumptions": "a string, not a list"}),
        json.dumps(
            {"assumptions": [{"text": 5}, {"text": None}, {"id": "a"}, "text", None, ["text"]]}
        ),
    ],
    ids=[
        "not-json",
        "not-utf-8",
        "nested-deeply",
        "a-json-list",
        "a-json-string",
        "no-assumptions",
        "assumptions-an-object",
        "assumptions-a-string",
        "items-without-a-string-text",
    ],
)
def test_a_candidate_without_usable_assumptions_gives_none_and_no_note(
    tmp_path: Path, candidate: str | bytes
) -> None:
    runs = with_candidate(tmp_path, "03-call-centre", candidate)
    summary = json.loads(summarize(runs, CORPUS).to_json())

    assert summary["scenarios"][1]["assumptions"] == []
    assert summary["notes"] == []
    assert section_of(runs)[1:4] == [
        "  03-call-centre",
        f"    unsupported: {QUOTE_03}",
        "    assumption: none",
    ]


def test_only_the_items_with_a_string_text_count(tmp_path: Path) -> None:
    items = [{"text": 5}, {"text": "kept"}, "text", {"id": "b"}, {"text": "also kept"}]
    runs = with_candidate(tmp_path, "03-call-centre", json.dumps({"assumptions": items}))

    assert rows_of(runs)[1]["assumptions"] == ["kept", "also kept"]


def test_a_missing_candidate_gives_no_assumption_and_a_note(tmp_path: Path) -> None:
    runs = copy_runs(tmp_path)
    (runs / "03-call-centre" / "candidate.txt").unlink()

    summary = json.loads(summarize(runs, CORPUS).to_json())

    assert summary["scenarios"][1]["assumptions"] == []
    assert summary["notes"] == ["03-call-centre: candidate.txt not found; assumptions not shown"]
    assert summary["not_scored"] == []
    assert section_of(runs)[3] == "    assumption: none"
    assert (
        "Note: 03-call-centre: candidate.txt not found; assumptions not shown"
        in summarize(runs, CORPUS).to_text().splitlines()
    )


# --- ADR-0008 D9: the recall after look-alike ids ------------------------------------------------

OPUS = "claude-opus-5-5"
SUPERMARKET = "05-supermarket-checkouts"
LOOKALIKE_LINE = (
    "Recall after look-alike ids (not the gate): 0.295 (13/44); 1 of 4 invalid candidates made "
    "valid (05-supermarket-checkouts)"
)
# The written ids quote a Greek capital verbatim, which ruff reads as Latin (RUF001).
CALL_CENTRE_IDS = "/constraints/8/params/resource_ids"


def lookalike_runs(tmp_path: Path) -> Path:
    """Five recorded runs (``write_answer``): 01's truth; 05 with ``T2`` (Latin T) where its
    truth has Greek Tau; 04 with ``Z1`` (no look-alike); 06 answering ``I cannot.``; 03 with
    Greek Alpha and ``03`` (a look-alike of its Latin ``A03``) and ``Z9``."""
    runs = tmp_path / OPUS
    answers = {
        "01-pharmacy": truth_text("01-pharmacy"),
        "03-call-centre": written(
            "03-call-centre",
            (f"{CALL_CENTRE_IDS}/0", "Α03"),  # noqa: RUF001 Greek Alpha
            (f"{CALL_CENTRE_IDS}/1", "Z9"),
        ),
        "04-restaurant": written("04-restaurant", ("/constraints/6/params/resource_ids/0", "Z1")),
        SUPERMARKET: written(SUPERMARKET, ("/constraints/6/params/resource_ids/1", "T2")),
        "06-ambulance-crews": "I cannot.",
    }
    for name, text in answers.items():
        scenario = load_scenario(CORPUS / name)
        write_answer(runs / name, scenario, OPUS, anthropic_message(OPUS, text))
    return runs


def test_a_lookalike_id_makes_one_data_level_candidate_valid(tmp_path: Path) -> None:
    runs = lookalike_runs(tmp_path)

    summary = summarize(runs, CORPUS)

    assert summary.lookalike_valid == ((SUPERMARKET, 8),)
    assert summary.lookalike_counts == (13, 44)
    assert (summary.matched_count, summary.truth_count) == (5, 44)  # the gate is unchanged
    lines = summary.to_text().splitlines()
    valid_only = next(i for i, line in enumerate(lines) if line.startswith("Recall over valid"))
    assert lines[valid_only + 1] == LOOKALIKE_LINE
    assert json.loads(summary.to_json())["recall_lookalike"] == {
        "invalid": 4,
        "made_valid": [{"matched_count": 8, "scenario": SUPERMARKET}],
        "matched": 13,
        "recall": 13 / 44,
        "truth": 44,
    }
    assert summary.notes == ()


def test_a_scenario_that_cannot_be_loaded_gives_a_note_and_no_replacement(
    tmp_path: Path,
) -> None:
    runs = lookalike_runs(tmp_path)
    corpus = corpus_of(tmp_path, [path.name for path in runs.iterdir()])
    (corpus / SUPERMARKET / "truth.json").write_text("{}", encoding="utf-8")

    summary = summarize(runs, corpus)

    assert summary.lookalike_valid == ()
    assert summary.notes == (
        f"{SUPERMARKET}: the scenario cannot be loaded; look-alike ids not replaced",
    )
    assert (
        "Recall after look-alike ids (not the gate): 0.114 (5/44); 0 of 4 invalid candidates made "
        "valid"
    ) in summary.to_text().splitlines()


def test_a_missing_candidate_is_not_replaced_and_has_only_the_assumptions_note(
    tmp_path: Path,
) -> None:
    runs = lookalike_runs(tmp_path)
    (runs / SUPERMARKET / "candidate.txt").unlink()

    summary = summarize(runs, CORPUS)

    assert summary.lookalike_valid == ()
    assert summary.notes == (f"{SUPERMARKET}: candidate.txt not found; assumptions not shown",)


def test_a_candidate_that_is_not_utf8_is_not_replaced(tmp_path: Path) -> None:
    runs = lookalike_runs(tmp_path)
    (runs / SUPERMARKET / "candidate.txt").write_bytes(b"\xff")

    summary = summarize(runs, CORPUS)

    assert summary.lookalike_valid == ()
    assert summary.notes == ()
