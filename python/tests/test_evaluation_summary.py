"""The summary of recorded Test A runs (ADR-0007 §7 «Records» and its amendment; TASK-005c1 D4,
TASK-005c2 E4).

The fixtures in ``fixtures/test_a_runs/claude-opus-5-5`` are real answer shapes for three
scenarios: 01 with the truth in a ```` ```json ```` fence (``end_turn``), 03 with the imperfect
candidate of the compare tests as bare text (``end_turn``), and 04 with a refusal. Their
``candidate.txt`` and ``comparison.json`` are what ``rescore_run`` derives from ``response.json``,
so they cannot drift from the rules or from ``compare.py``.
"""

import json
import shutil
from pathlib import Path

import pytest

from potakt.evaluation import (
    LONG_PROMPT_PRICES_USD_PER_MILLION,
    PRICES_USD_PER_MILLION,
    load_scenario,
    model_directory_name,
    rescore_run,
    summarize,
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
