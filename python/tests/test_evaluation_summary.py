"""The summary of recorded Test A runs (ADR-0007 §7 «Records»; TASK-005c1 D4).

The fixtures in ``fixtures/test_a_runs/claude-opus-5-5`` are hand-made runs for three scenarios:
01 with the truth as candidate, 03 with the imperfect candidate of the compare tests, and 04 with
an answer that is not JSON. Their ``comparison.json`` files are checked against the metric, so
they cannot drift from ``compare.py``.
"""

import json
import shutil
from pathlib import Path

import pytest

from potakt.evaluation import (
    PRICES_USD_PER_MILLION,
    evaluate_candidate,
    load_scenario,
    model_directory_name,
    summarize,
)

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "docs" / "validation" / "scenarios"
RUNS = Path(__file__).resolve().parent / "fixtures" / "test_a_runs"
OPUS_RUNS = RUNS / "claude-opus-5-5"
GOLDEN = Path(__file__).resolve().parent / "golden"
FIXTURE_SCENARIOS = ["01-pharmacy", "03-call-centre", "04-restaurant"]


@pytest.mark.parametrize("name", FIXTURE_SCENARIOS)
def test_fixture_comparisons_are_what_the_metric_gives_for_their_candidates(name: str) -> None:
    run = OPUS_RUNS / name
    candidate = (run / "candidate.txt").read_text(encoding="utf-8")

    result = evaluate_candidate(load_scenario(CORPUS / name), candidate)

    assert result.to_json().encode("utf-8") == (run / "comparison.json").read_bytes()


def test_summary_json_matches_the_golden_byte_for_byte() -> None:
    summary = summarize(OPUS_RUNS, CORPUS)

    assert summary.to_json().encode("utf-8") == (GOLDEN / "test-a-summary.json").read_bytes()


def test_summary_text_matches_the_golden_byte_for_byte() -> None:
    summary = summarize(OPUS_RUNS, CORPUS)

    assert summary.to_text().encode("utf-8") == (GOLDEN / "test-a-summary.txt").read_bytes()


def test_summary_numbers_match_the_hand_computation() -> None:
    # 01: 5 of 5 (el). 03: 5 of 8 (en). 04: invalid, 0 of 12 (en). Micro-averaged over 25.
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
    }
    assert document["usage"] == {"input_tokens": 36000, "output_tokens": 18000}
    assert document["cost_usd"] == 0.504  # (36000 * 4 + 18000 * 20) / 1e6
    assert (document["not_scored"], document["notes"]) == ([], [])


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


def test_files_next_to_the_scenario_directories_are_ignored(tmp_path: Path) -> None:
    runs = copy_runs(tmp_path)
    (runs / "summary.json").write_text("{}", encoding="utf-8")

    assert summarize(runs, CORPUS).to_json() == summarize(OPUS_RUNS, CORPUS).to_json()


def test_a_missing_runs_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a directory"):
        summarize(tmp_path / "nope", CORPUS)


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


def test_prices_are_pinned_per_million_tokens() -> None:
    assert PRICES_USD_PER_MILLION == {
        "claude-opus-5-5": (4.0, 20.0),
        "claude-haiku-5-5": (0.1, 0.5),
        "gemini-3.8-flash": (0.0, 0.0),
    }


def test_haiku_and_gemini_costs(tmp_path: Path) -> None:
    haiku = one_run(
        tmp_path / "a",
        "claude-haiku-5-5",
        {"usage": {"input_tokens": 1_000_000, "output_tokens": 2_000_000}},
    )
    gemini = one_run(
        tmp_path / "b",
        "gemini-3.8-flash",
        {"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 10}},
    )

    assert json.loads(summarize(haiku, CORPUS).to_json())["cost_usd"] == 1.1
    assert json.loads(summarize(gemini, CORPUS).to_json())["cost_usd"] == 0.0


def test_a_model_without_a_price_has_no_cost(tmp_path: Path) -> None:
    runs = one_run(tmp_path, "some-other-model", {"usage": {"input_tokens": 1, "output_tokens": 1}})

    summary = summarize(runs, CORPUS)

    assert json.loads(summary.to_json())["cost_usd"] is None
    assert "cost unknown" in summary.to_text()


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
