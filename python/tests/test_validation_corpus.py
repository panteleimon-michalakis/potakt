"""Integrity of the Phase 0 validation corpus (ADR-0007 §2-§4, TASK-005a D4).

Parametrized over every directory in ``docs/validation/scenarios/``: each scenario loads, its
truth and reference solution agree with its metadata, its request follows the writing rules and its
disruption really disrupts the reference solution.
"""

import json
import re
from datetime import date, datetime, time, timedelta
from functools import cache
from pathlib import Path

import pytest
from typer.testing import CliRunner

from potakt.checker import EVALUATORS, check
from potakt.cli import app
from potakt.evaluation import LoadedModel, LoadedScenario, compare_models, load_scenario
from potakt.model import load_instance, parse_decision_model, validate_data

CORPUS = Path(__file__).resolve().parents[2] / "docs" / "validation" / "scenarios"
SCENARIOS = sorted(path.name for path in CORPUS.iterdir() if path.is_dir())

TEMPLATES = sorted(EVALUATORS)
SCORED = [template for template in TEMPLATES if template != "coverage"]
LEAK_WORDS = ["hard", "soft", "scope", "weight", "constraint"]
# Template names as written and with spaces; every word also in the plural.
_LEAK_TERMS = sorted({*TEMPLATES, *(t.replace("_", " ") for t in TEMPLATES), *LEAK_WORDS})
LEAK = re.compile(
    r"\b(?:" + "|".join(re.escape(term) for term in _LEAK_TERMS) + r")s?\b", re.IGNORECASE
)
KEBAB = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
BANDS = {"easy": (4, 6), "medium": (7, 9), "hard": (10, 14)}
TIMEZONES = {"el": {"Europe/Athens"}, "en": {"Europe/London", "America/New_York"}}


@cache
def loaded(name: str) -> LoadedScenario:
    return load_scenario(CORPUS / name)


def normalized(text: str) -> str:
    return " ".join(text.split())


def scored_rules(scenario: LoadedScenario) -> list[str]:
    return [rule.id for rule in scenario.truth.constraints if rule.template != "coverage"]


def greek_letters(text: str) -> tuple[int, int]:
    """(Greek letters, all letters)."""
    letters = [ch for ch in text if ch.isalpha()]
    greek = [ch for ch in letters if "Ͱ" <= ch <= "Ͽ" or "ἀ" <= ch <= "῿"]
    return len(greek), len(letters)


def test_the_corpus_is_not_empty() -> None:
    assert SCENARIOS


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("A hard limit.", ["hard"]),
        ("Soft rules, WEIGHTS and Constraints.", ["Soft", "WEIGHTS", "Constraints"]),
        ("in scope", ["scope"]),
        ("no overlap please", ["no overlap"]),
        ("max_per_period and min per period", ["max_per_period", "min per period"]),
        ("Keep them apart, or together; check the balance.", ["apart", "together", "balance"]),
        ("her preferences, his preference", ["preferences", "preference"]),
        ("unavailable, coverage, stability", ["unavailable", "coverage", "stability"]),
        ("hardly a softer apartment; no overlapping; balanced", []),
        ("Ιδανικά λιγότερες, απαγορεύεται οπωσδήποτε: 25 ώρες", []),
    ],
)
def test_the_leak_pattern_finds_catalog_words_as_whole_words(text: str, found: list[str]) -> None:
    assert LEAK.findall(text) == found


def test_greek_letters_are_counted_among_all_letters() -> None:
    assert greek_letters("Καλημέρα, Anna! ἀρχή 42") == (12, 16)


@pytest.mark.parametrize("name", SCENARIOS)
def test_scenario_loads_and_its_id_is_the_directory_name(name: str) -> None:
    scenario = loaded(name)

    assert scenario.scenario.id == name
    assert re.fullmatch(r"\d{2}-[a-z0-9]+(-[a-z0-9]+)*", name)


@pytest.mark.parametrize("name", SCENARIOS)
def test_csv_files_are_exactly_the_csv_files_present_and_the_request_names_them(
    name: str,
) -> None:
    scenario = loaded(name)
    present = sorted(p.name for p in (CORPUS / name).iterdir() if p.name.endswith(".csv"))

    assert sorted(scenario.scenario.csv_files) == present
    for file_name in present:
        assert file_name in scenario.request, file_name


@pytest.mark.parametrize("name", SCENARIOS)
def test_truth_is_a_valid_decision_model_0_2(name: str) -> None:
    scenario = loaded(name)

    assert scenario.truth.schema_version == "0.2"
    assert scenario.truth.language == scenario.scenario.language
    assert validate_data(scenario.truth, scenario.instance) == []


@pytest.mark.parametrize("name", SCENARIOS)
def test_potakt_validate_accepts_the_truth(name: str) -> None:
    result = CliRunner().invoke(app, ["validate", str(CORPUS / name / "truth.json")])

    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("OK: ")


@pytest.mark.parametrize("name", SCENARIOS)
def test_reference_solution_is_feasible_and_scores_the_reference_score(name: str) -> None:
    scenario = loaded(name)

    score = check(scenario.truth, scenario.instance, scenario.solution).score

    assert score.hard == 0
    reference = scenario.scenario.reference_score
    assert (score.hard, score.soft) == (reference.hard, reference.soft)


@pytest.mark.parametrize("name", SCENARIOS)
def test_every_scored_rule_quotes_the_request_verbatim(name: str) -> None:
    scenario = loaded(name)
    request = normalized(scenario.request)

    for rule in scenario.truth.constraints:
        provenance = rule.provenance
        if rule.template != "coverage":
            assert provenance.kind == "utterance", rule.id
        if provenance.quote is not None:
            assert normalized(provenance.quote) in request, (rule.id, provenance.quote)
    for unsupported in scenario.scenario.unsupported:
        assert normalized(unsupported.quote) in request, unsupported.quote


@pytest.mark.parametrize("name", SCENARIOS)
def test_request_uses_no_catalog_vocabulary(name: str) -> None:
    found = LEAK.findall(loaded(name).request)

    assert found == []


@pytest.mark.parametrize("name", SCENARIOS)
def test_request_and_user_facing_truth_strings_are_in_the_scenario_language(name: str) -> None:
    scenario = loaded(name)
    language = scenario.scenario.language
    texts = [scenario.truth.title, *(rule.rationale for rule in scenario.truth.constraints)]

    greek, letters = greek_letters(scenario.request)
    if language == "el":
        assert greek > letters / 2, (greek, letters)
        assert all(greek_letters(text)[0] > 0 for text in texts)
    else:
        assert greek == 0
        assert all(greek_letters(text)[0] == 0 for text in texts)


@pytest.mark.parametrize("name", SCENARIOS)
def test_ids_are_ascii_kebab_case(name: str) -> None:
    truth = loaded(name).truth
    ids = [truth.id, *(r.id for r in truth.constraints), *(a.id for a in truth.assumptions)]

    assert [item for item in ids if not KEBAB.fullmatch(item)] == []


@pytest.mark.parametrize("name", SCENARIOS)
def test_size_horizon_and_timezone_follow_the_corpus_rules(name: str) -> None:
    scenario = loaded(name)
    truth, instance = scenario.truth, scenario.instance

    assert 5 <= len(instance.resources) <= 12
    assert 14 <= len(instance.tasks) <= 42
    start, end = truth.horizon.start, truth.horizon.end
    assert start.weekday() == 0
    assert (end - start).days == 6
    assert date(2026, 11, 1) <= start and end <= date(2026, 12, 31)
    assert truth.timezone in TIMEZONES[scenario.scenario.language]


@pytest.mark.parametrize("name", SCENARIOS)
def test_scored_constraints_fit_the_difficulty(name: str) -> None:
    scenario = loaded(name)
    low, high = BANDS[scenario.scenario.difficulty]

    count = len(scored_rules(scenario))

    assert count >= 4
    assert low <= count <= high, count


@pytest.mark.parametrize("name", SCENARIOS)
def test_the_disruption_makes_the_reference_solution_infeasible(name: str) -> None:
    scenario = loaded(name)
    disruption = scenario.scenario.disruption
    horizon = scenario.truth.horizon
    assert disruption.resource_id in {r.id for r in scenario.instance.resources}
    assert datetime.combine(horizon.start, time()) <= disruption.start < disruption.end
    assert disruption.end <= datetime.combine(horizon.end + timedelta(days=1), time())
    document = json.loads(scenario.truth.model_dump_json())
    window = {
        "resource_id": disruption.resource_id,
        "start": disruption.start.isoformat(timespec="minutes"),
        "end": disruption.end.isoformat(timespec="minutes"),
    }
    document["constraints"].append(
        {
            "id": "disruption",
            "template": "unavailable",
            "hardness": "hard",
            "params": {"windows": [window]},
            "rationale": "The scenario's disruption.",
            "provenance": {"kind": "assumption"},
        }
    )
    disrupted = parse_decision_model(json.dumps(document, ensure_ascii=False))
    instance = load_instance(disrupted, CORPUS / name)
    assert validate_data(disrupted, instance) == []

    result = check(disrupted, instance, scenario.solution)

    assert result.rule("disruption").amount > 0
    assert result.score.hard > 0


@pytest.mark.parametrize("name", SCENARIOS)
def test_the_truth_matches_itself(name: str) -> None:
    scenario = loaded(name)
    truth = LoadedModel(scenario.truth, scenario.instance)

    result = compare_models(truth, truth)

    assert result.truth_count == len(scored_rules(scenario))
    assert (result.recall, result.precision) == (1.0, 1.0)
