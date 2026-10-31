"""The example scenarios under ``examples/`` are valid ground-truth Decision Models."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from potakt.checker import CheckResult, Score, check
from potakt.cli import app
from potakt.model import (
    DecisionModel,
    ProblemInstance,
    Solution,
    load_decision_model,
    load_instance,
    load_solution,
    parse_decision_model,
    validate_data,
    validate_solution,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
EXAMPLES = {"clinic-el": "el", "retail-en": "en"}


def model_file(name: str) -> Path:
    return EXAMPLES_DIR / name / "model.json"


def load_example(name: str) -> DecisionModel:
    return load_decision_model(model_file(name))


@pytest.mark.parametrize("name", EXAMPLES)
def test_example_loads_and_validates_without_issues(name: str) -> None:
    dm = load_example(name)
    instance = load_instance(dm, model_file(name).parent)

    assert validate_data(dm, instance) == []
    assert dm.language == EXAMPLES[name]
    assert instance.resources
    assert instance.tasks


@pytest.mark.parametrize("name", EXAMPLES)
def test_example_round_trips_through_json(name: str) -> None:
    dm = load_example(name)

    assert parse_decision_model(dm.model_dump_json()) == dm


@pytest.mark.parametrize("name", EXAMPLES)
def test_every_rule_quotes_the_description_or_states_a_reason(name: str) -> None:
    dm = load_example(name)
    description = (EXAMPLES_DIR / name / "description.md").read_text(encoding="utf-8")

    for rule in dm.constraints:
        provenance = rule.provenance
        if provenance.kind == "utterance":
            assert provenance.quote is not None
            assert provenance.quote in description, (rule.id, provenance.quote)
        else:
            assert provenance.kind in ("default", "assumption"), rule.id
            assert rule.rationale.strip(), rule.id


def test_greek_text_round_trips_in_description_quotes_and_csv() -> None:
    dm = load_example("clinic-el")
    instance = load_instance(dm, model_file("clinic-el").parent)
    description = (EXAMPLES_DIR / "clinic-el" / "description.md").read_text(encoding="utf-8")

    assert "νοσηλευτές" in description
    assert any(
        rule.provenance.quote and "βάρδια" in rule.provenance.quote for rule in dm.constraints
    )
    names = [resource.attributes["name"] for resource in instance.resources]
    assert "Μαρία Παπαδοπούλου" in names
    shift_types = {str(task.attributes["type"]) for task in instance.tasks}
    assert shift_types == {"πρωί", "απόγευμα", "νύχτα"}
    reloaded = parse_decision_model(dm.model_dump_json())
    assert reloaded.title == dm.title


@pytest.mark.parametrize("name", EXAMPLES)
def test_cli_validates_example(name: str) -> None:
    result = CliRunner().invoke(app, ["validate", str(model_file(name))])

    assert result.exit_code == 0, result.stdout
    assert result.stdout.startswith("OK: ")


# --- reference solutions (independent cross-check, TASK-003) ---------------------------------

# Scores computed by the reviewer's independent implementation of ADR-0005.
REFERENCE_SCORES = {"clinic-el": Score(hard=0, soft=57), "retail-en": Score(hard=0, soft=2)}


def solution_file(name: str) -> Path:
    return EXAMPLES_DIR / name / "solution.json"


def load_reference(name: str) -> tuple[DecisionModel, ProblemInstance, Solution]:
    dm = load_example(name)
    return dm, load_instance(dm, model_file(name).parent), load_solution(solution_file(name))


def with_changes(solution: Solution, changes: dict[tuple[str, str], str]) -> Solution:
    """Replace the resource of each ``(task_id, resource_id)`` pair in ``changes``."""
    assignments = tuple(
        a.model_copy(update={"resource_id": changes.get((a.task_id, a.resource_id), a.resource_id)})
        for a in solution.assignments
    )
    return solution.model_copy(update={"assignments": assignments})


def hard_violations(result: CheckResult) -> dict[str, list[tuple[object, ...]]]:
    return {
        rule.rule_id: [(v.resource_id, v.task_ids, v.period, v.amount) for v in rule.violations]
        for rule in result.rules
        if rule.hardness == "hard" and rule.amount
    }


@pytest.mark.parametrize("name", EXAMPLES)
def test_reference_solution_is_valid_for_its_model(name: str) -> None:
    dm, instance, solution = load_reference(name)

    assert validate_solution(dm, instance, solution) == []
    assert solution.metadata is not None
    assert solution.metadata.produced_by == "reviewer: independent CP-SAT probe"


@pytest.mark.parametrize("name", EXAMPLES)
def test_reference_solution_matches_the_independent_score(name: str) -> None:
    result = check(*load_reference(name))

    assert result.score == REFERENCE_SCORES[name]


def test_clinic_reference_soft_breakdown() -> None:
    result = check(*load_reference("clinic-el"))

    contributions = {rule.rule_id: rule.contribution for rule in result.rules if rule.amount}
    assert contributions == {
        "max-five-shifts-per-week": 50,
        "eleni-avoids-nights": 3,
        "balance-nights-als": 4,
    }
    weekly = result.rule("max-five-shifts-per-week").violations
    assert [(v.resource_id, v.period, v.amount) for v in weekly] == [
        (nurse, "2026-W45", 1) for nurse in ("n1", "n2", "n3", "n4", "n6")
    ]
    assert result.rule("balance-nights-als").violations[0].message == (
        "tasks per resource range from 1 (n2) to 3 (n1, n3)"
    )


def test_retail_reference_soft_breakdown() -> None:
    result = check(*load_reference("retail-en"))

    contributions = {rule.rule_id: rule.contribution for rule in result.rules if rule.amount}
    assert contributions == {"balance-weekend-shifts": 2}


def test_giving_maria_a_wednesday_shift_breaks_only_her_leave() -> None:
    dm, instance, solution = load_reference("clinic-el")
    # Maria (n1) takes Giorgos' (n5) Wednesday afternoon; he takes her Thursday morning.
    swapped = with_changes(
        solution, {("2026-11-04-apogevma", "n5"): "n1", ("2026-11-05-proi", "n1"): "n5"}
    )

    result = check(dm, instance, swapped)

    assert hard_violations(result) == {
        "maria-leave-wed": [("n1", ("2026-11-04-apogevma",), None, 1)]
    }
    assert result.score == Score(hard=1, soft=57)


def test_putting_emma_on_chloes_shift_breaks_only_apart() -> None:
    dm, instance, solution = load_reference("retail-en")
    # Emma (s5) replaces Alice (s1) on Friday's closing shift, next to her sister Chloe (s3).
    changed = with_changes(solution, {("fri-close", "s1"): "s5"})

    result = check(dm, instance, changed)

    assert hard_violations(result) == {"chloe-emma-apart": [(None, ("fri-close",), None, 1)]}
    # Alice drops to 24 h: 8 started hours under her 32 contract hours, x 3; weekend balance 2.
    assert result.score == Score(hard=1, soft=26)


@pytest.mark.parametrize("name", EXAMPLES)
def test_cli_checks_reference_solution(name: str) -> None:
    result = CliRunner().invoke(app, ["check", str(model_file(name)), str(solution_file(name))])

    assert result.exit_code == 0, result.output
    assert result.stdout.startswith(f"{load_example(name).title} (")
    score = REFERENCE_SCORES[name]
    assert f"Score: hard {score.hard}, soft {score.soft} (feasible)\n" in result.stdout
