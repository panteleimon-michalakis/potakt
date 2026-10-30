"""The example scenarios under ``examples/`` are valid ground-truth Decision Models."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from potakt.cli import app
from potakt.model import (
    DecisionModel,
    load_decision_model,
    load_instance,
    parse_decision_model,
    validate_data,
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
