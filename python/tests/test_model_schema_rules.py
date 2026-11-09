"""The committed JSON Schema against the validator (TASK-005e1, ADR-0008 D2).

The schema is what a model sees (ADR-0007 §7 «Prompt»), the validator is the reference. The
differential test checks, with an independent draft 2020-12 implementation (``jsonschema``), that
the schema accepts every Decision Model the validator accepts in the repository and every valid
mutation of the corpus truths, so that showing a rule in the schema never changes the language.
"""

import json
from pathlib import Path
from typing import Any, cast, get_args

from hypothesis import assume, given, settings
from hypothesis import strategies as st
from jsonschema import Draft202012Validator

from potakt.evaluation import CORPUS_PATH
from potakt.model import DecisionModelError, parse_decision_model
from potakt.model.catalog import _HardOrSoft

PYTHON = Path(__file__).resolve().parents[1]
REPO = PYTHON.parent
SCHEMA_FILE = REPO / "spec" / "decision-model.schema.json"
SCHEMA: dict[str, Any] = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
VALIDATOR = Draft202012Validator(SCHEMA)
FIXTURES = PYTHON / "tests" / "fixtures"

# Templates that take either hardness, read from the catalog (``_HardOrSoft`` subclasses).
HARD_OR_SOFT = frozenset(
    get_args(cls.model_fields["template"].annotation)[0] for cls in _HardOrSoft.__subclasses__()
)

Document = dict[str, Any]


def _parses(text: str) -> bool:
    try:
        parse_decision_model(text)
    except DecisionModelError:
        return False
    return True


def _truth_paths() -> list[Path]:
    return sorted(CORPUS_PATH.glob("*/truth.json"))


def _fixture_models() -> list[Path]:
    """Every file under ``tests/fixtures`` whose text is a Decision Model the validator accepts."""
    candidates = sorted(
        path for path in FIXTURES.rglob("*") if path.is_file() and path.suffix in (".json", ".txt")
    )
    return [path for path in candidates if _parses(path.read_text(encoding="utf-8-sig"))]


def repository_models() -> list[Path]:
    """The ten corpus truths, both examples and every DM fixture that parses."""
    return [*_truth_paths(), *sorted(REPO.glob("examples/*/model.json")), *_fixture_models()]


def schema_errors(document: Any) -> list[str]:
    """The schema's error messages for ``document``, each with its JSON path; empty if valid."""
    return [
        f"{error.json_path}: {error.message}"
        for error in sorted(VALIDATOR.iter_errors(document), key=lambda e: e.json_path)
    ]


def test_the_committed_schema_is_a_valid_draft_2020_12_schema() -> None:
    assert SCHEMA["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    Draft202012Validator.check_schema(SCHEMA)


def test_the_repository_has_the_models_the_differential_test_needs() -> None:
    names = [path.relative_to(REPO).as_posix() for path in repository_models()]

    assert len(_truth_paths()) == 10
    assert "examples/clinic-el/model.json" in names
    assert "examples/retail-en/model.json" in names
    assert len(_fixture_models()) >= 1, "no DM fixture parses: the fixture part would be vacuous"


def test_the_hard_or_soft_templates_are_the_catalog_ones() -> None:
    expected = {
        "requires_attribute",
        "no_overlap",
        "min_rest",
        "max_per_period",
        "min_per_period",
        "max_consecutive_days",
        "unavailable",
        "apart",
    }

    assert sorted(HARD_OR_SOFT) == sorted(expected)


def test_the_schema_accepts_every_model_the_validator_accepts_in_the_repository() -> None:
    rejected = {}
    for path in repository_models():
        text = path.read_text(encoding="utf-8-sig")
        parse_decision_model(text)  # each one is valid for the validator
        errors = schema_errors(json.loads(text))
        if errors:
            rejected[path.relative_to(REPO).as_posix()] = errors

    assert rejected == {}


TRUTHS: list[Document] = [json.loads(path.read_text(encoding="utf-8")) for path in _truth_paths()]


@st.composite
def valid_mutations(draw: st.DrawFn) -> Document:
    """A corpus truth with some non-coverage rules dropped, the rules shuffled and some
    hard-or-soft rules flipped (hard: weight removed; soft: weight 1-10000)."""
    truth = cast("Document", json.loads(json.dumps(draw(st.sampled_from(TRUTHS)))))
    rules: list[Document] = truth["constraints"]
    kept = [
        rule
        for rule in rules
        if rule["template"] == "coverage" or not draw(st.booleans(), label="drop")
    ]
    kept = draw(st.permutations(kept), label="order")
    for rule in kept:
        if rule["template"] in HARD_OR_SOFT and draw(st.booleans(), label="flip"):
            if rule["hardness"] == "hard":
                rule["hardness"] = "soft"
                rule["weight"] = draw(st.integers(1, 10_000), label="weight")
            else:
                rule["hardness"] = "hard"
                rule.pop("weight", None)
    truth["constraints"] = kept
    return truth


@settings(deadline=None, max_examples=200)
@given(document=valid_mutations())
def test_the_schema_accepts_every_valid_mutation_of_the_truths(document: Document) -> None:
    # ``assume``: Hypothesis fails the test if too few mutations parse, so it is never vacuous.
    assume(_parses(json.dumps(document, ensure_ascii=False)))

    assert schema_errors(document) == []
