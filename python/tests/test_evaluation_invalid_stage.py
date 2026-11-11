"""The stage at which an invalid candidate failed (for ADR-0008 D3 (a)).

One corpus slip per stage, decided by the step that failed and, for a structural error, by the
type of the Pydantic error behind the first issue; never by message text.
"""

import json
from functools import cache
from typing import Any, get_args

import pytest
from pydantic import field_validator

from potakt.evaluation import (
    CORPUS_PATH,
    Comparison,
    InvalidStage,
    LoadedScenario,
    evaluate_candidate,
    load_model,
    load_scenario,
)
from potakt.evaluation.compare import InvalidModel
from potakt.evaluation.records import Answer, score_answer
from potakt.model import DecisionModelError
from potakt.model.common import SOFT_RULE_NEEDS_WEIGHT, StrictModel
from potakt.model.loader import _parse_json  # pyright: ignore[reportPrivateUsage]

S06 = "06-ambulance-crews"


@cache
def scenario() -> LoadedScenario:
    return load_scenario(CORPUS_PATH / S06)


def truth_text() -> str:
    return (CORPUS_PATH / S06 / "truth.json").read_text(encoding="utf-8")


def truth_document() -> dict[str, Any]:
    return json.loads(truth_text())


def evaluate(document: object) -> Comparison:
    return evaluate_candidate(scenario(), json.dumps(document, ensure_ascii=False))


def version_as_string() -> dict[str, Any]:
    document = truth_document()
    document["version"] = str(document["version"])
    return document


def weight_on_a_hard_rule() -> dict[str, Any]:
    document = truth_document()
    next(c for c in document["constraints"] if c["id"] == "rest-11h")["weight"] = 2
    return document


def source_with_a_folder() -> dict[str, Any]:
    document = truth_document()
    binding = document["data"]["resources"]
    binding["source"] = f"data/{binding['source']}"
    return document


def unknown_resource_in_a_baseline() -> dict[str, Any]:
    document = truth_document()
    rule = next(c for c in document["constraints"] if c["template"] == "stability")
    rule["params"]["baseline"][0]["resource_id"] = "E99"  # no such crew member
    return document


def prose_before_the_fence() -> Comparison:
    text = f"Here is the Decision Model:\n\n```json\n{truth_text()}\n```\n"
    _, comparison = score_answer(scenario(), Answer(text, "end_turn", True))
    return comparison


SLIPS = {
    "json": prose_before_the_fence,
    "schema": lambda: evaluate(version_as_string()),
    "rule": lambda: evaluate(weight_on_a_hard_rule()),
    "data_file": lambda: evaluate(source_with_a_folder()),
    "data": lambda: evaluate(unknown_resource_in_a_baseline()),
}


@pytest.mark.parametrize(("stage", "slip"), SLIPS.items(), ids=SLIPS)
def test_one_corpus_slip_per_stage(stage: str, slip: Any) -> None:
    result: Comparison = slip()

    assert not result.valid
    assert result.invalid_stage == stage
    assert json.loads(result.to_json())["invalid_stage"] == stage


def test_the_five_stages_are_exactly_the_literal_s_values() -> None:
    stages = {slip().invalid_stage for slip in SLIPS.values()}

    assert get_args(InvalidStage) == ("json", "schema", "rule", "data_file", "data")
    assert stages == set(get_args(InvalidStage))


def test_a_valid_candidate_has_no_stage() -> None:
    result = evaluate(truth_document())

    assert result.valid
    assert result.invalid_stage is None
    assert json.loads(result.to_json())["invalid_stage"] is None


def test_an_answer_that_did_not_end_normally_has_no_stage() -> None:
    # No step ran: the answer's stop reason classifies it, not a validation stage.
    _, result = score_answer(scenario(), Answer(truth_text(), "max_tokens", False))

    assert not result.valid
    assert result.invalid_stage is None


def with_constraint_change(change: Any) -> dict[str, Any]:
    document = truth_document()
    change(document["constraints"])
    return document


def with_data_resources_weight() -> dict[str, Any]:
    document = truth_document()
    document["data"]["resources"]["weight"] = 1
    return document


MORE = {
    "not JSON": ("{", "json"),
    "a JSON array": ("[]", "json"),
    "a JSON string": ('"model"', "json"),
    "nested too deep": ("[" * 70 + "]" * 70, "json"),
    "an integer too long": ('{"version": ' + "9" * 5000 + "}", "json"),
    "an empty object": ("{}", "schema"),
    "an unknown template": (
        with_constraint_change(lambda c: c[1].update(template="requires_skill")),
        "schema",
    ),
    # The error is Pydantic's `extra_forbidden`, but the message is our rule's wording for a
    # constraint's own `weight`, so it is a `rule` like the same slip on a template with both
    # hardnesses; a `weight` that is not a constraint's own keeps Pydantic's message: `schema`.
    "weight on coverage": (with_constraint_change(lambda c: c[0].update(weight=1)), "rule"),
    "a weight outside the constraints": (with_data_resources_weight(), "schema"),
    "a duplicate constraint id": (
        with_constraint_change(lambda c: c[1].update(id=c[2]["id"])),
        "rule",
    ),
}


TITLE = '"title": "'
# Texts that `json.loads` accepts but Pydantic's own JSON parser does not, or parses differently.
# `json` covers the raw checks, `json.loads`, a top level that is not an object and Pydantic's own
# `json_invalid`; Pydantic parses `NaN`, whose error is `int_type`, so that one stays `schema`.
MORE |= {
    "a lone surrogate (Pydantic: json_invalid)": (
        truth_text().replace(TITLE, TITLE + "\\ud800", 1),
        "json",
    ),
    "NaN as the version (Pydantic: int_type)": (
        truth_text().replace('"version": 1,', '"version": NaN,', 1),
        "schema",
    ),
}


@pytest.mark.parametrize(("candidate", "stage"), MORE.values(), ids=MORE)
def test_more_slips_land_in_their_stage(candidate: object, stage: str) -> None:
    text = candidate if isinstance(candidate, str) else json.dumps(candidate, ensure_ascii=False)

    result = evaluate_candidate(scenario(), text)

    assert not result.valid
    assert result.invalid_stage == stage


def test_a_soft_only_rule_without_a_weight_is_a_rule() -> None:
    # `preference` is soft-only, so Pydantic reports `missing` for the absent weight before any
    # validator runs; the message is our rule's wording (as on a template with both hardnesses).
    s01 = load_scenario(CORPUS_PATH / "01-pharmacy")
    document = json.loads((CORPUS_PATH / "01-pharmacy" / "truth.json").read_text(encoding="utf-8"))
    rule = next(c for c in document["constraints"] if c["id"] == "stelios-avoids-afternoons")
    assert rule["template"] == "preference"
    del rule["weight"]

    result = evaluate_candidate(s01, json.dumps(document, ensure_ascii=False))

    assert not result.valid
    assert result.invalid_stage == "rule"
    assert [issue.message for issue in result.issues] == [SOFT_RULE_NEEDS_WEIGHT]


def test_a_data_file_the_loader_cannot_read_is_a_data_file_stage() -> None:
    document = truth_document()
    document["data"]["tasks"]["source"] = "missing.csv"

    candidate = load_model(json.dumps(document, ensure_ascii=False), CORPUS_PATH / S06)

    assert isinstance(candidate, InvalidModel)
    assert candidate.stage == "data_file"


class _Checked(StrictModel):
    value: int

    @field_validator("value")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be positive")
        return value


@pytest.mark.parametrize(
    ("text", "stage"),
    [('{"value": 0}', "rule"), ('{"value": "1"}', "schema"), ('{"value": 1', "json")],
)
def test_a_validator_s_value_error_is_a_rule_and_a_type_error_is_schema(
    text: str, stage: str
) -> None:
    with pytest.raises(DecisionModelError) as caught:
        _parse_json(text, _Checked)

    assert caught.value.stage == stage


def test_an_error_raised_outside_the_json_parser_has_no_stage() -> None:
    assert DecisionModelError([]).stage is None
