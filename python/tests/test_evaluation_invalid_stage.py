"""The stage at which an invalid candidate failed (for ADR-0008 D3 (a)).

One corpus slip per stage, decided by the step that failed and, for a structural error, by the
type of the Pydantic error behind the first issue; never by message text.
"""

import copy
import json
from functools import cache
from random import Random
from typing import Any, cast, get_args

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
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


# --- the invariant, over random slips ---------------------------------------------------------

CORPUS_NAMES = sorted(
    path.name for path in CORPUS_PATH.iterdir() if (path / "truth.json").is_file()
)
# Values a careless model could put where a number, a string or an object belongs. The surrogate
# is a lone one: `json.dumps` writes it as `\ud800`, which `json.loads` accepts and Pydantic's own
# JSON parser rejects (`json_invalid`).
ODD_VALUES: list[Any] = [None, True, False, -1, 0, 10**12, 1.5, "", " ", "x", [], {}, "\ud800"]
BAD_SOURCES = ["", "missing.csv", "../escape.csv", "data/nul.csv", "NUL", "a/b/c.csv"]
SEEDS = st.integers(min_value=0, max_value=2**32 - 1).map(Random)


def node_paths(node: Any, prefix: tuple[str | int, ...] = ()) -> list[tuple[str | int, ...]]:
    """The path of every node of a JSON document, the root (the empty path) first."""
    found = [prefix]
    if isinstance(node, dict):
        for key, value in cast("dict[str, Any]", node).items():
            found += node_paths(value, (*prefix, key))
    elif isinstance(node, list):
        for index, value in enumerate(cast("list[Any]", node)):
            found += node_paths(value, (*prefix, index))
    return found


def node_at(document: Any, path: tuple[str | int, ...]) -> Any:
    for step in path:
        document = document[step]
    return document


def put(document: Any, path: tuple[str | int, ...], value: Any) -> Any:
    """``document`` with the node at ``path`` replaced (the root: ``value`` itself)."""
    if not path:
        return value
    node_at(document, path[:-1])[path[-1]] = value
    return document


def replace_a_leaf(document: Any, rng: Random) -> str:
    leaves = [p for p in node_paths(document) if not isinstance(node_at(document, p), dict | list)]
    return dumps(put(document, rng.choice(leaves), rng.choice(ODD_VALUES)))


def delete_a_key(document: Any, rng: Random) -> str:
    path = rng.choice(node_paths(document)[1:])
    del node_at(document, path[:-1])[path[-1]]
    return dumps(document)


def add_an_extra_key(document: Any, rng: Random) -> str:
    objects = [p for p in node_paths(document) if isinstance(node_at(document, p), dict)]
    node_at(document, rng.choice(objects))["surprise"] = 1
    return dumps(document)


def wrap_in_an_array(document: Any, rng: Random) -> str:
    path = rng.choice(node_paths(document))
    return dumps(put(document, path, [node_at(document, path)]))


def cut_the_text(document: Any, rng: Random) -> str:
    text = dumps(document)
    return text[: rng.randrange(len(text))]


def duplicate_a_constraint(document: Any, rng: Random) -> str:
    document["constraints"].append(copy.deepcopy(rng.choice(document["constraints"])))
    return dumps(document)


def put_an_unknown_resource_id(document: Any, rng: Random) -> str:
    ids = [
        p
        for p in node_paths(document)
        if p and (p[-1] == "resource_id" or (len(p) > 1 and p[-2] == "resource_ids"))
    ]
    if ids:
        document = put(document, rng.choice(ids), "ZZ99")
    return dumps(document)


def change_a_source(document: Any, rng: Random) -> str:
    sources = [p for p in node_paths(document) if p and p[-1] == "source"]
    return dumps(put(document, rng.choice(sources), rng.choice(BAD_SOURCES)))


def dumps(document: Any) -> str:
    return json.dumps(document)  # ASCII only: a lone surrogate is written as an escape


SLIPS_AT_RANDOM = {
    function.__name__: function
    for function in (
        replace_a_leaf,
        delete_a_key,
        add_an_extra_key,
        wrap_in_an_array,
        cut_the_text,
        duplicate_a_constraint,
        put_an_unknown_resource_id,
        change_a_source,
    )
}


@cache
def corpus_scenario(name: str) -> LoadedScenario:
    return load_scenario(CORPUS_PATH / name)


@settings(max_examples=300, deadline=None)
@given(name=st.sampled_from(CORPUS_NAMES), slip=st.sampled_from(sorted(SLIPS_AT_RANDOM)), rng=SEEDS)
def test_every_invalid_candidate_has_a_stage_and_a_valid_one_has_none(
    name: str, slip: str, rng: Random
) -> None:
    truth = json.loads((CORPUS_PATH / name / "truth.json").read_text(encoding="utf-8"))
    text = SLIPS_AT_RANDOM[slip](truth, rng)

    result = evaluate_candidate(corpus_scenario(name), text)

    assert result.valid == (result.invalid_stage is None)
    assert result.invalid_stage is None or result.invalid_stage in get_args(InvalidStage)
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
    if not isinstance(parsed, dict):
        assert result.invalid_stage == "json"
