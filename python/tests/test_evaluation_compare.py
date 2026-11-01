"""The Test A metric: denotations, one-to-one matching and diagnostics (ADR-0007 §5, D5)."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from checker_helpers import make_instance, rule
from hypothesis import given, settings
from hypothesis import strategies as st
from test_checker_properties import Case, cases, model

from potakt.evaluation import (
    Comparison,
    InvalidModel,
    LoadedModel,
    compare_models,
    denote,
    load_model,
    load_scenario,
)

RESOURCES = """id,name,contract,max_hours,skills,level,badge
a1,Ana,full,40,opening;closing,senior,1
a2,Ben,full,40,opening,junior,0
a3,Cleo,part,24,closing,senior,1
a4,Dev,part,24,,junior,0
"""

TASKS = """id,start,end,demand,kind,needs,min_level,badge_needed
mon-am,2026-11-02T07:00,2026-11-02T15:00,1,am,opening,senior,1
mon-pm,2026-11-02T15:00,2026-11-02T23:00,1,pm,closing,,
tue-am,2026-11-03T07:00,2026-11-03T15:00,1,am,opening,,
tue-pm,2026-11-03T15:00,2026-11-03T23:00,1,pm,closing,senior,
sat-am,2026-11-07T09:00,2026-11-07T17:00,2,am,,,1
sun-am,2026-11-08T09:00,2026-11-08T17:00,2,am,,,
"""

WINDOWS = """person,from,to
a1,2026-11-03T00:00,2026-11-04T00:00
"""

RESOURCE_ATTRIBUTES: dict[str, Any] = {
    "contract": {"column": "contract", "type": "str"},
    "max_hours": {"column": "max_hours", "type": "float"},
    "skills": {"column": "skills", "type": "list", "separator": ";"},
    "level": {"column": "level", "type": "str"},
}
TASK_ATTRIBUTES: dict[str, Any] = {
    "kind": {"column": "kind", "type": "str"},
    "needs": {"column": "needs", "type": "str"},
    "min_level": {"column": "min_level", "type": "str"},
}

FULL_TIME = {"resources": [{"attribute": "contract", "op": "eq", "value": "full"}]}
WEEKEND = {"tasks": [{"attribute": "weekday", "op": "in", "value": ["sat", "sun"]}]}
TUESDAY = {"resource_id": "a1", "start": "2026-11-03T00:00", "end": "2026-11-04T00:00"}
WINDOWS_SOURCE = {"source": "windows.csv", "resource_id": "person", "start": "from", "end": "to"}
SKILLS = {"task_attribute": "needs", "resource_attribute": "skills", "match": "contains"}
LEVEL = {"task_attribute": "min_level", "resource_attribute": "level", "match": "equals"}


def model_doc(
    *rules: dict[str, Any],
    resource_attributes: dict[str, Any] | None = None,
    task_attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    coverage = rule("coverage", id="coverage")
    return {
        "schema_version": "0.2",
        "id": "compare-test",
        "version": 1,
        "status": "draft",
        "title": "Compare test",
        "language": "en",
        "archetype": "assignment",
        "timezone": "Europe/London",
        "horizon": {"start": "2026-11-02", "end": "2026-11-08"},
        "data": {
            "resources": {
                "source": "staff.csv",
                "id": "id",
                "attributes": resource_attributes or RESOURCE_ATTRIBUTES,
            },
            "tasks": {
                "source": "shifts.csv",
                "id": "id",
                "start": "start",
                "end": "end",
                "demand": "demand",
                "attributes": task_attributes or TASK_ATTRIBUTES,
            },
        },
        "constraints": [coverage, *rules],
    }


@pytest.fixture
def data(tmp_path: Path) -> Path:
    (tmp_path / "staff.csv").write_text(RESOURCES, encoding="utf-8")
    (tmp_path / "shifts.csv").write_text(TASKS, encoding="utf-8")
    (tmp_path / "windows.csv").write_text(WINDOWS, encoding="utf-8")
    return tmp_path


def loaded(document: dict[str, Any], directory: Path) -> LoadedModel:
    result = load_model(json.dumps(document), directory)
    assert isinstance(result, LoadedModel), result
    return result


def compare(truth: dict[str, Any], candidate: dict[str, Any], directory: Path) -> Comparison:
    return compare_models(loaded(truth, directory), load_model(json.dumps(candidate), directory))


def one(template: str, hardness: str = "hard", **fields: Any) -> dict[str, Any]:
    return rule(template, hardness, **fields)


# --- equivalent rewrites must match -----------------------------------------------------------


EQUIVALENT: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "resource scope, other conditions, same extension": (
        one(
            "max_per_period",
            params={"measure": "hours", "period": "week", "limit": 40},
            scope=FULL_TIME,
        ),
        one(
            "max_per_period",
            params={"measure": "hours", "period": "week", "limit": 40},
            scope={"resources": [{"attribute": "id", "op": "in", "value": ["a2", "a1"]}]},
        ),
    ),
    "task scope, other conditions, same extension": (
        one("balance", "soft", params={"measure": "count"}, scope=WEEKEND),
        one(
            "balance",
            "soft",
            params={"measure": "count"},
            scope={"tasks": [{"attribute": "date", "op": "gte", "value": "2026-11-07"}]},
        ),
    ),
    "literal limit vs attribute limit with equal values": (
        one(
            "max_per_period",
            params={"measure": "hours", "period": "week", "limit": 40},
            scope=FULL_TIME,
        ),
        one(
            "max_per_period",
            params={"measure": "hours", "period": "week", "limit": {"attribute": "max_hours"}},
            scope=FULL_TIME,
        ),
    ),
    "16 vs 16.0 count limit": (
        one("min_per_period", "soft", params={"measure": "count", "period": "horizon", "limit": 2}),
        one(
            "min_per_period", "soft", params={"measure": "count", "period": "horizon", "limit": 2.0}
        ),
    ),
    "11 vs 11.0 hours": (
        one("min_rest", params={"hours": 11}),
        one("min_rest", params={"hours": 11.0}),
    ),
    "list order in a scope": (
        one("preference", "soft", params={"kind": "avoid"}, scope=WEEKEND),
        one(
            "preference",
            "soft",
            params={"kind": "avoid"},
            scope={"tasks": [{"attribute": "weekday", "op": "in", "value": ["sun", "sat"]}]},
        ),
    ),
    "inline vs CSV windows": (
        one("unavailable", params={"windows": [TUESDAY]}),
        one("unavailable", params={"source": WINDOWS_SOURCE}),
    ),
    "split vs merged windows": (
        one("unavailable", params={"windows": [TUESDAY]}),
        one(
            "unavailable",
            params={
                "windows": [
                    {"resource_id": "a1", "start": "2026-11-03T12:00", "end": "2026-11-04T00:00"},
                    {"resource_id": "a1", "start": "2026-11-03T00:00", "end": "2026-11-03T12:00"},
                ]
            },
        ),
    ),
    "overlapping vs merged windows, inline plus CSV": (
        one("unavailable", params={"windows": [TUESDAY]}),
        one(
            "unavailable",
            params={
                "windows": [
                    {"resource_id": "a1", "start": "2026-11-03T06:00", "end": "2026-11-03T18:00"}
                ],
                "source": WINDOWS_SOURCE,
            },
        ),
    ),
    "windows of out-of-scope resources are inert": (
        one("unavailable", params={"windows": [TUESDAY]}, scope=FULL_TIME),
        one(
            "unavailable",
            params={
                "windows": [
                    TUESDAY,
                    {"resource_id": "a3", "start": "2026-11-02T00:00", "end": "2026-11-03T00:00"},
                ]
            },
            scope=FULL_TIME,
        ),
    ),
    "pair order in together": (
        one("together", "soft", params={"resource_ids": ["a1", "a2"]}),
        one("together", "soft", params={"resource_ids": ["a2", "a1"]}),
    ),
    "pair order in apart": (
        one("apart", params={"resource_ids": ["a3", "a4"]}),
        one("apart", params={"resource_ids": ["a4", "a3"]}),
    ),
    "baseline order and out-of-scope baseline pairs": (
        one(
            "stability",
            "soft",
            params={
                "baseline": [
                    {"task_id": "mon-am", "resource_id": "a1"},
                    {"task_id": "mon-pm", "resource_id": "a3"},
                ]
            },
            scope=FULL_TIME,
        ),
        one(
            "stability",
            "soft",
            params={
                "baseline": [
                    {"task_id": "mon-pm", "resource_id": "a3"},
                    {"task_id": "mon-am", "resource_id": "a1"},
                    {"task_id": "tue-am", "resource_id": "a4"},
                ]
            },
            scope=FULL_TIME,
        ),
    ),
    "equals vs contains on a single-valued resource column": (
        one("requires_attribute", params=LEVEL),
        one("requires_attribute", params={**LEVEL, "match": "contains"}),
    ),
    "ids, weights, provenance, rationale and status": (
        one("max_consecutive_days", "soft", weight=2, params={"days": 5}),
        {
            **one("max_consecutive_days", "soft", weight=9, params={"days": 5}),
            "id": "five-days-max",
            "rationale": "Another explanation.",
            "provenance": {"kind": "utterance", "quote": "no more than five days in a row"},
            "status": "confirmed",
        },
    ),
}


@pytest.mark.parametrize(("truth_rule", "candidate_rule"), EQUIVALENT.values(), ids=EQUIVALENT)
def test_equivalent_rewrites_match(
    data: Path, truth_rule: dict[str, Any], candidate_rule: dict[str, Any]
) -> None:
    result = compare(model_doc(truth_rule), model_doc(candidate_rule), data)

    assert result.valid
    assert [(m.truth_id, m.candidate_id) for m in result.matched] == [
        (truth_rule["id"], candidate_rule["id"])
    ]
    assert (result.recall, result.precision) == (1.0, 1.0)


def test_renamed_attributes_and_a_list_binding_of_a_single_valued_column_match(data: Path) -> None:
    truth = model_doc(
        one("requires_attribute", params=SKILLS), one("requires_attribute", id="lvl", params=LEVEL)
    )
    renamed_resources = {
        "abilities": {"column": "skills", "type": "list", "separator": ";"},
        "seniority": {"column": "level", "type": "list"},
    }
    renamed_tasks = {
        "required_skill": {"column": "needs", "type": "list"},
        "level_needed": {"column": "min_level", "type": "str"},
    }
    candidate = model_doc(
        one(
            "requires_attribute",
            id="skill-rule",
            params={
                "task_attribute": "required_skill",
                "resource_attribute": "abilities",
                "match": "contains",
            },
        ),
        one(
            "requires_attribute",
            id="level-rule",
            params={
                "task_attribute": "level_needed",
                "resource_attribute": "seniority",
                "match": "equals",
            },
        ),
        resource_attributes=renamed_resources,
        task_attributes=renamed_tasks,
    )

    result = compare(truth, candidate, data)

    assert sorted((m.truth_id, m.candidate_id) for m in result.matched) == [
        ("lvl", "level-rule"),
        ("requires_attribute", "skill-rule"),
    ]


def test_a_list_column_bound_as_a_string_reads_other_values_and_does_not_match(data: Path) -> None:
    truth = model_doc(one("requires_attribute", params=SKILLS))
    as_string = {**RESOURCE_ATTRIBUTES, "skills": {"column": "skills", "type": "str"}}
    candidate = model_doc(one("requires_attribute", params=SKILLS), resource_attributes=as_string)

    result = compare(truth, candidate, data)

    assert result.matched == ()
    assert [u.category for u in result.unmatched_truth] == ["wrong_parameters"]


# --- mutations must not match, with the right category -----------------------------------------

LIMIT_40 = {"measure": "hours", "period": "week", "limit": 40}

MUTATIONS: dict[str, tuple[dict[str, Any], dict[str, Any], str]] = {
    "hardness flipped (hard to soft)": (
        one("max_per_period", params=LIMIT_40),
        one("max_per_period", "soft", params=LIMIT_40),
        "wrong_hardness",
    ),
    "hardness flipped (soft to hard)": (
        one("min_rest", "soft", params={"hours": 11}),
        one("min_rest", params={"hours": 11}),
        "wrong_hardness",
    ),
    "min_rest hours": (
        one("min_rest", params={"hours": 11}),
        one("min_rest", params={"hours": 12}),
        "wrong_parameters",
    ),
    "limit value": (
        one("max_per_period", params=LIMIT_40),
        one("max_per_period", params={**LIMIT_40, "limit": 41}),
        "wrong_parameters",
    ),
    "limit measure": (
        one("max_per_period", params=LIMIT_40),
        one("max_per_period", params={**LIMIT_40, "measure": "count"}),
        "wrong_parameters",
    ),
    "limit period": (
        one("max_per_period", params=LIMIT_40),
        one("max_per_period", params={**LIMIT_40, "period": "horizon"}),
        "wrong_parameters",
    ),
    "attribute limit with values unequal to the literal": (
        one("max_per_period", params=LIMIT_40),
        one("max_per_period", params={**LIMIT_40, "limit": {"attribute": "max_hours"}}),
        "wrong_parameters",
    ),
    "min_per_period limit": (
        one("min_per_period", "soft", params={"measure": "count", "period": "week", "limit": 2}),
        one("min_per_period", "soft", params={"measure": "count", "period": "week", "limit": 3}),
        "wrong_parameters",
    ),
    "max_consecutive_days days": (
        one("max_consecutive_days", params={"days": 5}),
        one("max_consecutive_days", params={"days": 6}),
        "wrong_parameters",
    ),
    "window shifted by an hour": (
        one("unavailable", params={"windows": [TUESDAY]}),
        one("unavailable", params={"windows": [{**TUESDAY, "end": "2026-11-04T01:00"}]}),
        "wrong_parameters",
    ),
    "window for another person": (
        one("unavailable", params={"windows": [TUESDAY]}),
        one("unavailable", params={"windows": [{**TUESDAY, "resource_id": "a2"}]}),
        "wrong_parameters",
    ),
    "preference kind": (
        one("preference", "soft", params={"kind": "avoid"}, scope=WEEKEND),
        one("preference", "soft", params={"kind": "prefer"}, scope=WEEKEND),
        "wrong_parameters",
    ),
    "balance measure": (
        one("balance", "soft", params={"measure": "count"}),
        one("balance", "soft", params={"measure": "hours"}),
        "wrong_parameters",
    ),
    "together pair": (
        one("together", "soft", params={"resource_ids": ["a1", "a2"]}),
        one("together", "soft", params={"resource_ids": ["a1", "a3"]}),
        "wrong_parameters",
    ),
    "apart pair": (
        one("apart", params={"resource_ids": ["a3", "a4"]}),
        one("apart", params={"resource_ids": ["a2", "a4"]}),
        "wrong_parameters",
    ),
    "baseline pair": (
        one("stability", "soft", params={"baseline": [{"task_id": "mon-am", "resource_id": "a1"}]}),
        one("stability", "soft", params={"baseline": [{"task_id": "mon-am", "resource_id": "a2"}]}),
        "wrong_parameters",
    ),
    "match contains vs equals on a multi-valued column": (
        one("requires_attribute", params=SKILLS),
        one("requires_attribute", params={**SKILLS, "match": "equals"}),
        "wrong_parameters",
    ),
    "requirement read from another column": (
        one("requires_attribute", params=SKILLS),
        one("requires_attribute", params={**SKILLS, "task_attribute": "kind"}),
        "wrong_parameters",
    ),
    "one resource more in scope": (
        one("max_per_period", params=LIMIT_40, scope=FULL_TIME),
        one(
            "max_per_period",
            params=LIMIT_40,
            scope={"resources": [{"attribute": "id", "op": "in", "value": ["a1", "a2", "a3"]}]},
        ),
        "wrong_scope",
    ),
    "one resource less in scope": (
        one("max_per_period", params=LIMIT_40, scope=FULL_TIME),
        one(
            "max_per_period",
            params=LIMIT_40,
            scope={"resources": [{"attribute": "id", "op": "eq", "value": "a1"}]},
        ),
        "wrong_scope",
    ),
    "one task less in scope": (
        one("balance", "soft", params={"measure": "count"}, scope=WEEKEND),
        one(
            "balance",
            "soft",
            params={"measure": "count"},
            scope={"tasks": [{"attribute": "weekday", "op": "eq", "value": "sat"}]},
        ),
        "wrong_scope",
    ),
    "scope and hardness both differ": (
        one("no_overlap"),
        one("no_overlap", "soft", scope=FULL_TIME),
        "wrong_scope",
    ),
    "parameters and hardness both differ": (
        one("min_rest", params={"hours": 11}),
        one("min_rest", "soft", params={"hours": 10}),
        "wrong_parameters",
    ),
    "a different template": (
        one("max_per_period", params=LIMIT_40),
        one("min_per_period", params=LIMIT_40),
        "missing",
    ),
}


@pytest.mark.parametrize(
    ("truth_rule", "candidate_rule", "category"), MUTATIONS.values(), ids=MUTATIONS
)
def test_mutations_do_not_match_and_are_classified(
    data: Path, truth_rule: dict[str, Any], candidate_rule: dict[str, Any], category: str
) -> None:
    candidate_rule = {**candidate_rule, "id": "candidate"}

    result = compare(model_doc(truth_rule), model_doc(candidate_rule), data)

    assert result.valid
    assert result.matched == ()
    (missed,) = result.unmatched_truth
    assert (missed.truth_id, missed.category) == (truth_rule["id"], category)
    assert missed.closest_candidate_id == (None if category == "missing" else "candidate")
    assert [u.candidate_id for u in result.unmatched_candidate] == ["candidate"]
    assert (result.recall, result.precision) == (0.0, 0.0)


# --- matching ------------------------------------------------------------------------------------


def test_matching_is_one_to_one_over_equal_denotations(data: Path) -> None:
    twice = model_doc(
        one("min_rest", id="rest-a", params={"hours": 11}),
        one("min_rest", id="rest-b", params={"hours": 11}),
    )
    once = model_doc(one("min_rest", id="rest", params={"hours": 11}))

    fewer = compare(twice, once, data)
    more = compare(once, twice, data)

    assert [(m.truth_id, m.candidate_id) for m in fewer.matched] == [("rest-a", "rest")]
    assert [(u.truth_id, u.category) for u in fewer.unmatched_truth] == [("rest-b", "missing")]
    assert (fewer.recall, fewer.precision) == (0.5, 1.0)
    assert [u.candidate_id for u in more.unmatched_candidate] == ["rest-b"]
    assert (more.recall, more.precision) == (1.0, 0.5)


def test_diagnostics_pair_each_candidate_once_closest_first(data: Path) -> None:
    truth = model_doc(
        one("max_per_period", id="day", params={"measure": "count", "period": "day", "limit": 1}),
        one("max_per_period", id="week", params=LIMIT_40),
    )
    candidate = model_doc(
        one("max_per_period", id="week-soft", hardness="soft", weight=1, params=LIMIT_40)
    )

    result = compare(truth, candidate, data)

    assert [(u.truth_id, u.category, u.closest_candidate_id) for u in result.unmatched_truth] == [
        ("day", "missing", None),
        ("week", "wrong_hardness", "week-soft"),
    ]


def test_a_hardness_only_difference_is_closer_than_a_parameter_difference(data: Path) -> None:
    truth = model_doc(one("min_rest", id="rest", params={"hours": 11}))
    candidate = model_doc(
        one("min_rest", id="rest-12", params={"hours": 12}),
        one("min_rest", id="rest-soft", hardness="soft", weight=1, params={"hours": 11}),
    )

    result = compare(truth, candidate, data)

    (missed,) = result.unmatched_truth
    assert (missed.category, missed.closest_candidate_id) == ("wrong_hardness", "rest-soft")


def test_coverage_is_never_scored(data: Path) -> None:
    truth = model_doc()
    candidate = model_doc()
    candidate["constraints"][0]["scope"] = WEEKEND

    result = compare(truth, candidate, data)

    assert (result.truth_count, result.candidate_count) == (0, 0)
    assert (result.recall, result.precision) == (None, None)
    assert result.matched == result.unmatched_truth == result.unmatched_candidate == ()


BADGE = {"task_attribute": "badge_needed", "resource_attribute": "badge", "match": "contains"}


def _badge_rule(resource_type: str, task_type: str) -> dict[str, Any]:
    return model_doc(
        one("requires_attribute", params=BADGE),
        resource_attributes={"badge": {"column": "badge", "type": resource_type}},
        task_attributes={"badge_needed": {"column": "badge_needed", "type": task_type}},
    )


def test_requirement_items_compare_numbers_numerically(data: Path) -> None:
    result = compare(_badge_rule("int", "int"), _badge_rule("float", "int"), data)

    assert result.matched_count == 1


def test_requirement_items_never_equate_a_boolean_with_a_number(data: Path) -> None:
    result = compare(_badge_rule("int", "int"), _badge_rule("bool", "bool"), data)

    assert result.matched_count == 0
    assert [u.category for u in result.unmatched_truth] == ["wrong_parameters"]


def test_denote_refuses_an_unvalidated_per_resource_limit(data: Path) -> None:
    doc = model_doc(one("max_per_period", params={**LIMIT_40, "limit": {"attribute": "max_hours"}}))
    truth = loaded(doc, data)
    resources = tuple(r.model_copy(update={"attributes": {}}) for r in truth.instance.resources)
    unvalidated = truth.instance.model_copy(update={"resources": resources})

    with pytest.raises(ValueError, match="no numeric 'max_hours'; run validate_data first"):
        denote(truth.dm.constraints[1], truth.dm, unvalidated)


def test_denote_reads_the_scope_extension_and_normalized_parameters(data: Path) -> None:
    doc = model_doc(
        one(
            "max_per_period",
            params={**LIMIT_40, "limit": {"attribute": "max_hours"}},
            scope=WEEKEND,
        )
    )
    truth = loaded(doc, data)
    rule_ = truth.dm.constraints[1]

    denotation = denote(rule_, truth.dm, truth.instance)

    assert denotation.template == "max_per_period"
    assert denotation.hardness == "hard"
    assert denotation.resources == {"a1", "a2", "a3", "a4"}
    assert denotation.tasks == {"sat-am", "sun-am"}
    assert dict(denotation.params) == {
        "limits": frozenset({("a1", 2400), ("a2", 2400), ("a3", 1440), ("a4", 1440)}),
        "measure": "hours",
        "period": "week",
    }
    assert hash(denotation) == hash(denote(rule_, truth.dm, truth.instance))


# --- invalid candidates ---------------------------------------------------------------------------

INVALID = {
    "not JSON": "{",
    "structurally invalid": json.dumps(
        model_doc({**one("balance", "soft", params={"measure": "count"}), "weight": None})
    ),
    "unknown attribute in a scope": json.dumps(
        model_doc(
            one(
                "no_overlap",
                scope={"resources": [{"attribute": "team", "op": "eq", "value": "red"}]},
            )
        )
    ),
    "missing data file": json.dumps(
        model_doc(one("unavailable", params={"source": {**WINDOWS_SOURCE, "source": "off.csv"}}))
    ),
}


@pytest.mark.parametrize("text", INVALID.values(), ids=INVALID)
def test_an_invalid_candidate_recovers_nothing_and_lists_its_issues(data: Path, text: str) -> None:
    truth = loaded(model_doc(one("min_rest", params={"hours": 11}), one("no_overlap")), data)

    candidate = load_model(text, data)
    result = compare_models(truth, candidate)

    assert isinstance(candidate, InvalidModel)
    assert not result.valid
    assert result.issues == candidate.issues != ()
    assert (result.truth_count, result.candidate_count, result.matched_count) == (2, 0, 0)
    assert (result.recall, result.precision) == (0.0, None)
    assert {(u.truth_id, u.category) for u in result.unmatched_truth} == {
        ("min_rest", "missing"),
        ("no_overlap", "missing"),
    }
    document = json.loads(result.to_json())
    assert document["valid"] is False
    assert document["issues"] == [{"path": i.path, "message": i.message} for i in candidate.issues]


# --- deterministic JSON ---------------------------------------------------------------------------


def test_comparison_json_is_sorted_and_independent_of_constraint_order(data: Path) -> None:
    rules = [
        one("min_rest", params={"hours": 11}),
        one("max_per_period", params=LIMIT_40),
        one("apart", params={"resource_ids": ["a3", "a4"]}),
        one("balance", "soft", params={"measure": "count"}, scope=WEEKEND),
    ]
    truth = model_doc(*rules)
    candidate = model_doc(
        rules[3],
        {**rules[1], "hardness": "soft", "weight": 1},
        rules[0],
        one("max_consecutive_days", params={"days": 5}),
    )
    reordered = copy.deepcopy(candidate)
    reordered["constraints"][1:] = reversed(reordered["constraints"][1:])

    text = compare(truth, candidate, data).to_json()

    assert text == compare(truth, reordered, data).to_json()
    assert text.endswith("}\n") and "\r" not in text
    document = json.loads(text)
    assert list(document) == sorted(document)
    assert [m["truth_id"] for m in document["matched"]] == ["balance", "min_rest"]
    assert document["categories"] == {
        "missing": 1,
        "wrong_hardness": 1,
        "wrong_parameters": 0,
        "wrong_scope": 0,
    }
    assert (document["recall"], document["precision"]) == (0.5, 0.5)
    assert document["matched_count"] == 2


# --- properties ---------------------------------------------------------------------------------


def _loaded(case: Case, document: dict[str, Any]) -> LoadedModel:
    _, people, work, _ = case
    dm = model(document)
    return LoadedModel(dm, make_instance(dm, people, work))


@settings(max_examples=60, deadline=None)
@given(cases(), st.data())
def test_renamed_ids_and_shuffled_order_give_full_scores(case: Case, data: st.DataObject) -> None:
    document = case[0]
    renamed = copy.deepcopy(document)
    for constraint in renamed["constraints"]:
        constraint["id"] = f"renamed-{constraint['id']}"
    renamed["constraints"] = data.draw(st.permutations(renamed["constraints"]))

    result = compare_models(_loaded(case, document), _loaded(case, renamed))

    assert result.truth_count == result.candidate_count > 0
    assert (result.recall, result.precision) == (1.0, 1.0)
    assert result.unmatched_truth == result.unmatched_candidate == ()


@settings(max_examples=60, deadline=None)
@given(cases(), st.data())
def test_removing_one_scored_constraint_costs_exactly_one_match(
    case: Case, data: st.DataObject
) -> None:
    document = case[0]
    scored = [i for i, c in enumerate(document["constraints"]) if c["template"] != "coverage"]
    removed = data.draw(st.sampled_from(scored))
    fewer = copy.deepcopy(document)
    del fewer["constraints"][removed]
    n = len(scored)

    result = compare_models(_loaded(case, document), _loaded(case, fewer))

    assert result.matched_count == n - 1
    assert result.recall == (n - 1) / n
    assert result.precision == 1.0


# --- golden: a hand-made imperfect candidate for scenario 03 ------------------------------------

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "compare"
SCENARIO_03 = Path(__file__).resolve().parents[2] / "docs/validation/scenarios/03-call-centre"


def test_imperfect_candidate_for_scenario_03_matches_the_golden_byte_for_byte() -> None:
    # One missing rule, one wrong hardness, one wrong parameter and one extra rule; renamed ids and
    # attributes, a list binding, a float limit column and rewritten scopes must still match.
    scenario = load_scenario(SCENARIO_03)
    truth = LoadedModel(scenario.truth, scenario.instance)
    text = (FIXTURES / "03-imperfect.json").read_text(encoding="utf-8")

    result = compare_models(truth, load_model(text, SCENARIO_03))

    assert result.valid
    assert (result.truth_count, result.candidate_count, result.matched_count) == (8, 8, 5)
    assert (result.recall, result.precision) == (0.625, 0.625)
    assert [(u.truth_id, u.category, u.closest_candidate_id) for u in result.unmatched_truth] == [
        ("leo-shadows-priya", "missing", None),
        ("max-five-days-in-a-row", "wrong_parameters", "c3-days-in-a-row"),
        ("rest-12h", "wrong_hardness", "c5-rest"),
    ]
    assert [u.candidate_id for u in result.unmatched_candidate] == [
        "c3-days-in-a-row",
        "c5-rest",
        "c8-one-shift-a-day",
    ]
    golden = (FIXTURES / "03-imperfect.comparison.json").read_bytes()
    assert result.to_json().encode("utf-8") == golden
