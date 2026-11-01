"""The Test A metric: denotations, one-to-one matching and diagnostics (ADR-0007 §5, D5)."""

import copy
import json
from datetime import date
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

# `duty` equals `needs` on weekdays and also asks for an opener at the weekend.
TASKS = """id,start,end,demand,kind,needs,min_level,badge_needed,duty
mon-am,2026-11-02T07:00,2026-11-02T15:00,1,am,opening,senior,1,opening
mon-pm,2026-11-02T15:00,2026-11-02T23:00,1,pm,closing,,,closing
tue-am,2026-11-03T07:00,2026-11-03T15:00,1,am,opening,,,opening
tue-pm,2026-11-03T15:00,2026-11-03T23:00,1,pm,closing,senior,,closing
sat-am,2026-11-07T09:00,2026-11-07T17:00,2,am,,,1,opening
sun-am,2026-11-08T09:00,2026-11-08T17:00,2,am,,,,opening
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
    "duty": {"column": "duty", "type": "str"},
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
    "week vs horizon on a horizon of exactly one ISO week": (
        one("max_per_period", params={"measure": "hours", "period": "week", "limit": 40}),
        one("max_per_period", params={"measure": "hours", "period": "horizon", "limit": 40}),
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
    "window ending T23:59 vs the next day T00:00": (
        one("unavailable", params={"windows": [TUESDAY]}),
        one("unavailable", params={"windows": [{**TUESDAY, "end": "2026-11-03T23:59"}]}),
    ),
    "windows that differ only where no task is affected": (
        one("unavailable", params={"windows": [TUESDAY]}),
        one(
            "unavailable",
            params={
                "windows": [{**TUESDAY, "start": "2026-11-02T23:00", "end": "2026-11-04T07:00"}]
            },
        ),
    ),
    "redundant resource scope on unavailable": (
        one("unavailable", params={"windows": [TUESDAY]}),
        one(
            "unavailable",
            params={"windows": [TUESDAY]},
            scope={"resources": [{"attribute": "id", "op": "eq", "value": "a1"}]},
        ),
    ),
    "redundant task scope on requires_attribute": (
        one("requires_attribute", params=SKILLS),
        one(
            "requires_attribute",
            params=SKILLS,
            scope={"tasks": [{"attribute": "weekday", "op": "in", "value": ["mon", "tue"]}]},
        ),
    ),
    "redundant resource scope on requires_attribute": (
        one("requires_attribute", params=LEVEL),
        one(
            "requires_attribute",
            params=LEVEL,
            scope={"resources": [{"attribute": "level", "op": "eq", "value": "junior"}]},
        ),
    ),
    "requirement through another column and a condition, same forbidden pairs": (
        one("requires_attribute", params=SKILLS),
        one(
            "requires_attribute",
            params={**SKILLS, "task_attribute": "duty"},
            scope={"tasks": [{"attribute": "weekday", "op": "not_in", "value": ["sat", "sun"]}]},
        ),
    ),
    "redundant resource scope on apart": (
        one("apart", params={"resource_ids": ["a3", "a4"]}),
        one(
            "apart",
            params={"resource_ids": ["a3", "a4"]},
            scope={"resources": [{"attribute": "contract", "op": "eq", "value": "part"}]},
        ),
    ),
    "inert apart rules with the same pair": (
        one("apart", params={"resource_ids": ["a3", "a4"]}, scope=FULL_TIME),
        one(
            "apart",
            params={"resource_ids": ["a4", "a3"]},
            scope={"resources": [{"attribute": "id", "op": "eq", "value": "a3"}], **WEEKEND},
        ),
    ),
    "redundant resource scope on together": (
        one("together", "soft", params={"resource_ids": ["a1", "a2"]}, scope=WEEKEND),
        one(
            "together",
            "soft",
            params={"resource_ids": ["a2", "a1"]},
            scope={**FULL_TIME, **WEEKEND},
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
        one("max_per_period", params={**LIMIT_40, "period": "day"}),
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
    "window one hour earlier: one extra forbidden pair": (
        one("unavailable", params={"windows": [TUESDAY]}),
        one("unavailable", params={"windows": [{**TUESDAY, "start": "2026-11-02T22:00"}]}),
        "wrong_parameters",
    ),
    "window ending as a task starts: one forbidden pair missing": (
        one("unavailable", params={"windows": [TUESDAY]}),
        one("unavailable", params={"windows": [{**TUESDAY, "end": "2026-11-03T15:00"}]}),
        "wrong_parameters",
    ),
    "unavailable scoped away from its person": (
        one("unavailable", params={"windows": [TUESDAY]}),
        one(
            "unavailable",
            params={"windows": [TUESDAY]},
            scope={"resources": [{"attribute": "id", "op": "ne", "value": "a1"}]},
        ),
        "wrong_parameters",
    ),
    "requirement on one task fewer": (
        one("requires_attribute", params=SKILLS),
        one(
            "requires_attribute",
            params=SKILLS,
            scope={"tasks": [{"attribute": "id", "op": "ne", "value": "mon-am"}]},
        ),
        "wrong_parameters",
    ),
    "requirement for one resource fewer": (
        one("requires_attribute", params=SKILLS),
        one(
            "requires_attribute",
            params=SKILLS,
            scope={"resources": [{"attribute": "id", "op": "ne", "value": "a4"}]},
        ),
        "wrong_parameters",
    ),
    "requirement through another column that also binds the weekend": (
        one("requires_attribute", params=SKILLS),
        one("requires_attribute", params={**SKILLS, "task_attribute": "duty"}),
        "wrong_parameters",
    ),
    "apart on fewer tasks": (
        one("apart", params={"resource_ids": ["a3", "a4"]}),
        one("apart", params={"resource_ids": ["a3", "a4"]}, scope=WEEKEND),
        "wrong_parameters",
    ),
    "inert vs active apart with the same pair": (
        one("apart", params={"resource_ids": ["a3", "a4"]}),
        one("apart", params={"resource_ids": ["a3", "a4"]}, scope=FULL_TIME),
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


def test_a_per_period_rule_on_a_longer_horizon_does_not_match(data: Path) -> None:
    # One more (empty) day is one more period that min_per_period checks: different behaviour.
    daily = one("min_per_period", "soft", params={"measure": "count", "period": "day", "limit": 1})
    longer = model_doc({**daily, "id": "candidate"})
    longer["horizon"] = {"start": "2026-11-02", "end": "2026-11-09"}

    result = compare(model_doc(daily), longer, data)

    assert result.matched == ()
    assert [(u.category, u.closest_candidate_id) for u in result.unmatched_truth] == [
        ("wrong_parameters", "candidate")
    ]


def test_a_longer_horizon_does_not_change_rules_without_periods(data: Path) -> None:
    longer = model_doc(one("min_rest", params={"hours": 11}))
    longer["horizon"] = {"start": "2026-11-01", "end": "2026-11-09"}

    result = compare(model_doc(one("min_rest", params={"hours": 11})), longer, data)

    assert result.matched_count == 1


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


def test_a_boolean_binding_on_both_sides_forbids_the_same_pairs(data: Path) -> None:
    result = compare(_badge_rule("int", "int"), _badge_rule("bool", "bool"), data)

    assert result.matched_count == 1


def test_requirement_items_never_equate_a_boolean_with_a_number(data: Path) -> None:
    # Holders read `true`, the tasks require `1`: nobody qualifies, so more pairs are forbidden.
    result = compare(_badge_rule("int", "int"), _badge_rule("bool", "int"), data)

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
        "periods": ((date(2026, 11, 2), date(2026, 11, 8)),),
    }
    assert hash(denotation) == hash(denote(rule_, truth.dm, truth.instance))


def _denoted(data: Path, rule_: dict[str, Any]) -> tuple[frozenset[str], frozenset[str], Any]:
    truth = loaded(model_doc(rule_), data)
    denotation = denote(truth.dm.constraints[1], truth.dm, truth.instance)
    return denotation.resources, denotation.tasks, dict(denotation.params)


def test_denote_folds_the_scope_into_the_forbidden_pairs(data: Path) -> None:
    requirement = _denoted(data, one("requires_attribute", params=SKILLS, scope=FULL_TIME))
    absence = _denoted(data, one("unavailable", params={"source": WINDOWS_SOURCE}))

    assert requirement == (
        frozenset(),
        frozenset(),
        {"forbidden": frozenset({("a2", "mon-pm"), ("a2", "tue-pm")})},
    )
    assert absence == (
        frozenset(),
        frozenset(),
        {"forbidden": frozenset({("a1", "tue-am"), ("a1", "tue-pm")})},
    )


def test_denote_reads_a_pair_rule_as_its_pair_and_tasks_or_as_inert(data: Path) -> None:
    active = _denoted(data, one("apart", params={"resource_ids": ["a4", "a3"]}, scope=WEEKEND))
    inert = _denoted(
        data, one("together", "soft", params={"resource_ids": ["a1", "a3"]}, scope=FULL_TIME)
    )

    assert active == (
        frozenset(),
        frozenset(),
        {"resource_ids": frozenset({"a3", "a4"}), "tasks": frozenset({"sat-am", "sun-am"})},
    )
    assert inert == (
        frozenset(),
        frozenset(),
        {"resource_ids": frozenset({"a1", "a3"}), "tasks": None},
    )


# --- recall with absences merged (ADR-0007 §5.7) --------------------------------------------------

BEN_MONDAY = {"resource_id": "a2", "start": "2026-11-02T00:00", "end": "2026-11-03T00:00"}


def absence(rule_id: str, *windows: dict[str, str], hardness: str = "hard") -> dict[str, Any]:
    return one("unavailable", hardness, id=rule_id, params={"windows": list(windows)})


def recalls(result: Comparison) -> tuple[tuple[int, int], tuple[int, int]]:
    """(matched, truth) of the primary recall and of the recall with absences merged."""
    return (
        (result.matched_count, result.truth_count),
        (result.merged_matched_count, result.merged_truth_count),
    )


def test_one_rule_for_two_stated_absences_is_recovered_only_when_merged(data: Path) -> None:
    truth = model_doc(absence("ana", TUESDAY), absence("ben", BEN_MONDAY))
    candidate = model_doc(absence("both", TUESDAY, BEN_MONDAY))

    result = compare(truth, candidate, data)

    assert recalls(result) == ((0, 2), (1, 1))
    assert (result.recall, result.merged_recall) == (0.0, 1.0)


IDENTICAL_GROUPING = {
    "two absence rules on both sides": (
        [
            absence("ana", TUESDAY),
            absence("ben", BEN_MONDAY),
            one("min_rest", params={"hours": 11}),
        ],
        [absence("x", TUESDAY), absence("y", BEN_MONDAY), one("min_rest", params={"hours": 11})],
        ((3, 3), (2, 2)),
    ),
    "one absence rule on both sides": (
        [absence("both", TUESDAY, BEN_MONDAY), one("min_rest", params={"hours": 11})],
        [absence("x", BEN_MONDAY, TUESDAY), one("min_rest", params={"hours": 12})],
        ((1, 2), (1, 2)),
    ),
}


@pytest.mark.parametrize(
    ("truth_rules", "candidate_rules", "expected"),
    IDENTICAL_GROUPING.values(),
    ids=IDENTICAL_GROUPING,
)
def test_identical_grouping_gives_the_same_recall(
    data: Path,
    truth_rules: list[dict[str, Any]],
    candidate_rules: list[dict[str, Any]],
    expected: tuple[tuple[int, int], tuple[int, int]],
) -> None:
    result = compare(model_doc(*truth_rules), model_doc(*candidate_rules), data)

    assert recalls(result) == expected
    assert result.merged_recall == result.recall


MISSING_ABSENCE = {
    "one of two absences": (
        [absence("ana", TUESDAY), absence("ben", BEN_MONDAY)],
        [absence("x", TUESDAY)],
        (1, 2),
    ),
    "the only absence": ([absence("ana", TUESDAY)], [], (0, 1)),
}


@pytest.mark.parametrize(
    ("truth_rules", "candidate_rules", "primary"), MISSING_ABSENCE.values(), ids=MISSING_ABSENCE
)
def test_a_missing_absence_is_not_recovered_when_merged(
    data: Path,
    truth_rules: list[dict[str, Any]],
    candidate_rules: list[dict[str, Any]],
    primary: tuple[int, int],
) -> None:
    result = compare(model_doc(*truth_rules), model_doc(*candidate_rules), data)

    assert recalls(result) == (primary, (0, 1))
    assert result.merged_recall == 0.0


def test_absences_are_merged_per_hardness(data: Path) -> None:
    truth = model_doc(absence("ana", TUESDAY), absence("ben", BEN_MONDAY, hardness="soft"))
    merged_hard = model_doc(absence("both", TUESDAY, BEN_MONDAY))
    same = model_doc(absence("x", TUESDAY), absence("y", BEN_MONDAY, hardness="soft"))

    assert recalls(compare(truth, merged_hard, data)) == ((0, 2), (0, 2))
    assert recalls(compare(truth, same, data)) == ((2, 2), (2, 2))


def test_an_invalid_candidate_recovers_nothing_when_merged_either(data: Path) -> None:
    truth = loaded(model_doc(absence("ana", TUESDAY), absence("ben", BEN_MONDAY)), data)

    result = compare_models(truth, load_model("{", data))

    assert recalls(result) == ((0, 2), (0, 1))
    assert (result.recall, result.merged_recall) == (0.0, 0.0)


def test_the_merged_recall_is_in_the_json_next_to_the_primary_recall(data: Path) -> None:
    truth = model_doc(absence("ana", TUESDAY), absence("ben", BEN_MONDAY))
    candidate = model_doc(absence("both", TUESDAY, BEN_MONDAY))

    document = json.loads(compare(truth, candidate, data).to_json())

    assert (document["recall"], document["merged_recall"]) == (0.0, 1.0)
    assert (document["merged_matched_count"], document["merged_truth_count"]) == (1, 1)


def test_the_merged_recall_is_undefined_without_truth_constraints(data: Path) -> None:
    result = compare(model_doc(), model_doc(absence("x", TUESDAY)), data)

    assert recalls(result) == ((0, 0), (0, 0))
    assert (result.recall, result.merged_recall) == (None, None)


def test_an_extra_candidate_absence_can_lower_the_merged_recall(data: Path) -> None:
    # ADR-0007 §5.7: the merged recall is not a pure relaxation of the primary recall.
    cleo_saturday = {"resource_id": "a3", "start": "2026-11-07T00:00", "end": "2026-11-08T00:00"}
    truth = model_doc(absence("ana", TUESDAY), absence("ben", BEN_MONDAY))
    candidate = model_doc(
        absence("x", TUESDAY), absence("y", BEN_MONDAY), absence("z", cleo_saturday)
    )

    result = compare(truth, candidate, data)

    assert recalls(result) == ((2, 2), (0, 1))


def test_equal_truth_rules_count_with_their_multiplicity_when_merged(data: Path) -> None:
    # Review finding: the merged truth count must count repeated denotations like the matches do.
    twice = model_doc(
        one("min_rest", id="rest-a", params={"hours": 11}),
        one("min_rest", id="rest-b", params={"hours": 11}),
        absence("ana", TUESDAY),
        absence("ana-again", TUESDAY),
    )
    once = model_doc(one("min_rest", id="rest", params={"hours": 11}), absence("x", TUESDAY))

    itself = compare(twice, twice, data)
    fewer = compare(twice, once, data)
    invalid = compare_models(loaded(twice, data), load_model("{", data))

    assert recalls(itself) == ((4, 4), (3, 3))
    assert itself.merged_recall == 1.0
    assert recalls(fewer) == ((2, 4), (2, 3))
    assert recalls(invalid) == ((0, 4), (0, 3))


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
