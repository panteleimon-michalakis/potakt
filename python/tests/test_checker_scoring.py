"""Scoring, result structure and JSON output of the checker (ADR-0005 §5)."""

import json
import typing

import pytest
from checker_helpers import make_dm, make_instance, make_solution, res, rule, run, task

from potakt.checker import EVALUATORS, CheckResult, Score, check
from potakt.model import Constraint, DecisionModel

RESOURCES = [res("r1", contract="full_time"), res("r2", contract="part_time")]
TASKS = [
    task("t1", "2026-11-02T07:00", "2026-11-02T15:00", demand=2),
    task("t2", "2026-11-02T10:00", "2026-11-02T18:00"),
]


def scored_dm() -> DecisionModel:
    return make_dm(
        rule("no_overlap"),
        rule("max_per_period", "soft", 4, params={"measure": "count", "period": "day", "limit": 1}),
        rule("preference", "soft", 3, params={"kind": "prefer"}, id="likes-all"),
        rule("apart", "soft", 2, params={"resource_ids": ["r1", "r2"]}),
    )


def test_every_rule_is_listed_in_model_order_even_without_violations() -> None:
    dm = scored_dm()

    result = run(dm, RESOURCES, TASKS, {"t1": ["r1", "r2"]})

    assert [r.rule_id for r in result.rules] == [
        "coverage",
        "no_overlap",
        "max_per_period",
        "likes-all",
        "apart",
    ]
    assert [r.template for r in result.rules] == [
        "coverage",
        "no_overlap",
        "max_per_period",
        "preference",
        "apart",
    ]
    assert result.rule("no_overlap").amount == 0
    assert result.rule("no_overlap").violations == ()


def test_hard_sums_amounts_and_soft_sums_weighted_amounts() -> None:
    dm = scored_dm()

    result = run(dm, RESOURCES, TASKS, {"t1": ["r1", "r2"], "t2": ["r1"]})

    # coverage 0; no_overlap 1 (r1: t1/t2); max_per_period 1 (r1 has 2 on Monday) x 4;
    # prefer 3 assignments x 3 as a reward; apart 1 (t1) x 2.
    assert [(r.rule_id, r.hardness, r.weight, r.amount, r.contribution) for r in result.rules] == [
        ("coverage", "hard", None, 0, 0),
        ("no_overlap", "hard", None, 1, 1),
        ("max_per_period", "soft", 4, 1, 4),
        ("likes-all", "soft", 3, 3, -9),
        ("apart", "soft", 2, 1, 2),
    ]
    assert result.score == Score(hard=1, soft=-3)
    assert not result.score.feasible


def test_feasible_means_no_hard_amount() -> None:
    dm = scored_dm()

    result = run(dm, [*RESOURCES, res("r3")], TASKS, {"t1": ["r1", "r2"], "t2": ["r3"]})

    assert result.score == Score(hard=0, soft=-7)
    assert result.score.feasible


def test_scores_compare_lexicographically_hard_first() -> None:
    assert Score(hard=0, soft=100) < Score(hard=1, soft=-100)
    assert Score(hard=1, soft=-5) < Score(hard=1, soft=0)
    assert sorted([Score(2, 0), Score(0, 3), Score(0, -1)]) == [
        Score(0, -1),
        Score(0, 3),
        Score(2, 0),
    ]


def test_every_template_has_an_evaluator() -> None:
    union = typing.get_args(typing.get_args(Constraint)[0])
    templates = {typing.get_args(member.model_fields["template"].annotation)[0] for member in union}

    assert set(EVALUATORS) == templates
    assert len(templates) == 13


def test_unknown_rule_id_raises_key_error() -> None:
    result = run(scored_dm(), RESOURCES, TASKS, {})

    with pytest.raises(KeyError, match="nope"):
        result.rule("nope")


def test_check_rejects_solutions_with_unknown_ids() -> None:
    dm = scored_dm()
    instance = make_instance(dm, RESOURCES, TASKS)

    with pytest.raises(ValueError, match="validate_solution"):
        check(dm, instance, make_solution(dm, {"t9": ["r1"]}))
    with pytest.raises(ValueError, match="validate_solution"):
        check(dm, instance, make_solution(dm, {"t1": ["r9"]}))


# --- JSON ------------------------------------------------------------------------------------


def json_result() -> CheckResult:
    return run(scored_dm(), RESOURCES, TASKS, {"t1": ["r1"], "t2": ["r1"]})


def test_result_serializes_to_a_plain_json_document() -> None:
    document = json_result().to_dict()

    assert document["score"] == {"hard": 2, "soft": -2, "feasible": False}
    coverage = document["rules"][0]
    assert coverage == {
        "rule_id": "coverage",
        "template": "coverage",
        "hardness": "hard",
        "weight": None,
        "amount": 1,
        "contribution": 1,
        "violations": [
            {
                "resource_id": None,
                "task_ids": ["t1"],
                "period": None,
                "amount": 1,
                "message": "task t1: 1 assigned, demand 2",
            }
        ],
    }
    assert json.loads(json_result().to_json()) == document


def test_json_output_is_byte_identical_for_the_same_inputs() -> None:
    first = json_result().to_json()
    second = json_result().to_json()

    assert first == second
    assert first.endswith("}\n")


def test_json_keeps_non_ascii_text() -> None:
    dm = make_dm(rule("no_overlap"))
    tasks = [
        task("πρωί", "2026-11-02T07:00", "2026-11-02T15:00"),
        task("νύχτα", "2026-11-02T14:00", "2026-11-02T22:00"),
    ]

    text = run(dm, RESOURCES, tasks, {"πρωί": ["r1"], "νύχτα": ["r1"]}).to_json()

    assert "r1: νύχτα overlaps πρωί" in text
