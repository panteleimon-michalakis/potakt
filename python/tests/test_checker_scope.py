"""Scope conditions and attribute comparisons (ADR-0005 §2)."""

from typing import Any

import pytest

from potakt.checker.scope import condition_holds, in_scope
from potakt.model import Condition

ATTRIBUTES: dict[str, Any] = {
    "id": "r1",
    "contract": "full_time",
    "skills": ["als", "opening"],
    "level": 2,
    "rate": 1.5,
    "senior": True,
    "date": "2026-11-04",
}


def holds(attribute: str, op: str, value: object) -> bool:
    condition = Condition.model_validate({"attribute": attribute, "op": op, "value": value})
    return condition_holds(condition, ATTRIBUTES)


@pytest.mark.parametrize(
    ("attribute", "op", "value", "expected"),
    [
        ("contract", "eq", "full_time", True),
        ("contract", "ne", "full_time", False),
        ("level", "eq", 2.0, True),
        ("senior", "eq", True, True),
        ("senior", "eq", 1, False),
        ("level", "eq", True, False),
        ("level", "eq", "2", False),
        ("skills", "eq", ["als", "opening"], True),
        ("skills", "eq", ["opening", "als"], False),
        ("contract", "in", ["part_time", "full_time"], True),
        ("contract", "not_in", ["part_time", "full_time"], False),
        ("contract", "not_in", ["part_time"], True),
        ("skills", "in", ["als"], False),
        ("skills", "contains", "als", True),
        ("skills", "contains", "closing", False),
        ("contract", "contains", "full_time", True),
        ("contract", "contains", "full", False),
        ("level", "gte", 2, True),
        ("level", "lte", 1.5, False),
        ("rate", "lte", 1.5, True),
        ("date", "gte", "2026-11-04", True),
        ("date", "lte", "2026-11-03", False),
        ("level", "gte", "1", False),
        ("date", "lte", 3, False),
        ("senior", "gte", 0, False),
        ("skills", "gte", "a", False),
        ("id", "eq", "r1", True),
    ],
)
def test_condition_semantics(attribute: str, op: str, value: object, expected: bool) -> None:
    assert holds(attribute, op, value) is expected


@pytest.mark.parametrize(
    ("op", "value"),
    [
        ("eq", "x"),
        ("ne", "x"),
        ("in", ["x"]),
        ("not_in", ["x"]),
        ("contains", "x"),
        ("gte", 0),
        ("lte", 0),
    ],
)
def test_conditions_on_absent_attributes_are_false(op: str, value: object) -> None:
    assert holds("missing", op, value) is False


def test_conditions_are_combined_with_and_and_empty_means_all() -> None:
    full_time = Condition(attribute="contract", op="eq", value="full_time")
    senior = Condition(attribute="senior", op="eq", value=True)
    junior = Condition(attribute="senior", op="eq", value=False)

    assert in_scope([], ATTRIBUTES)
    assert in_scope([full_time, senior], ATTRIBUTES)
    assert not in_scope([full_time, junior], ATTRIBUTES)
