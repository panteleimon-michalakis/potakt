"""Scope conditions and attribute comparisons (ADR-0005 §2, amended by ADR-0006 §3)."""

from collections.abc import Mapping, Sequence
from typing import Literal, TypeGuard

from potakt.model import Condition
from potakt.model.common import AttributeValue, Scalar


def _scalars_equal(left: Scalar, right: Scalar) -> bool:
    """A boolean only equals a boolean; numbers compare numerically."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return left == right


def values_equal(left: AttributeValue, right: AttributeValue) -> bool:
    """The one equality of the DM (ADR-0006 §3).

    If either value is a list, both are read as lists (a scalar is a one-item list) and must have
    the same items, order and repetitions ignored; otherwise scalars compare as usual.
    """
    if isinstance(left, list) or isinstance(right, list):
        return same_items(left, right)
    return _scalars_equal(left, right)


def as_items(value: AttributeValue) -> list[Scalar]:
    """A list attribute as is; a scalar as a one-item list."""
    return list(value) if isinstance(value, list) else [value]


def contains_all(items: AttributeValue, required: AttributeValue) -> bool:
    """Every required item equals some item of ``items``."""
    have = as_items(items)
    return all(any(_scalars_equal(item, need) for item in have) for need in as_items(required))


def same_items(left: AttributeValue, right: AttributeValue) -> bool:
    """The same items, order and repetitions ignored."""
    return contains_all(left, right) and contains_all(right, left)


def _is_number(value: AttributeValue) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _ordered(op: Literal["gte", "lte"], left: AttributeValue, right: AttributeValue) -> bool:
    """``gte`` / ``lte`` of two numbers or two strings; any other combination is false."""
    if isinstance(left, str) and isinstance(right, str):
        return left >= right if op == "gte" else left <= right
    if _is_number(left) and _is_number(right):
        return left >= right if op == "gte" else left <= right
    return False


def condition_holds(condition: Condition, attributes: Mapping[str, AttributeValue]) -> bool:
    """Whether one condition holds; a condition on an absent attribute never holds."""
    if condition.attribute not in attributes:
        return False
    actual, value = attributes[condition.attribute], condition.value
    match condition.op:
        case "eq":
            return values_equal(actual, value)
        case "ne":
            return not values_equal(actual, value)
        case "in":
            return any(values_equal(actual, option) for option in as_items(value))
        case "not_in":
            return not any(values_equal(actual, option) for option in as_items(value))
        case "contains":
            return contains_all(actual, value)
        case "gte" | "lte":
            return _ordered(condition.op, actual, value)


def in_scope(conditions: Sequence[Condition], attributes: Mapping[str, AttributeValue]) -> bool:
    """All conditions hold (an empty list selects everything)."""
    return all(condition_holds(condition, attributes) for condition in conditions)
