"""Validation issues and their conversion from Pydantic errors (ADR-0004, validation levels)."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

from pydantic import ValidationError


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One problem in a Decision Model: a JSON pointer into the DM and a readable message."""

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


class DecisionModelError(Exception):
    """Raised when a Decision Model or its data cannot be loaded; carries every issue found."""

    def __init__(self, issues: Iterable[ValidationIssue]) -> None:
        self.issues: tuple[ValidationIssue, ...] = tuple(issues)
        super().__init__("\n".join(str(issue) for issue in self.issues))


def json_pointer(parts: Iterable[str | int]) -> str:
    """Render path segments as an RFC 6901 JSON pointer ("/" for the document root)."""
    escaped = [str(part).replace("~", "~0").replace("/", "~1") for part in parts]
    return "/" + "/".join(escaped)


def issues_from_pydantic(error: ValidationError, document: object) -> list[ValidationIssue]:
    """Convert Pydantic errors into issues whose paths point into ``document``.

    Pydantic locations also contain union tags (the ``template`` of a constraint) and union
    member names; walking the input document drops those so the path is a real JSON pointer.
    Messages of errors that land on the same path (alternatives of a union) are merged.
    """
    merged: dict[str, list[str]] = {}
    for detail in error.errors(include_url=False):
        path = json_pointer(_document_path(detail["loc"], document))
        messages = merged.setdefault(path, [])
        if detail["msg"] not in messages:
            messages.append(detail["msg"])
    return [ValidationIssue(path, "; ".join(messages)) for path, messages in merged.items()]


def _document_path(loc: tuple[str | int, ...], document: object) -> list[str | int]:
    parts: list[str | int] = []
    node: object = document
    for position, item in enumerate(loc):
        is_last = position == len(loc) - 1
        if isinstance(node, dict):
            mapping = cast("dict[str, Any]", node)
            if isinstance(item, str) and item in mapping:
                parts.append(item)
                node = mapping[item]
            elif mapping.get("template") == item:
                continue  # discriminated-union tag
            elif is_last:
                parts.append(item)  # a missing required field
                node = None
        elif isinstance(node, list) and isinstance(item, int):
            items = cast("list[Any]", node)
            if 0 <= item < len(items):
                parts.append(item)
                node = items[item]
        # Anything else is an internal marker (union member, validator name): not in the document.
    return parts
