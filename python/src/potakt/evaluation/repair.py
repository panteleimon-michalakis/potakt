"""One repair turn of an invalid candidate (ADR-0008 D7; TASK-005e5 decision 5).

After a model's first corpus run is complete, each invalid candidate whose answer holds text gets
one repair request: the pinned first request with two turns appended, the answer as the model's
turn and one user message that lists the issues of the first ``comparison.json``. The repaired
answer is scored like a first answer and reported apart; it never decides the gate. Library code
only: no network, no credentials.
"""

import copy
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from potakt.evaluation.records import (
    COMPARISON_FILE,
    RESPONSE_FILE,
    UNANSWERED_FILE,
    answer_of,
    parse_response,
)
from potakt.evaluation.test_a import ANTHROPIC_PREFIX, check_registered
from potakt.evaluation.transport import RunStopped
from potakt.model import json_nesting_exceeds

__all__ = [
    "REPAIR_FOOTER",
    "REPAIR_HEADER",
    "NotSent",
    "RepairNeed",
    "repair_need",
    "repair_payload",
    "repair_text",
]

# ADR-0008 D7: the first and the last line of the repair message.
REPAIR_HEADER = "The Decision Model is not valid:"
REPAIR_FOOTER = "Answer with the corrected JSON object only."
# Why an invalid candidate gets no repair request.
NotSent = Literal["no answer", "empty answer"]
CANNOT_READ = "{name}: cannot read {path}"
NOT_A_COMPARISON = "{name}: {path} is not a valid comparison"


def _one_line(text: str) -> str:
    """``text`` with every ``\\r\\n``, ``\\r`` and ``\\n`` turned into one space."""
    return text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")


def repair_text(issues: Sequence[tuple[str, str]]) -> str:
    """The user message of a repair request (ADR-0008 D7): ``REPAIR_HEADER``, one line
    ``<path>: <message>`` per issue in the given order, then ``REPAIR_FOOTER``, joined by
    ``\\n``, with no line feed at the end. A line break inside a path or a message becomes one
    space, so each issue stays on one line."""
    lines = [f"{_one_line(path)}: {_one_line(message)}" for path, message in issues]
    return "\n".join([REPAIR_HEADER, *lines, REPAIR_FOOTER])


def repair_payload(
    first: dict[str, object], model: str, answer: str, issues: Sequence[tuple[str, str]]
) -> dict[str, object]:
    """The repair request of ``model`` (ADR-0008 D7): a deep copy of the first request
    ``first`` with two messages appended. Claude: ``{"role": "assistant", "content": answer}``
    and ``{"role": "user", "content": repair_text(issues)}`` after ``messages``; Gemini:
    ``{"role": "model", "parts": [{"text": answer}]}`` and ``{"role": "user", "parts":
    [{"text": repair_text(issues)}]}`` after ``contents``. Nothing else changes: no thinking
    blocks and no thought signatures (Gemini requires them only for function calls, and the
    model's turn is the answer text). ``UnknownModelError`` for an unregistered model."""
    check_registered(model)
    payload = copy.deepcopy(first)
    message = repair_text(issues)
    if model.startswith(ANTHROPIC_PREFIX):
        turns = cast(list[object], payload["messages"])
        turns += [{"role": "assistant", "content": answer}, {"role": "user", "content": message}]
    else:
        turns = cast(list[object], payload["contents"])
        turns += [
            {"role": "model", "parts": [{"text": answer}]},
            {"role": "user", "parts": [{"text": message}]},
        ]
    return payload


@dataclass(frozen=True, slots=True)
class RepairNeed:
    """An invalid candidate that gets a repair request: the answer text exactly as
    ``answer_of`` gives it (not ``candidate.txt``) and the issues of its ``comparison.json`` as
    (path, message) pairs, in order."""

    answer: str
    issues: tuple[tuple[str, str], ...]


def repair_need(model: str, first_run: Path) -> RepairNeed | NotSent | None:
    """What one scenario directory of a first run needs (TASK-005e5 decision 5): ``None`` when
    its ``comparison.json`` says valid; ``"no answer"`` when it has ``unanswered.json``
    (ADR-0008 D6); ``"empty answer"`` when the answer text holds no character other than white
    space (the Claude API refuses a white-space-only text block), a malformed answer included;
    otherwise a ``RepairNeed``.

    ``response.json`` is read as ``rescore_run`` reads it (``parse_response``: bytes that are not
    JSON are a malformed answer). ``RunStopped`` naming the scenario and the file, never the
    error's text, for a ``response.json`` that cannot be read (an ``OSError``) and for a
    ``comparison.json`` that cannot be read or parsed: a complete first run never has one."""
    name = first_run.name
    valid, issues = _comparison(name, first_run / COMPARISON_FILE)
    if valid:
        return None
    if (first_run / UNANSWERED_FILE).is_file():
        return "no answer"
    path = first_run / RESPONSE_FILE
    try:
        raw = path.read_bytes()
    except OSError:
        raise RunStopped(CANNOT_READ.format(name=name, path=path)) from None
    text = answer_of(model, parse_response(raw)).text
    if not text.strip():
        return "empty answer"
    return RepairNeed(text, issues)


def _comparison(name: str, path: Path) -> tuple[bool, tuple[tuple[str, str], ...]]:
    """``valid`` and the (path, message) of every issue of one ``comparison.json``, whose
    nesting our code checks before ``json.loads`` (CLAUDE.md DoD 6)."""
    try:
        data = path.read_bytes()
    except OSError:
        raise RunStopped(CANNOT_READ.format(name=name, path=path)) from None
    invalid = RunStopped(NOT_A_COMPARISON.format(name=name, path=path))
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise invalid from None
    if json_nesting_exceeds(text):
        raise invalid
    try:
        document: object = json.loads(text)
    except ValueError:
        raise invalid from None
    if not isinstance(document, dict):
        raise invalid
    fields = cast(dict[str, object], document)
    valid, items = fields.get("valid"), fields.get("issues")
    if not isinstance(valid, bool) or not isinstance(items, list):
        raise invalid
    issues: list[tuple[str, str]] = []
    for item in cast(list[object], items):
        issue = cast(dict[str, object], item) if isinstance(item, dict) else {}
        where, message = issue.get("path"), issue.get("message")
        if not isinstance(where, str) or not isinstance(message, str):
            raise invalid
        issues.append((where, message))
    return valid, tuple(issues)
