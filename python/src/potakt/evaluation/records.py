"""Test A answers and run records (ADR-0007 §7 and its amendment «Answer text», «Records»).

An ``Answer`` is what a raw API response says: its text and how it stopped. The candidate and its
comparison are derived from ``response.json`` alone, so a run can be rescored offline after a
harness fix without a new request. Library code only: no network, no credentials.

Every write is atomic: the bytes go to ``<name>.tmp`` in the same directory, which then replaces
``<name>`` (``os.replace``). Bytes, not text mode, so line endings are LF on every OS; a leftover
``*.tmp`` is simply overwritten by the next write.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from potakt.evaluation.compare import Comparison, InvalidModel, LoadedModel, compare_models
from potakt.evaluation.scenario import LoadedScenario
from potakt.evaluation.test_a import (
    ANTHROPIC_PREFIX,
    REGISTERED_MODELS,
    check_registered,
    evaluate_candidate,
    extract_candidate,
    render_payload,
)
from potakt.model.validation import issue_at

__all__ = [
    "CANDIDATE_FILE",
    "COMPARISON_FILE",
    "REGISTERED_MODELS",
    "REQUEST_FILE",
    "RESPONSE_FILE",
    "Answer",
    "answer_of",
    "rescore_run",
    "score_answer",
    "write_answer",
    "write_atomic",
    "write_request",
]

REQUEST_FILE = "request.json"
RESPONSE_FILE = "response.json"
CANDIDATE_FILE = "candidate.txt"
COMPARISON_FILE = "comparison.json"
_TEMPORARY_SUFFIX = ".tmp"
ANTHROPIC_NORMAL_END = "end_turn"
GEMINI_NORMAL_END = "STOP"


@dataclass(frozen=True, slots=True)
class Answer:
    """The answer text (text blocks joined without a separator) and how the answer stopped."""

    text: str
    stop: str
    normal: bool


_MALFORMED = Answer("", "malformed", False)


class _Malformed(Exception):
    """A field of the response has the wrong type."""


def answer_of(model: str, response: object) -> Answer:
    """The answer in a raw response of ``model`` (ADR-0007 §7 amendment «Answer text»).

    Never raises for any response: a field with the wrong type gives
    ``Answer("", "malformed", False)``. ``model`` must be registered (``UnknownModelError``
    otherwise); Claude models read the Anthropic Messages shape, Gemini the ``generateContent``
    shape.
    """
    check_registered(model)
    try:
        if model.startswith(ANTHROPIC_PREFIX):
            return _anthropic(response)
        return _gemini(response)
    except _Malformed:
        return _MALFORMED


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _Malformed
    return cast(dict[str, object], value)


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise _Malformed
    return cast(list[object], value)


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise _Malformed
    return value


def _anthropic(response: object) -> Answer:
    """The ``text`` of the content blocks of type ``text``; thinking blocks are ignored."""
    message = _object(response)
    stop = _string(message.get("stop_reason"))
    texts: list[str] = []
    for item in _list(message.get("content")):
        block = _object(item)
        if _string(block.get("type")) == "text":
            texts.append(_string(block.get("text")))
    return Answer("".join(texts), stop, stop == ANTHROPIC_NORMAL_END)


def _gemini(response: object) -> Answer:
    """The text parts of the first candidate that are not thoughts; a blocked prompt or no
    candidate is not a normal end."""
    document = _object(response)
    feedback = document.get("promptFeedback")
    if feedback is not None:
        reason = _object(feedback).get("blockReason")
        if reason is not None:
            return Answer("", f"blocked:{_string(reason)}", False)
    candidates = document.get("candidates")
    if candidates is None or not _list(candidates):
        return Answer("", "no-candidate", False)
    first = _object(_list(candidates)[0])
    stop = _string(first.get("finishReason"))
    content = first.get("content")
    parts = _list(_object(content).get("parts", [])) if content is not None else []
    texts: list[str] = []
    for item in parts:
        part = _object(item)
        thought = part.get("thought", False)
        if not isinstance(thought, bool):
            raise _Malformed
        if "text" in part and not thought:
            texts.append(_string(part["text"]))
    return Answer("".join(texts), stop, stop == GEMINI_NORMAL_END)


def score_answer(scenario: LoadedScenario, answer: Answer) -> tuple[str, Comparison]:
    """The candidate text of an answer and its comparison (ADR-0007 §7 «Answer»).

    An answer that did not end normally is an invalid candidate, whatever its text.
    """
    candidate = extract_candidate(answer.text)
    if answer.normal:
        return candidate, evaluate_candidate(scenario, candidate)
    truth = LoadedModel(scenario.truth, scenario.instance)
    message = f"the answer did not end normally (stop: {answer.stop})"
    return candidate, compare_models(truth, InvalidModel((issue_at((), message),)))


def write_request(run_dir: Path, payload: dict[str, Any]) -> None:
    """Write ``request.json`` as ``render_payload(payload)``; the same bytes again are accepted.

    Raises ``FileExistsError`` instead of replacing an existing request with different bytes.
    """
    data = render_payload(payload).encode("utf-8")
    path = run_dir / REQUEST_FILE
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"{path} exists with a different request")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(path, data)


def write_answer(
    run_dir: Path, scenario: LoadedScenario, model: str, response: object
) -> Comparison:
    """Record a response: ``response.json`` first, then the derived ``candidate.txt`` and
    ``comparison.json``. Returns the comparison.

    Raises ``FileExistsError`` if ``response.json`` exists: a recorded answer is never replaced.
    If deriving fails, ``response.json`` stays and ``rescore_run`` completes the record.
    """
    path = run_dir / RESPONSE_FILE
    if path.exists():
        raise FileExistsError(f"{path} exists: a recorded answer is never replaced")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(path, _response_bytes(response))
    candidate, comparison = score_answer(scenario, answer_of(model, response))
    write_atomic(run_dir / CANDIDATE_FILE, candidate.encode("utf-8"))
    write_atomic(run_dir / COMPARISON_FILE, comparison.to_json().encode("utf-8"))
    return comparison


def rescore_run(run_dir: Path, scenario: LoadedScenario, model: str) -> tuple[str, ...]:
    """Recompute ``candidate.txt`` and ``comparison.json`` from ``response.json`` alone.

    Writes only the files that are missing or whose bytes differ and returns their names (``()``
    when the run is unchanged). Never touches ``response.json`` or ``request.json``. A
    ``response.json`` that is not JSON is a malformed answer. Raises ``FileNotFoundError`` if
    there is no ``response.json``.
    """
    raw = (run_dir / RESPONSE_FILE).read_bytes()
    try:
        response: object = json.loads(raw)
    except (ValueError, RecursionError):
        response = None  # not JSON: a malformed answer
    candidate, comparison = score_answer(scenario, answer_of(model, response))
    written: list[str] = []
    for name, data in (
        (CANDIDATE_FILE, candidate.encode("utf-8")),
        (COMPARISON_FILE, comparison.to_json().encode("utf-8")),
    ):
        path = run_dir / name
        if not path.is_file() or path.read_bytes() != data:
            write_atomic(path, data)
            written.append(name)
    return tuple(written)


def _response_bytes(response: object) -> bytes:
    text = json.dumps(response, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    return text.encode("utf-8")


def write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (see the module docstring); also used by the runner."""
    temporary = path.with_name(path.name + _TEMPORARY_SUFFIX)
    temporary.write_bytes(data)
    os.replace(temporary, path)
