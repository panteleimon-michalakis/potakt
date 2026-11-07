"""Test A answers and run records (ADR-0007 §7 and its amendment «Answer text», «Records»).

An ``Answer`` is what a raw API response says: its text and how it stopped. The candidate and its
comparison are derived from ``response.json`` alone, so a run can be rescored offline after a
harness fix without a new request. Library code only: no network, no credentials.

Every write is atomic: the bytes go to ``<name>.tmp`` in the same directory, which then replaces
``<name>`` (``os.replace``); the file and, where possible, the directory are synced to disk
(``os.fsync``). Bytes, not text mode, so line endings are LF on every OS; a leftover ``*.tmp`` is
simply overwritten by the next write. An open or a replace that Windows refuses for a moment
(``PermissionError``) is retried, about 3.3 s at most (TASK-005c3g G3); an answer whose replace
still fails stays in ``response.json.tmp``.
"""

import json
import os
import time
from collections.abc import Callable
from contextlib import suppress
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
    "REFUSED_ATTEMPTS",
    "REGISTERED_MODELS",
    "REQUEST_FILE",
    "RESPONSE_FILE",
    "UNSAVED_RESPONSE_FILE",
    "Answer",
    "answer_of",
    "refused_pause",
    "rescore_run",
    "retry_refused",
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
# TASK-005c3g G5: where ``write_answer`` keeps an answer whose replace failed.
UNSAVED_RESPONSE_FILE = RESPONSE_FILE + _TEMPORARY_SUFFIX
# TASK-005c3g G3: attempts of an open, replace or remove that Windows refuses (PermissionError).
REFUSED_ATTEMPTS = 10
ANTHROPIC_NORMAL_END = "end_turn"
GEMINI_NORMAL_END = "STOP"

Sleep = Callable[[float], object]


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

    Never raises for any response: a field with the wrong type, or a text or stop that is not
    valid Unicode, gives ``Answer("", "malformed", False)``. ``model`` must be registered
    (``UnknownModelError`` otherwise); Claude models read the Anthropic Messages shape, Gemini
    the ``generateContent`` shape.
    """
    check_registered(model)
    try:
        answer = _anthropic(response) if model.startswith(ANTHROPIC_PREFIX) else _gemini(response)
    except _Malformed:
        return _MALFORMED
    # ADR-0007 §7 amendment of TASK-005c3d (H8): text or a stop that is not valid Unicode (an
    # unpaired surrogate, which json.loads accepts) is malformed.
    if _surrogate(answer.text) or _surrogate(answer.stop):
        return _MALFORMED
    return answer


def _surrogate(text: str) -> bool:
    """``text`` holds a code point in U+D800-U+DFFF: not valid Unicode, no UTF-8 for it."""
    return any("\ud800" <= char <= "\udfff" for char in text)


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
    write_atomic(path, _response_bytes(response), keep=True)  # TASK-005c3g G5
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
    """UTF-8 JSON with sorted keys; if the response holds an unpaired surrogate (no UTF-8 exists
    for it), ASCII JSON with every non-ASCII character escaped instead, which ``json.loads``
    turns back into exactly the same object (TASK-005c3d H8)."""
    text = json.dumps(response, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError:
        escaped = json.dumps(response, indent=2, ensure_ascii=True, sort_keys=True) + "\n"
        return escaped.encode("ascii")


def refused_pause(attempt: int) -> float:
    """Seconds slept after the failed attempt ``attempt`` (0-based) of ``retry_refused``."""
    return min(0.05 * 2**attempt, 0.5)


def retry_refused[T](action: Callable[[], T], sleep: Sleep | None = None) -> T:
    """``action()``, attempted up to ``REFUSED_ATTEMPTS`` times while it raises
    ``PermissionError`` (TASK-005c3g G3): Windows refuses to open, replace or remove a file that
    another program holds for a moment (WinError 5 or 32: an antivirus, an indexer, an editor).
    ``refused_pause`` is slept between two attempts, about 3.3 s in all (``sleep``, by default
    ``time.sleep``); the last ``PermissionError`` is raised. Any other exception is raised at once.
    The same on every operating system."""
    pause = time.sleep if sleep is None else sleep
    for attempt in range(REFUSED_ATTEMPTS - 1):
        try:
            return action()
        except PermissionError:
            pause(refused_pause(attempt))
    return action()


def write_atomic(
    path: Path, data: bytes, *, sleep: Sleep | None = None, keep: bool = False
) -> None:
    """Write ``data`` to ``path`` atomically and durably (see the module docstring); also used by
    the runner.

    TASK-005c3d H11: the temporary file is written, flushed, ``os.fsync``-ed and closed before
    ``os.replace``; then the directory is synced where the OS allows it. A failure before the
    replace (the file fsync included) removes the temporary file and is raised: a write that may
    not be durable is never reported as done.

    TASK-005c3g G3: the open of the temporary file and ``os.replace`` are retried while they raise
    ``PermissionError`` (``retry_refused``, with ``sleep``). A replace that still fails removes the
    temporary file and is raised, unless ``keep``: then the complete, synced temporary file stays
    (G5: an answer that could not be saved is never lost; the next run stops on it).
    """
    temporary = path.with_name(path.name + _TEMPORARY_SUFFIX)
    handle = retry_refused(lambda: temporary.open("wb"), sleep)
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise
    try:
        retry_refused(lambda: os.replace(temporary, path), sleep)
    except BaseException:
        if not keep:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
        raise
    _sync_directory(path.parent)


def _sync_directory(directory: Path) -> None:
    """Make a replace in ``directory`` durable: ``os.fsync`` of the directory itself (POSIX).
    Any ``OSError`` is ignored: Windows cannot open a directory this way, and its ``os.replace``
    is a single file-system operation."""
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        with suppress(OSError):
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
