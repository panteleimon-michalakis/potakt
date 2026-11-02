"""The Test A runs (ADR-0007 §7 and its amendment «Transport», TASK-005c3 F2, F3).

``run_anthropic`` sends every pending scenario of the corpus to a Claude model through the Message
Batches API and records each answer with ``write_answer``. A scenario is pending while its run
directory has no ``response.json``; a recorded answer is never sent again.

The batches of a model live in ``<model dir>/batches.json`` and each batch's raw results in
``<model dir>/batch-<id>.results.jsonl``: files next to the scenario directories, which the summary
and ``eval rescore`` ignore (they read directories only). The API key is only ever placed in a
request header; it is never written, printed or put in a URL, and every ``RunStopped`` message is
redacted of it.
"""

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import cast

from potakt.evaluation.compare import Comparison
from potakt.evaluation.records import RESPONSE_FILE, answer_of, write_answer, write_request
from potakt.evaluation.scenario import METADATA_FILE, LoadedScenario, ScenarioError, load_scenario
from potakt.evaluation.summary import model_directory_name
from potakt.evaluation.test_a import (
    ANTHROPIC_PREFIX,
    check_registered,
    render_test_a,
    request_payload,
)
from potakt.evaluation.transport import (
    ATTEMPTS,
    RETRIED_STATUSES,
    HttpResponse,
    RunStopped,
    Transport,
    TransportError,
    base_delay,
    check_url,
    describe_response,
    redact,
    retry_delay,
    send_with_retries,
    server_hint,
)

__all__ = [
    "ANTHROPIC_VERSION",
    "BATCHES_URL",
    "BATCH_STATE_FILE",
    "run_anthropic",
]

ANTHROPIC_VERSION = "2026-10-21"  # ADR-0007 §7 amendment «Transport»
# Like potakt.__version__; read here because potakt.evaluation never imports the potakt package.
USER_AGENT = f"potakt/{version('potakt')}"
BATCHES_URL = "https://api.anthropic.com/v1/messages/batches"
BATCH_STATE_FILE = "batches.json"
CALL_TIMEOUT_SECONDS = 60.0  # create, poll and list
RESULTS_TIMEOUT_SECONDS = 300.0
ADOPTION_WAIT_SECONDS = 10
ADOPTION_WINDOW = timedelta(minutes=10)
LIST_LIMIT = 20
_CUSTOM_ID = re.compile(r"[a-zA-Z0-9_-]{1,64}")
_BATCH_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
_INSTANT = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.\d+)?(Z|[+-]\d{2}:\d{2})")
_STATUSES = ("in_progress", "canceling", "ended")
_COUNTS = ("processing", "succeeded", "errored", "expired", "canceled")
_RESULT_TYPES = ("succeeded", "errored", "expired", "canceled")
_ERROR_TEXT_LIMIT = 300
_TEMPORARY_SUFFIX = ".tmp"

Sleep = Callable[[float], object]
Echo = Callable[[str], object]


# --- files -----------------------------------------------------------------------------------


def _write_atomic(path: Path, data: bytes) -> None:
    """Bytes to ``<name>.tmp``, then replace ``<name>`` (like ``records``): LF on every OS."""
    temporary = path.with_name(path.name + _TEMPORARY_SUFFIX)
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _results_file(batch_id: str) -> str:
    return f"batch-{batch_id}.results.jsonl"


@dataclass(frozen=True, slots=True)
class _Batch:
    """One entry of ``batches.json``: ``state`` is ``open`` until its results are recorded."""

    id: str
    scenarios: tuple[str, ...]
    created_at: str
    state: str


def _load_state(model_dir: Path) -> list[_Batch]:
    path = model_dir / BATCH_STATE_FILE
    if not path.is_file():
        return []
    invalid = RunStopped(f"{path} is not a valid batch state; it was not changed")
    try:
        document: object = json.loads(path.read_bytes())
    except (ValueError, RecursionError):
        raise invalid from None
    entries = _field(document, "batches")
    if not isinstance(entries, list):
        raise invalid
    batches: list[_Batch] = []
    for entry in cast(list[object], entries):
        batch_id = _field(entry, "id")
        scenarios = _field(entry, "scenarios")
        created_at = _field(entry, "created_at")
        state = _field(entry, "state")
        if not (
            isinstance(batch_id, str)
            and _BATCH_ID.fullmatch(batch_id)
            and isinstance(scenarios, list)
            and all(isinstance(name, str) for name in cast(list[object], scenarios))
            and isinstance(created_at, str)
            and state in ("open", "done")
        ):
            raise invalid
        names = tuple(cast(list[str], scenarios))
        batches.append(_Batch(batch_id, names, created_at, cast(str, state)))
    return batches


def _save_state(model_dir: Path, batches: list[_Batch]) -> None:
    document = {
        "batches": [
            {
                "id": batch.id,
                "scenarios": list(batch.scenarios),
                "created_at": batch.created_at,
                "state": batch.state,
            }
            for batch in batches
        ]
    }
    text = json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(model_dir / BATCH_STATE_FILE, text.encode("utf-8"))


# --- the corpus ------------------------------------------------------------------------------


def _corpus_names(corpus_dir: Path) -> list[str]:
    """The corpus scenarios (subdirectories with a ``scenario.json``) in UTF-8 byte order."""
    if not corpus_dir.is_dir():
        raise RunStopped(f"{corpus_dir} is not a directory")
    return sorted(
        (path.name for path in corpus_dir.iterdir() if (path / METADATA_FILE).is_file()),
        key=lambda name: name.encode("utf-8"),
    )


def pending_scenarios(corpus_dir: Path, model_dir: Path) -> list[str]:
    """The corpus scenarios whose run directory has no ``response.json``, in UTF-8 byte order."""
    return [
        name
        for name in _corpus_names(corpus_dir)
        if not (model_dir / name / RESPONSE_FILE).exists()
    ]


def _load(corpus_dir: Path, name: str) -> LoadedScenario:
    try:
        return load_scenario(corpus_dir / name)
    except ScenarioError as error:
        raise RunStopped(f"{name}: invalid scenario: {error.issues[0]}") from None


def _prepare(corpus_dir: Path, model_dir: Path, model: str, name: str) -> dict[str, object]:
    """Write ``request.json`` of one scenario (the ``eval prompt`` payload) and return it."""
    payload = request_payload(render_test_a(_load(corpus_dir, name)), model)
    try:
        write_request(model_dir / name, payload)
    except FileExistsError as error:
        raise RunStopped(str(error)) from None
    return payload


def recorded_line(name: str, model: str, response: object, comparison: Comparison) -> str:
    """``<id>: recorded (<stop>, valid|invalid, recall <r:.3f>)`` (F2 step 5)."""
    stop = answer_of(model, response).stop
    validity = "valid" if comparison.valid else "invalid"
    recall = "n/a" if comparison.recall is None else f"{comparison.recall:.3f}"
    return f"{name}: recorded ({stop}, {validity}, recall {recall})"


# --- API objects -----------------------------------------------------------------------------


def _json(body: bytes) -> object:
    try:
        return json.loads(body)
    except (ValueError, RecursionError):
        return None


def _field(document: object, name: str) -> object:
    return cast(dict[str, object], document).get(name) if isinstance(document, dict) else None


@dataclass(frozen=True, slots=True)
class _BatchStatus:
    """The fields of a Message Batch object that the runner reads."""

    id: str
    status: str
    counts: dict[str, int]
    results_url: str | None
    created_at: str

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _batch_status(document: object) -> _BatchStatus | None:
    batch_id = _field(document, "id")
    status = _field(document, "processing_status")
    raw_counts = _field(document, "request_counts")
    results_url = _field(document, "results_url")
    created_at = _field(document, "created_at")
    if not (
        isinstance(batch_id, str)
        and _BATCH_ID.fullmatch(batch_id)
        and status in _STATUSES
        and isinstance(raw_counts, dict)
        and (results_url is None or isinstance(results_url, str))
        and isinstance(created_at, str)
    ):
        return None
    counts: dict[str, int] = {}
    for name in _COUNTS:
        value = cast(dict[str, object], raw_counts).get(name, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return None
        counts[name] = value
    return _BatchStatus(batch_id, cast(str, status), counts, results_url, created_at)


def _instant(text: str) -> datetime | None:
    """An RFC 3339 time with at most microseconds kept; ``None`` if it is not one."""
    matched = _INSTANT.fullmatch(text)
    if matched is None:
        return None
    fraction = (matched.group(2) or "")[:7]
    zone = "+00:00" if matched.group(3) == "Z" else matched.group(3)
    try:
        return datetime.fromisoformat(matched.group(1) + fraction + zone)
    except ValueError:
        return None


# --- the Anthropic batch runner --------------------------------------------------------------


def anthropic_headers(key: str) -> dict[str, str]:
    """The request headers of every Message Batches call (F2)."""
    return {
        "x-api-key": key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
        "user-agent": USER_AGENT,
    }


@dataclass
class _AnthropicRun:
    model: str
    corpus_dir: Path
    model_dir: Path
    headers: dict[str, str]
    transport: Transport
    sleep: Sleep
    now: Callable[[], datetime]
    poll_seconds: float
    echo: Echo

    def run(self) -> None:
        pending = pending_scenarios(self.corpus_dir, self.model_dir)
        if not pending:
            self.echo("nothing to run")
            return
        batches = _load_state(self.model_dir)
        batch = next((batch for batch in batches if batch.state == "open"), None)
        if batch is not None:
            self.echo(f"resuming batch {batch.id}")
        else:
            self._next_attempt(pending, batches)
            batch = self._create(pending, batches)
        while True:
            self._complete(batch)
            pending = pending_scenarios(self.corpus_dir, self.model_dir)
            if not pending:
                return
            batches = _load_state(self.model_dir)
            self.sleep(base_delay(self._next_attempt(pending, batches)))
            batch = self._create(pending, batches)

    def _next_attempt(self, pending: list[str], batches: list[_Batch]) -> int:
        """The attempt number of the next batch for ``pending``; ``RunStopped`` if a scenario
        was already sent in ``ATTEMPTS`` batches."""
        sent = {name: sum(name in batch.scenarios for batch in batches) for name in pending}
        exhausted = [name for name in pending if sent[name] >= ATTEMPTS]
        if exhausted:
            verb = "was" if len(exhausted) == 1 else "were"
            raise RunStopped(
                f"{', '.join(exhausted)} {verb} sent in {ATTEMPTS} batches without an answer; "
                "it is not sent again"
            )
        return max(sent.values()) + 1

    # -- create (F2 step 3, F3) --

    def _create(self, pending: list[str], batches: list[_Batch]) -> _Batch:
        for name in pending:
            if not _CUSTOM_ID.fullmatch(name):
                raise RunStopped(f"{name}: not a valid custom_id for a batch request")
        requests = [
            {
                "custom_id": name,
                "params": _prepare(self.corpus_dir, self.model_dir, self.model, name),
            }
            for name in pending
        ]
        body = json.dumps({"requests": requests}, ensure_ascii=False).encode("utf-8")
        started = self.now()
        known = {batch.id for batch in batches}
        last = ""
        for attempt in range(1, ATTEMPTS + 1):
            hint: int | None = None
            lost = False
            try:
                response = self.transport.send(
                    "POST", BATCHES_URL, self.headers, body, CALL_TIMEOUT_SECONDS
                )
            except TransportError as error:
                last, lost = str(error), True
            else:
                status = _batch_status(_json(response.body)) if response.status == 200 else None
                if status is not None:
                    return self._record_batch(batches, status, pending, "created")
                if response.status == 200:
                    last, lost = "HTTP 200 without a readable batch object", True
                elif response.status in RETRIED_STATUSES:
                    last, hint = describe_response(response), server_hint(response)
                else:
                    raise RunStopped(f"creating a batch failed: {describe_response(response)}")
            if lost:
                adopted = self._adopt(started, known, len(pending))
                if adopted is not None:
                    return self._record_batch(batches, adopted, pending, "adopted")
            if attempt < ATTEMPTS:
                self.sleep(retry_delay(attempt + 1, hint))
        raise RunStopped(f"creating a batch failed {ATTEMPTS} times; last: {last}")

    def _record_batch(
        self, batches: list[_Batch], status: _BatchStatus, pending: list[str], how: str
    ) -> _Batch:
        """Write the new batch to ``batches.json`` at once, before any poll."""
        batch = _Batch(status.id, tuple(pending), status.created_at, "open")
        _save_state(self.model_dir, [*batches, batch])
        if how == "adopted":
            self.echo(f"adopted batch {batch.id} created by the interrupted request")
        else:
            self.echo(f"batch {batch.id} created for {len(pending)} scenarios")
        return batch

    def _adopt(self, started: datetime, known: set[str], count: int) -> _BatchStatus | None:
        """F3: after a create without a response, find the batch it may have created."""
        self.sleep(ADOPTION_WAIT_SECONDS)
        response = self._get(f"{BATCHES_URL}?limit={LIST_LIMIT}", CALL_TIMEOUT_SECONDS)
        if response.status != 200:
            raise RunStopped(f"listing batches failed: {describe_response(response)}")
        data = _field(_json(response.body), "data")
        if not isinstance(data, list):
            raise RunStopped("listing batches failed: unexpected list object")
        earliest = started - ADOPTION_WINDOW
        candidates: list[_BatchStatus] = []
        for item in cast(list[object], data):
            status = _batch_status(item)
            created = _instant(status.created_at) if status is not None else None
            if status is None or created is None:
                raise RunStopped("listing batches failed: unexpected batch object")
            if status.id not in known and status.total == count and created >= earliest:
                candidates.append(status)
        if len(candidates) > 1:
            ids = ", ".join(sorted(candidate.id for candidate in candidates))
            raise RunStopped(
                f"the create response was lost and {len(candidates)} recent batches could be "
                f"it: {ids}; cancel the extra ones in the Claude Console, then run again"
            )
        return candidates[0] if candidates else None

    # -- poll, download, record (F2 steps 4, 5) --

    def _get(self, url: str, timeout: float) -> HttpResponse:
        return send_with_retries(
            self.transport, "GET", url, self.headers, None, timeout, sleep=self.sleep
        )

    def _poll(self, batch_id: str) -> _BatchStatus:
        shown = ""
        while True:
            response = self._get(f"{BATCHES_URL}/{batch_id}", CALL_TIMEOUT_SECONDS)
            if response.status != 200:
                raise RunStopped(f"polling batch {batch_id} failed: {describe_response(response)}")
            status = _batch_status(_json(response.body))
            if status is None or status.id != batch_id:
                raise RunStopped(f"polling batch {batch_id} failed: unexpected batch object")
            counts = ", ".join(f"{name} {status.counts[name]}" for name in _COUNTS)
            line = f"batch {batch_id}: {status.status} ({counts})"
            if line != shown:
                self.echo(line)
                shown = line
            if status.status == "ended":
                return status
            self.sleep(self.poll_seconds)

    def _complete(self, batch: _Batch) -> None:
        status = self._poll(batch.id)
        if status.results_url is None:
            raise RunStopped(f"batch {batch.id} ended without a results_url")
        try:
            check_url(status.results_url)
        except ValueError as error:
            raise RunStopped(f"batch {batch.id}: results_url {error}") from None
        response = self._get(status.results_url, RESULTS_TIMEOUT_SECONDS)
        if response.status != 200:
            raise RunStopped(
                f"downloading the results of batch {batch.id} failed: {describe_response(response)}"
            )
        _write_atomic(self.model_dir / _results_file(batch.id), response.body)
        problems = self._record_results(batch, response.body)
        batches = _load_state(self.model_dir)
        done = [replace(item, state="done") if item.id == batch.id else item for item in batches]
        _save_state(self.model_dir, done)
        if problems:
            raise RunStopped("; ".join(problems))

    def _record_results(self, batch: _Batch, raw: bytes) -> list[str]:
        """Record every usable line, in scenario order; return the problems found."""
        results: dict[str, dict[str, object]] = {}
        problems: list[str] = []
        for number, line in enumerate(raw.split(b"\n"), start=1):
            if not line.strip():
                continue
            prefix = f"batch {batch.id}: results line {number}"
            document = _json(line)
            if not isinstance(document, dict):
                problems.append(f"{prefix} is not a JSON object")
                continue
            entry = cast(dict[str, object], document)
            custom_id, result = entry.get("custom_id"), entry.get("result")
            if not isinstance(custom_id, str) or custom_id not in batch.scenarios:
                shown = (
                    custom_id
                    if isinstance(custom_id, str) and _CUSTOM_ID.fullmatch(custom_id)
                    else "an unknown id"
                )
                problems.append(f"{prefix} names {shown}, not a scenario of the batch")
            elif custom_id in results:
                problems.append(f"{prefix} names {custom_id} twice")
            elif not isinstance(_field(result, "type"), str):
                problems.append(f"{prefix} has no result")
            elif _field(result, "type") not in _RESULT_TYPES:
                problems.append(f"{prefix} has an unknown result type")
            else:
                results[custom_id] = cast(dict[str, object], result)
        for name in batch.scenarios:
            if name in results:
                problem = self._record_result(name, results[name])
                if problem:
                    problems.append(problem)
        return problems

    def _record_result(self, name: str, result: dict[str, object]) -> str | None:
        run_dir = self.model_dir / name
        kind = cast(str, result["type"])
        if (run_dir / RESPONSE_FILE).exists():
            self.echo(f"{name}: already recorded")
            return None
        if kind == "succeeded":
            message = result.get("message")
            if not isinstance(message, dict):
                return f"{name}: a succeeded result without a message"
            response = cast(dict[str, object], message)
            comparison = write_answer(run_dir, _load(self.corpus_dir, name), self.model, response)
            self.echo(recorded_line(name, self.model, response, comparison))
            return None
        if kind == "errored":
            error = _field(_field(result, "error"), "error")
            error_type, text = _field(error, "type"), _field(error, "message")
            if error_type == "invalid_request_error":
                detail = text[:_ERROR_TEXT_LIMIT] if isinstance(text, str) else "no message"
                return f"{name}: invalid_request_error: {detail}"
            shown = error_type if isinstance(error_type, str) else "unknown error"
            self.echo(f"{name}: errored ({shown}); pending")
            return None
        self.echo(f"{name}: {kind}; pending")  # expired or canceled
        return None


def run_anthropic(
    model: str,
    corpus_dir: Path,
    runs_dir: Path,
    key: str,
    transport: Transport,
    *,
    sleep: Sleep,
    now: Callable[[], datetime],
    poll_seconds: float = 60,
    echo: Echo,
) -> None:
    """Run every pending scenario of ``corpus_dir`` on a Claude model through the Message
    Batches API (TASK-005c3 F2, F3) and record the answers under
    ``runs_dir / model_directory_name(model)``.

    Returns once every corpus scenario is recorded. Raises ``RunStopped`` (its message is safe to
    print and never contains ``key``) when the run must stop; running the same call again
    resumes. Raises ``ValueError`` for a model that is not a registered Claude model.
    """
    check_registered(model)
    if not model.startswith(ANTHROPIC_PREFIX):
        raise ValueError(f"{model} is not a Claude model")
    run = _AnthropicRun(
        model=model,
        corpus_dir=corpus_dir,
        model_dir=runs_dir / model_directory_name(model),
        headers=anthropic_headers(key),
        transport=transport,
        sleep=sleep,
        now=now,
        poll_seconds=poll_seconds,
        echo=echo,
    )
    try:
        run.run()
    except RunStopped as error:
        raise RunStopped(redact(str(error), [key])) from None
