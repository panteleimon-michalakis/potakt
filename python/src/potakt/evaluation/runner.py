"""The Test A runs (ADR-0007 §7 and its amendment «Transport», TASK-005c3 F2-F4).

``run_anthropic`` sends every pending scenario of the corpus to a Claude model through the Message
Batches API, ``run_gemini`` sends them to Gemini one ``generateContent`` call at a time; both
record each answer with ``write_answer``. A scenario is pending while its run
directory has no ``response.json``; a recorded answer is never sent again.

The batches of a model live in ``<model dir>/batches.json`` and each batch's raw results in
``<model dir>/batch-<id>.results.jsonl``: files next to the scenario directories, which the summary
and ``eval rescore`` ignore (they read directories only). The API key is only ever placed in a
request header; it is never written, printed or put in a URL, and every ``RunStopped`` message is
redacted of it.

A batch is never paid for twice (TASK-005c3b D1, D2, TASK-005c3c E2, E3): a ``creating`` entry is
saved before every create call, so a create whose outcome is unclear (no response, a 3xx, a 408,
any status of 500 or more, an unreadable 200, an interruption or a crash) is reconciled against
the list of batches, by one list call at least 120 seconds after the later of the call and its
answer, before anything else is sent; the next run starts with that reconciliation.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import cast

from potakt.evaluation.compare import Comparison
from potakt.evaluation.records import (
    CANDIDATE_FILE,
    COMPARISON_FILE,
    REQUEST_FILE,
    RESPONSE_FILE,
    answer_of,
    rescore_run,
    write_answer,
    write_atomic,
    write_request,
)
from potakt.evaluation.scenario import METADATA_FILE, LoadedScenario, ScenarioError, load_scenario
from potakt.evaluation.summary import (
    LONG_PROMPT_PRICES_USD_PER_MILLION,
    PRICES_USD_PER_MILLION,
    model_directory_name,
    summarize,
)
from potakt.evaluation.test_a import (
    ANTHROPIC_MAX_TOKENS,
    ANTHROPIC_PREFIX,
    STRONG_MODEL,
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
    secret_values,
    send_with_retries,
    server_hint,
    server_text,
)

__all__ = [
    "ANTHROPIC_VERSION",
    "BATCHES_URL",
    "BATCH_STATE_FILE",
    "FIRST_MODEL",
    "GEMINI_URL",
    "RUNS_PATH",
    "USER_AGENT",
    "check_order",
    "cost_upper_bound",
    "pending_scenarios",
    "recorded_scenarios",
    "run_anthropic",
    "run_gemini",
    "run_plan",
]

ANTHROPIC_VERSION = "2026-10-21"  # ADR-0007 §7 amendment «Transport»
# Like potakt.__version__; read here because potakt.evaluation never imports the potakt package.
USER_AGENT = f"potakt/{version('potakt')}"
BATCHES_URL = "https://api.anthropic.com/v1/messages/batches"
BATCH_STATE_FILE = "batches.json"
CALL_TIMEOUT_SECONDS = 60.0  # create, poll and list
RESULTS_TIMEOUT_SECONDS = 300.0
# ADR-0007 §7: Gemini ``generateContent``, the key in a header (TASK-005c3 F4).
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_TIMEOUT_SECONDS = 600.0
# The records of ADR-0007 §7 «Records», resolved from the repository root like CORPUS_PATH.
RUNS_PATH = Path(__file__).resolve().parents[4] / "docs" / "validation" / "runs" / "test-a"
# ADR-0007 §7 amendment «Order»: the strong model runs only after this one has a complete run.
FIRST_MODEL = "claude-haiku-5-5"
# TASK-005c3c E3 (ADR-0007 §7 amendment «Unclear create»): the one list call of a reconciliation
# comes this long after the later of the create call and its answer.
RECONCILE_DELAY_SECONDS = 120
ADOPTION_WINDOW = timedelta(minutes=10)  # clock skew allowed between us and the API
LIST_LIMIT = 100
# TASK-005c3b D5: an unchanged status line is shown again after this many polls (10 minutes at
# the default 60 s), so the owner sees that the run is alive.
HEARTBEAT_POLLS = 10
# TASK-005c3b D6: the endings of the stops that must not be run again blindly.
DO_NOT_RUN_AGAIN = "; do not run again: send this message to the reviewer"
TO_THE_REVIEWER = "; send {name} to the reviewer before running again"
_STATES = ("creating", "open", "done", "failed")
_ACTIVE = ("creating", "open")  # at most one entry is in one of these states
_WITH_BATCH = ("open", "done")  # the states whose entry names a batch
_ENTRY_KEYS = frozenset({"id", "scenarios", "started_at", "created_at", "state"})
_CUSTOM_ID = re.compile(r"[a-zA-Z0-9_-]{1,64}")
_BATCH_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
_INSTANT = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.\d+)?(Z|[+-]\d{2}:\d{2})")
_STARTED_AT = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
SEVERAL_CANDIDATES = (
    "{count} recent batches could be the one created by an earlier create request: {ids}; in the "
    "Claude Console (Batches) cancel all but the oldest, then run the same command again: it "
    "adopts the one left"
)
_STATUSES = ("in_progress", "canceling", "ended")
_COUNTS = ("processing", "succeeded", "errored", "expired", "canceled")
_RESULT_TYPES = ("succeeded", "errored", "expired", "canceled")
_ERROR_TYPE = re.compile(r"[a-z_]{1,64}")  # an API error type, safe to echo

Sleep = Callable[[float], object]
Echo = Callable[[str], object]


# --- files -----------------------------------------------------------------------------------


def _results_file(batch_id: str) -> str:
    return f"batch-{batch_id}.results.jsonl"


@dataclass(frozen=True, slots=True)
class _Batch:
    """One entry of ``batches.json`` (TASK-005c3b D1).

    ``state`` is ``creating`` while a create call may be in flight or its outcome is unknown,
    ``open`` once the batch exists and its results are not recorded yet, ``done`` once they are,
    and ``failed`` when it is proven that no batch was created. ``id`` and ``created_at`` (the
    batch's) are set for ``open`` and ``done`` only; ``started_at`` is the UTC instant just before
    the last create call, in whole seconds (``2026-11-02T07:15:03Z``).
    """

    id: str | None
    scenarios: tuple[str, ...]
    started_at: str
    created_at: str | None
    state: str


def _stamp(moment: datetime) -> str:
    """``moment`` as ``started_at``: UTC, whole seconds, ``Z``."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _started(text: str) -> datetime | None:
    """A ``started_at`` value read back; ``None`` if it is not one."""
    if not _STARTED_AT.fullmatch(text):
        return None
    try:
        return datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError:
        return None


def _entry(item: object) -> _Batch | None:
    """One entry of ``batches.json`` with every key and type checked for its state."""
    if not isinstance(item, dict) or frozenset(cast(dict[str, object], item)) != _ENTRY_KEYS:
        return None
    entry = cast(dict[str, object], item)
    batch_id, scenarios, state = entry["id"], entry["scenarios"], entry["state"]
    started_at, created_at = entry["started_at"], entry["created_at"]
    if not (
        state in _STATES
        and isinstance(scenarios, list)
        and all(isinstance(name, str) for name in cast(list[object], scenarios))
        and isinstance(started_at, str)
        and _started(started_at) is not None
    ):
        return None
    if state in _WITH_BATCH:
        if not (
            isinstance(batch_id, str)
            and _BATCH_ID.fullmatch(batch_id)
            and isinstance(created_at, str)
        ):
            return None
    elif batch_id is not None or created_at is not None:
        return None
    names = tuple(cast(list[str], scenarios))
    return _Batch(batch_id, names, started_at, created_at, cast(str, state))


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
    for item in cast(list[object], entries):
        batch = _entry(item)
        if batch is None:
            raise invalid
        batches.append(batch)
    if sum(batch.state in _ACTIVE for batch in batches) > 1:
        raise invalid
    return batches


def _all_batch_ids(runs_dir: Path) -> set[str]:
    """The batch ids in the ``batches.json`` of every model directory under ``runs_dir`` (it
    exists: ``batches.json`` is written before any batch is created)."""
    return {
        batch.id
        for model_dir in runs_dir.iterdir()
        if (model_dir / BATCH_STATE_FILE).is_file()
        for batch in _load_state(model_dir)
        if batch.id is not None
    }


def _save_state(model_dir: Path, batches: list[_Batch]) -> None:
    document = {
        "batches": [
            {
                "id": batch.id,
                "scenarios": list(batch.scenarios),
                "started_at": batch.started_at,
                "created_at": batch.created_at,
                "state": batch.state,
            }
            for batch in batches
        ]
    }
    text = json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    model_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(model_dir / BATCH_STATE_FILE, text.encode("utf-8"))


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


def recorded_scenarios(corpus_dir: Path, model_dir: Path) -> list[str]:
    """The corpus scenarios with a recorded ``response.json``, in UTF-8 byte order."""
    return [
        name for name in _corpus_names(corpus_dir) if (model_dir / name / RESPONSE_FILE).exists()
    ]


def _complete_records(corpus_dir: Path, model_dir: Path, model: str, echo: Echo) -> None:
    """Derive the missing ``candidate.txt`` / ``comparison.json`` of every recorded scenario
    (a run interrupted after ``response.json``) with ``rescore_run``; nothing is sent."""
    for name in recorded_scenarios(corpus_dir, model_dir):
        run_dir = model_dir / name
        if (run_dir / CANDIDATE_FILE).is_file() and (run_dir / COMPARISON_FILE).is_file():
            continue
        written = rescore_run(run_dir, _load(corpus_dir, name), model)
        echo(f"{name}: record completed ({', '.join(written)})")


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
    cancelled: bool  # ``cancel_initiated_at`` is set: the owner (or anyone) cancelled it

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _batch_status(document: object) -> _BatchStatus | None:
    batch_id = _field(document, "id")
    status = _field(document, "processing_status")
    raw_counts = _field(document, "request_counts")
    results_url = _field(document, "results_url")
    created_at = _field(document, "created_at")
    # "Specified only if cancellation was initiated": a missing field is null (no cancel).
    cancel = _field(document, "cancel_initiated_at")
    if not (
        isinstance(batch_id, str)
        and _BATCH_ID.fullmatch(batch_id)
        and status in _STATUSES
        and isinstance(raw_counts, dict)
        and (results_url is None or isinstance(results_url, str))
        and isinstance(created_at, str)
        and (cancel is None or isinstance(cancel, str))
    ):
        return None
    counts: dict[str, int] = {}
    for name in _COUNTS:
        value = cast(dict[str, object], raw_counts).get(name, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return None
        counts[name] = value
    return _BatchStatus(
        batch_id, cast(str, status), counts, results_url, created_at, cancel is not None
    )


def _unclear(status: int) -> bool:
    """TASK-005c3c E2: a create answered with ``status`` (no readable batch object) may have
    created a batch. Only a 4xx other than 408 proves that none exists (429 included)."""
    return not (400 <= status < 500 and status != 408)


def _repeated(status: int) -> bool:
    """A create answered with ``status`` (no readable batch object, and no batch found) is sent
    again: ADR-0007 §7 «Failures before an answer», plus an unreadable 200."""
    return status == 200 or status in RETRIED_STATUSES


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
    headers: dict[str, str] = field(repr=False)  # carries the key: never in a repr
    transport: Transport
    sleep: Sleep
    now: Callable[[], datetime]
    poll_seconds: float
    echo: Echo

    def run(self) -> None:
        _complete_records(self.corpus_dir, self.model_dir, self.model, self.echo)
        adopted = self._reconcile_earlier()
        pending = pending_scenarios(self.corpus_dir, self.model_dir)
        if not pending and not adopted:
            self.echo("nothing to run")
            return
        batches = _load_state(self.model_dir)
        batch = next((batch for batch in batches if batch.state == "open"), None)
        if batch is None:
            self._next_attempt(pending, batches)
            batch = self._create(pending, batches)
        elif not adopted:
            self.echo(f"resuming batch {batch.id}")
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
        was already sent in ``ATTEMPTS`` batches (``open`` or ``done`` entries: a ``failed``
        create was never a batch)."""
        sent = {
            name: sum(name in batch.scenarios for batch in batches if batch.state in _WITH_BATCH)
            for name in pending
        }
        exhausted = [name for name in pending if sent[name] >= ATTEMPTS]
        if exhausted:
            verb = "was" if len(exhausted) == 1 else "were"
            raise RunStopped(
                f"{', '.join(exhausted)} {verb} sent in {ATTEMPTS} batches without an answer; "
                "it is not sent again"
            )
        return max(sent.values()) + 1

    # -- create and reconcile (TASK-005c3b D2) --

    def _create(self, pending: list[str], batches: list[_Batch]) -> _Batch:
        """Create one batch for ``pending``; a ``creating`` entry is saved before every POST.

        TASK-005c3c E2: a 429 is no batch and is retried; any other 4xx except 408 is no batch
        and stops the run; every other answer (no response, a 3xx, a 408, any status of 500 or
        more, a 200 without a readable batch object, any other status) is unclear and
        reconciled. When the reconciliation finds no batch, the answers that ADR-0007 §7 repeats
        (no response, ``RETRIED_STATUSES``, an unreadable 200) are retried and the others stop
        the run. A ``KeyboardInterrupt`` is not caught: the ``creating`` entry on disk is
        reconciled by the next run.
        """
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
        index = len(batches)
        entry = _Batch(None, tuple(pending), "", None, "creating")
        batches = [*batches, entry]
        last = ""
        for attempt in range(1, ATTEMPTS + 1):
            # started_at is the instant just before this create call: the waits of a later
            # reconciliation, in this run or the next, count from the last request sent.
            entry = replace(entry, started_at=_stamp(self.now()))
            batches[index] = entry
            _save_state(self.model_dir, batches)
            hint: int | None = None
            response: HttpResponse | None = None
            try:
                response = self.transport.send(
                    "POST", BATCHES_URL, self.headers, body, CALL_TIMEOUT_SECONDS
                )
            except TransportError as error:
                last = str(error)
            else:
                status = _batch_status(_json(response.body)) if response.status == 200 else None
                if status is not None:
                    return self._opened(batches, index, status, adopted=False)
                last, hint = self._describe(response), server_hint(response)
                if response.status == 200:
                    last = "HTTP 200 without a readable batch object"
            if response is None or _unclear(response.status):
                # A gateway or a proxy may answer after the API created the batch.
                found = self._reconcile(entry, answered=self.now())
                if found is not None:
                    return self._opened(batches, index, found, adopted=True)
            if response is not None and not _repeated(response.status):
                batches[index] = replace(entry, state="failed")
                _save_state(self.model_dir, batches)
                raise RunStopped(f"creating a batch failed: {self._describe(response)}")
            if attempt < ATTEMPTS:
                self.sleep(retry_delay(attempt + 1, hint))
        # The last answer was a 429, or the last reconciliation found no batch.
        batches[index] = replace(entry, state="failed")
        _save_state(self.model_dir, batches)
        raise RunStopped(f"creating a batch failed {ATTEMPTS} times; last: {last}")

    def _opened(
        self, batches: list[_Batch], index: int, status: _BatchStatus, *, adopted: bool
    ) -> _Batch:
        """The entry at ``index`` becomes ``open`` with the batch's id, saved at once, before
        any poll."""
        batch = replace(batches[index], id=status.id, created_at=status.created_at, state="open")
        batches[index] = batch
        _save_state(self.model_dir, batches)
        if adopted:
            self.echo(f"adopted batch {status.id} created by an earlier create request")
        else:
            self.echo(f"batch {status.id} created for {len(batch.scenarios)} scenarios")
        return batch

    def _reconcile_earlier(self) -> bool:
        """At the start of every run: reconcile a ``creating`` entry left by an earlier run.
        ``True`` if its batch was found and adopted (the entry is now ``open``)."""
        batches = _load_state(self.model_dir)
        index = next((i for i, batch in enumerate(batches) if batch.state == "creating"), None)
        if index is None:
            return False
        entry = batches[index]
        found = self._reconcile(entry)
        if found is not None:
            self._opened(batches, index, found, adopted=True)
            return True
        batches[index] = replace(entry, state="failed")
        _save_state(self.model_dir, batches)
        self.echo(
            f"no batch was created by the earlier create request (started {entry.started_at})"
        )
        return False

    def _reconcile(self, entry: _Batch, answered: datetime | None = None) -> _BatchStatus | None:
        """The batch an unclear create of ``entry`` created, or ``None`` if it created none.

        TASK-005c3c E3: one list call, at ``t0 + RECONCILE_DELAY_SECONDS``, where ``t0`` is the
        later of ``started_at`` and ``answered`` (the instant the unclear answer arrived, in this
        run; a leftover entry of an earlier run has none). The wait is computed once from
        ``now()`` and slept in one call, or skipped if it is not positive. Exactly one candidate
        is adopted; several stop the run.
        """
        started = cast(datetime, _started(entry.started_at))  # validated or written by _stamp
        t0 = started if answered is None else max(started, answered)
        wait = (t0 + timedelta(seconds=RECONCILE_DELAY_SECONDS) - self.now()).total_seconds()
        if wait > 0:
            self.sleep(wait)
        candidates = self._candidates(entry, started)
        if len(candidates) > 1:
            ids = ", ".join(sorted(candidate.id for candidate in candidates))
            raise RunStopped(SEVERAL_CANDIDATES.format(count=len(candidates), ids=ids))
        return candidates[0] if candidates else None

    def _candidates(self, entry: _Batch, started: datetime) -> list[_BatchStatus]:
        """The listed batches that could be the one ``entry`` created: not cancelled, recorded
        by no model, with as many requests as ``entry`` has scenarios, created at most
        ``ADOPTION_WINDOW`` before ``started_at``."""
        response = self._get(f"{BATCHES_URL}?limit={LIST_LIMIT}", CALL_TIMEOUT_SECONDS)
        if response.status != 200:
            raise RunStopped(f"listing batches failed: {self._describe(response)}")
        data = _field(_json(response.body), "data")
        if not isinstance(data, list):
            raise RunStopped("listing batches failed: unexpected list object")
        # Every batch any model recorded: another model's batch of the same size is never this one.
        known = _all_batch_ids(self.model_dir.parent)
        earliest = started - ADOPTION_WINDOW
        candidates: list[_BatchStatus] = []
        for item in cast(list[object], data):
            status = _batch_status(item)
            created = _instant(status.created_at) if status is not None else None
            if status is None or created is None:
                raise RunStopped("listing batches failed: unexpected batch object")
            if (
                not status.cancelled
                and status.id not in known
                and status.total == len(entry.scenarios)
                and created >= earliest
            ):
                candidates.append(status)
        return candidates

    # -- poll, download, record (F2 steps 4, 5) --

    def _describe(self, response: HttpResponse) -> str:
        return describe_response(response, secret_values(self.headers))

    def _get(self, url: str, timeout: float) -> HttpResponse:
        return send_with_retries(
            self.transport, "GET", url, self.headers, None, timeout, sleep=self.sleep
        )

    def _poll(self, batch_id: str) -> _BatchStatus:
        """Poll until the batch ends. The status line is printed when it changes and again after
        every ``HEARTBEAT_POLLS`` polls without a change (polls, not the clock: deterministic)."""
        shown = ""
        unchanged = 0
        while True:
            response = self._get(f"{BATCHES_URL}/{batch_id}", CALL_TIMEOUT_SECONDS)
            if response.status != 200:
                raise RunStopped(f"polling batch {batch_id} failed: {self._describe(response)}")
            status = _batch_status(_json(response.body))
            if status is None or status.id != batch_id:
                raise RunStopped(f"polling batch {batch_id} failed: unexpected batch object")
            counts = ", ".join(f"{name} {status.counts[name]}" for name in _COUNTS)
            line = f"batch {batch_id}: {status.status} ({counts})"
            unchanged = unchanged + 1 if line == shown else 0
            if line != shown or unchanged == HEARTBEAT_POLLS:
                self.echo(line)
                shown, unchanged = line, 0
            if status.status == "ended":
                return status
            self.sleep(self.poll_seconds)

    def _complete(self, batch: _Batch) -> None:
        batch_id = cast(str, batch.id)  # an open entry always names its batch (_entry)
        status = self._poll(batch_id)
        if status.results_url is None:
            raise RunStopped(f"batch {batch_id} ended without a results_url")
        try:
            check_url(status.results_url)
        except ValueError as error:
            raise RunStopped(f"batch {batch_id}: results_url {server_text(str(error))}") from None
        response = self._get(status.results_url, RESULTS_TIMEOUT_SECONDS)
        if response.status != 200:
            raise RunStopped(
                f"downloading the results of batch {batch_id} failed: {self._describe(response)}"
            )
        write_atomic(self.model_dir / _results_file(batch_id), response.body)
        problems, results_problem = self._record_results(batch, batch_id, response.body)
        batches = _load_state(self.model_dir)
        done = [replace(item, state="done") if item.id == batch_id else item for item in batches]
        _save_state(self.model_dir, done)
        if problems:
            # D6: running again would send the affected scenarios in a new, paid batch.
            ending = TO_THE_REVIEWER.format(name=_results_file(batch_id)) if results_problem else ""
            raise RunStopped("; ".join(problems) + ending)

    def _record_results(self, batch: _Batch, batch_id: str, raw: bytes) -> tuple[list[str], bool]:
        """Record every usable line, in scenario order; return the problems found and whether
        one of them is in the results file itself (a malformed line or a missing result)."""
        results: dict[str, dict[str, object]] = {}
        problems: list[str] = []
        for number, line in enumerate(raw.split(b"\n"), start=1):
            if not line.strip():
                continue
            prefix = f"batch {batch_id}: results line {number}"
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
        # D3: a scenario with no line would silently stay pending and be paid for again.
        problems += [
            f"batch {batch_id}: no result for {name}"
            for name in batch.scenarios
            if name not in results
        ]
        results_problem = bool(problems)
        for name in batch.scenarios:
            if name in results:
                problem, malformed = self._record_result(name, results[name])
                if problem:
                    problems.append(problem)
                    results_problem = results_problem or malformed
        return problems, results_problem

    def _record_result(self, name: str, result: dict[str, object]) -> tuple[str | None, bool]:
        """Record one result line; return a problem, if any, and whether it is a malformed
        line."""
        run_dir = self.model_dir / name
        kind = cast(str, result["type"])
        if (run_dir / RESPONSE_FILE).exists():
            self.echo(f"{name}: already recorded")
            return None, False
        if kind == "succeeded":
            message = result.get("message")
            if not isinstance(message, dict):
                return f"{name}: a succeeded result without a message", True
            response = cast(dict[str, object], message)
            comparison = write_answer(run_dir, _load(self.corpus_dir, name), self.model, response)
            self.echo(recorded_line(name, self.model, response, comparison))
            return None, False
        if kind == "errored":
            error = _field(_field(result, "error"), "error")
            error_type, text = _field(error, "type"), _field(error, "message")
            if error_type == "invalid_request_error":
                secrets = secret_values(self.headers)
                detail = server_text(text, secrets) if isinstance(text, str) else "no message"
                # D6: the same request would fail again.
                return f"{name}: invalid_request_error: {detail}{DO_NOT_RUN_AGAIN}", False
            known_type = isinstance(error_type, str) and _ERROR_TYPE.fullmatch(error_type)
            shown = error_type if known_type else "unknown error"
            self.echo(f"{name}: errored ({shown}); pending")
            return None, False
        self.echo(f"{name}: {kind}; pending")  # expired or canceled
        return None, False


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


# --- the Gemini runner (F4) ------------------------------------------------------------------


def gemini_headers(key: str) -> dict[str, str]:
    """The request headers of a ``generateContent`` call: the key only in ``x-goog-api-key``,
    never in the URL (no ``?key=``)."""
    return {"x-goog-api-key": key, "content-type": "application/json", "user-agent": USER_AGENT}


class _JsonObjectOnly:
    """A 200 whose body is not a JSON object counts as no response (retried like one)."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    def send(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
    ) -> HttpResponse:
        response = self._transport.send(method, url, headers, body, timeout)
        if response.status == 200 and not isinstance(_json(response.body), dict):
            raise TransportError("HTTP 200 without a JSON object")
        return response


def _run_gemini(
    model: str,
    corpus_dir: Path,
    model_dir: Path,
    headers: dict[str, str],
    transport: Transport,
    sleep: Sleep,
    echo: Echo,
) -> None:
    _complete_records(corpus_dir, model_dir, model, echo)
    pending = pending_scenarios(corpus_dir, model_dir)
    if not pending:
        echo("nothing to run")
        return
    url = GEMINI_URL.format(model=model)
    checked = _JsonObjectOnly(transport)
    for name in pending:
        _prepare(corpus_dir, model_dir, model, name)
        body = (model_dir / name / REQUEST_FILE).read_bytes()  # exactly the recorded request
        try:
            response = send_with_retries(
                checked, "POST", url, headers, body, GEMINI_TIMEOUT_SECONDS, sleep=sleep
            )
        except RunStopped as error:
            raise RunStopped(f"{name}: {error}") from None
        if response.status != 200:
            raise RunStopped(f"{name}: {describe_response(response, secret_values(headers))}")
        document = cast(dict[str, object], _json(response.body))
        comparison = write_answer(model_dir / name, _load(corpus_dir, name), model, document)
        echo(recorded_line(name, model, document, comparison))


def run_gemini(
    model: str,
    corpus_dir: Path,
    runs_dir: Path,
    key: str,
    transport: Transport,
    *,
    sleep: Sleep,
    echo: Echo,
) -> None:
    """Run every pending scenario of ``corpus_dir`` on a Gemini model, one ``generateContent``
    call at a time in UTF-8 byte order (TASK-005c3 F4), and record each answer under
    ``runs_dir / model_directory_name(model)``.

    A 200 with a JSON object is recorded, a blocked prompt included (an invalid answer). Returns
    once every corpus scenario is recorded; raises ``RunStopped`` (never containing ``key``)
    when the run must stop, and running again sends only the scenarios still pending. Raises
    ``ValueError`` for a model that is not a registered Gemini model.
    """
    check_registered(model)
    if model.startswith(ANTHROPIC_PREFIX):
        raise ValueError(f"{model} is not a Gemini model")
    model_dir = runs_dir / model_directory_name(model)
    try:
        _run_gemini(model, corpus_dir, model_dir, gemini_headers(key), transport, sleep, echo)
    except RunStopped as error:
        raise RunStopped(redact(str(error), [key])) from None


# --- before a run: the order rule and the plan (F5) ------------------------------------------


def check_order(model: str, runs_dir: Path, corpus_dir: Path) -> None:
    """ADR-0007 §7 amendment «Order»: ``RunStopped`` for ``claude-opus-5-5`` unless
    ``claude-haiku-5-5`` has a complete run in ``runs_dir``; every other model may run."""
    if model != STRONG_MODEL:
        return
    first_dir = runs_dir / model_directory_name(FIRST_MODEL)
    if first_dir.is_dir():
        summary = summarize(first_dir, corpus_dir)
        missing = list(dict.fromkeys([*summary.missing, *summary.not_scored]))
    else:
        missing = _corpus_names(corpus_dir)
    if missing:
        raise RunStopped(
            f"{model} runs only after {FIRST_MODEL} has a complete run "
            f"(missing: {', '.join(missing)})"
        )


def cost_upper_bound(model: str, corpus_dir: Path, scenarios: list[str]) -> float:
    """USD: for each scenario, the UTF-8 bytes of system + user as input tokens (a byte is at
    least one token's worth of text) and ``max_tokens`` as output tokens, at the pinned batch
    prices of a Claude model. A scenario whose bytes pass a long-prompt threshold is priced at
    that tier, so the bound holds whatever the token count."""
    price_in, price_out = PRICES_USD_PER_MILLION[model]
    long_prompt = LONG_PROMPT_PRICES_USD_PER_MILLION.get(model)
    total = 0.0
    for name in scenarios:
        prompt = render_test_a(_load(corpus_dir, name))
        input_bytes = len(prompt.system.encode("utf-8")) + len(prompt.user.encode("utf-8"))
        prices = (price_in, price_out)
        if long_prompt is not None and input_bytes > long_prompt[0]:
            prices = long_prompt[1]
        total += input_bytes * prices[0] + ANTHROPIC_MAX_TOKENS[model] * prices[1]
    return total / 1_000_000


def run_plan(model: str, corpus_dir: Path, runs_dir: Path) -> str:
    """What ``eval run`` would do for ``model``, one fact per line, ending with ``\\n``. Reads
    no key and sends nothing."""
    model_dir = runs_dir / model_directory_name(model)
    pending = pending_scenarios(corpus_dir, model_dir)
    recorded = recorded_scenarios(corpus_dir, model_dir)
    lines = [
        f"Test A run plan: {model}",
        f"Runs directory: {model_dir}",
        f"Pending ({len(pending)}): {', '.join(pending) or 'none'}",
        f"Recorded ({len(recorded)}): {', '.join(recorded) or 'none'}",
    ]
    if model.startswith(ANTHROPIC_PREFIX):
        # One entry at most is creating or open (_load_state); the dry run never lists batches.
        for batch in _load_state(model_dir):
            count = len(batch.scenarios)
            if batch.state == "open":
                lines.append(f"Open batch: {batch.id} ({count} scenarios), resumed by --yes")
            elif batch.state == "creating":
                lines.append(
                    f"Unconfirmed create request: started {batch.started_at} ({count} scenarios); "
                    "--yes checks for its batch before sending anything"
                )
        cost = cost_upper_bound(model, corpus_dir, pending)
        lines.append(
            f"Cost upper bound: ${cost:.4f} ({len(pending)} requests; input: the UTF-8 bytes of "
            f"the prompt, output: max_tokens {ANTHROPIC_MAX_TOKENS[model]}; batch prices pinned "
            "on 2026-11-02)"
        )
    else:
        lines.append("Cost upper bound: $0 (free tier)")
    lines.append("Run again with --yes to send.")
    return "\n".join(lines) + "\n"
