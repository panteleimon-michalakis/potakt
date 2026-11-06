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
import math
import os
import re
from collections.abc import Callable
from contextlib import suppress
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
    EFFORT,
    GEMINI_MAX_OUTPUT_TOKENS,
    INSTRUCTION_PATH,
    STRONG_MODEL,
    Prompt,
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
    "GEMINI_MODEL_URL",
    "GEMINI_URL",
    "LOCK_FILE",
    "MODELS_URL",
    "RUNS_PATH",
    "USER_AGENT",
    "RunLock",
    "audit_batches",
    "check_order",
    "corpus_scenarios",
    "cost_upper_bound",
    "lock_text",
    "pending_scenarios",
    "preflight",
    "recorded_scenarios",
    "run_anthropic",
    "run_gemini",
    "run_plan",
    "take_lock",
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
# TASK-005c3d H6: the batch list is read page by page back to an instant, at most this many pages.
LIST_PAGES = 10
# TASK-005c3d H4: no reconcile sleep is longer than this, whatever ``started_at`` says.
MAX_RECONCILE_WAIT_SECONDS = RECONCILE_DELAY_SECONDS + CALL_TIMEOUT_SECONDS
WAITING = "waiting {seconds} s before looking for a batch the create request may have made"
FUTURE_STOP = (
    "batches.json has an entry started at {started_at}, in the future: the clock or the file is "
    "wrong; send batches.json to the reviewer"
)
TOO_MANY_BATCHES = (
    f"more than {LIST_PAGES * LIST_LIMIT} batches were created recently in this workspace; look "
    "in the Claude Console (Batches) and send this message to the reviewer"
)
# TASK-005c3b D5: an unchanged status line is shown again after this many polls (10 minutes at
# the default 60 s), so the owner sees that the run is alive.
HEARTBEAT_POLLS = 10
# TASK-005c3c E4 (ADR-0007 §7 amendment «Held batch»): the ending of the stop that holds a batch,
# and the stop of every later run until the owner releases it.
HELD = "; batch {id} is held: send batch-{id}.results.jsonl and this message to the reviewer"
HELD_STOP = (
    "batch {id} is held for the reviewer: {problems}; after the reviewer's decision, run the "
    "same command with --release {id}"
)
# TASK-005c3d H12: a held batch stores and prints at most this many problems, then one more line.
MAX_PROBLEMS = 20
MORE_PROBLEMS = "\N{HORIZONTAL ELLIPSIS} and {count} more problems (see batch-{id}.results.jsonl)"
# TASK-005c3c E5: the lock file of ``eval run --yes`` in the runs directory of all models.
LOCK_FILE = ".eval-run.lock"
_LOCK_SHOWN = 120  # characters of a lock file shown in a message
_STATES = ("creating", "open", "held", "done", "failed")
_ACTIVE = ("creating", "open")  # at most one entry is in one of these states
_WITH_BATCH = ("open", "held", "done")  # the states whose entry names a batch
_ENTRY_KEYS = frozenset({"id", "scenarios", "started_at", "created_at", "state", "problems"})
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
    ``held`` (TASK-005c3c E4) when the batch ended and its results were recorded where possible
    but ``problems`` need the reviewer, and ``failed`` when it is proven that no batch was
    created. ``id`` and ``created_at`` (the batch's) are set for ``open``, ``held`` and ``done``
    only; ``started_at`` is the UTC instant just before the last create call, in whole seconds
    (``2026-11-02T07:15:03Z``). ``problems`` is empty except for a held batch, and for a ``done``
    one that was held and then released.
    """

    id: str | None
    scenarios: tuple[str, ...]
    started_at: str
    created_at: str | None
    state: str
    problems: tuple[str, ...] = ()


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
    started_at, created_at, problems = entry["started_at"], entry["created_at"], entry["problems"]
    if not (
        state in _STATES
        and isinstance(scenarios, list)
        and all(isinstance(name, str) for name in cast(list[object], scenarios))
        and isinstance(started_at, str)
        and _started(started_at) is not None
        and isinstance(problems, list)
        and all(
            isinstance(text, str) and text and text.isprintable()
            for text in cast(list[object], problems)
        )
    ):
        return None
    texts = tuple(cast(list[str], problems))
    # E4: problems only on a held batch (never empty) or on a done one (held, then released);
    # H12: at most MAX_PROBLEMS and the line that counts the others.
    if (state == "held" and not texts) or (texts and state not in ("held", "done")):
        return None
    if len(texts) > MAX_PROBLEMS + 1:
        return None
    if state in _WITH_BATCH:
        if not (
            isinstance(batch_id, str)
            and _BATCH_ID.fullmatch(batch_id)
            and isinstance(created_at, str)
            and _instant(created_at) is not None  # TASK-005c3e J2
        ):
            return None
    elif batch_id is not None or created_at is not None:
        return None
    names = tuple(cast(list[str], scenarios))
    return _Batch(batch_id, names, started_at, created_at, cast(str, state), texts)


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
                "problems": list(batch.problems),
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


def corpus_scenarios(corpus_dir: Path) -> list[str]:
    """The corpus scenarios (subdirectories with a ``scenario.json``) in UTF-8 byte order."""
    return _corpus_names(corpus_dir)


UNEXPECTED_RECORD = "{name}: could not be recorded ({kind})"
UNEXPECTED_COMPLETION = (
    "{name}: the record could not be completed ({kind}); send this message to the reviewer"
)
TO_REVIEWER = "; send this message to the reviewer"


def _complete_records(corpus_dir: Path, model_dir: Path, model: str, echo: Echo) -> None:
    """Derive the missing ``candidate.txt`` / ``comparison.json`` of every recorded scenario
    (a run interrupted after ``response.json``) with ``rescore_run``; nothing is sent. An
    unexpected exception stops the run naming only its type (TASK-005c3d H9)."""
    for name in recorded_scenarios(corpus_dir, model_dir):
        run_dir = model_dir / name
        if (run_dir / CANDIDATE_FILE).is_file() and (run_dir / COMPARISON_FILE).is_file():
            continue
        scenario = _load(corpus_dir, name)
        try:
            written = rescore_run(run_dir, scenario, model)
        except Exception as error:  # never its text: it may hold anything (H9)
            kind = type(error).__name__
            raise RunStopped(UNEXPECTED_COMPLETION.format(name=name, kind=kind)) from None
        echo(f"{name}: record completed ({', '.join(written)})")


def _load(corpus_dir: Path, name: str) -> LoadedScenario:
    try:
        return load_scenario(corpus_dir / name)
    except ScenarioError as error:
        raise RunStopped(f"{name}: invalid scenario: {error.issues[0]}") from None


def _render(corpus_dir: Path, name: str) -> Prompt:
    """The Test A prompt of one scenario; ``RunStopped`` (TASK-005c3d H10) if the instruction
    file cannot be read (the wording of ``eval prompt``) or the bundle cannot be built (for
    example an undecodable CSV file)."""
    scenario = _load(corpus_dir, name)
    try:
        return render_test_a(scenario)
    except OSError as error:
        path = error.filename or INSTRUCTION_PATH
        reason = error.strerror or type(error).__name__
        raise RunStopped(f"cannot read the instruction file {path}: {reason}") from None
    except ValueError as error:
        raise RunStopped(f"{name}: {error}") from None


def _prepare(corpus_dir: Path, model_dir: Path, model: str, name: str) -> dict[str, object]:
    """Write ``request.json`` of one scenario (the ``eval prompt`` payload) and return it."""
    payload = request_payload(_render(corpus_dir, name), model)
    path = model_dir / name / REQUEST_FILE
    try:
        write_request(model_dir / name, payload)
    except FileExistsError as error:
        raise RunStopped(str(error)) from None
    except OSError as error:
        reason = error.strerror or type(error).__name__
        raise RunStopped(f"cannot write {path}: {reason}") from None
    return payload


def recorded_line(name: str, model: str, response: object, comparison: Comparison) -> str:
    """``<id>: recorded (<stop>, valid|invalid, recall <r:.3f>)`` (F2 step 5); the stop is a
    server string, shown through ``server_text`` (TASK-005c3d H8)."""
    stop = server_text(answer_of(model, response).stop)
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
    """The batch object ``document``; ``None`` if a field the runner reads is missing or odd,
    ``created_at`` included: it must be an RFC 3339 instant (``_instant``, TASK-005c3e J2), so a
    stored ``created_at`` is always one, and printable."""
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
        and _instant(created_at) is not None
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


@dataclass(frozen=True, slots=True)
class _Listed:
    """A batch of the list, with its ``created_at`` read as an instant."""

    status: _BatchStatus
    created: datetime


def _read_page(document: object) -> tuple[list[_Listed], bool]:
    """One page of the batch list and its ``has_more`` (TASK-005c3d H6); ``ValueError``
    (``unexpected list object`` or ``unexpected batch object``) unless ``data`` is a list of
    readable batches and ``has_more`` a boolean. Shared by the reconciliation, the audit and
    ``eval preflight`` (TASK-005c3e J7)."""
    data = _field(document, "data")
    if not isinstance(data, list):
        raise ValueError("unexpected list object")
    page: list[_Listed] = []
    for item in cast(list[object], data):
        status = _batch_status(item)
        if status is None:
            raise ValueError("unexpected batch object")
        created = cast(datetime, _instant(status.created_at))  # read by _batch_status
        page.append(_Listed(status, created))
    has_more = _field(document, "has_more")
    if not isinstance(has_more, bool):
        raise ValueError("unexpected list object")
    return page, has_more


def _t0(entry: _Batch, answered: datetime | None) -> datetime:
    """The instant a reconciliation of ``entry`` counts from (TASK-005c3c E3, TASK-005c3d H3):
    the later of ``started_at`` and the unclear answer in this run; for a leftover entry of an
    earlier run (no answer known) ``started_at + CALL_TIMEOUT_SECONDS``, the latest instant its
    create call could have ended."""
    started = cast(datetime, _started(entry.started_at))  # validated or written by _stamp
    if answered is None:
        return started + timedelta(seconds=CALL_TIMEOUT_SECONDS)
    return max(started, answered)


def _is_candidate(listed: _Listed, entry: _Batch, t_end: datetime, known: set[str]) -> bool:
    """TASK-005c3d H2: ``listed`` could be the batch an unclear create of ``entry`` made: not
    cancelled, recorded by no model (``known``), with as many requests as ``entry`` has
    scenarios, and created in the closed window from ``started_at - ADOPTION_WINDOW`` to
    ``t_end + ADOPTION_WINDOW``. ``t_end`` is ``_t0`` of the reconciliation. The one rule of the
    reconciliation and of the audit."""
    started = cast(datetime, _started(entry.started_at))
    return (
        not listed.status.cancelled
        and listed.status.id not in known
        and listed.status.total == len(entry.scenarios)
        and started - ADOPTION_WINDOW <= listed.created <= t_end + ADOPTION_WINDOW
    )


# --- the Anthropic batch runner --------------------------------------------------------------


def anthropic_headers(key: str) -> dict[str, str]:
    """The request headers of every Message Batches call (F2): the key as ``Authorization:
    Bearer <key>``, the documented method (TASK-005c3d H7); no ``anthropic-workspace-id``, as
    the key is restricted to one workspace."""
    return {
        "authorization": f"Bearer {key}",
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
    release: str | None = None

    def run(self) -> None:
        if self.release is not None:
            self._release(self.release)
        # E4: a held batch stops every run before anything is reconciled, polled or sent.
        held = next((b for b in _load_state(self.model_dir) if b.state == "held"), None)
        if held is not None:
            raise RunStopped(HELD_STOP.format(id=held.id, problems="; ".join(held.problems)))
        self._check_started_at()
        _complete_records(self.corpus_dir, self.model_dir, self.model, self.echo)
        adopted = self._reconcile_earlier()
        pending = pending_scenarios(self.corpus_dir, self.model_dir)
        batches = _load_state(self.model_dir)
        batch = next((batch for batch in batches if batch.state == "open"), None)
        if not pending and batch is None:
            self.echo("nothing to run")
            return
        if batch is None:
            self._next_attempt(pending, batches)
            batch = self._create(pending, batches)
        elif not adopted:
            self.echo(f"resuming batch {batch.id}")
        if not pending:
            # E6: an open batch is resumed and closed even when nothing is pending.
            self._complete(batch)
            self.echo("nothing to run")
            return
        while True:
            self._complete(batch)
            pending = pending_scenarios(self.corpus_dir, self.model_dir)
            if not pending:
                return
            batches = _load_state(self.model_dir)
            self.sleep(base_delay(self._next_attempt(pending, batches)))
            batch = self._create(pending, batches)

    def _check_started_at(self) -> None:
        """TASK-005c3d H4: ``RunStopped`` before any call if an entry, of any state, started
        more than ``ADOPTION_WINDOW`` after ``now()``: the clock moved back or the file was
        edited, so no wait or window computed from it can be trusted."""
        latest = self.now() + ADOPTION_WINDOW
        for batch in _load_state(self.model_dir):
            if cast(datetime, _started(batch.started_at)) > latest:
                raise RunStopped(FUTURE_STOP.format(started_at=batch.started_at))

    def _release(self, batch_id: str) -> None:
        """E4: the held entry ``batch_id`` becomes ``done`` (its problems kept, for the record)
        after the reviewer's decision; ``RunStopped`` if no held entry has that id."""
        batches = _load_state(self.model_dir)
        index = next(
            (i for i, b in enumerate(batches) if b.state == "held" and b.id == batch_id), None
        )
        if index is None:
            raise RunStopped(f"{server_text(batch_id)} is not a held batch of {self.model}")
        batches[index] = replace(batches[index], state="done")
        _save_state(self.model_dir, batches)
        self.echo(f"batch {batch_id} released after review")

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

        TASK-005c3c E3, TASK-005c3d H2-H6: one listing (``_list_back_to``, every page back to
        the window) at ``t0 + RECONCILE_DELAY_SECONDS`` (``_t0``). The wait is computed once
        from ``now()``, capped at ``MAX_RECONCILE_WAIT_SECONDS``, announced and slept in one
        call, or skipped if it is not positive. Exactly one candidate (``_is_candidate``) is
        adopted; several stop the run.
        """
        t0 = _t0(entry, answered)
        wait = (t0 + timedelta(seconds=RECONCILE_DELAY_SECONDS) - self.now()).total_seconds()
        wait = min(wait, MAX_RECONCILE_WAIT_SECONDS)
        if wait > 0:
            self.echo(WAITING.format(seconds=math.ceil(wait)))
            self.sleep(wait)
        started = cast(datetime, _started(entry.started_at))
        listed = self._list_back_to(started - ADOPTION_WINDOW)
        # Every batch any model recorded: another model's batch of the same size is never this one.
        known = _all_batch_ids(self.model_dir.parent)
        candidates = [item.status for item in listed if _is_candidate(item, entry, t0, known)]
        if len(candidates) > 1:
            ids = ", ".join(sorted(candidate.id for candidate in candidates))
            raise RunStopped(SEVERAL_CANDIDATES.format(count=len(candidates), ids=ids))
        return candidates[0] if candidates else None

    def _list_back_to(self, instant: datetime) -> list[_Listed]:
        """TASK-005c3d H6: the listed batches, newest first, page by page until a page is the
        last one or ends with a batch created before ``instant``; at most ``LIST_PAGES`` pages.

        Each page goes through ``send_with_retries``. A page that is not a 200 with a list of
        readable batches, a ``has_more`` that is not a boolean, or a page to follow whose
        ``last_id`` is not the id of its last batch stops the run; so does a list that would need
        more pages (fail closed: never "none found").
        """
        url = f"{BATCHES_URL}?limit={LIST_LIMIT}"
        listed: list[_Listed] = []
        for _ in range(LIST_PAGES):
            response = self._get(url, CALL_TIMEOUT_SECONDS)
            if response.status != 200:
                raise RunStopped(f"listing batches failed: {self._describe(response)}")
            document = _json(response.body)
            try:
                page, has_more = _read_page(document)
            except ValueError as error:
                raise RunStopped(f"listing batches failed: {error}") from None
            listed += page
            if not has_more or (page and page[-1].created < instant):
                return listed
            last_id = _field(document, "last_id")
            if not (
                page
                and isinstance(last_id, str)
                and _BATCH_ID.fullmatch(last_id)
                and last_id == page[-1].status.id
            ):
                raise RunStopped("listing batches failed: unexpected list object")
            url = f"{BATCHES_URL}?limit={LIST_LIMIT}&after_id={last_id}"
        raise RunStopped(TOO_MANY_BATCHES)

    def audit(self) -> None:
        """TASK-005c3d H13, TASK-005c3e J3: the body of ``audit_batches`` (see there)."""
        batches = _load_state(self.model_dir)
        if not batches:
            return
        # Every model's entries, read before the wait: an invalid state stops the audit at once.
        every = _every_entry(self.model_dir.parent)
        started = [cast(datetime, _started(batch.started_at)) for batch in batches]
        target = max(started) + timedelta(seconds=AUDIT_DELAY_SECONDS)
        wait = min((target - self.now()).total_seconds(), AUDIT_DELAY_SECONDS)
        if wait > 0:
            self.echo(AUDIT_WAITING.format(seconds=math.ceil(wait)))
            self.sleep(wait)
        # J3: back to the first create request of any model, so a late duplicate of another
        # model, listed only after that model's own audit, is still named.
        earliest = min(cast(datetime, _started(batch.started_at)) for batch in every)
        earliest -= ADOPTION_WINDOW
        listed = self._list_back_to(earliest)
        known = {batch.id for batch in every if batch.id is not None}
        pending = [batch for batch in every if batch.state == "creating"]
        for item in listed:
            if (
                item.status.cancelled
                or item.status.id in known
                or item.created < earliest
                or any(_is_candidate(item, entry, _t0(entry, None), known) for entry in pending)
            ):
                continue
            created = server_text(item.status.created_at)  # an RFC 3339 instant (_instant)
            line = UNRECORDED.format(id=item.status.id, count=item.status.total, created=created)
            self.echo(line)

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
            check_url(status.results_url, self.headers)
        except ValueError as error:
            raise RunStopped(f"batch {batch_id}: results_url {server_text(str(error))}") from None
        response = self._get(status.results_url, RESULTS_TIMEOUT_SECONDS)
        if response.status != 200:
            raise RunStopped(
                f"downloading the results of batch {batch_id} failed: {self._describe(response)}"
            )
        write_atomic(self.model_dir / _results_file(batch_id), response.body)
        problems = self._record_results(batch, batch_id, response.body)
        if len(problems) > MAX_PROBLEMS:  # H12: bounded in batches.json and in every message
            more = MORE_PROBLEMS.format(count=len(problems) - MAX_PROBLEMS, id=batch_id)
            problems = [*problems[:MAX_PROBLEMS], more]
        # E4: a problem holds the batch, so no run sends its scenarios again (a new, paid batch)
        # until the owner releases it after the reviewer's decision.
        state = "held" if problems else "done"
        batches = [
            replace(item, state=state, problems=tuple(problems)) if item.id == batch_id else item
            for item in _load_state(self.model_dir)
        ]
        _save_state(self.model_dir, batches)
        if problems:
            raise RunStopped("; ".join(problems) + HELD.format(id=batch_id))

    def _record_results(self, batch: _Batch, batch_id: str, raw: bytes) -> list[str]:
        """Record every usable line, in scenario order; return the problems found (each safe to
        print and to write: cleaned, the key redacted)."""
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
        for name in batch.scenarios:
            if name in results:
                problem = self._recorded(name, results[name])
                if problem:
                    problems.append(problem)
        return problems

    def _recorded(self, name: str, result: dict[str, object]) -> str | None:
        """``_record_result``, with any exception of one result turned into its problem
        (TASK-005c3d H9), so the other results are still recorded and the batch is held, never
        left open: a ``RunStopped`` gives its message, prefixed with ``<name>: `` unless it already
        starts with it (TASK-005c3e J6), any other ``Exception`` only its type (its text could
        hold anything). A ``KeyboardInterrupt`` is not caught."""
        try:
            return self._record_result(name, result)
        except RunStopped as error:
            text = server_text(str(error), secret_values(self.headers))
            return text if text.startswith(f"{name}: ") else f"{name}: {text}"
        except Exception as error:
            return UNEXPECTED_RECORD.format(name=name, kind=type(error).__name__)

    def _record_result(self, name: str, result: dict[str, object]) -> str | None:
        """Record one result line; return a problem, if any."""
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
                # The same request would fail again: the reviewer decides.
                secrets = secret_values(self.headers)
                detail = server_text(text, secrets) if isinstance(text, str) else "no message"
                return f"{name}: invalid_request_error: {detail}"
            known_type = isinstance(error_type, str) and _ERROR_TYPE.fullmatch(error_type)
            shown = error_type if known_type else "unknown error"
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
    release: str | None = None,
) -> None:
    """Run every pending scenario of ``corpus_dir`` on a Claude model through the Message
    Batches API (TASK-005c3 F2, F3) and record the answers under
    ``runs_dir / model_directory_name(model)``.

    Returns once every corpus scenario is recorded. Raises ``RunStopped`` (its message is safe to
    print and never contains ``key``) when the run must stop; running the same call again
    resumes, except while a batch is held (TASK-005c3c E4): then every run stops before any
    call until ``release`` names that batch, which marks it ``done`` first. Raises
    ``ValueError`` for a model that is not a registered Claude model.
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
        release=release,
    )
    try:
        run.run()
    except RunStopped as error:
        raise RunStopped(redact(str(error), [key])) from None


# --- the audit of batches that no run recorded (TASK-005c3d H13) -----------------------------

AUDIT_DELAY_SECONDS = 300
AUDIT_WAITING = "waiting {seconds} s before checking for batches that no run recorded"
UNRECORDED = (
    "warning: batch {id} ({count} requests, created {created}) is recorded by no run: it may be "
    "a paid duplicate; send this warning to the reviewer, who decides whether to cancel it in the "
    "Claude Console (Batches)"
)
AUDIT_FAILED = (
    "warning: could not check for batches that no run recorded ({reason}): look in the Claude "
    "Console (Batches)"
)
AUDIT_INTERRUPTED = "audit interrupted: look in the Claude Console (Batches)"


def _every_entry(runs_dir: Path) -> list[_Batch]:
    """The entries of the ``batches.json`` of every model directory under ``runs_dir``
    (``RunStopped`` if one is invalid). Its ``creating`` entries are the pending creates: the
    next run of that model reconciles each of them and may adopt one of its candidates."""
    return [
        batch
        for model_dir in runs_dir.iterdir()
        if (model_dir / BATCH_STATE_FILE).is_file()
        for batch in _load_state(model_dir)
    ]


def audit_batches(
    model: str,
    runs_dir: Path,
    key: str,
    transport: Transport,
    *,
    sleep: Sleep,
    now: Callable[[], datetime],
    echo: Echo,
) -> None:
    """TASK-005c3d H13 (ADR-0007 §7 amendment of TASK-005c3d): after a run of a Claude model
    that called the API and was not interrupted, name every listed batch that no run recorded.

    Nothing if the model's ``batches.json`` has no entry. Otherwise it reads the ``batches.json``
    of every model, waits until ``AUDIT_DELAY_SECONDS`` after this model's latest ``started_at``
    (at most that long, announced), lists the batches back to the earliest ``started_at`` of any
    model minus ``ADOPTION_WINDOW`` (``_list_back_to``; TASK-005c3e J3) and prints one warning for
    each batch created since then that is not cancelled, is in no model's ``batches.json`` and is
    no candidate (``_is_candidate``) of a ``creating`` entry of any model, which the next run
    reconciles. It never cancels anything and never writes: the records,
    ``batches.json`` and the caller's exit code stay as they are. A failure prints a warning
    instead (its reason redacted of ``key``; an unexpected exception by its type only), and a
    ``KeyboardInterrupt`` ends only the audit. Every line goes to ``echo``.
    """
    check_registered(model)
    run = _AnthropicRun(
        model=model,
        corpus_dir=runs_dir,  # unused: the audit reads no scenario
        model_dir=runs_dir / model_directory_name(model),
        headers=anthropic_headers(key),
        transport=transport,
        sleep=sleep,
        now=now,
        poll_seconds=0,
        echo=echo,
    )
    try:
        run.audit()
    except RunStopped as error:
        echo(AUDIT_FAILED.format(reason=redact(str(error), secret_values(run.headers))))
    except KeyboardInterrupt:
        echo(AUDIT_INTERRUPTED)
    except Exception as error:  # never its text: the audit must not change the outcome
        echo(AUDIT_FAILED.format(reason=type(error).__name__))


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
        scenario = _load(corpus_dir, name)
        try:
            comparison = write_answer(model_dir / name, scenario, model, document)
            line = recorded_line(name, model, document, comparison)
        except Exception as error:  # never its text (TASK-005c3d H9)
            kind = type(error).__name__
            raise RunStopped(UNEXPECTED_RECORD.format(name=name, kind=kind) + TO_REVIEWER) from None
        echo(line)


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


# --- eval preflight: free checks of the key and the model (TASK-005c3e J7) --------------------

MODELS_URL = "https://api.anthropic.com/v1/models"  # + "/<id>": "Get a Model", free
# Gemini ``models.get``, on the API version of GEMINI_URL.
GEMINI_MODEL_URL = GEMINI_URL.removesuffix(":generateContent")
NOT_REPORTED = "not reported"
PREFLIGHT_PASSED = "preflight passed: no prompt was sent and nothing was charged"


def _failed(check: str, reason: str) -> RunStopped:
    """A failed preflight check: a definite contradiction (the reason is safe to print)."""
    return RunStopped(f"preflight failed: {check}: {reason}")


def _nested(document: object, *names: str) -> object:
    for name in names:
        document = _field(document, name)
    return document


def _positive(value: object) -> int | None:
    """``value`` if it is a positive integer; ``None`` (not reported) otherwise: the documented
    example answers ``max_tokens: 0``, a placeholder."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


@dataclass(frozen=True, slots=True)
class _Preflight:
    model: str
    headers: dict[str, str] = field(repr=False)  # carries the key: never in a repr
    transport: Transport
    sleep: Sleep
    echo: Echo

    def shown(self, value: object) -> str:
        """A server string, safe to print; ``not reported`` if it is not a non-empty string."""
        if isinstance(value, str) and value:
            return server_text(value, secret_values(self.headers))
        return NOT_REPORTED

    def get(self, url: str, check: str) -> dict[str, object]:
        """One read-only call through ``send_with_retries``: its JSON object, or the failed
        ``check`` for any other answer. ``check_url`` refusing the URL or a key header (H7) stops
        before any call."""
        try:
            response = send_with_retries(
                self.transport,
                "GET",
                url,
                self.headers,
                None,
                CALL_TIMEOUT_SECONDS,
                sleep=self.sleep,
            )
        except ValueError as error:  # its message names the host only
            raise RunStopped(str(error)) from None
        if response.status != 200:
            raise _failed(check, describe_response(response, secret_values(self.headers)))
        document = _json(response.body)
        if not isinstance(document, dict):
            raise _failed(check, "HTTP 200 without a JSON object")
        return cast(dict[str, object], document)

    def supported(self, check: str, value: object, reason: str) -> None:
        """A capability: ``false`` fails ``check``, ``true`` is supported, anything else is not
        reported."""
        if value is False:
            raise _failed(check, reason)
        self.echo(f"{check}: {'supported' if value is True else NOT_REPORTED}")

    def limit(self, check: str, value: object, wanted: int) -> None:
        """An output limit: a positive integer below ``wanted`` fails ``check``."""
        allowed = _positive(value)
        if allowed is not None and allowed < wanted:
            raise _failed(check, f"the model allows {allowed}, the run asks for {wanted}")
        shown = NOT_REPORTED if allowed is None else str(allowed)
        self.echo(f"{check}: {shown} (the run asks for {wanted})")

    def claude(self) -> None:
        document = self.get(f"{MODELS_URL}/{self.model}", "model metadata")
        self.echo("key: accepted by api.anthropic.com")
        model_id = document.get("id")
        if model_id is None:
            raise _failed("model", "no id in the answer")
        if model_id != self.model:  # an id of any other value or type is no answer for it
            if not isinstance(model_id, str):
                named = "an id that is not a string"
            else:
                named = self.shown(model_id) if model_id else "an empty id"
            raise _failed("model", f"the answer names {named}, not {self.model}")
        name, created = self.shown(document.get("display_name")), document.get("created_at")
        lifecycle = self.shown(document.get("lifecycle"))
        self.echo(
            f"model: {self.model} ({name}), created {self.shown(created)}, lifecycle {lifecycle}"
        )
        capabilities = document.get("capabilities")
        batch = _nested(capabilities, "batch", "supported")
        self.supported("batch", batch, "not supported by this model")
        effort = _nested(capabilities, "effort", EFFORT, "supported")
        reason = f"not supported by this model (the run sends effort {EFFORT})"
        self.supported(f"effort {EFFORT}", effort, reason)
        self.limit("max_tokens", document.get("max_tokens"), ANTHROPIC_MAX_TOKENS[self.model])
        listing = self.get(f"{BATCHES_URL}?limit=1", "batch list")
        try:
            page, _ = _read_page(listing)  # as the reconciliation and the audit read a page
        except ValueError as error:
            raise _failed("batch list", str(error)) from None
        count = f"{len(page)} batch" if len(page) == 1 else f"{len(page)} batches"
        self.echo(f"batch list: readable ({count} on the first page)")

    def gemini(self) -> None:
        document = self.get(GEMINI_MODEL_URL.format(model=self.model), "model metadata")
        self.echo("key: accepted by generativelanguage.googleapis.com")
        name, expected = document.get("name"), f"models/{self.model}"
        if isinstance(name, str) and name != expected:
            raise _failed("model", f"the answer names {self.shown(name)}, not {expected}")
        display, version = document.get("displayName"), document.get("version")
        self.echo(
            f"model: {self.shown(name)} ({self.shown(display)}), version {self.shown(version)}"
        )
        methods = document.get("supportedGenerationMethods")
        if isinstance(methods, list) and "generateContent" not in methods:
            raise _failed("generateContent", "not in supportedGenerationMethods")
        shown = "supported" if isinstance(methods, list) else NOT_REPORTED
        self.echo(f"generateContent: {shown}")
        output = document.get("outputTokenLimit")
        self.limit("outputTokenLimit", output, GEMINI_MAX_OUTPUT_TOKENS)


def preflight(model: str, key: str, transport: Transport, *, sleep: Sleep, echo: Echo) -> None:
    """``potakt eval preflight`` (TASK-005c3e J7, ADR-0007 §7 amendment of TASK-005c3e): check
    the key and ``model`` with free, read-only calls before a paid run. No prompt is sent, no
    file is written, no lock is taken and no audit runs.

    A Claude model: ``GET MODELS_URL/<id>`` and one page of the batch list (``limit=1``), read
    as ``_read_page`` reads one; Gemini: ``GET GEMINI_MODEL_URL``. Each call goes through
    ``send_with_retries`` with the run's headers. One line per check goes to ``echo``, every
    server string through ``server_text``. A check fails only on a definite contradiction (a
    missing field, or one of an unexpected type, is ``not reported``; the Claude ``id`` alone
    must be the requested one); the first failure raises ``RunStopped("preflight failed: <check>:
    <reason>")``. A stop of ``send_with_retries`` is a ``RunStopped`` too; every message is
    redacted of ``key``. Raises ``UnknownModelError`` for an unregistered model.
    """
    check_registered(model)
    anthropic = model.startswith(ANTHROPIC_PREFIX)
    headers = anthropic_headers(key) if anthropic else gemini_headers(key)
    checks = _Preflight(model, headers, transport, sleep, echo)
    try:
        if anthropic:
            checks.claude()
        else:
            checks.gemini()
    except RunStopped as error:
        raise RunStopped(redact(str(error), [key])) from None
    echo(PREFLIGHT_PASSED)


# --- one run at a time (TASK-005c3c E5) --------------------------------------------------------

LOCKED = (
    "another eval run is active ({contents}), or one was cut off: if no other window runs eval "
    "run, delete {path} and run again"
)


def lock_text(runs_dir: Path) -> str | None:
    """The contents of the lock file in ``runs_dir``, safe to print; ``None`` if there is none."""
    path = runs_dir / LOCK_FILE
    try:
        with path.open("rb") as handle:
            data = handle.read(_LOCK_SHOWN * 4)
    except FileNotFoundError:
        return None
    except OSError:
        return "unreadable"
    return server_text(data.decode("utf-8", "replace").strip(), limit=_LOCK_SHOWN)


@dataclass(frozen=True, slots=True)
class RunLock:
    """The lock file of one ``eval run --yes`` and the line it wrote."""

    path: Path
    line: str

    def release(self) -> str | None:
        """Remove the lock if it is still this run's. A warning, safe to print, if it now holds
        another run's line (the owner deleted it as stale while this run was still running, and
        started another) or if it cannot be removed; ``None`` otherwise (TASK-005c3c E9)."""
        try:
            current = self.path.read_bytes().decode("utf-8", "replace")
        except FileNotFoundError:
            return None
        except OSError:
            current = self.line  # unreadable: try to remove it anyway
        if current != self.line:
            shown = server_text(current.strip(), limit=_LOCK_SHOWN)
            return (
                f"the lock {self.path} now belongs to another run ({shown}); it was left in place"
            )
        try:
            self.path.unlink(missing_ok=True)
        except OSError as error:
            reason = error.strerror or type(error).__name__
            return (
                f"could not remove the lock {self.path} ({reason}): delete it before the next run"
            )
        return None


def take_lock(runs_dir: Path, model: str, now: datetime) -> RunLock:
    """Create the lock file of ``eval run --yes`` exclusively (``open("x")``: the same on every
    operating system) with one line, ``<model> <UTC start>``.

    ``RunStopped`` if it exists: another run is active, or one was cut off before it could
    remove the file (a crash); also if it cannot be created or written (then it is removed).
    The caller releases it however the run ends.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / LOCK_FILE
    line = f"{model} {_stamp(now)}\n"
    try:
        handle = path.open("x", encoding="utf-8", newline="\n")
    except OSError as error:
        if path.exists() or isinstance(error, FileExistsError):
            contents = lock_text(runs_dir)
            shown = "gone" if contents is None else contents  # removed in the meantime
            raise RunStopped(LOCKED.format(contents=shown, path=path)) from None
        raise RunStopped(f"cannot create {path}: {error.strerror or error}") from None
    try:
        with handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())  # TASK-005c3d H11: durable before the run goes on
    except OSError as error:
        with suppress(OSError):
            path.unlink(missing_ok=True)  # an empty or partial lock would block the next run
        raise RunStopped(f"cannot write {path}: {error.strerror or error}") from None
    return RunLock(path, line)


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
        prompt = _render(corpus_dir, name)
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
            elif batch.state == "held":
                lines.append(
                    f"Held batch: {batch.id} ({count} scenarios): {'; '.join(batch.problems)}; "
                    f"--yes --release {batch.id} after the reviewer's decision"
                )
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
    lock = lock_text(runs_dir)
    if lock is not None:
        lines.append(f"Lock: {lock} (another run may be active)")
    lines.append("Run again with --yes to send.")
    return "\n".join(lines) + "\n"
