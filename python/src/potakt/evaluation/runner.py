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

Every ``batches.json`` entry has the key ``ended_at`` (ADR-0008 D5): a ``held`` or ``done`` entry
keeps the batch object's ``ended_at``, verbatim, from the poll that first showed the batch ended,
when that is an RFC 3339 instant; it is ``null`` otherwise, and an odd or missing value never
stops a run. A release keeps it. It is a record only and never changes a score.

A batch is never paid for twice (TASK-005c3b D1, D2, TASK-005c3c E2, E3): a ``creating`` entry is
saved before every create call, so a create whose outcome is unclear (no response, a 3xx, a 408,
any status of 500 or more, an unreadable 200, an interruption or a crash) is reconciled against
the list of batches, by listing every page back to the window (``_list_back_to``) at least 120
seconds after the later of the call and its answer, or, for an entry left by an earlier run, at
least 180 seconds after its ``started_at``, before anything else is sent; the next run starts
with that reconciliation.
"""

import email.utils
import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import cast

import potakt.evaluation.test_a as test_a_module
from potakt.evaluation.compare import Comparison
from potakt.evaluation.records import (
    CANDIDATE_FILE,
    COMPARISON_FILE,
    RECEIVED_FILE,
    REQUEST_FILE,
    RESPONSE_FILE,
    UNANSWERED_FILE,
    UNSAVED_RESPONSE_FILE,
    answer_of,
    present,
    rescore_run,
    retry_refused,
    write_answer,
    write_atomic,
    write_request,
    write_unanswered,
)
from potakt.evaluation.repair import RepairNeed, repair_need, repair_payload
from potakt.evaluation.scenario import METADATA_FILE, LoadedScenario, ScenarioError, load_scenario
from potakt.evaluation.summary import (
    LONG_PROMPT_PRICES_USD_PER_MILLION,
    PRICES_USD_PER_MILLION,
    model_directory_name,
    unscored,
)
from potakt.evaluation.test_a import (
    ANTHROPIC_MAX_TOKENS,
    ANTHROPIC_PREFIX,
    EFFORT,
    GEMINI_MAX_OUTPUT_TOKENS,
    STRONG_MODEL,
    Prompt,
    check_registered,
    render_payload,
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
    "DEV_RUNS_PATH",
    "DEV_RUNS_ROOT",
    "FIRST_MODEL",
    "GEMINI_MODEL_URL",
    "GEMINI_URL",
    "LOCK_FILE",
    "MODELS_URL",
    "REPAIR_DIR",
    "REPEAT_DIRS",
    "RUNS_PATH",
    "RUN_KINDS",
    "USER_AGENT",
    "RunLock",
    "audit_batches",
    "check_frozen",
    "check_order",
    "check_pins",
    "check_set",
    "corpus_scenarios",
    "cost_upper_bound",
    "lock_text",
    "pending_scenarios",
    "preflight",
    "recorded_scenarios",
    "repair_cost_upper_bound",
    "run_anthropic",
    "run_directory",
    "run_gemini",
    "run_plan",
    "take_lock",
    "unanswered_scenarios",
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
# ADR-0008 D1 (amendment of TASK-005e6): the records of the development runs; a later round after a
# revision records in a directory of its own under DEV_RUNS_ROOT.
DEV_RUNS_ROOT = RUNS_PATH.parent / "dev"
DEV_RUNS_PATH = DEV_RUNS_ROOT / "round-1"
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
_ENDED = ("held", "done")  # the states whose entry may keep the batch's ended_at (ADR-0008 D5)
_ENTRY_KEYS = frozenset(
    {"id", "scenarios", "started_at", "created_at", "state", "problems", "ended_at"}
)
_CUSTOM_ID = re.compile(r"[a-zA-Z0-9_-]{1,64}")
_BATCH_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
_INSTANT = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.\d+)?(Z|[+-]\d{2}:\d{2})")
_STARTED_AT = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
SEVERAL_CANDIDATES = (
    "{count} recent batches could be the one created by an earlier create request: {ids}; in the "
    "Claude Console (Batches) cancel all but the oldest, then run the same command again: it "
    "adopts the one left; a batch that has already ended cannot be cancelled: then send this "
    "message to the reviewer"
)
# TASK-005c3h H6: the reviewer's acknowledged batches, in the runs directory of all models.
ACKNOWLEDGED_FILE = "acknowledged-batches.txt"
_ACKNOWLEDGED_LINE = re.compile(r"([A-Za-z0-9_-]{1,128})(?:[ \t]+.*)?")  # _BATCH_ID, a comment
NOT_UTF8 = "{path} is not UTF-8: write it with the command in the runbook; it was not changed"
NOT_A_BATCH_ID = "{path} line {number} is not a batch id; it was not changed"
# TASK-005c3h H6: the check of the workspace before a model's first batch.
WORKSPACE_WAITING = (
    "waiting {seconds} s before checking the workspace for batches that no run recorded"
)
WORKSPACE_TOO_MANY = (
    f"the workspace lists more than {LIST_PAGES * LIST_LIMIT} batches: this check cannot see them "
    "all; nothing was sent; send this message to the reviewer"
)
UNKNOWN_BATCHES = (
    "the workspace has {noun} {ids} that no batches.json in {runs} records and "
    "acknowledged-batches.txt does not name; nothing was sent: if the records of an earlier run "
    "were lost, restore them before running again; otherwise send this message to the reviewer"
)
UNKNOWN_SHOWN = 5  # ids named by UNKNOWN_BATCHES, then "and <n> more"
# TASK-005c3h H6 (TASK-005c3s S3-8): an answer kept in a done batch's results, its records lost.
LOST_RECORDS = (
    "{name}: its answer is in {file} but its records are missing: restore them from git before "
    "running again; nothing was sent"
)
LOST_UNREADABLE = (
    "cannot read the kept results {path}: {reason}; nothing was sent: restore it from git or send "
    "this message to the reviewer"
)
_STATUSES = ("in_progress", "canceling", "ended")
_COUNTS = ("processing", "succeeded", "errored", "expired", "canceled")
_RESULT_TYPES = ("succeeded", "errored", "expired", "canceled")
_ERROR_TYPE = re.compile(r"[a-z_]{1,64}")  # an API error type, safe to echo
# TASK-005c3g G7: the errored results that are sent again (not billed: the API's own failures);
# a result errored with any other type, an unknown one included, holds its batch.
TRANSIENT_ERRORS = frozenset({"api_error", "overloaded_error", "rate_limit_error", "timeout_error"})

Sleep = Callable[[float], object]
Echo = Callable[[str], object]


# --- the laptop clock against the API's (TASK-005c3g G2) ---------------------------------------

# The one free call whose ``Date`` header is read before any other request of a Claude run.
CLOCK_URL = f"{BATCHES_URL}?limit=1"
MAX_CLOCK_SKEW = timedelta(minutes=2)
CLOCK_STOP = (
    "the laptop clock differs from the API's by {seconds} s (more than {limit} s): sync the "
    "Windows clock (Settings, Time & language, Date & time, Sync now) and run again; nothing was "
    "sent"
)
NO_DATE = (
    "the API answer has no readable Date header, so the laptop clock cannot be checked; send this "
    "message to the reviewer"
)


def _api_time(response: HttpResponse) -> datetime | None:
    """The instant of the response's ``Date`` header (RFC 9110 §6.6.1, read with
    ``email.utils.parsedate_to_datetime``); ``None`` if it is missing, unreadable or has no time
    zone (``-0000``)."""
    text = response.headers.get("date", "")
    if not text:
        return None
    try:
        moment = email.utils.parsedate_to_datetime(text)
    except (ValueError, TypeError, IndexError, OverflowError):
        return None
    return moment if moment.tzinfo is not None else None


def _skew(response: HttpResponse, arrived: datetime) -> timedelta | None:
    """The laptop clock minus the API's, read when the answer arrived; ``None`` without a
    readable ``Date`` header."""
    api = _api_time(response)
    return None if api is None else arrived - api


def _skew_seconds(skew: timedelta) -> int:
    """The size of a skew in whole seconds, rounded up."""
    return math.ceil(abs(skew.total_seconds()))


def _clock_stop(skew: timedelta) -> str | None:
    """``CLOCK_STOP`` if the skew is beyond ``MAX_CLOCK_SKEW`` in either direction."""
    if abs(skew) <= MAX_CLOCK_SKEW:
        return None
    limit = int(MAX_CLOCK_SKEW.total_seconds())
    return CLOCK_STOP.format(seconds=_skew_seconds(skew), limit=limit)


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

    ``ended_at`` (ADR-0008 D5) is the batch object's ``ended_at``, exactly as the poll that first
    showed the batch ended sent it, when that is an RFC 3339 instant (``_instant``); it is set
    only when the entry becomes ``held`` or ``done``, kept by a release, and ``None`` on every
    other entry and when the API gave none or an odd value. It is a record only: it never
    changes a score or stops a run.
    """

    id: str | None
    scenarios: tuple[str, ...]
    started_at: str
    created_at: str | None
    state: str
    problems: tuple[str, ...] = ()
    ended_at: str | None = None


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
    ended_at = entry["ended_at"]
    # ADR-0008 D5: null, or an instant on an ended (held or done) entry only.
    if ended_at is not None and not (
        state in _ENDED and isinstance(ended_at, str) and _instant(ended_at) is not None
    ):
        return None
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
    return _Batch(batch_id, names, started_at, created_at, cast(str, state), texts, ended_at)


def _load_state(model_dir: Path) -> list[_Batch]:
    path = model_dir / BATCH_STATE_FILE
    if not present(path):  # TASK-005c3h H8: an unreadable state is never "no state"
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


# TASK-005e5 decision 3: the run kinds besides the first run, each a directory of the runs
# directory that holds one directory per model, like the runs directory itself.
REPAIR_DIR = "repair"
REPEAT_DIRS = ("repeat-2", "repeat-3")
RUN_KINDS = (REPAIR_DIR, *REPEAT_DIRS)


def run_directory(runs_dir: Path, model: str, kind: str | None) -> Path:
    """The run directory of ``model`` for a run of ``kind``: ``runs_dir / <model dir>`` for the
    first run (``None``), ``runs_dir / <kind> / <model dir>`` for ``repair``, ``repeat-2`` and
    ``repeat-3``; ``ValueError`` for any other kind. The lock and ``acknowledged-batches.txt``
    stay in ``runs_dir`` for every kind."""
    if kind is None:
        return runs_dir / model_directory_name(model)
    if kind not in RUN_KINDS:
        raise ValueError(f"{kind!r} is not a run kind (use {', '.join(RUN_KINDS)} or None)")
    return runs_dir / kind / model_directory_name(model)


def _model_directories(runs_dir: Path) -> list[Path]:
    """Every entry directly under ``runs_dir`` and directly under ``runs_dir / <kind>`` for each
    kind directory that exists (TASK-005e5 decision 3); none if ``runs_dir`` does not exist yet
    (TASK-005c3h H6: the workspace check of a first run may come before anything was written).
    Nothing deeper is listed, and a ``batches.json`` anywhere else is never read."""
    try:
        entries = list(runs_dir.iterdir())
    except FileNotFoundError:
        return []
    for kind in RUN_KINDS:
        try:
            entries += list((runs_dir / kind).iterdir())
        except (FileNotFoundError, NotADirectoryError):
            continue
    return entries


def _all_batch_ids(runs_dir: Path) -> set[str]:
    """The batch ids in the ``batches.json`` of every model directory of ``runs_dir``, of every
    run kind (``_model_directories``)."""
    return {
        batch.id
        for model_dir in _model_directories(runs_dir)
        if present(model_dir / BATCH_STATE_FILE)
        for batch in _load_state(model_dir)
        if batch.id is not None
    }


def _acknowledged(runs_dir: Path, sleep: Sleep) -> set[str]:
    """TASK-005c3h H6: the batch ids of ``<runs>/acknowledged-batches.txt``, written only on the
    reviewer's decision; a missing file is none.

    UTF-8, a byte order mark at the start ignored (TASK-005c3s S3-2: PowerShell 5.1 writes one),
    lines ending with ``\\n`` or ``\\r\\n``. Each line is blank, a ``#`` comment, or a batch id
    optionally followed by spaces or tabs and a free comment. Read with ``retry_refused``.
    ``RunStopped`` (never echoing a line's text) for bytes that are not UTF-8, a line that is
    none of these, or a file that cannot be read."""
    path = runs_dir / ACKNOWLEDGED_FILE
    try:
        raw = retry_refused(path.read_bytes, sleep)
    except FileNotFoundError:
        return set()
    except OSError as error:
        reason = error.strerror or type(error).__name__
        raise RunStopped(f"cannot read {path}: {reason}") from None
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise RunStopped(NOT_UTF8.format(path=path)) from None
    found: set[str] = set()
    for number, line in enumerate(text.split("\n"), start=1):
        stripped = line.strip(" \t\r")
        if not stripped or stripped.startswith("#"):
            continue
        matched = _ACKNOWLEDGED_LINE.fullmatch(stripped)
        # H11 review: a CR inside a line (a bare-CR line break) would hide the next id.
        if matched is None or "\r" in stripped:
            raise RunStopped(NOT_A_BATCH_ID.format(path=path, number=number))
        found.add(matched.group(1))
    return found


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
                "ended_at": batch.ended_at,
            }
            for batch in batches
        ]
    }
    text = json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    model_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(model_dir / BATCH_STATE_FILE, text.encode("utf-8"))


# --- a scenario that never gets an answer (ADR-0008 D6, TASK-005e5 decision 4) ----------------

# After ATTEMPTS batches without an answer, one more batch, at least ANSWER_WAIT after the
# started_at of the last of them; after it, the scenario counts as an invalid candidate.
LAST_BATCH = ATTEMPTS + 1
ANSWER_WAIT = timedelta(hours=24)
LAST_BATCH_WAIT = (
    "{names} {verb} sent in {attempts} batches without an answer; ADR-0008 D6 sends {pronoun} "
    "once more from {time}: run the same command again then; nothing was sent"
)
UNANSWERED_LINE = (
    "{name}: no answer after {count} batches; it counts as an invalid candidate (ADR-0008 D6)"
)


def _sent_in(name: str, batches: list[_Batch]) -> list[_Batch]:
    """The entries of ``batches`` (in file order) that sent scenario ``name`` in a batch: those
    in state ``open``, ``held`` or ``done`` that list it (a ``creating`` or ``failed`` entry
    never was a batch)."""
    return [batch for batch in batches if batch.state in _WITH_BATCH and name in batch.scenarios]


def _last_batch_from(sent: list[_Batch]) -> datetime:
    """ADR-0008 D6: the instant from which a scenario sent in ``ATTEMPTS`` batches goes into its
    last batch: ``ANSWER_WAIT`` after the ``started_at`` of the last of ``sent`` (the fourth
    batch)."""
    return cast(datetime, _started(sent[-1].started_at)) + ANSWER_WAIT  # validated (_entry)


# --- the corpus ------------------------------------------------------------------------------


def _corpus_names(corpus_dir: Path) -> list[str]:
    """The corpus scenarios (subdirectories with a ``scenario.json``) in UTF-8 byte order."""
    if not corpus_dir.is_dir():
        raise RunStopped(f"{corpus_dir} is not a directory")
    return sorted(
        (path.name for path in corpus_dir.iterdir() if (path / METADATA_FILE).is_file()),
        key=lambda name: name.encode("utf-8"),
    )


def _unanswered(run_dir: Path) -> bool:
    """TASK-005e5 decision 4 (ADR-0008 D6): the run directory has ``unanswered.json``."""
    return present(run_dir / UNANSWERED_FILE)


def pending_scenarios(corpus_dir: Path, model_dir: Path) -> list[str]:
    """The corpus scenarios whose run directory has neither ``response.json`` nor
    ``unanswered.json`` (ADR-0008 D6, TASK-005e5 decision 4), in UTF-8 byte order (``present``:
    one that cannot be stat'ed raises, TASK-005c3h H8)."""
    return [
        name
        for name in _corpus_names(corpus_dir)
        if not present(model_dir / name / RESPONSE_FILE) and not _unanswered(model_dir / name)
    ]


def recorded_scenarios(corpus_dir: Path, model_dir: Path) -> list[str]:
    """The corpus scenarios with a recorded ``response.json`` and no ``unanswered.json``, in
    UTF-8 byte order."""
    return [
        name
        for name in _corpus_names(corpus_dir)
        if present(model_dir / name / RESPONSE_FILE) and not _unanswered(model_dir / name)
    ]


def unanswered_scenarios(corpus_dir: Path, model_dir: Path) -> list[str]:
    """TASK-005e5 decision 4: the corpus scenarios whose run directory has ``unanswered.json``
    (ADR-0008 D6), in UTF-8 byte order."""
    return [name for name in _corpus_names(corpus_dir) if _unanswered(model_dir / name)]


# --- what a run sends (TASK-005e5 decision 6) --------------------------------------------------


def _first_dir(runs_dir: Path, model: str, kind: str | None) -> Path | None:
    """The first run's directory that a repair run reads (ADR-0008 D7); ``None`` for every
    other kind, whose requests are the pinned first requests themselves."""
    return run_directory(runs_dir, model, None) if kind == REPAIR_DIR else None


def _repair_needs(corpus_dir: Path, model: str, first_dir: Path) -> dict[str, RepairNeed | str]:
    """``repair_need`` of every corpus scenario of the first run in ``first_dir`` that is not
    valid there, in UTF-8 byte order (an answer and issues, ``no answer`` or ``empty answer``)."""
    needs: dict[str, RepairNeed | str] = {}
    for name in _corpus_names(corpus_dir):
        need = repair_need(model, first_dir / name)
        if need is not None:
            needs[name] = need
    return needs


def _pending(corpus_dir: Path, model: str, model_dir: Path, first_dir: Path | None) -> list[str]:
    """The scenarios a run still sends, in UTF-8 byte order. A first or repeat run
    (``first_dir`` ``None``): ``pending_scenarios``. A repair run: the scenarios of the first run
    whose ``repair_need`` is an answer and its issues, and whose repair run directory has neither
    ``response.json`` nor ``unanswered.json`` (TASK-005e5 decision 6)."""
    if first_dir is None:
        return pending_scenarios(corpus_dir, model_dir)
    return [
        name
        for name, need in _repair_needs(corpus_dir, model, first_dir).items()
        if isinstance(need, RepairNeed)
        and not present(model_dir / name / RESPONSE_FILE)
        and not _unanswered(model_dir / name)
    ]


def _payload(corpus_dir: Path, model: str, name: str, first_dir: Path | None) -> dict[str, object]:
    """The request of scenario ``name``: the first request, rendered and checked against its pin
    (``_pinned_payload``: a changed prompt, schema or data file still stops the run); for a
    repair run, ``repair_payload`` of it with the first answer's text and issues (no pin exists
    for a repair request). ``RunStopped`` for a scenario that needs no repair."""
    first = _pinned_payload(corpus_dir, name, model)
    if first_dir is None:
        return first
    need = repair_need(model, first_dir / name)
    if not isinstance(need, RepairNeed):
        raise RunStopped(f"{name}: needs no repair request")
    return repair_payload(first, model, need.answer, need.issues)


def corpus_scenarios(corpus_dir: Path) -> list[str]:
    """The corpus scenarios (subdirectories with a ``scenario.json``) in UTF-8 byte order."""
    return _corpus_names(corpus_dir)


UNEXPECTED_RECORD = "{name}: could not be recorded ({kind})"
# TASK-005c3h H8 (the review's #3, TASK-005c3s S3-6): response.json was saved, a derived file was
# not. A held Claude batch is completed only when it is released; Gemini by the next run.
RECORDED_INCOMPLETE = "{name}: recorded, but its record could not be completed ({kind}); {when}"
COMPLETED_ON_RELEASE = "it is completed when the batch is released"
COMPLETED_NEXT_RUN = "the next run completes it"


def _record_failure(run_dir: Path, name: str, kind: str, when: str) -> str:
    """The message of a recording that raised an exception of type ``kind``: ``RECORDED_INCOMPLETE``
    if ``response.json`` was saved before it, else ``UNEXPECTED_RECORD``. Never raises: a
    ``response.json`` that cannot even be stat'ed counts as not saved (the batch is held, or the
    Gemini run stops, either way)."""
    try:
        saved = present(run_dir / RESPONSE_FILE)
    except OSError:
        saved = False
    if saved:
        return RECORDED_INCOMPLETE.format(name=name, kind=kind, when=when)
    return UNEXPECTED_RECORD.format(name=name, kind=kind)


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


# TASK-005c3g G5: an answer whose replace failed stays in its temporary file (``write_answer``).
UNSAVED = (
    "{name}: an answer that could not be saved is kept in {path}; nothing was sent; send this "
    "message to the reviewer and do not delete or rename the file"
)


# TASK-005c3h H5: a leftover of a scenario of the open batch, whose results are downloaded again.
REPLACED = "{name}: a leftover {file} will be replaced by the answer of batch {id}"


def _check_unsaved(pending: list[str], model_dir: Path, open_batch: _Batch | None) -> list[str]:
    """TASK-005c3g G5: ``RunStopped`` before anything is sent if a pending scenario (of
    ``pending``, what the run sends: ``_pending``) keeps an answer that could not be saved
    (``response.json.tmp``): sending it again would pay twice and lose the first answer. A
    recorded scenario's leftover temporary file is ignored.

    TASK-005c3h H5: a pending scenario of ``open_batch`` (the ``open`` entry of a Claude model;
    ``None`` for Gemini) is no stop: the batch's results are downloaded again and ``write_answer``
    overwrites the file. Returns the ``REPLACED`` line of each such scenario."""
    exempt = () if open_batch is None else open_batch.scenarios
    lines: list[str] = []
    for name in pending:
        path = model_dir / name / UNSAVED_RESPONSE_FILE
        if not present(path):
            continue
        if name not in exempt:
            raise RunStopped(UNSAVED.format(name=name, path=path))
        batch_id = cast(str, cast(_Batch, open_batch).id)  # an open entry names its batch
        lines.append(REPLACED.format(name=name, file=UNSAVED_RESPONSE_FILE, id=batch_id))
    return lines


# TASK-005c3g G4: the release of a held batch and its kept results.
STILL_HELD = "; batch {id} is still held: send this message to the reviewer"
# TASK-005c3h H2: a kept results file that cannot be read, or that holds no result of the batch,
# keeps the batch held: releasing it would send again answers the file may hold.
KEPT_UNREADABLE = "cannot read the kept results {path}: {reason}" + STILL_HELD
KEPT_NOT_RESULTS = "{path} holds no result of batch {id}" + STILL_HELD
RECORDED_FROM_KEPT = "{name}: recorded from the kept results of batch {id}"
# TASK-005c3h H3: the same --release command, run again after a stop that followed the release.
ALREADY_RELEASED = "batch {id} was already released"


def _kept_results(raw: bytes) -> dict[str, list[object]]:
    """The ``result`` of every line of a kept results file, by its ``custom_id``; a line that is
    no JSON object with a string ``custom_id`` names no scenario."""
    found: dict[str, list[object]] = {}
    for line in raw.split(b"\n"):
        if not line.strip():
            continue
        document = _json(line)
        custom_id = _field(document, "custom_id")
        if isinstance(custom_id, str):
            found.setdefault(custom_id, []).append(_field(document, "result"))
    return found


def _succeeded_message(result: object) -> dict[str, object] | None:
    """The ``message`` object of a ``succeeded`` result; ``None`` for any other result."""
    message = _field(result, "message")
    if _field(result, "type") != "succeeded" or not isinstance(message, dict):
        return None
    return cast(dict[str, object], message)


def _load(corpus_dir: Path, name: str) -> LoadedScenario:
    try:
        return load_scenario(corpus_dir / name)
    except ScenarioError as error:
        raise RunStopped(f"{name}: invalid scenario: {error.issues[0]}") from None


def _render(corpus_dir: Path, name: str) -> Prompt:
    """The Test A prompt of one scenario; ``RunStopped`` (TASK-005c3d H10) if the instruction
    file cannot be read (the wording of ``eval prompt``) or the bundle cannot be built (for
    example an undecodable CSV file). TASK-005c3h H8 (the review's #10): an ``OSError`` that
    names another file (a CSV file of the scenario) names that file and the scenario."""
    scenario = _load(corpus_dir, name)
    try:
        return render_test_a(scenario)
    except OSError as error:
        reason = error.strerror or type(error).__name__
        instruction = test_a_module.INSTRUCTION_PATH  # the file render_test_a reads
        if error.filename is not None and Path(error.filename) != instruction:
            raise RunStopped(f"{name}: cannot read {error.filename}: {reason}") from None
        path = error.filename or instruction
        raise RunStopped(f"cannot read the instruction file {path}: {reason}") from None
    except ValueError as error:
        raise RunStopped(f"{name}: {error}") from None


def _prepare(
    corpus_dir: Path, model_dir: Path, model: str, name: str, first_dir: Path | None
) -> dict[str, object]:
    """Write ``request.json`` of one scenario (``_payload``: the ``eval prompt`` payload, or its
    repair request) and return it. TASK-005c3h H7: the first payload is checked against its pin
    first (``_pinned_payload``), so a request that differs is never written or sent."""
    payload = _payload(corpus_dir, model, name, first_dir)
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
    ended_at: str | None = None  # ADR-0008 D5: an RFC 3339 instant (``_instant``), else None

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _batch_status(document: object) -> _BatchStatus | None:
    """The batch object ``document``; ``None`` if a field the runner reads is missing or odd,
    ``created_at`` included: it must be an RFC 3339 instant (``_instant``, TASK-005c3e J2), so a
    stored ``created_at`` is always one, and printable. ``ended_at`` (ADR-0008 D5) is the
    object's ``ended_at`` when that is a string that ``_instant`` reads, else ``None``: a
    missing or odd one never makes the object unreadable."""
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
    ended_at = _field(document, "ended_at")
    ended = ended_at if isinstance(ended_at, str) and _instant(ended_at) is not None else None
    return _BatchStatus(
        batch_id, cast(str, status), counts, results_url, created_at, cancel is not None, ended
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
    # TASK-005e5 decision 3: the runs directory of every kind, given to the run (never derived
    # from ``model_dir``, whose parent is ``<runs>/<kind>`` for a repair or repeat run).
    runs_dir: Path
    headers: dict[str, str] = field(repr=False)  # carries the key: never in a repr
    transport: Transport
    sleep: Sleep
    now: Callable[[], datetime]
    poll_seconds: float
    echo: Echo
    release: str | None = None
    clock_transport: Transport | None = None  # G2: the clock call's, ``transport`` if None
    # TASK-005e5 decision 6: the first run's directory of a repair run (``_first_dir``).
    first_dir: Path | None = None

    def _pending(self) -> list[str]:
        return _pending(self.corpus_dir, self.model, self.model_dir, self.first_dir)

    def run(self) -> None:
        started = self.now()  # TASK-005c3h H6: the workspace is listed 120 s after this
        if self.release is not None:
            self._release(self.release)
        # E4: a held batch stops every run before anything is reconciled, polled or sent.
        held = next((b for b in _load_state(self.model_dir) if b.state == "held"), None)
        if held is not None:
            raise RunStopped(HELD_STOP.format(id=held.id, problems="; ".join(held.problems)))
        # G2: before any other request; TASK-005c3h H4: also before the future check, so a slow
        # laptop clock gets the clock stop.
        self._check_clock()
        self._check_started_at()
        opened = next((b for b in _load_state(self.model_dir) if b.state == "open"), None)
        for line in _check_unsaved(self._pending(), self.model_dir, opened):  # G5, H5
            self.echo(line)
        _complete_records(self.corpus_dir, self.model_dir, self.model, self.echo)
        adopted = self._reconcile_earlier()
        pending = self._pending()
        batches = _load_state(self.model_dir)
        batch = next((batch for batch in batches if batch.state == "open"), None)
        if not pending and batch is None:
            self.echo("nothing to run")
            return
        if batch is None:
            if not batches:
                self._check_workspace(started)  # H6: before the model's first batch
            pending = self._settle_unanswered(pending, batches)  # TASK-005e5: ADR-0008 D6
            if not pending:
                return
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
            pending = self._pending()
            if not pending:
                return
            batches = _load_state(self.model_dir)
            pending = self._settle_unanswered(pending, batches)  # TASK-005e5: ADR-0008 D6
            if not pending:
                return
            self.sleep(base_delay(self._next_attempt(pending, batches)))
            batch = self._create(pending, batches)

    def _check_clock(self) -> None:
        """TASK-005c3g G2: one ``GET CLOCK_URL`` (free) through ``send_with_retries`` before any
        other request; ``RunStopped`` unless it is a 200 whose ``Date`` header, compared with
        ``now()`` when the answer arrived, is within ``MAX_CLOCK_SKEW``. A laptop clock that is
        wrong by more would shift every reconcile and audit window (the audit's p02)."""
        transport = self.transport if self.clock_transport is None else self.clock_transport
        response = send_with_retries(
            transport, "GET", CLOCK_URL, self.headers, None, CALL_TIMEOUT_SECONDS, sleep=self.sleep
        )
        arrived = self.now()
        if response.status != 200:
            raise RunStopped(f"listing batches failed: {self._describe(response)}")
        skew = _skew(response, arrived)
        if skew is None:
            raise RunStopped(NO_DATE)
        stop = _clock_stop(skew)
        if stop is not None:
            raise RunStopped(stop)

    def _check_started_at(self) -> None:
        """TASK-005c3d H4: ``RunStopped`` before any call if an entry, of any state, started
        more than ``ADOPTION_WINDOW`` after ``now()``: the clock moved back or the file was
        edited, so no wait or window computed from it can be trusted."""
        latest = self.now() + ADOPTION_WINDOW
        for batch in _load_state(self.model_dir):
            if cast(datetime, _started(batch.started_at)) > latest:
                raise RunStopped(FUTURE_STOP.format(started_at=batch.started_at))

    def _check_workspace(self, started: datetime) -> None:
        """TASK-005c3h H6 (the review's #21): before the first batch of a model with no entry
        in its ``batches.json``, ``RunStopped`` if the workspace lists a batch that is not
        cancelled, is in no model's ``batches.json`` and is not acknowledged: the records of an
        earlier run may be lost, and its scenarios would be paid again.

        First it sleeps, once and announced, until ``RECONCILE_DELAY_SECONDS`` after ``started``
        (the start of ``run()``; at most that long), so a batch created before this run started
        is listed (TASK-005c3s S3-1); then it lists every page (``_list_back_to`` with no lower
        bound), and fails closed beyond ``LIST_PAGES``. The ids are named in list order, newest
        first, through ``server_text``."""
        runs_dir = self.runs_dir  # TASK-005e5 decision 3: every kind's records are known
        # Read before the wait, as the audit does: an invalid file stops at once, with no call.
        known = _all_batch_ids(runs_dir) | _acknowledged(runs_dir, self.sleep)
        target = started + timedelta(seconds=RECONCILE_DELAY_SECONDS)
        wait = min((target - self.now()).total_seconds(), RECONCILE_DELAY_SECONDS)
        if wait > 0:
            self.echo(WORKSPACE_WAITING.format(seconds=math.ceil(wait)))
            self.sleep(wait)
        listed = self._list_back_to(datetime.min.replace(tzinfo=UTC), WORKSPACE_TOO_MANY)
        unknown = [
            server_text(item.status.id)
            for item in listed
            if not item.status.cancelled and item.status.id not in known
        ]
        if not unknown:
            return
        shown = ", ".join(unknown[:UNKNOWN_SHOWN])
        if len(unknown) > UNKNOWN_SHOWN:
            shown += f" and {len(unknown) - UNKNOWN_SHOWN} more"
        noun = "batch" if len(unknown) == 1 else "batches"
        raise RunStopped(UNKNOWN_BATCHES.format(noun=noun, ids=shown, runs=runs_dir))

    def _check_kept_answers(self, pending: list[str]) -> None:
        """TASK-005c3h H6 (TASK-005c3s S3-8), before every create: ``RunStopped`` if the kept
        results of a ``done`` entry hold exactly one line for a pending scenario of that entry,
        a ``succeeded`` one with a message object (the rule of ``_record_kept``): its answer was
        recorded and then lost, and would be paid again. Each kept file is read with
        ``retry_refused``; one that cannot be read stops too. A scenario that a release left
        pending on purpose (no such line, or a file the reviewer emptied) never stops here."""
        for batch in _load_state(self.model_dir):
            names = [name for name in pending if name in batch.scenarios]
            if batch.state != "done" or not names:
                continue
            file = _results_file(cast(str, batch.id))  # a done entry names its batch (_entry)
            path = self.model_dir / file
            try:
                raw = retry_refused(path.read_bytes, self.sleep)
            except OSError as error:
                reason = error.strerror or type(error).__name__
                raise RunStopped(LOST_UNREADABLE.format(path=path, reason=reason)) from None
            lines = _kept_results(raw)
            for name in names:
                found = lines.get(name, [])
                if len(found) == 1 and _succeeded_message(found[0]) is not None:
                    raise RunStopped(LOST_RECORDS.format(name=name, file=file))

    def _release(self, batch_id: str) -> None:
        """E4: the held entry ``batch_id`` becomes ``done`` (its problems kept, for the record)
        after the reviewer's decision; ``RunStopped`` if no held entry has that id. TASK-005c3g
        G4: first, ``_record_kept`` records what the batch's kept results already hold.

        TASK-005c3h H3: a ``done`` entry with that id and problems was held and released before
        (a stop after the release, such as the clock check, advises the same command again): it
        is announced and the run goes on as one without ``--release``."""
        batches = _load_state(self.model_dir)
        if any(b.state == "done" and b.problems and b.id == batch_id for b in batches):
            self.echo(ALREADY_RELEASED.format(id=batch_id))
            return
        index = next(
            (i for i, b in enumerate(batches) if b.state == "held" and b.id == batch_id), None
        )
        if index is None:
            raise RunStopped(f"{server_text(batch_id)} is not a held batch of {self.model}")
        self._record_kept(batches[index])
        batches[index] = replace(batches[index], state="done")
        _save_state(self.model_dir, batches)
        self.echo(f"batch {batch_id} released after review")

    def _record_kept(self, batch: _Batch) -> None:
        """TASK-005c3g G4: before a held ``batch`` is released, record each of its scenarios
        without ``response.json`` whose kept ``batch-<id>.results.jsonl`` has exactly one line,
        a ``succeeded`` one with a message object, as ``_record_result`` does; nothing is sent.
        The others stay pending and are sent again. If a recording fails, the batch stays held
        and ``RunStopped`` names each failed scenario (H9 wording).

        TASK-005c3h H2: the file is read with ``retry_refused``. ``RunStopped`` before anything
        is recorded, saved or sent, the batch still held, if it cannot be read (a missing file
        included), or if it has non-blank lines but none is a result of a scenario of the batch
        (an array, a pretty-printed file: TASK-005c3s S3-3). A file of whitespace only is the
        reviewer's "send them all again": nothing is recorded. Malformed lines, unknown ids and
        duplicates never refuse the file: they are why a batch is held."""
        batch_id = cast(str, batch.id)  # a held entry always names its batch (_entry)
        pending = [n for n in batch.scenarios if not present(self.model_dir / n / RESPONSE_FILE)]
        if not pending:
            return
        path = self.model_dir / _results_file(batch_id)
        try:
            raw = retry_refused(path.read_bytes, self.sleep)
        except OSError as error:
            reason = error.strerror or type(error).__name__
            raise RunStopped(
                KEPT_UNREADABLE.format(path=path, reason=reason, id=batch_id)
            ) from None
        lines = _kept_results(raw)
        if raw.strip() and not any(name in batch.scenarios for name in lines):
            raise RunStopped(KEPT_NOT_RESULTS.format(path=path, id=batch_id))
        failures: list[str] = []
        for name in pending:
            found = lines.get(name, [])
            message = _succeeded_message(found[0]) if len(found) == 1 else None
            if message is None:
                continue
            try:
                write_answer(
                    self.model_dir / name, _load(self.corpus_dir, name), self.model, message
                )
            except Exception as error:  # never its text (H9)
                failures.append(self._problem(name, error))
                continue
            self.echo(RECORDED_FROM_KEPT.format(name=name, id=batch_id))
        if failures:
            raise RunStopped("; ".join(failures) + STILL_HELD.format(id=batch_id))

    def _settle_unanswered(self, pending: list[str], batches: list[_Batch]) -> list[str]:
        """TASK-005e5 decision 4 (ADR-0008 D6), before the next batch is created: each pending
        scenario already sent in ``LAST_BATCH`` batches or more (``_sent_in``) is never sent
        again. ``write_unanswered`` records it (``comparison.json``, then ``unanswered.json``
        with the ids of those entries in the order of ``batches.json``), ``UNANSWERED_LINE`` is
        printed, and the run goes on with the others, which are returned. A failure of either
        write is raised: the scenario stays pending and the next run writes both again.

        First, as before every create, ``_check_kept_answers`` (TASK-005c3h H6): a scenario whose
        answer is in the kept results of a done batch while its records are lost stops the run
        for a restore, never counted unanswered (TASK-005e5 self-review)."""
        self._check_kept_answers(
            [name for name in pending if len(_sent_in(name, batches)) >= LAST_BATCH]
        )
        left: list[str] = []
        for name in pending:
            sent = _sent_in(name, batches)
            if len(sent) < LAST_BATCH:
                left.append(name)
                continue
            ids = [cast(str, batch.id) for batch in sent]  # a sent entry names its batch
            write_unanswered(self.model_dir / name, _load(self.corpus_dir, name), ids)
            self.echo(UNANSWERED_LINE.format(name=name, count=LAST_BATCH))
        return left

    def _next_attempt(self, pending: list[str], batches: list[_Batch]) -> int:
        """The attempt number of the next batch for ``pending`` (a ``failed`` create was never a
        batch: ``_sent_in``). TASK-005e5 decision 4 (ADR-0008 D6): ``RunStopped`` with
        ``LAST_BATCH_WAIT``, before anything is sent, while ``now()`` is earlier than
        ``_last_batch_from`` of a scenario already sent in ``ATTEMPTS`` batches; the stop names
        every such scenario in UTF-8 byte order and the latest of their times. From that time
        on (equal included) such a scenario goes into the next batch with every other pending
        one."""
        sent = {name: _sent_in(name, batches) for name in pending}
        now = self.now()
        waiting = {
            name: _last_batch_from(entries)
            for name, entries in sent.items()
            if len(entries) == ATTEMPTS and now < _last_batch_from(entries)
        }
        if waiting:
            names = sorted(waiting, key=lambda name: name.encode("utf-8"))
            one = len(names) == 1
            raise RunStopped(
                LAST_BATCH_WAIT.format(
                    names=", ".join(names),
                    verb="was" if one else "were",
                    attempts=ATTEMPTS,
                    pronoun="it" if one else "them",
                    time=_stamp(max(waiting.values())),
                )
            )
        return max(len(entries) for entries in sent.values()) + 1

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
        self._check_kept_answers(pending)  # TASK-005c3h H6 (S3-8): before every create
        # TASK-005c3h H7: every request against its pin before any request.json is written.
        _check_requests(self.corpus_dir, pending, self.model, self.first_dir)
        requests = [
            {
                "custom_id": name,
                "params": _prepare(
                    self.corpus_dir, self.model_dir, self.model, name, self.first_dir
                ),
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
        # Every batch any model recorded: another model's batch of the same size is never this one;
        # TASK-005c3h H6 (S3-7): nor an acknowledged one, so acknowledging all but one of several
        # ended candidates lets the same command adopt the one left.
        known = _all_batch_ids(self.runs_dir) | _acknowledged(self.runs_dir, self.sleep)
        candidates = [item.status for item in listed if _is_candidate(item, entry, t0, known)]
        if len(candidates) > 1:
            ids = ", ".join(sorted(candidate.id for candidate in candidates))
            raise RunStopped(SEVERAL_CANDIDATES.format(count=len(candidates), ids=ids))
        return candidates[0] if candidates else None

    def _list_back_to(self, instant: datetime, too_many: str = TOO_MANY_BATCHES) -> list[_Listed]:
        """TASK-005c3d H6: the listed batches, newest first, page by page until a page is the
        last one or ends with a batch created before ``instant``; at most ``LIST_PAGES`` pages.

        Each page goes through ``send_with_retries``. A page that is not a 200 with a list of
        readable batches, a ``has_more`` that is not a boolean, or a page to follow whose
        ``last_id`` is not the id of its last batch stops the run; so does a list that would need
        more pages, with ``too_many`` (fail closed: never "none found").
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
        raise RunStopped(too_many)

    def audit(self) -> None:
        """TASK-005c3d H13, TASK-005c3e J3: the body of ``audit_batches`` (see there)."""
        batches = _load_state(self.model_dir)
        if not batches:
            return
        # Every model's entries, read before the wait: an invalid state stops the audit at once;
        # TASK-005c3h H11 review: the acknowledgements too (never named, H6).
        every = _every_entry(self.runs_dir)
        acknowledged = _acknowledged(self.runs_dir, self.sleep)
        started = [cast(datetime, _started(batch.started_at)) for batch in batches]
        target = max(started) + timedelta(seconds=AUDIT_DELAY_SECONDS)
        wait = min((target - self.now()).total_seconds(), AUDIT_DELAY_SECONDS)
        if wait > 0:
            self.echo(AUDIT_WAITING.format(seconds=math.ceil(wait)))
            self.sleep(wait)
        # J3: back to the first create request of any model, so a late duplicate of another
        # model, listed only after that model's own audit, is still named; G2: and back to the
        # earliest created_at of a recorded batch (the API's clock), whichever is earlier.
        instants = [cast(datetime, _started(batch.started_at)) for batch in every]
        instants += [
            cast(datetime, _instant(batch.created_at))  # validated by _entry (J2)
            for batch in every
            if batch.created_at is not None
        ]
        earliest = min(instants) - ADOPTION_WINDOW
        listed = self._list_back_to(earliest)
        known = {batch.id for batch in every if batch.id is not None}
        known |= acknowledged  # TASK-005c3h H6: never named
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
        # until the owner releases it after the reviewer's decision. ADR-0008 D5: the entry keeps
        # the ended_at of the poll that saw the batch ended.
        state = "held" if problems else "done"
        batches = [
            replace(item, state=state, problems=tuple(problems), ended_at=status.ended_at)
            if item.id == batch_id
            else item
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
        except Exception as error:
            return self._problem(name, error)

    def _problem(self, name: str, error: Exception) -> str:
        """The problem of a scenario whose record failed with ``error`` (H9, J6): a
        ``RunStopped`` gives its message, prefixed with ``<name>: `` unless it starts with it;
        any other exception only its type."""
        if isinstance(error, RunStopped):
            text = server_text(str(error), secret_values(self.headers))
            return text if text.startswith(f"{name}: ") else f"{name}: {text}"
        kind = type(error).__name__
        return _record_failure(self.model_dir / name, name, kind, COMPLETED_ON_RELEASE)

    def _record_result(self, name: str, result: dict[str, object]) -> str | None:
        """Record one result line; return a problem, if any."""
        run_dir = self.model_dir / name
        kind = cast(str, result["type"])
        if present(run_dir / RESPONSE_FILE):
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
            if not (isinstance(error_type, str) and error_type in TRANSIENT_ERRORS):
                # TASK-005c3g G7: a repeat cannot fix it (or it is unknown): the reviewer decides.
                return f"{name}: errored ({shown}); it would fail again"
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
    clock_transport: Transport | None = None,
    kind: str | None = None,
) -> None:
    """Run every pending scenario of ``corpus_dir`` on a Claude model through the Message
    Batches API (TASK-005c3 F2, F3) and record the answers under
    ``run_directory(runs_dir, model, kind)`` (TASK-005e5 decision 3: ``runs_dir / <model dir>``
    for the first run, ``runs_dir / <kind> / <model dir>`` for a repair or repeat run; the
    batch checks read every kind's ``batches.json`` and ``runs_dir / acknowledged-batches.txt``).
    A repair run (TASK-005e5 decision 6, ADR-0008 D7) sends the repair request of each invalid
    candidate of the first run that has an answer (``_pending``, ``_payload``); every other rule
    of a run is the same.

    Returns once every corpus scenario is recorded. Raises ``RunStopped`` (its message is safe to
    print and never contains ``key``) when the run must stop; running the same call again
    resumes, except while a batch is held (TASK-005c3c E4): then every run stops before any
    call until ``release`` names that batch, which marks it ``done`` first (after recording what
    its kept results hold, TASK-005c3g G4). Before any other request the laptop clock is checked
    against the API's ``Date`` header (TASK-005c3g G2), through ``clock_transport`` if given (the
    CLI passes one that is not counted for the audit), else ``transport``. Raises ``ValueError``
    for a model that is not a registered Claude model.
    """
    check_registered(model)
    if not model.startswith(ANTHROPIC_PREFIX):
        raise ValueError(f"{model} is not a Claude model")
    run = _AnthropicRun(
        model=model,
        corpus_dir=corpus_dir,
        model_dir=run_directory(runs_dir, model, kind),
        runs_dir=runs_dir,
        headers=anthropic_headers(key),
        transport=transport,
        sleep=sleep,
        now=now,
        poll_seconds=poll_seconds,
        echo=echo,
        release=release,
        clock_transport=clock_transport,
        first_dir=_first_dir(runs_dir, model, kind),
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
    """The entries of the ``batches.json`` of every model directory of ``runs_dir``, of every run
    kind (``_model_directories``; ``RunStopped`` if one is invalid). Its ``creating`` entries are
    the pending creates: the next run of that model and kind reconciles each of them and may
    adopt one of its candidates."""
    return [
        batch
        for model_dir in _model_directories(runs_dir)
        if present(model_dir / BATCH_STATE_FILE)
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
    kind: str | None = None,
) -> None:
    """TASK-005c3d H13 (ADR-0007 §7 amendment of TASK-005c3d): after a run of a Claude model
    that called the API and was not interrupted, name every listed batch that no run recorded.

    Nothing if the ``batches.json`` of the run's directory (``run_directory(runs_dir, model,
    kind)``, TASK-005e5 decision 3) has no entry; "every model" below means every model
    directory of every kind. Otherwise it reads the ``batches.json``
    of every model, waits until ``AUDIT_DELAY_SECONDS`` after this model's latest ``started_at``
    (at most that long, announced), lists the batches back to the earliest ``started_at`` of any
    model, or the earliest ``created_at`` of a recorded batch if that is earlier (TASK-005c3g G2),
    minus ``ADOPTION_WINDOW`` (``_list_back_to``; TASK-005c3e J3) and prints one warning for
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
        model_dir=run_directory(runs_dir, model, kind),
        runs_dir=runs_dir,
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
    now: Callable[[], datetime],
    echo: Echo,
    first_dir: Path | None,
) -> None:
    _complete_records(corpus_dir, model_dir, model, echo)
    pending = _pending(corpus_dir, model, model_dir, first_dir)  # TASK-005e5 decision 6
    if not pending:
        echo("nothing to run")
        return
    _check_unsaved(pending, model_dir, None)  # TASK-005c3g G5: before the first request
    # TASK-005c3h H7 (S3-4): every request against its pin before the first is sent; _prepare
    # checks each one again when it is built.
    _check_requests(corpus_dir, pending, model, first_dir)
    url = GEMINI_URL.format(model=model)
    checked = _JsonObjectOnly(transport)
    for name in pending:
        _prepare(corpus_dir, model_dir, model, name, first_dir)
        body = (model_dir / name / REQUEST_FILE).read_bytes()  # exactly the recorded request
        try:
            response = send_with_retries(
                checked, "POST", url, headers, body, GEMINI_TIMEOUT_SECONDS, sleep=sleep
            )
        except RunStopped as error:
            raise RunStopped(f"{name}: {error}") from None
        if response.status != 200:
            raise RunStopped(f"{name}: {describe_response(response, secret_values(headers))}")
        received = json.dumps({"received_at": _stamp(now())}, indent=2) + "\n"  # ADR-0008 D5
        document = cast(dict[str, object], _json(response.body))
        scenario = _load(corpus_dir, name)
        try:
            # Before response.json: every recorded answer has its file, and the next attempt
            # replaces the file of an attempt that was not recorded.
            write_atomic(model_dir / name / RECEIVED_FILE, received.encode("utf-8"))
            comparison = write_answer(model_dir / name, scenario, model, document)
            line = recorded_line(name, model, document, comparison)
        except Exception as error:  # never its text (TASK-005c3d H9)
            kind = type(error).__name__
            failure = _record_failure(model_dir / name, name, kind, COMPLETED_NEXT_RUN)
            raise RunStopped(failure + TO_REVIEWER) from None
        echo(line)


def run_gemini(
    model: str,
    corpus_dir: Path,
    runs_dir: Path,
    key: str,
    transport: Transport,
    *,
    sleep: Sleep,
    now: Callable[[], datetime],
    echo: Echo,
    kind: str | None = None,
) -> None:
    """Run every pending scenario of ``corpus_dir`` on a Gemini model, one ``generateContent``
    call at a time in UTF-8 byte order (TASK-005c3 F4), and record each answer under
    ``run_directory(runs_dir, model, kind)`` (TASK-005e5 decisions 3 and 6: a repair run sends
    the repair request of each invalid candidate of the first run that has an answer).

    A 200 with a JSON object is recorded, a blocked prompt included (an invalid answer). ADR-0008
    D5: once such an answer arrived, ``now()`` is read once and ``received.json``
    (``{"received_at": <UTC, whole seconds>}``) is written just before ``response.json``, in the
    same step, so a failure there is the same stop and every recorded answer has its file; the
    next attempt replaces the file of an attempt that was not recorded. A failed or retried call
    writes none, nor does a Claude run; ``eval rescore`` and the summary never read it. Returns
    once every corpus scenario is recorded; raises ``RunStopped`` (never containing ``key``)
    when the run must stop, and running again sends only the scenarios still pending. Raises
    ``ValueError`` for a model that is not a registered Gemini model.
    """
    check_registered(model)
    if model.startswith(ANTHROPIC_PREFIX):
        raise ValueError(f"{model} is not a Gemini model")
    model_dir = run_directory(runs_dir, model, kind)
    first_dir = _first_dir(runs_dir, model, kind)
    try:
        headers = gemini_headers(key)
        _run_gemini(model, corpus_dir, model_dir, headers, transport, sleep, now, echo, first_dir)
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
    now: Callable[[], datetime]

    def shown(self, value: object) -> str:
        """A server string, safe to print; ``not reported`` if it is not a non-empty string."""
        if isinstance(value, str) and value:
            return server_text(value, secret_values(self.headers))
        return NOT_REPORTED

    def get(self, url: str, check: str) -> tuple[dict[str, object], timedelta | None]:
        """One read-only call through ``send_with_retries``: its JSON object and the clock skew
        read when it arrived (``_skew``; ``None`` without a readable ``Date``), or the failed
        ``check`` for any other answer. ``check_url`` refusing the URL or a key header (H7) stops
        before any call, naming the host only; TASK-005c3g G10: any other ``ValueError`` (from a
        transport) is not caught here, so the CLI names its type only."""
        try:
            check_url(url, self.headers)
        except ValueError as error:  # its message names the host only
            raise RunStopped(str(error)) from None
        response = send_with_retries(
            self.transport, "GET", url, self.headers, None, CALL_TIMEOUT_SECONDS, sleep=self.sleep
        )
        arrived = self.now()
        if response.status != 200:
            raise _failed(check, describe_response(response, secret_values(self.headers)))
        document = _json(response.body)
        if not isinstance(document, dict):
            raise _failed(check, "HTTP 200 without a JSON object")
        return cast(dict[str, object], document), _skew(response, arrived)

    def clock(self, skew: timedelta | None) -> None:
        """TASK-005c3g G2: the laptop clock against the ``Date`` of the Models API answer; a
        missing or unreadable header is not reported (the run checks it again)."""
        if skew is None:
            self.echo(f"clock: {NOT_REPORTED}")
            return
        stop = _clock_stop(skew)
        if stop is not None:
            raise _failed("clock", stop)
        self.echo(f"clock: within {_skew_seconds(skew)} s of api.anthropic.com")

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
        document, skew = self.get(f"{MODELS_URL}/{self.model}", "model metadata")
        self.echo("key: accepted by api.anthropic.com")
        self.clock(skew)
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
        listing, _ = self.get(CLOCK_URL, "batch list")  # the run's clock call (G2)
        try:
            page, _ = _read_page(listing)  # as the reconciliation and the audit read a page
        except ValueError as error:
            raise _failed("batch list", str(error)) from None
        count = f"{len(page)} batch" if len(page) == 1 else f"{len(page)} batches"
        self.echo(f"batch list: readable ({count} on the first page)")

    def gemini(self) -> None:
        document, _ = self.get(GEMINI_MODEL_URL.format(model=self.model), "model metadata")
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


def preflight(
    model: str,
    key: str,
    transport: Transport,
    *,
    sleep: Sleep,
    echo: Echo,
    now: Callable[[], datetime],
) -> None:
    """``potakt eval preflight`` (TASK-005c3e J7, ADR-0007 §7 amendment of TASK-005c3e): check
    the key and ``model`` with free, read-only calls before a paid run. No prompt is sent, no
    file is written, no lock is taken and no audit runs.

    A Claude model: ``GET MODELS_URL/<id>``, whose ``Date`` header is compared with ``now()``
    (the ``clock`` check, TASK-005c3g G2), and one page of the batch list (``limit=1``), read
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
    checks = _Preflight(model, headers, transport, sleep, echo, now)
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

    def release(self, *, sleep: Sleep | None = None) -> str | None:
        """Remove the lock if it is still this run's. A warning, safe to print, if it now holds
        another run's line (the owner deleted it as stale while this run was still running, and
        started another) or if it cannot be removed; ``None`` otherwise (TASK-005c3c E9). A
        removal refused with ``PermissionError`` is retried like a replace of ``write_atomic``
        (``retry_refused``, with ``sleep``; TASK-005c3g G3)."""
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
            retry_refused(lambda: self.path.unlink(missing_ok=True), sleep)
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


# --- before a run: the pins, the order rule and the plan (F5) ---------------------------------

# ADR-0008 D1 (amendment of TASK-005e6): False until the commit tagged phase0-freeze, which sets it
# to True; until then eval run --yes sends only the development set (--dev). Read when called.
CORPUS_FROZEN = False
NOT_FROZEN = (
    "no corpus scenario is sent before the commit tagged phase0-freeze (ADR-0008 D1); run the "
    "development set with --dev; nothing was sent"
)
NOT_FROZEN_PLAN = (
    "Not sent before the freeze: --yes is refused until the commit tagged phase0-freeze "
    "(ADR-0008 D1); the development set runs with --dev."
)


def check_frozen(dev: bool) -> None:
    """ADR-0008 D1 (amendment of TASK-005e6): ``RunStopped`` (``NOT_FROZEN``) for a run without
    ``--dev`` while ``CORPUS_FROZEN`` is false. Reads the flag when called."""
    if not dev and not CORPUS_FROZEN:
        raise RunStopped(NOT_FROZEN)


# TASK-005c3g G8: the SHA256 values that ADR-0008 Appendix A pins (it revises ADR-0007 §7; a test
# compares them with the ADR), checked by every eval run before anything is rendered.
INSTRUCTION_SHA256 = "2dcb497384a35ae227ccc8df46349d6179162687093ac16e0cd48be9f406263c"
SCHEMA_SHA256 = "afd54559d52d492c8ce49914a886be49b2f377187d11098f8fff7395dec9c02c"
PIN_DIFFERS = (
    "the {what} differs from the one ADR-0008 Appendix A pins (SHA256 {pinned}); nothing was sent"
)
# TASK-005c3h H7 (ADR-0007 §7, amendment of TASK-005c3h): the SHA256 of the UTF-8 bytes of each
# request.json (``render_payload``), by (model, scenario), re-rendered on 2026-11-10 (TASK-005e1;
# first rendered by TASK-005c3g G8; the test of the payloads compares this table with its own).
# Each request is checked against it when it is built, before it is written or sent.
PAYLOAD_SHA256: Mapping[tuple[str, str], str] = MappingProxyType(
    {
        (model, name): digest
        for model, digests in {
            "claude-haiku-5-5": {
                "01-pharmacy": "3702e6913ec0d626028c283e3e03d9a364d5835de17213c1219dc00ecfda0d0d",
                "02-hotel-reception": (
                    "dfec88ac78989f21f921bfcc1c7bff086f6f035ff5e107bb70ebbefe6446d4e0"
                ),
                "03-call-centre": (
                    "8b8761dd4362d66c8b7185dcfea97b22fc0ba517ba1b66ff37664ba93449de79"
                ),
                "04-restaurant": "dc4e68bb1f8cdc61698431094aa083a3ac4a75680540bac5f127fcedb1f682cb",
                "05-supermarket-checkouts": (
                    "6c2259c82ff4fcb4e867d85bd611f9f15cffb2d867eb1cda0eb007eb8bc5bd50"
                ),
                "06-ambulance-crews": (
                    "91569eb9d7241983bbe4ea6bd44a3970570b81b73462fa33a0c2c8e0aef5828f"
                ),
                "07-security-guards": (
                    "47f6d6cf8c4675cb570c6ce03bef8be40ca8da04b598c0c100b07361222df053"
                ),
                "08-warehouse": "1525adf054f55a200f021c08e2b40f76d9d5755d6f6063465f048cdc416b1f68",
                "09-care-home": "7f8283ca6b26c8cddc5019dfa1e166044f88617f62a5a3349a90578104d9fca5",
                "10-municipal-nursery": (
                    "3a719e8884209c4d0b6989bc1aaaf10596b3b84f1788bad38aaf83fbae29014e"
                ),
            },
            "gemini-3.8-flash": {
                "01-pharmacy": "86b758384fb8811e0ecfff73509c6b38aee54fd5cda975f3652bbca558b025b1",
                "02-hotel-reception": (
                    "50fac1af77904284a0643e1dca2cdcef4d5119c9c802d0afc996a5dedd1a2a4c"
                ),
                "03-call-centre": (
                    "551192810c8e6cf7bd9c6cb0a9d96b209ca1725b8bb3e781f9fb41b4dcfd0836"
                ),
                "04-restaurant": "6a899c76a5231f254fbba306c4cb08bcbbea6615e30a5c4f13ccb967d64f8241",
                "05-supermarket-checkouts": (
                    "7d20478e0a6cda78ee79031a4f379225eb926f8908424a51cdc2cd0b312182d0"
                ),
                "06-ambulance-crews": (
                    "79717ecc42927fbcc52872092150c0ff80399668f427f2789a1bd6c2e617cafd"
                ),
                "07-security-guards": (
                    "a50462e093d1c3d1eb9f0801d53fdb7badbf4950e3c6bd41209e806bb97662b3"
                ),
                "08-warehouse": "30c6533b32e759d035ec5cdfe853068bf3fdf216036d80621533fe0e0d9333cd",
                "09-care-home": "3a386574e09650fef04513c3712881887d4685e440e299bb0283441dd97df62c",
                "10-municipal-nursery": (
                    "e819d76a8b487840f15a70a32449bc5c7ae65c2b1fa066b168d6bfcd351f8ee5"
                ),
            },
            "claude-opus-5-5": {
                "01-pharmacy": "6ffdad29d405c47c577321813e46264bfb69ee9705f0d8445dbfd716b2d7418a",
                "02-hotel-reception": (
                    "97d1a0c22fddabced95674ff6b9560430d1f5d83e294777e0eeae37539375f96"
                ),
                "03-call-centre": (
                    "6edfc3d28f5018c4529b1696ed7f29f4594425bbba56f8ede3ca4c2dbd266877"
                ),
                "04-restaurant": "c43339b196bce5fc1f30684490456760485aebc1a54d08b07ae926e53f1e36ec",
                "05-supermarket-checkouts": (
                    "0e45865310df919f9ca74152250765c4ec405bc51241b1bec2da52279a564507"
                ),
                "06-ambulance-crews": (
                    "bceca2befc110f5be82600dad0074cd08b1e463f6c04a5883d734580d278391c"
                ),
                "07-security-guards": (
                    "180ae4c2c7dab4f47ddd553dd15e81fe8fbf954140595806266e88da27935d7b"
                ),
                "08-warehouse": "d515ac77bfe5e9bf49e847e0f21507be1610c851347f1baadbf341590487859c",
                "09-care-home": "5c1ca20accfbddf970e7d1378a8f040e52893659586023ff60bb6522c7b7728d",
                "10-municipal-nursery": (
                    "b2dc1d24eba5c709c5d298cb2968dc6d5b3ec1ecd99ec8ae16f23e9c77966a30"
                ),
            },
        }.items()
        for name, digest in digests.items()
    }
)
# ADR-0008 D1 and Appendix A (TASK-005e6): the SHA256 of the UTF-8 bytes of each development
# request.json (``render_payload``), by (model, scenario), rendered on 2026-11-27 by the reviewer;
# the test of the payloads compares this table with its own.
DEV_PAYLOAD_SHA256: Mapping[tuple[str, str], str] = MappingProxyType(
    {
        (model, name): digest
        for model, digests in {
            "claude-haiku-5-5": {
                "91-swimming-pool": (
                    "836adb0369c5ef9ceb006cc023b7fe2cff00776acfff94a1dbcd559188c451af"
                ),
                "92-museum-front-desk": (
                    "2eaeec98cadf5abb106b2927182588af61e67ae796b36f5b8c104b36c41d17de"
                ),
                "93-fuel-station": (
                    "023eb530e872dd278633210e6fddce9e2201880f05b653a6b5bafeb701f782a7"
                ),
            },
            "gemini-3.8-flash": {
                "91-swimming-pool": (
                    "06998f2806bddb730107eab138da06d78a56349ea480bea478d85691464ee30c"
                ),
                "92-museum-front-desk": (
                    "f4bd1cfe711d99ec6d617129e51f12f022c339aef9d7c9da90d20c5b142d5329"
                ),
                "93-fuel-station": (
                    "d783b9e6fa5fea27bdeb66815a1b9a09e36c0a24e831822ccf43724857a71717"
                ),
            },
            "claude-opus-5-5": {
                "91-swimming-pool": (
                    "c8eb5532b7b141ff467ed5cb6c4d7c1724c1e6ae6753fd3b295b452afc75bc53"
                ),
                "92-museum-front-desk": (
                    "ed311a4ad20212e1a7f2604b72042431263290f75228a7a2cceaae806f269602"
                ),
                "93-fuel-station": (
                    "290c98af0fc82745fa9235099160720afd701dc0f19cd08c14accb0d95080ef1"
                ),
            },
        }.items()
        for name, digest in digests.items()
    }
)
REQUEST_DIFFERS = (
    "{name}: the request for {model} differs from the one the harness pins (SHA256 {pinned}); "
    "nothing was sent"
)
REQUEST_UNPINNED = "{name}: no request is pinned for {model}; nothing was sent"


def _pinned_payload(corpus_dir: Path, name: str, model: str) -> dict[str, object]:
    """The request payload of scenario ``name`` for ``model`` (the ``eval prompt`` payload), as
    rendered now; ``RunStopped`` unless the SHA256 of its UTF-8 bytes (``render_payload``, the
    bytes of ``request.json``) is the one ``PAYLOAD_SHA256`` pins (TASK-005c3h H7), or, for a
    scenario it does not pin, ``DEV_PAYLOAD_SHA256`` (TASK-005e6, ADR-0008 D1). Reads both tables
    when called. Nothing is written."""
    payload = request_payload(_render(corpus_dir, name), model)
    pinned = PAYLOAD_SHA256.get((model, name))
    if pinned is None:
        pinned = DEV_PAYLOAD_SHA256.get((model, name))
    if pinned is None:
        raise RunStopped(REQUEST_UNPINNED.format(name=name, model=model))
    digest = hashlib.sha256(render_payload(payload).encode("utf-8")).hexdigest()
    if digest != pinned:
        raise RunStopped(REQUEST_DIFFERS.format(name=name, model=model, pinned=pinned))
    return payload


def _check_requests(
    corpus_dir: Path, names: list[str], model: str, first_dir: Path | None = None
) -> None:
    """TASK-005c3h H7 (TASK-005c3s S3-4): every request of ``names`` checked against its pin
    before the first is sent (Gemini, the dry run); for a repair run (``first_dir``) the first
    request under each repair request (``_payload``). Nothing is written."""
    for name in names:
        _payload(corpus_dir, model, name, first_dir)


def check_pins() -> None:
    """TASK-005c3g G8: ``RunStopped`` unless the prompt file (its bytes) and the schema section
    (the UTF-8 text that ``render_test_a`` puts after the schema heading) have the SHA256 values
    that ADR-0008 Appendix A (which revises ADR-0007 §7) pins, computed as the tests of ``test_a``
    compute them. Reads the module attributes of ``test_a`` that ``render_test_a`` uses, so it
    checks exactly what is sent. An unreadable prompt file stops with the wording of
    ``_render``."""
    path = test_a_module.INSTRUCTION_PATH
    try:
        data = path.read_bytes()
    except OSError as error:
        reason = error.strerror or type(error).__name__
        shown = error.filename or path
        raise RunStopped(f"cannot read the instruction file {shown}: {reason}") from None
    if hashlib.sha256(data).hexdigest() != INSTRUCTION_SHA256:
        raise RunStopped(PIN_DIFFERS.format(what="prompt file", pinned=INSTRUCTION_SHA256))
    schema = test_a_module.render_json_schema().encode("utf-8")
    if hashlib.sha256(schema).hexdigest() != SCHEMA_SHA256:
        raise RunStopped(PIN_DIFFERS.format(what="schema section", pinned=SCHEMA_SHA256))


# TASK-005e6 decision 3 (ADR-0008 D1 and its amendment): the corpus and the development set never
# mix, in what a run sends or where it records.
DEV_SCENARIOS_WITHOUT_DEV = (
    "{names}: development scenarios run only with --dev (ADR-0008 D1); nothing was sent"
)
CORPUS_SCENARIOS_WITH_DEV = (
    "{names}: corpus scenarios never run with --dev (ADR-0008 D1); nothing was sent"
)
DEV_RUNS_WITHOUT_DEV = (
    "{runs} is under {root}, where only --dev runs record (ADR-0008 D1); nothing was sent"
)
CORPUS_RUNS_WITH_DEV = (
    "{runs} is under {root}, where --dev runs never record (ADR-0008 D1); nothing was sent"
)


def _under(path: Path, root: Path) -> bool:
    """``path`` is ``root`` or under it: ``os.path.normcase(os.path.abspath(...))`` of both is
    equal, or the path's starts with the root's followed by ``os.sep``; so ``..`` parts, a path
    relative to the working directory and, on Windows, letter case do not matter."""
    target = os.path.normcase(os.path.abspath(path))
    base = os.path.normcase(os.path.abspath(root))
    return target == base or target.startswith(base + os.sep)


def check_set(corpus_dir: Path, runs_dir: Path, dev: bool) -> None:
    """TASK-005e6 decision 3 (ADR-0008 D1): ``RunStopped`` with the first of these that applies:
    without ``--dev``, a scenario of ``corpus_dir`` that ``DEV_PAYLOAD_SHA256`` names
    (``DEV_SCENARIOS_WITHOUT_DEV``); with it, one that ``PAYLOAD_SHA256`` names
    (``CORPUS_SCENARIOS_WITH_DEV``); without it, ``runs_dir`` at or under ``DEV_RUNS_ROOT``
    (``DEV_RUNS_WITHOUT_DEV``); with it, ``runs_dir`` at or under ``RUNS_PATH``
    (``CORPUS_RUNS_WITH_DEV``). Scenarios are named in UTF-8 byte order; the tables and both
    roots are read when called. Nothing is read but the names of ``corpus_dir``."""
    other = {scenario for _, scenario in (PAYLOAD_SHA256 if dev else DEV_PAYLOAD_SHA256)}
    found = [name for name in _corpus_names(corpus_dir) if name in other]
    if found:
        message = CORPUS_SCENARIOS_WITH_DEV if dev else DEV_SCENARIOS_WITHOUT_DEV
        raise RunStopped(message.format(names=", ".join(found)))
    root = RUNS_PATH if dev else DEV_RUNS_ROOT
    if _under(runs_dir, root):
        message = CORPUS_RUNS_WITH_DEV if dev else DEV_RUNS_WITHOUT_DEV
        raise RunStopped(message.format(runs=runs_dir, root=str(root)))


def _missing(run_dir: Path, corpus_dir: Path) -> list[str]:
    """The corpus scenarios without a scored run in ``run_dir`` (``unscored``); every corpus
    scenario if ``run_dir`` does not exist."""
    if not run_dir.is_dir():
        return _corpus_names(corpus_dir)
    return unscored(run_dir, corpus_dir)


REPAIR_ORDER = "{model} repairs run only after {model} has a complete run (missing: {names})"
# TASK-005e5 decision 8 (ADR-0008 D8): the repeats of the strong model, one after the other.
REPEAT_2_ORDER = "repeat 2 of {model} runs only after {model} has a complete run (missing: {names})"
REPEAT_3_ORDER = "repeat 3 of {model} runs only after repeat 2 is complete (missing: {names})"


def check_order(model: str, runs_dir: Path, corpus_dir: Path, kind: str | None = None) -> None:
    """ADR-0007 §7 amendment «Order»: ``RunStopped`` for a first run of ``claude-opus-5-5``
    unless ``claude-haiku-5-5`` has a complete run in ``runs_dir``; every other model's first run
    may run. TASK-005e5 decision 6 (ADR-0008 D7): a repair run of a model needs that model's own
    first run complete (``REPAIR_ORDER``), and no other model's run. Decision 8 (ADR-0008 D8):
    repeat 2 needs the first run complete (``REPEAT_2_ORDER``), repeat 3 needs repeat 2
    complete (``REPEAT_3_ORDER``)."""
    orders = {
        REPAIR_DIR: (None, REPAIR_ORDER),
        REPEAT_DIRS[0]: (None, REPEAT_2_ORDER),
        REPEAT_DIRS[1]: (REPEAT_DIRS[0], REPEAT_3_ORDER),
    }
    if kind is not None:
        needed, message = orders[kind]
        missing = _missing(run_directory(runs_dir, model, needed), corpus_dir)
        if missing:
            raise RunStopped(message.format(model=model, names=", ".join(missing)))
        return
    if model != STRONG_MODEL:
        return
    missing = _missing(runs_dir / model_directory_name(FIRST_MODEL), corpus_dir)
    if missing:
        raise RunStopped(
            f"{model} runs only after {FIRST_MODEL} has a complete run "
            f"(missing: {', '.join(missing)})"
        )


def cost_upper_bound(model: str, corpus_dir: Path, scenarios: list[str]) -> float:
    """USD: for each scenario, the UTF-8 bytes of system + user as input tokens (a byte is at
    least one token's worth of text) and ``max_tokens`` as output tokens, at the pinned batch
    prices of a Claude model. A scenario whose bytes pass a long-prompt threshold is priced at
    that tier. It counts only the prompt bytes and ``max_tokens``: the system prompt that the API
    adds when thinking is active and the message framing are not counted, but covered by the
    slack between bytes and tokens (a token holds several bytes of this text) (TASK-005c3g G10)."""
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


def repair_cost_upper_bound(
    model: str, corpus_dir: Path, scenarios: list[str], first_dir: Path
) -> float:
    """USD, TASK-005e5 decision 6: like ``cost_upper_bound``, but the input of each request is the
    UTF-8 bytes of the system text and of the text of every message of its repair request
    (``_payload``), so the answer and the issues count too."""
    price_in, price_out = PRICES_USD_PER_MILLION[model]
    long_prompt = LONG_PROMPT_PRICES_USD_PER_MILLION.get(model)
    total = 0.0
    for name in scenarios:
        payload = _payload(corpus_dir, model, name, first_dir)
        messages = cast(list[dict[str, str]], payload["messages"])
        input_bytes = len(cast(str, payload["system"]).encode("utf-8"))
        input_bytes += sum(len(message["content"].encode("utf-8")) for message in messages)
        prices = (price_in, price_out)
        if long_prompt is not None and input_bytes > long_prompt[0]:
            prices = long_prompt[1]
        total += input_bytes * prices[0] + ANTHROPIC_MAX_TOKENS[model] * prices[1]
    return total / 1_000_000


def _unanswered_plan(
    corpus_dir: Path, model_dir: Path, pending: list[str], batches: list[_Batch]
) -> list[str]:
    """TASK-005e5 decision 4 (ADR-0008 D6), the dry run's lines after ``Recorded``: the pending
    scenarios sent in ``ATTEMPTS`` batches, whose next batch is their last, from the latest of
    their times (``_last_batch_from``, as the stop names it), and the scenarios recorded as
    unanswered; each line only when it names a scenario."""
    waiting = {
        name: _last_batch_from(sent)
        for name in pending
        if len(sent := _sent_in(name, batches)) == ATTEMPTS
    }
    lines: list[str] = []
    if waiting:
        time = _stamp(max(waiting.values()))
        lines.append(LAST_BATCH_PLAN.format(time=time, names=", ".join(waiting)))
    unanswered = unanswered_scenarios(corpus_dir, model_dir)
    if unanswered:
        lines.append(UNANSWERED_PLAN.format(names=", ".join(unanswered)))
    return lines


# TASK-005e5 decisions 4 and 6: the dry run's lines for ADR-0008 D6 and for a repair run.
LAST_BATCH_PLAN = "Last batch (ADR-0008 D6) from {time}: {names}"
UNANSWERED_PLAN = "Unanswered (ADR-0008 D6): {names}"
NOT_SENT_PLAN = "Not sent ({reason}): {names}"
NOT_SENT_REASONS = (("empty answer", "empty answer"), ("no answer", "no answer, ADR-0008 D6"))


GEMINI_COST_LINE = (
    "Cost upper bound: $0 on the free tier (the key's project must be on the free tier: runbook §1)"
)


def _kind_label(kind: str) -> str:
    """The run kind in the dry run's first line: ``repair``, ``repeat 2``, ``repeat 3``."""
    return kind.replace("-", " ")


def _not_sent_plan(corpus_dir: Path, model: str, first_dir: Path | None) -> list[str]:
    """TASK-005e5 decision 6, the dry run of a repair run: the invalid candidates of the first
    run that get no repair request, by reason; each line only when it names a scenario."""
    if first_dir is None:
        return []
    needs = _repair_needs(corpus_dir, model, first_dir)
    lines: list[str] = []
    for need, label in NOT_SENT_REASONS:
        names = [name for name, found in needs.items() if found == need]
        if names:
            lines.append(NOT_SENT_PLAN.format(reason=label, names=", ".join(names)))
    return lines


def run_plan(
    model: str, corpus_dir: Path, runs_dir: Path, kind: str | None = None, *, dev: bool = False
) -> str:
    """What ``eval run`` would do for ``model`` and a run of ``kind`` (TASK-005e5 decisions 3
    and 6), one fact per line, ending with ``\\n``. Reads no key and sends nothing. The first
    line names the model and, in parentheses, ``development set`` for a ``--dev`` run
    (TASK-005e6) and the kind, joined by ``, ``.

    TASK-005c3h H5: ``RunStopped`` (the ``--yes`` message) for a leftover ``response.json.tmp``
    that ``--yes`` would stop on; a leftover of the open batch adds its ``REPLACED`` line. While
    a batch is held the leftovers are not checked: ``--yes`` stops on the held batch first, and
    ``--release`` first records what the kept results hold. H7: ``RunStopped`` for a pending
    request that differs from its pin or has none (``_check_requests``; for a repair run, the
    first request under each repair request)."""
    model_dir = run_directory(runs_dir, model, kind)
    first_dir = _first_dir(runs_dir, model, kind)
    anthropic = model.startswith(ANTHROPIC_PREFIX)
    batches = _load_state(model_dir) if anthropic else []
    pending = _pending(corpus_dir, model, model_dir, first_dir)
    replaced: list[str] = []
    if not any(batch.state == "held" for batch in batches):
        opened = next((batch for batch in batches if batch.state == "open"), None)
        replaced = _check_unsaved(pending, model_dir, opened)
    # TASK-005c3h H7: as --yes checks them
    _check_requests(corpus_dir, pending, model, first_dir)
    recorded = recorded_scenarios(corpus_dir, model_dir)
    parts = [*(["development set"] if dev else []), *([] if kind is None else [_kind_label(kind)])]
    lines = [
        f"Test A run plan: {model}" + (f" ({', '.join(parts)})" if parts else ""),
        f"Runs directory: {model_dir}",
        f"Pending ({len(pending)}): {', '.join(pending) or 'none'}",
        f"Recorded ({len(recorded)}): {', '.join(recorded) or 'none'}",
        *_not_sent_plan(corpus_dir, model, first_dir),
        *_unanswered_plan(corpus_dir, model_dir, pending, batches),
    ]
    if anthropic:
        # One entry at most is creating or open (_load_state); the dry run never lists batches.
        for batch in batches:
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
        lines += replaced
        if first_dir is None:
            cost = cost_upper_bound(model, corpus_dir, pending)
            counted = "the prompt"
        else:  # TASK-005e5 decision 6: the answer and the issues are input too
            cost = repair_cost_upper_bound(model, corpus_dir, pending, first_dir)
            counted = "every message"
        lines.append(
            f"Cost upper bound: ${cost:.4f} ({len(pending)} requests; input: the UTF-8 bytes of "
            f"{counted}, output: max_tokens {ANTHROPIC_MAX_TOKENS[model]}; batch prices pinned "
            "on 2026-11-02)"
        )
    else:
        # TASK-005c3g G9: no API shows the key's tier; a billing-linked project pays.
        lines.append(GEMINI_COST_LINE)
    lock = lock_text(runs_dir)
    if lock is not None:
        lines.append(f"Lock: {lock} (another run may be active)")
    # TASK-005e6 decision 4 (ADR-0008 D1): before the freeze, --yes sends no corpus scenario.
    lines.append(
        NOT_FROZEN_PLAN if not dev and not CORPUS_FROZEN else "Run again with --yes to send."
    )
    return "\n".join(lines) + "\n"
