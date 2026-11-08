"""The Anthropic Test A runner (TASK-005c3 F2, F3): Message Batches through a fake server.

``FakeBatches`` plays the Message Batches API: it creates numbered batches, answers polls with
``in_progress`` and then ``ended``, lists them and serves each batch's JSONL results, built from an
``outcome`` function of (batch number, scenario id). No test opens a connection. The clock of a
run is its ``Sleeps``: ``now()`` moves only by the seconds slept (TASK-005c3b D2).
"""

import email.utils
import http.client
import json
import math
import os
import shutil
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pytest
from runner_helpers import (
    ALL_SCENARIOS,
    KEY,
    NOW,
    STARTED_AT,
    Echo,
    RefusedReplace,
    Sleeps,
    anthropic_message,
    batch_entry,
    corpus_copy,
    files_containing,
    truth_text,
    write_state,
)
from typer.testing import CliRunner

import potakt.evaluation.records as records_module
import potakt.evaluation.runner as runner_module
import potakt.evaluation.test_a as test_a_module
from potakt import __version__
from potakt.cli import app
from potakt.evaluation import (
    CORPUS_PATH,
    load_scenario,
    render_test_a,
    request_payload,
    summarize,
    write_answer,
    write_request,
)
from potakt.evaluation.records import refused_pause
from potakt.evaluation.runner import (
    ANTHROPIC_VERSION,
    BATCH_STATE_FILE,
    BATCHES_URL,
    HEARTBEAT_POLLS,
    run_anthropic,
)
from potakt.evaluation.transport import HttpResponse, RunStopped, TransportError, UrllibTransport

HAIKU = "claude-haiku-5-5"
OPUS = "claude-opus-5-5"
CREATED_AT = "2026-11-02T10:00:05.123456Z"
LIST_URL = f"{BATCHES_URL}?limit=100"

Outcome = Callable[[int, str], dict[str, Any]]
API_CLOCK = "<the API's clock>"  # FakeBatches.date_header: the fake's own clock
# TASK-005c3g G2: the one free call that reads the API's clock before any other request. Spelled
# out here (not imported) so a changed URL in the runner fails a test instead of following it.
CLOCK_URL = f"{BATCHES_URL}?limit=1"


def succeeded(text: str | None = None, model: str = HAIKU) -> dict[str, Any]:
    message = anthropic_message(model) if text is None else anthropic_message(model, text)
    return {"type": "succeeded", "message": message}


def errored(kind: str, text: str = "details") -> dict[str, Any]:
    return {"type": "errored", "error": {"type": "error", "error": {"type": kind, "message": text}}}


def all_succeed(number: int, scenario: str) -> dict[str, Any]:
    return succeeded()


@dataclass
class Call:
    method: str
    url: str
    headers: dict[str, str]
    body: bytes | None
    timeout: float


@dataclass(frozen=True)
class Created:
    """A create step: the server creates the batch, then answers ``response`` (a gateway's 5xx
    after the API accepted the request)."""

    response: HttpResponse


CreateStep = str | HttpResponse | Created


@dataclass
class FakeBatches:
    """The Message Batches API, scripted."""

    outcome: Outcome = all_succeed
    polls_in_progress: int = 1
    # Consumed by the create calls, in order, before normal creates: "lost" registers the batch
    # and raises TransportError (the response was lost); "lost-none" raises without creating it;
    # "interrupt" raises KeyboardInterrupt before the server sees the call; "interrupt-created"
    # registers the batch, then raises KeyboardInterrupt; "duplicated" registers two batches (a
    # proxy retried the call), the second missing from the list for 60 s more, then raises
    # TransportError; "duplicated-staggered" the same, the second created ``stagger`` s (40 by
    # default) after the first (TASK-005c3d R4); an HttpResponse is returned as is (no batch);
    # Created(response) registers the batch, then returns the response.
    create_script: list[CreateStep] = field(default_factory=list[CreateStep])
    stagger: float = 40.0
    # Extra listed batches: ``newest`` before the fake's own batches, ``listing`` after them.
    newest: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    listing: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    list_pages: list[HttpResponse] = field(default_factory=list[HttpResponse])  # answered first
    # A batch created by a create call is missing from the list for ``lag`` seconds of the clock
    # (plus its entry in ``extra_lag``).
    lag: float = 0.0
    extra_lag: dict[str, float] = field(default_factory=dict[str, float])
    posted_at: dict[str, datetime] = field(default_factory=dict[str, datetime])
    list_times: list[datetime] = field(default_factory=list[datetime])  # the clock at each list
    # Seconds the clock moves while a create call is answered (``clock.advance``).
    create_seconds: float = 0.0
    clock: Sleeps | None = None
    stamp_with_clock: bool = False  # created_at from the clock instead of CREATED_AT
    cancelled: set[str] = field(default_factory=set[str])
    omit_cancel_field: bool = False  # batch objects without cancel_initiated_at
    list_failures: list[TransportError] = field(default_factory=list[TransportError])
    results_base: str = BATCHES_URL
    raw_results: dict[str, bytes] = field(default_factory=dict[str, bytes])
    on_first_poll: Callable[[], None] | None = None
    poll_failure: HttpResponse | TransportError | None = None  # every poll answers this
    list_answer: HttpResponse | None = None  # the list call answers this instead
    results_answer: HttpResponse | None = None  # every results download answers this instead
    calls: list[Call] = field(default_factory=list[Call])
    batches: dict[str, list[str]] = field(default_factory=dict[str, list[str]])
    polled: dict[str, int] = field(default_factory=dict[str, int])
    created_at: dict[str, str] = field(default_factory=dict[str, str])
    posted: set[str] = field(default_factory=set[str])  # the batches created by create calls
    lists: int = 0
    # TASK-005c3g G2: every answer carries a ``date`` header from the API's clock (``now()``, at
    # whole seconds, as RFC 9110 IMF-fixdate); ``None`` sends none, any other string as it is.
    date_header: str | None = API_CLOCK
    # The clock calls (``CLOCK_URL``) answer these first, then an empty page.
    clock_answers: list[HttpResponse] = field(default_factory=list[HttpResponse])
    clock_seconds: float = 0.0  # seconds both clocks move while a clock call is answered
    clock_checks: list[datetime] = field(default_factory=list[datetime])  # API time of each
    # TASK-005c3h H6: ``newest``, ``listing``, the batches registered without a create call and
    # the scripted list answers (``list_pages``, ``list_answer``, ``list_failures``) appear only
    # after the first create call, so the workspace check of a first run does not see them: for
    # tests of the reconciliation, whose other batches stand for batches created meanwhile.
    extras_after_create: bool = False

    def register(self, scenarios: list[str], created_at: str | None = None) -> str:
        """A new batch, created at ``created_at``: by default ``CREATED_AT``, or the clock's
        time when ``stamp_with_clock`` is set."""
        if created_at is None:
            created_at = CREATED_AT
            if self.stamp_with_clock and self.clock is not None:
                created_at = self.clock.now().isoformat().replace("+00:00", "Z")
        batch_id = f"msgbatch_{len(self.batches) + 1:02d}"
        self.batches[batch_id] = scenarios
        self.created_at[batch_id] = created_at
        return batch_id

    def number(self, batch_id: str) -> int:
        return list(self.batches).index(batch_id) + 1

    def batch(self, batch_id: str) -> dict[str, Any]:
        scenarios = self.batches[batch_id]
        cancelled = batch_id in self.cancelled
        ended = cancelled or self.polled.get(batch_id, 0) > self.polls_in_progress
        counts = dict.fromkeys(("processing", "succeeded", "errored", "canceled", "expired"), 0)
        if cancelled:
            counts["canceled"] = len(scenarios)
        elif ended:
            for scenario in scenarios:
                counts[self.outcome(self.number(batch_id), scenario)["type"]] += 1
        else:
            counts["processing"] = len(scenarios)
        document: dict[str, Any] = {
            "id": batch_id,
            "type": "message_batch",
            "processing_status": "ended" if ended else "in_progress",
            "request_counts": counts,
            "created_at": self.created_at[batch_id],
            "ended_at": None,
            "expires_at": "2026-11-03T10:00:05Z",
            "cancel_initiated_at": "2026-11-02T10:00:30Z" if cancelled else None,
            "results_url": f"{self.results_base}/{batch_id}/results" if ended else None,
        }
        if self.omit_cancel_field:
            del document["cancel_initiated_at"]
        return document

    def results(self, batch_id: str) -> bytes:
        if batch_id in self.raw_results:
            return self.raw_results[batch_id]
        lines = [
            json.dumps({"custom_id": s, "result": self.outcome(self.number(batch_id), s)})
            for s in reversed(self.batches[batch_id])  # any order (the API guarantees none)
        ]
        return ("\n".join(lines) + "\n").encode("utf-8")

    def send(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
    ) -> HttpResponse:
        response = self._answer(method, url, headers, body, timeout)
        if self.date_header is None:
            return response
        date = self.date_header
        if date == API_CLOCK:
            date = email.utils.format_datetime(self.now().replace(microsecond=0), usegmt=True)
        return replace(response, headers={**response.headers, "date": date})

    def _answer(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
    ) -> HttpResponse:
        self.calls.append(Call(method, url, headers, body, timeout))
        if method == "GET" and url == CLOCK_URL:
            if self.clock is not None:
                self.clock.advance(self.clock_seconds)
            self.clock_checks.append(self.now())
            if self.clock_answers:
                return self.clock_answers.pop(0)
            return ok({"data": [], "has_more": False, "first_id": None, "last_id": None})
        if method == "POST" and url == BATCHES_URL:
            return self._create(body)
        if method == "GET" and url.startswith(LIST_URL):
            extras = not self.extras_after_create or bool(self.creates())
            if self.list_failures and extras:
                raise self.list_failures.pop(0)
            self.lists += 1
            now = self.now()
            self.list_times.append(now)
            if self.list_pages and extras:
                return self.list_pages.pop(0)
            if self.list_answer is not None and extras:
                return self.list_answer
            data = (
                (self.newest if extras else [])
                + [
                    self.batch(batch_id)
                    for batch_id in reversed(self.batches)
                    if (batch_id not in self.posted and extras)
                    or (
                        batch_id in self.posted
                        and now
                        >= self.posted_at[batch_id]
                        + timedelta(seconds=self.lag + self.extra_lag.get(batch_id, 0.0))
                    )
                ]
                + (self.listing if extras else [])
            )
            return ok(page_of(data, url))
        if method == "GET" and url.endswith("/results"):
            if self.results_answer is not None:
                return self.results_answer
            return HttpResponse(200, {}, self.results(url.split("/")[-2]))
        if method == "GET" and url.startswith(BATCHES_URL + "/"):
            batch_id = url.rsplit("/", 1)[1]
            if isinstance(self.poll_failure, TransportError):
                raise self.poll_failure
            if self.poll_failure is not None:
                return self.poll_failure
            if not self.polled and self.on_first_poll is not None:
                self.on_first_poll()
            self.polled[batch_id] = self.polled.get(batch_id, 0) + 1
            return ok(self.batch(batch_id))
        raise AssertionError(f"unexpected call {method} {url}")

    def _create(self, body: bytes | None) -> HttpResponse:
        assert body is not None
        requests = json.loads(body)["requests"]
        scenarios = [request["custom_id"] for request in requests]
        step = self.create_script.pop(0) if self.create_script else "create"
        if self.clock is not None:
            self.clock.advance(self.create_seconds)
        if step == "interrupt":
            raise KeyboardInterrupt
        if isinstance(step, HttpResponse):
            return step
        if step == "lost-none":
            raise TransportError("TimeoutError: no response received")
        batch_id = self.post(scenarios)
        if step == "duplicated":
            self.extra_lag[self.post(scenarios)] = 60.0
            raise TransportError("TimeoutError: no response received")
        if step == "duplicated-staggered":
            # R4: the proxy's second batch is created ``stagger`` s after the first.
            later = self.now() + timedelta(seconds=self.stagger)
            twin = self.register(scenarios, created_at=later.isoformat().replace("+00:00", "Z"))
            self.posted.add(twin)
            self.posted_at[twin] = later
            raise TransportError("TimeoutError: no response received")
        if isinstance(step, Created):
            return step.response
        if step == "lost":
            raise TransportError("TimeoutError: no response received")
        if step == "interrupt-created":
            raise KeyboardInterrupt
        return ok(self.batch(batch_id))

    def now(self) -> datetime:
        return self.clock.now() if self.clock is not None else NOW

    def post(self, scenarios: list[str]) -> str:
        """A batch created by a create call, at the clock's time."""
        batch_id = self.register(scenarios)
        self.posted.add(batch_id)
        self.posted_at[batch_id] = self.now()
        return batch_id

    def creates(self) -> list[list[str]]:
        """The scenarios of every create call, in order."""
        return [
            [r["custom_id"] for r in json.loads(call.body or b"{}")["requests"]]
            for call in self.calls
            if call.method == "POST"
        ]


def ok(document: object) -> HttpResponse:
    return HttpResponse(200, {"content-type": "application/json"}, json.dumps(document).encode())


def page_of(data: list[dict[str, Any]], url: str) -> dict[str, Any]:
    """One page of a list call, as the documented API answers it (TASK-005c3d H6): ``limit``
    (default 20) items after ``after_id``, with ``has_more``, ``first_id`` and ``last_id``."""
    query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
    limit = int(query.get("limit", "20"))
    after = query.get("after_id")
    start = 0 if after is None else [item["id"] for item in data].index(after) + 1
    chunk = data[start : start + limit]
    return {
        "data": chunk,
        "has_more": start + limit < len(data),
        "first_id": chunk[0]["id"] if chunk else None,
        "last_id": chunk[-1]["id"] if chunk else None,
    }


@dataclass
class Run:
    corpus: Path
    runs: Path
    server: FakeBatches
    sleeps: Sleeps
    echo: Echo
    model: str = HAIKU
    # TASK-005c3g G2: the laptop clock minus the API's (the server's ``Sleeps`` clock).
    skew: timedelta = timedelta(0)
    # TASK-005c3h H6: a run that will create the model's first batch first waits 120 s and lists
    # the workspace. With ``rewind``, such a run starts 120 s before ``NOW`` on the fake clock,
    # so after that wait the clock reads what the test states (``NOW``, ``STARTED_AT``); the
    # wait itself stays visible in ``sleeps``, ``echo`` and the calls.
    rewind: bool = True

    @property
    def model_dir(self) -> Path:
        return self.runs / self.model

    def now(self) -> datetime:
        """The runner's clock: the API's, plus ``skew``."""
        return self.sleeps.now() + self.skew

    def first_batch_ahead(self) -> bool:
        """The model has no entry in ``batches.json`` and a corpus scenario is pending (``False``
        for a state or a corpus that the runner itself refuses)."""
        state = self.model_dir / BATCH_STATE_FILE
        try:
            if state.exists() and json.loads(state.read_bytes())["batches"]:
                return False
            names = [path.name for path in self.corpus.iterdir()]
        except (OSError, ValueError, TypeError, KeyError):
            return False
        return any(not (self.model_dir / name / "response.json").exists() for name in names)

    def __call__(self, release: str | None = None) -> None:
        if self.rewind and self.first_batch_ahead():
            self.sleeps.advance(-runner_module.RECONCILE_DELAY_SECONDS)
        run_anthropic(
            self.model,
            self.corpus,
            self.runs,
            KEY,
            self.server,
            sleep=self.sleeps,
            now=self.now,
            poll_seconds=60,
            echo=self.echo,
            release=release,
        )

    def state(self) -> dict[str, Any]:
        return json.loads((self.model_dir / BATCH_STATE_FILE).read_bytes())

    def states(self) -> list[str]:
        return [batch["state"] for batch in self.state()["batches"]]

    def lists(self) -> int:
        """The listings: list calls without ``after_id`` (the first page)."""
        return sum(call.url == LIST_URL for call in self.server.calls)

    def pages(self) -> int:
        """Every list call, the later pages included."""
        return sum(call.url.startswith(LIST_URL) for call in self.server.calls)

    def batch_ids(self) -> list[str | None]:
        """The batch ids in ``batches.json``; none if the file was never written."""
        if not (self.model_dir / BATCH_STATE_FILE).exists():
            return []
        return [batch["id"] for batch in self.state()["batches"]]

    def recorded(self) -> list[str]:
        return sorted(p.parent.name for p in self.model_dir.glob("*/response.json"))


def make_run(tmp_path: Path, server: FakeBatches, names: tuple[str, ...] = ALL_SCENARIOS) -> Run:
    sleeps = Sleeps()
    server.clock = sleeps
    return Run(corpus_copy(tmp_path, names), tmp_path / "runs", server, sleeps, Echo())


# --- the happy path --------------------------------------------------------------------------


def test_the_happy_path_records_all_ten_scenarios(tmp_path: Path) -> None:
    server = FakeBatches(
        outcome=lambda n, s: succeeded(truth_text(s)) if s == "01-pharmacy" else succeeded()
    )
    run = make_run(tmp_path, server)

    def state_is_written() -> None:
        state = run.state()
        assert state["batches"] == [
            {
                "id": "msgbatch_01",
                "scenarios": list(ALL_SCENARIOS),
                "started_at": STARTED_AT,
                "created_at": CREATED_AT,
                "state": "open",
                "problems": [],
            }
        ]

    server.on_first_poll = state_is_written
    run()

    assert run.recorded() == list(ALL_SCENARIOS)
    assert run.state()["batches"][0]["state"] == "done"
    raw = run.model_dir / "batch-msgbatch_01.results.jsonl"
    assert raw.read_bytes() == server.results("msgbatch_01")
    assert [call.method for call in server.calls] == ["GET", "GET", "POST", "GET", "GET", "GET"]
    assert server.calls[0].url == CLOCK_URL  # TASK-005c3g G2: the clock first
    assert server.calls[1].url == LIST_URL  # TASK-005c3h H6: the workspace, before the create
    assert run.sleeps == [120, 60]
    assert run.echo[0] == WORKSPACE_WAIT.format(seconds=120)
    assert run.echo[1] == "batch msgbatch_01 created for 10 scenarios"
    assert run.echo[2] == (
        "batch msgbatch_01: in_progress "
        "(processing 10, succeeded 0, errored 0, expired 0, canceled 0)"
    )
    assert run.echo[3] == (
        "batch msgbatch_01: ended (processing 0, succeeded 10, errored 0, expired 0, canceled 0)"
    )
    assert run.echo[4] == "01-pharmacy: recorded (end_turn, valid, recall 1.000)"
    assert run.echo[5] == "02-hotel-reception: recorded (end_turn, invalid, recall 0.000)"
    assert len(run.echo) == 14


def test_summary_and_rescore_ignore_the_batch_files(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:3])
    run()
    files = sorted(path.name for path in run.model_dir.iterdir() if path.is_file())
    assert files == ["batch-msgbatch_01.results.jsonl", "batches.json"]

    summary = summarize(run.model_dir, run.corpus)
    rescored = CliRunner().invoke(
        app, ["eval", "rescore", str(run.model_dir), "--corpus", str(run.corpus)]
    )

    assert summary.complete
    assert summary.notes == ()
    assert summary.scenarios_run == 3
    assert rescored.exit_code == 0, rescored.stderr
    assert rescored.stdout == "".join(f"{name}: unchanged\n" for name in ALL_SCENARIOS[:3])


def test_each_request_json_is_the_eval_prompt_payload_and_is_sent_as_params(
    tmp_path: Path,
) -> None:
    run = make_run(tmp_path, FakeBatches())

    run()

    [create] = [call for call in run.server.calls if call.method == "POST"]
    requests = json.loads(create.body or b"")["requests"]
    assert [request["custom_id"] for request in requests] == list(ALL_SCENARIOS)
    cli = CliRunner()
    for request in requests:
        name = request["custom_id"]
        printed = cli.invoke(app, ["eval", "prompt", str(run.corpus / name), "--model", HAIKU])
        assert printed.exit_code == 0
        assert (run.model_dir / name / "request.json").read_bytes() == printed.stdout_bytes
        assert request["params"] == json.loads(printed.stdout_bytes)


def test_every_call_carries_the_key_the_pinned_version_and_the_timeouts(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches())

    run()

    assert ANTHROPIC_VERSION == "2026-10-21"
    for call in run.server.calls:
        # TASK-005c3d H7: the documented Bearer header; no x-api-key, no anthropic-workspace-id.
        assert call.headers == {
            "authorization": f"Bearer {KEY}",
            "anthropic-version": "2026-10-21",
            "content-type": "application/json",
            "user-agent": f"potakt/{__version__}",
        }
        assert KEY not in call.url
    timeouts = [
        (call.method, call.url.endswith("/results"), call.timeout) for call in run.server.calls
    ]
    assert timeouts == [
        ("GET", False, 60.0),  # TASK-005c3g G2: the clock call
        ("GET", False, 60.0),  # TASK-005c3h H6: the workspace list
        ("POST", False, 60.0),
        ("GET", False, 60.0),
        ("GET", False, 60.0),
        ("GET", True, 300.0),
    ]


def test_the_key_is_in_no_file_and_no_echoed_line(tmp_path: Path) -> None:
    server = FakeBatches(
        outcome=lambda n, s: errored("overloaded_error") if n == 1 and s < "03" else succeeded(),
        create_script=["lost-none"],
    )
    run = make_run(tmp_path, server)

    run()

    assert run.recorded() == list(ALL_SCENARIOS)
    assert files_containing(tmp_path, KEY) == []
    assert all(KEY not in line for line in run.echo)


def test_a_status_line_is_printed_only_when_it_changes(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(polls_in_progress=3), ALL_SCENARIOS[:2])

    run()

    assert run.sleeps == [120, 60, 60, 60]  # TASK-005c3h H6: the workspace wait first
    assert [line for line in run.echo if line.startswith("batch msgbatch_01:")] == [
        "batch msgbatch_01: in_progress "
        "(processing 2, succeeded 0, errored 0, expired 0, canceled 0)",
        "batch msgbatch_01: ended (processing 0, succeeded 2, errored 0, expired 0, canceled 0)",
    ]


def test_an_unchanged_status_line_is_repeated_after_every_ten_polls(tmp_path: Path) -> None:
    # D5: a heartbeat, counted in polls (not the clock): polls 1, 11 and 21 show the line.
    run = make_run(tmp_path, FakeBatches(polls_in_progress=25), ALL_SCENARIOS[:1])

    run()

    in_progress = (
        "batch msgbatch_01: in_progress "
        "(processing 1, succeeded 0, errored 0, expired 0, canceled 0)"
    )
    status_lines = [line for line in run.echo if line.startswith("batch msgbatch_01:")]
    assert status_lines == [in_progress] * 3 + [
        "batch msgbatch_01: ended (processing 0, succeeded 1, errored 0, expired 0, canceled 0)"
    ]
    assert HEARTBEAT_POLLS == 10


def test_nothing_pending_prints_nothing_to_run_and_sends_nothing(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:2])
    run()
    run.server.calls.clear()
    run.echo.clear()

    run()

    assert after_the_clock(run.server) == []
    assert run.echo == ["nothing to run"]


def after_the_clock(server: "FakeBatches") -> list[Call]:
    """The calls of a run after its first, which must be the clock call (TASK-005c3g G2): a run
    that stops "before any call" now makes that one free call first."""
    assert server.calls[0].url == CLOCK_URL
    return server.calls[1:]


def after_the_workspace(server: "FakeBatches") -> list[Call]:
    """The calls of a first run after the clock call and the workspace list (TASK-005c3h H6),
    which are free and come before the requests are rendered."""
    assert server.calls[1].url == LIST_URL
    return after_the_clock(server)[1:]


# --- resume and pending ----------------------------------------------------------------------


def test_an_open_batch_is_resumed_without_a_create_call(tmp_path: Path) -> None:
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:3])
    batch_id = server.register(list(ALL_SCENARIOS[:3]))
    write_state(run.model_dir, batch_entry(batch_id, ALL_SCENARIOS[:3], "open"))

    run()

    assert [call.method for call in after_the_clock(server)] == ["GET", "GET", "GET"]
    assert run.echo[0] == "resuming batch msgbatch_01"
    assert run.recorded() == list(ALL_SCENARIOS[:3])
    assert run.state()["batches"][0]["state"] == "done"


def test_recorded_scenarios_are_never_sent(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:4])
    for name in ALL_SCENARIOS[:2]:
        scenario = load_scenario(run.corpus / name)
        write_answer(run.model_dir / name, scenario, HAIKU, anthropic_message(HAIKU))

    run()

    assert run.server.creates() == [list(ALL_SCENARIOS[2:4])]
    assert run.recorded() == list(ALL_SCENARIOS[:4])


def test_errored_and_expired_are_resubmitted_in_a_second_batch(tmp_path: Path) -> None:
    def outcome(number: int, scenario: str) -> dict[str, Any]:
        if number == 1 and scenario == "02-hotel-reception":
            return errored("overloaded_error", "Overloaded")
        if number == 1 and scenario == "04-restaurant":
            return {"type": "expired"}
        if number == 1 and scenario == "05-supermarket-checkouts":
            return {"type": "canceled"}
        return succeeded()

    run = make_run(tmp_path, FakeBatches(outcome=outcome), ALL_SCENARIOS[:5])

    run()

    assert run.server.creates() == [
        list(ALL_SCENARIOS[:5]),
        ["02-hotel-reception", "04-restaurant", "05-supermarket-checkouts"],
    ]
    assert run.recorded() == list(ALL_SCENARIOS[:5])
    assert run.sleeps == [120, 60, 15, 60]  # TASK-005c3h H6: the workspace wait first
    assert [batch["state"] for batch in run.state()["batches"]] == ["done", "done"]
    assert "02-hotel-reception: errored (overloaded_error); pending" in run.echo
    assert "04-restaurant: expired; pending" in run.echo
    assert "05-supermarket-checkouts: canceled; pending" in run.echo


# TASK-005c3c E4: the ending of every stop that holds a batch.
HELD = (
    "; batch msgbatch_01 is held: send batch-msgbatch_01.results.jsonl and this message to the "
    "reviewer"
)


def problems_of(run: "Run", index: int = 0) -> list[str]:
    return run.state()["batches"][index]["problems"]


def test_a_scenario_missing_from_the_results_holds_the_batch(tmp_path: Path) -> None:
    # D3: it used to stay pending silently, and the next batch would pay for it again.
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:3])
    first = json.dumps({"custom_id": ALL_SCENARIOS[0], "result": succeeded()}) + "\n"
    server.raw_results["msgbatch_01"] = first.encode()

    with pytest.raises(RunStopped) as raised:
        run()

    problems = [
        "batch msgbatch_01: no result for 02-hotel-reception",
        "batch msgbatch_01: no result for 03-call-centre",
    ]
    assert str(raised.value) == "; ".join(problems) + HELD
    assert run.server.creates() == [list(ALL_SCENARIOS[:3])]
    assert run.recorded() == [ALL_SCENARIOS[0]]
    assert run.states() == ["held"]
    assert problems_of(run) == problems
    raw = run.model_dir / "batch-msgbatch_01.results.jsonl"
    assert raw.read_bytes() == first.encode()


def test_a_scenario_in_four_batches_is_not_resubmitted(tmp_path: Path) -> None:
    server = FakeBatches(
        outcome=lambda n, s: {"type": "expired"} if s == "02-hotel-reception" else succeeded()
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == (
        "02-hotel-reception was sent in 4 batches without an answer; it is not sent again"
    )
    assert server.creates() == [list(ALL_SCENARIOS[:2])] + [["02-hotel-reception"]] * 3
    assert run.sleeps == [120, 60, 15, 60, 45, 60, 135, 60]  # H6: the workspace wait first
    assert run.recorded() == ["01-pharmacy"]


def test_the_attempt_cap_holds_across_invocations(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:2])
    batches = [batch_entry(f"msgbatch_9{n}", ["02-hotel-reception"], "done") for n in range(4)]
    write_state(run.model_dir, *batches)

    with pytest.raises(RunStopped, match="02-hotel-reception was sent in 4 batches"):
        run()

    assert after_the_clock(run.server) == []


def test_a_failed_create_is_not_counted_as_a_batch(tmp_path: Path) -> None:
    # D1: a failed create was never a batch, so it does not use up one of the four.
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:2])
    batches = [batch_entry(f"msgbatch_9{n}", ["02-hotel-reception"], "done") for n in range(3)]
    failed = [batch_entry(None, ["02-hotel-reception"], "failed") for _ in range(3)]
    write_state(run.model_dir, *batches, *failed)

    run()

    assert run.server.creates() == [list(ALL_SCENARIOS[:2])]
    assert run.recorded() == list(ALL_SCENARIOS[:2])


# --- stops -----------------------------------------------------------------------------------


def test_an_invalid_request_error_stops_after_recording_the_other_lines(tmp_path: Path) -> None:
    server = FakeBatches(
        outcome=lambda n, s: (
            errored("invalid_request_error", "output_config: unknown field")
            if s == "02-hotel-reception"
            else succeeded()
        )
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:3])

    with pytest.raises(RunStopped) as raised:
        run()

    problem = "02-hotel-reception: invalid_request_error: output_config: unknown field"
    assert str(raised.value) == problem + HELD
    assert run.recorded() == ["01-pharmacy", "03-call-centre"]
    assert not (run.model_dir / "02-hotel-reception" / "response.json").exists()
    assert (run.model_dir / "02-hotel-reception" / "request.json").is_file()
    assert run.states() == ["held"]
    assert problems_of(run) == [problem]
    assert len(server.creates()) == 1


@pytest.mark.parametrize(
    "line",
    [
        b"not json",
        b'{"custom_id": "99-unknown", "result": {"type": "expired"}}',
        b'{"custom_id": "02-hotel-reception", "result": {"type": "mystery"}}',
        b'{"custom_id": "02-hotel-reception"}',
        b"[1]",
    ],
    ids=["not-json", "unknown-id", "unknown-type", "no-result", "not-an-object"],
)
def test_a_malformed_results_line_stops_after_the_raw_file_is_saved(
    tmp_path: Path, line: bytes
) -> None:
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    good = json.dumps({"custom_id": "01-pharmacy", "result": succeeded()}).encode()
    server.raw_results["msgbatch_01"] = good + b"\n" + line + b"\n"

    with pytest.raises(RunStopped, match=r"^batch msgbatch_01: results line 2 ") as raised:
        run()

    assert str(raised.value).endswith(
        "; batch msgbatch_01: no result for 02-hotel-reception" + HELD
    )
    raw = run.model_dir / "batch-msgbatch_01.results.jsonl"
    assert raw.read_bytes() == server.raw_results["msgbatch_01"]
    assert run.recorded() == ["01-pharmacy"]
    assert run.states() == ["held"]
    assert "; ".join(problems_of(run)) + HELD == str(raised.value)
    assert problems_of(run)[0].startswith("batch msgbatch_01: results line 2 ")


def test_a_duplicate_results_line_holds_the_batch(tmp_path: Path) -> None:
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    line = json.dumps({"custom_id": "01-pharmacy", "result": succeeded()}).encode()
    server.raw_results["msgbatch_01"] = line + b"\n" + line + b"\n"

    with pytest.raises(RunStopped) as raised:
        run()

    problem = "batch msgbatch_01: results line 2 names 01-pharmacy twice"
    assert str(raised.value) == problem + HELD
    assert run.recorded() == ["01-pharmacy"]
    assert run.states() == ["held"]
    assert problems_of(run) == [problem]


def test_an_invalid_request_error_and_a_missing_result_are_both_held(tmp_path: Path) -> None:
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    line = {"custom_id": "01-pharmacy", "result": errored("invalid_request_error", "bad")}
    server.raw_results["msgbatch_01"] = (json.dumps(line) + "\n").encode()

    with pytest.raises(RunStopped) as raised:
        run()

    problems = [
        "batch msgbatch_01: no result for 02-hotel-reception",
        "01-pharmacy: invalid_request_error: bad",
    ]
    assert str(raised.value) == "; ".join(problems) + HELD
    assert problems_of(run) == problems
    assert not hasattr(runner_module, "DO_NOT_RUN_AGAIN")
    assert not hasattr(runner_module, "TO_THE_REVIEWER")


def test_a_results_url_on_another_host_is_refused(tmp_path: Path) -> None:
    server = FakeBatches(results_base="https://evil.example.com/v1/messages/batches")
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped, match=r"refused: evil\.example\.com is not an API host"):
        run()

    assert not any("evil" in call.url for call in server.calls)
    assert run.recorded() == []


def test_a_results_url_on_the_google_host_is_refused_before_any_request(tmp_path: Path) -> None:
    # U3: an Anthropic results_url on the Google host would have carried the Anthropic key there.
    google = "https://generativelanguage.googleapis.com/v1/messages/batches"
    server = FakeBatches(results_base=google)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == (
        "batch msgbatch_01: results_url refused: generativelanguage.googleapis.com is not the API "
        "host of this key"
    )
    assert not any("googleapis" in call.url for call in server.calls)
    assert run.states() == ["open"]


@pytest.mark.parametrize("code", [400, 401, 403, 404, 409, 413, 422])
def test_a_create_refused_with_another_status_is_marked_failed_and_stops(
    tmp_path: Path, code: int
) -> None:
    body = json.dumps({"type": "error", "error": {"type": "some_error", "message": "bad"}})
    server = FakeBatches(create_script=[HttpResponse(code, {}, body.encode())])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == f"creating a batch failed: HTTP {code} (some_error: bad)"
    assert run.state()["batches"] == [batch_entry(None, ALL_SCENARIOS[:1], "failed")]
    assert run.sleeps == [120]  # TASK-005c3h H6: only the workspace wait
    assert run.lists() == 1  # only the workspace list: no reconciliation


def test_a_create_answered_with_429_is_simply_retried(tmp_path: Path) -> None:
    server = FakeBatches(create_script=[HttpResponse(429, {"retry-after": "20"}, b"")])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert len(server.creates()) == 2
    assert run.lists() == 1  # TASK-005c3h H6: the workspace list only
    assert run.sleeps == [120, 20, 60]
    assert run.states() == ["done"]


def test_four_429_answers_mark_the_create_failed(tmp_path: Path) -> None:
    server = FakeBatches(create_script=[HttpResponse(429, {}, b"")] * 4)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == "creating a batch failed 4 times; last: HTTP 429"
    assert run.sleeps == [120, 15, 45, 135]  # TASK-005c3h H6: the workspace wait first
    assert run.lists() == 1  # the workspace list only
    assert run.states() == ["failed"]


def test_a_poll_failure_stops_the_run_and_keeps_the_batch_open(tmp_path: Path) -> None:
    server = FakeBatches(poll_failure=TransportError("TimeoutError: no response received"))
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped, match="the request failed 4 times"):
        run()

    assert run.state()["batches"][0]["state"] == "open"
    assert run.sleeps == [120, 15, 45, 135]  # TASK-005c3h H6: the workspace wait first


def test_a_poll_answered_with_404_stops_the_run(tmp_path: Path) -> None:
    body = json.dumps({"type": "error", "error": {"type": "not_found_error", "message": "gone"}})
    server = FakeBatches(poll_failure=HttpResponse(404, {}, body.encode()))
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == (
        "polling batch msgbatch_01 failed: HTTP 404 (not_found_error: gone)"
    )


@pytest.mark.parametrize(
    "body",
    [b"[]", b"{}", b'{"id": "msgbatch_01", "processing_status": "weird"}', b"no"],
)
def test_an_unexpected_batch_object_stops_the_run(tmp_path: Path, body: bytes) -> None:
    server = FakeBatches(poll_failure=HttpResponse(200, {}, body))
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(
        RunStopped, match=r"^polling batch msgbatch_01 failed: unexpected batch object"
    ):
        run()


# --- F3: a create whose response was lost ------------------------------------------------------


def other_batch(batch_id: str, total: int, created: str) -> dict[str, Any]:
    counts = {"processing": total, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}
    return {
        "id": batch_id,
        "type": "message_batch",
        "processing_status": "in_progress",
        "request_counts": counts,
        "created_at": created,
        "results_url": None,
    }


def iso(moment: timedelta) -> str:
    return (NOW + moment).isoformat().replace("+00:00", "Z")


def test_a_lost_create_adopts_the_one_candidate_batch(tmp_path: Path) -> None:
    server = FakeBatches(
        create_script=["lost"],
        listing=[
            other_batch("msgbatch_old", 3, iso(-timedelta(minutes=11))),
            other_batch("msgbatch_size", 2, iso(-timedelta(minutes=1))),
        ],
        extras_after_create=True,
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:3])

    run()

    assert len(server.creates()) == 1
    assert "adopted batch msgbatch_01 created by an earlier create request" in run.echo
    assert run.state()["batches"] == [
        {
            "id": "msgbatch_01",
            "scenarios": list(ALL_SCENARIOS[:3]),
            "started_at": STARTED_AT,
            "created_at": CREATED_AT,
            "state": "done",
            "problems": [],
        }
    ]
    assert run.sleeps == [120, 120, 60]  # TASK-005c3h H6: the workspace wait, then as before
    assert run.lists() == 2
    assert run.recorded() == list(ALL_SCENARIOS[:3])


def test_a_batch_created_exactly_ten_minutes_before_is_a_candidate(tmp_path: Path) -> None:
    server = FakeBatches(create_script=["lost-none"], extras_after_create=True)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    # Created by the lost request at the edge of the window (the fake's ids are only names).
    server.batches["msgbatch_edge"] = list(ALL_SCENARIOS[:1])
    server.created_at["msgbatch_edge"] = iso(-timedelta(minutes=10))

    run()

    assert "adopted batch msgbatch_edge created by an earlier create request" in run.echo
    assert len(server.creates()) == 1


def test_a_lost_create_with_no_candidate_is_checked_once_then_retried(tmp_path: Path) -> None:
    server = FakeBatches(
        create_script=["lost-none"],
        listing=[other_batch("msgbatch_old", 1, iso(-timedelta(minutes=30)))],
        extras_after_create=True,
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert len(server.creates()) == 2
    # E3: one list call at started + 120 s, then the retry delay and the poll (after the H6
    # workspace wait and list).
    assert run.sleeps == [120, 120, 15, 60]
    assert run.lists() == 2
    assert server.list_times == [NOW, NOW + timedelta(seconds=120)]
    assert [batch["id"] for batch in run.state()["batches"]] == ["msgbatch_01"]
    # started_at is the instant just before the create call that created the batch (the retry).
    assert run.state()["batches"][0]["started_at"] == "2026-11-02T10:02:15Z"


def test_a_batch_already_in_the_state_is_no_candidate(tmp_path: Path) -> None:
    server = FakeBatches(create_script=["lost-none"])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    known = server.register(
        ["01-pharmacy", "02-hotel-reception"], created_at=iso(-timedelta(minutes=2))
    )
    write_state(run.model_dir, batch_entry(known, ALL_SCENARIOS[:2], "done"))

    run()

    assert len(server.creates()) == 2  # the lost one and its retry
    assert [batch["id"] for batch in run.state()["batches"]] == [known, "msgbatch_02"]


def test_two_candidates_stop_the_run_naming_both(tmp_path: Path) -> None:
    server = FakeBatches(
        create_script=["lost"],
        listing=[other_batch("msgbatch_twin", 1, iso(timedelta(seconds=1)))],
        extras_after_create=True,
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == (
        "2 recent batches could be the one created by an earlier create request: msgbatch_01, "
        "msgbatch_twin; in the Claude Console (Batches) cancel all but the oldest, then run the "
        "same command again: it adopts the one left; a batch that has already ended cannot be "
        "cancelled: then send this message to the reviewer"
    )
    assert run.state()["batches"] == [batch_entry(None, ALL_SCENARIOS[:1], "creating")]
    assert len(server.creates()) == 1


def test_four_lost_creates_stop_the_run_and_mark_the_create_failed(tmp_path: Path) -> None:
    server = FakeBatches(create_script=["lost-none"] * 4)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped, match=r"^creating a batch failed 4 times; last: TimeoutError"):
        run()

    assert len(server.creates()) == 4
    assert run.sleeps == [120, 120, 15, 120, 45, 120, 135, 120]  # H6: the workspace wait first
    assert run.lists() == 5
    assert run.states() == ["failed"]


def test_a_create_answered_200_without_a_batch_object_is_treated_as_lost(tmp_path: Path) -> None:
    server = FakeBatches(create_script=[HttpResponse(200, {}, b"<html>")])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert len(server.creates()) == 2
    assert run.lists() == 2  # TASK-005c3h H6: the workspace list, then the reconciliation
    assert run.sleeps == [120, 120, 15, 60]


@pytest.mark.parametrize(
    ("listing", "message"),
    [
        (HttpResponse(403, {}, b""), "listing batches failed: HTTP 403"),
        (ok({"data": "none"}), "listing batches failed: unexpected list object"),
        (ok({"data": [{"id": "x"}]}), "listing batches failed: unexpected batch object"),
        (
            ok({"data": [other_batch("msgbatch_x", 1, "yesterday")]}),
            "listing batches failed: unexpected batch object",
        ),
        (
            ok({"data": [other_batch("msgbatch_x", 1, "2026-13-45T10:00:00Z")]}),
            "listing batches failed: unexpected batch object",
        ),
    ],
)
def test_an_unusable_listing_stops_the_run(
    tmp_path: Path, listing: HttpResponse, message: str
) -> None:
    server = FakeBatches(create_script=["lost-none"], list_answer=listing, extras_after_create=True)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == message
    assert run.states() == ["creating"]  # the next run reconciles it again


def test_a_fractional_second_beyond_microseconds_is_read(tmp_path: Path) -> None:
    server = FakeBatches(create_script=["lost-none"], extras_after_create=True)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    server.batches["msgbatch_ns"] = list(ALL_SCENARIOS[:1])
    server.created_at["msgbatch_ns"] = "2026-11-02T12:00:05.123456789+02:00"

    run()

    assert "adopted batch msgbatch_ns created by an earlier create request" in run.echo


def test_a_negative_request_count_is_an_unexpected_batch_object(tmp_path: Path) -> None:
    batch = other_batch("msgbatch_01", 1, CREATED_AT)
    batch["request_counts"]["errored"] = -1
    server = FakeBatches(poll_failure=ok(batch))
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped, match="unexpected batch object"):
        run()


def test_an_ended_batch_without_a_results_url_stops(tmp_path: Path) -> None:
    batch = other_batch("msgbatch_01", 1, CREATED_AT) | {"processing_status": "ended"}
    server = FakeBatches(poll_failure=ok(batch))
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped, match=r"^batch msgbatch_01 ended without a results_url$"):
        run()


def test_a_failed_results_download_stops_and_keeps_the_batch_open(tmp_path: Path) -> None:
    server = FakeBatches(results_answer=HttpResponse(404, {}, b""))
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == "downloading the results of batch msgbatch_01 failed: HTTP 404"
    assert run.state()["batches"][0]["state"] == "open"


def test_a_line_for_an_already_recorded_scenario_is_skipped(tmp_path: Path) -> None:
    # A run interrupted after recording a line but before marking its batch done.
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    batch_id = server.register(list(ALL_SCENARIOS[:2]))
    scenario = load_scenario(run.corpus / ALL_SCENARIOS[0])
    write_answer(run.model_dir / ALL_SCENARIOS[0], scenario, HAIKU, anthropic_message(HAIKU))
    write_state(run.model_dir, batch_entry(batch_id, ALL_SCENARIOS[:2], "open"))

    run()

    assert f"{ALL_SCENARIOS[0]}: already recorded" in run.echo
    assert run.recorded() == list(ALL_SCENARIOS[:2])


def test_a_succeeded_result_without_a_message_holds_the_batch(tmp_path: Path) -> None:
    server = FakeBatches(outcome=lambda n, s: {"type": "succeeded"})
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == "01-pharmacy: a succeeded result without a message" + HELD
    assert run.recorded() == []
    assert run.states() == ["held"]
    assert problems_of(run) == ["01-pharmacy: a succeeded result without a message"]


def state_text(*entries: dict[str, Any]) -> str:
    return json.dumps({"batches": list(entries)})


OPEN = batch_entry("msgbatch_01", ["01-pharmacy"], "open")
CREATING = batch_entry(None, ["01-pharmacy"], "creating")
HELD_ENTRY = batch_entry("msgbatch_01", ["01-pharmacy"], "held", problems=["a problem"])


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '{"batches": {}}',
        '{"batches": [1]}',
        state_text(OPEN | {"id": "../x"}),
        state_text(OPEN | {"scenarios": [1]}),
        state_text(OPEN | {"scenarios": "01-pharmacy"}),
        state_text(OPEN | {"state": "lost"}),
        state_text(OPEN | {"id": None}),
        state_text(OPEN | {"created_at": None}),
        state_text(batch_entry("msgbatch_01", ["01-pharmacy"], "done") | {"created_at": 5}),
        state_text(CREATING | {"id": "msgbatch_01"}),
        state_text(CREATING | {"created_at": CREATED_AT}),
        state_text(batch_entry(None, ["01-pharmacy"], "failed") | {"id": "msgbatch_01"}),
        state_text(OPEN | {"started_at": "2026-11-02T10:00:00.5Z"}),
        state_text(OPEN | {"started_at": "2026-11-02T10:00:00+00:00"}),
        state_text(OPEN | {"started_at": "2026-13-02T10:00:00Z"}),
        state_text(OPEN | {"started_at": None}),
        state_text({key: value for key, value in OPEN.items() if key != "started_at"}),
        state_text(OPEN | {"extra": 1}),
        state_text(OPEN, CREATING),
        state_text(CREATING, CREATING),
        state_text(OPEN, OPEN | {"id": "msgbatch_02"}),
        # TASK-005c3c E4: the problems key and the held state.
        state_text({key: value for key, value in OPEN.items() if key != "problems"}),
        state_text(HELD_ENTRY | {"problems": []}),
        state_text(HELD_ENTRY | {"id": None}),
        state_text(HELD_ENTRY | {"created_at": None}),
        state_text(HELD_ENTRY | {"problems": "a problem"}),
        state_text(HELD_ENTRY | {"problems": [5]}),
        state_text(HELD_ENTRY | {"problems": [""]}),
        state_text(HELD_ENTRY | {"problems": ["bad\x1b[2J"]}),
        state_text(OPEN | {"problems": ["a problem"]}),
        state_text(CREATING | {"problems": ["a problem"]}),
        state_text(batch_entry(None, ["01-pharmacy"], "failed") | {"problems": ["a problem"]}),
        # TASK-005c3e J2: a stored created_at is an RFC 3339 instant.
        state_text(OPEN | {"created_at": "x"}),
        state_text(OPEN | {"created_at": "\ud83d"}),
        state_text(HELD_ENTRY | {"created_at": "yesterday"}),
        state_text(batch_entry("msgbatch_01", ["01-pharmacy"], "done") | {"created_at": ""}),
    ],
)
def test_an_invalid_batch_state_stops_and_is_kept(tmp_path: Path, text: str) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    run.model_dir.mkdir(parents=True)
    (run.model_dir / BATCH_STATE_FILE).write_text(text, encoding="utf-8")

    with pytest.raises(RunStopped, match="is not a valid batch state; it was not changed"):
        run()

    assert (run.model_dir / BATCH_STATE_FILE).read_text(encoding="utf-8") == text
    assert run.server.calls == []


def test_a_missing_corpus_stops(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ())
    run.corpus = tmp_path / "nope"

    with pytest.raises(RunStopped, match="is not a directory"):
        run()


def test_an_invalid_scenario_stops_before_any_call(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    (run.corpus / ALL_SCENARIOS[0] / "truth.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RunStopped, match=r"^01-pharmacy: invalid scenario: "):
        run()

    assert after_the_workspace(run.server) == []


def test_a_scenario_id_that_is_no_custom_id_stops(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    (run.corpus / "01 pharmacy").mkdir()
    (run.corpus / "01 pharmacy" / "scenario.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RunStopped, match="01 pharmacy: not a valid custom_id"):
        run()

    assert after_the_workspace(run.server) == []


def test_a_different_existing_request_stops(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    (run.model_dir / ALL_SCENARIOS[0]).mkdir(parents=True)
    (run.model_dir / ALL_SCENARIOS[0] / "request.json").write_bytes(b"{}\n")

    with pytest.raises(RunStopped, match="exists with a different request"):
        run()

    assert after_the_workspace(run.server) == []


def test_a_missing_instruction_file_stops_the_runner_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The CLI checks the prompt first (TASK-005c3g G8); the runner stops on its own too.
    missing = tmp_path / "test-a.md"
    monkeypatch.setattr(test_a_module, "INSTRUCTION_PATH", missing)
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped, match=r"^cannot read the instruction file .*test-a\.md: "):
        run()

    assert after_the_workspace(run.server) == []


def test_only_claude_models_run_on_batches(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"gemini-3\.8-flash is not a Claude model"):
        run_anthropic(
            "gemini-3.8-flash", tmp_path, tmp_path, KEY, FakeBatches(), sleep=Sleeps(),
            now=lambda: NOW, echo=Echo(),
        )  # fmt: skip


def test_the_stop_message_is_redacted_of_the_key(tmp_path: Path) -> None:
    body = json.dumps({"type": "error", "error": {"type": "authentication_error", "message": KEY}})
    server = FakeBatches(create_script=[HttpResponse(401, {}, body.encode())])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == (
        "creating a batch failed: HTTP 401 (authentication_error: [redacted])"
    )


# --- ultrareview findings ----------------------------------------------------------------------


def test_a_half_written_record_is_completed_before_anything_is_sent(tmp_path: Path) -> None:
    # A run interrupted after response.json and before the derived files were written.
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:2])
    scenario = load_scenario(run.corpus / "01-pharmacy")
    write_answer(run.model_dir / "01-pharmacy", scenario, HAIKU, anthropic_message(HAIKU))
    (run.model_dir / "01-pharmacy" / "comparison.json").unlink()

    run()

    assert run.echo[0] == "01-pharmacy: record completed (comparison.json)"
    assert (run.model_dir / "01-pharmacy" / "comparison.json").is_file()
    assert run.server.creates() == [["02-hotel-reception"]]
    assert summarize(run.model_dir, run.corpus).complete


def test_a_half_written_record_is_completed_when_nothing_is_pending(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    scenario = load_scenario(run.corpus / "01-pharmacy")
    write_answer(run.model_dir / "01-pharmacy", scenario, HAIKU, anthropic_message(HAIKU))
    (run.model_dir / "01-pharmacy" / "candidate.txt").unlink()
    (run.model_dir / "01-pharmacy" / "comparison.json").unlink()

    run()

    assert run.echo == [
        "01-pharmacy: record completed (candidate.txt, comparison.json)",
        "nothing to run",
    ]
    assert after_the_clock(run.server) == []


def test_an_interrupt_during_the_create_call_is_left_to_the_caller(tmp_path: Path) -> None:
    # D2 step 5: the creating entry is on disk, so the runner no longer catches the interrupt.
    run = make_run(tmp_path, FakeBatches(create_script=["interrupt-created"]), ALL_SCENARIOS[:1])

    with pytest.raises(KeyboardInterrupt):
        run()

    assert run.state()["batches"] == [batch_entry(None, ALL_SCENARIOS[:1], "creating")]
    assert not hasattr(runner_module, "INTERRUPTED_CREATE")


def test_an_interrupt_while_polling_is_left_to_the_caller(tmp_path: Path) -> None:
    server = FakeBatches(poll_failure=TransportError("unused"))
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    class Interrupting(Sleeps):
        """Ctrl+C during the first pause after the create (TASK-005c3h H6: the workspace wait
        before it passes)."""

        def __call__(self, seconds: float) -> None:
            if seconds != runner_module.RECONCILE_DELAY_SECONDS:
                raise KeyboardInterrupt
            super().__call__(seconds)

    run.sleeps = Interrupting()

    with pytest.raises(KeyboardInterrupt):
        run()

    assert run.state()["batches"][0]["state"] == "open"


# --- review findings (TASK-005c3 self-review) --------------------------------------------------


def test_a_key_across_the_cut_of_an_invalid_request_message_is_redacted(tmp_path: Path) -> None:
    # Minor 4: the message was cut to 300 characters before the key was redacted.
    text = "x" * 290 + KEY
    server = FakeBatches(outcome=lambda n, s: errored("invalid_request_error", text))
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == (
        f"01-pharmacy: invalid_request_error: {'x' * 290}[redacted]" + HELD
    )
    # The held problem is written redacted too (the raw results file keeps the server's bytes).
    assert KEY.encode() not in (run.model_dir / BATCH_STATE_FILE).read_bytes()
    assert problems_of(run) == [f"01-pharmacy: invalid_request_error: {'x' * 290}[redacted]"]


def test_a_server_message_echoing_the_bare_bearer_key_is_redacted_everywhere(
    tmp_path: Path,
) -> None:
    # TASK-005c3d H7: the header value is "Bearer <key>"; the server may echo the key alone.
    server = FakeBatches(
        outcome=lambda n, s: errored("invalid_request_error", f"key {KEY} is not allowed")
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    problem = "01-pharmacy: invalid_request_error: key [redacted] is not allowed"
    assert str(raised.value) == problem + HELD
    assert problems_of(run) == [problem]
    assert KEY.encode() not in (run.model_dir / BATCH_STATE_FILE).read_bytes()
    assert all(KEY not in line for line in run.echo)


def test_a_key_across_the_cut_of_an_http_error_is_redacted(tmp_path: Path) -> None:
    body = {"type": "error", "error": {"type": "permission_error", "message": "x" * 290 + KEY}}
    server = FakeBatches(create_script=[HttpResponse(403, {}, json.dumps(body).encode())])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert KEY[:5] not in str(raised.value)


@pytest.mark.parametrize(
    "kind",
    ["\x1b[2J", "a" * 65, 7, None, ["api_error"], {"type": "api_error"}],  # G14: unhashable too
)
def test_an_odd_error_type_is_not_echoed_and_holds_the_batch(tmp_path: Path, kind: object) -> None:
    # Minor 3: the error type of a result line was echoed as the server sent it. TASK-005c3g G7:
    # an unknown type holds the batch (it may fail again).
    def outcome(number: int, scenario: str) -> dict[str, Any]:
        if number > 1:
            return succeeded()
        return {"type": "errored", "error": {"type": "error", "error": {"type": kind}}}

    run = make_run(tmp_path, FakeBatches(outcome=outcome), ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == "01-pharmacy: errored (unknown error); it would fail again" + HELD
    assert run.states() == ["held"]


# --- TASK-005c3g G7: which errored results are sent again ------------------------------------

TRANSIENT = ["api_error", "overloaded_error", "rate_limit_error", "timeout_error"]
# The other documented types of a result (api/errors), two newer ones and an unknown one.
LASTING = [
    "invalid_request_error",
    "authentication_error",
    "billing_error",
    "permission_error",
    "not_found_error",
    "conflict_error",
    "request_too_large",
    "some_new_error",
]


def first_errored(kind: str) -> Outcome:
    """01-pharmacy errored with ``kind`` in the first batch; everything else succeeds."""

    def outcome(number: int, scenario: str) -> dict[str, Any]:
        if number == 1 and scenario == "01-pharmacy":
            return errored(kind, "details")
        return succeeded()

    return outcome


@pytest.mark.parametrize("kind", TRANSIENT)
def test_a_transient_error_type_is_sent_again_in_a_new_batch(tmp_path: Path, kind: str) -> None:
    run = make_run(tmp_path, FakeBatches(outcome=first_errored(kind)), ALL_SCENARIOS[:2])

    run()

    assert f"01-pharmacy: errored ({kind}); pending" in run.echo
    assert run.server.creates() == [list(ALL_SCENARIOS[:2]), ["01-pharmacy"]]
    assert run.states() == ["done", "done"]
    assert run.recorded() == list(ALL_SCENARIOS[:2])


@pytest.mark.parametrize("kind", LASTING)
def test_any_other_error_type_holds_the_batch(tmp_path: Path, kind: str) -> None:
    run = make_run(tmp_path, FakeBatches(outcome=first_errored(kind)), ALL_SCENARIOS[:2])

    with pytest.raises(RunStopped) as raised:
        run()

    problem = (
        "01-pharmacy: invalid_request_error: details"  # its message, as before (§8)
        if kind == "invalid_request_error"
        else f"01-pharmacy: errored ({kind}); it would fail again"
    )
    assert str(raised.value) == problem + HELD
    assert run.server.creates() == [list(ALL_SCENARIOS[:2])]
    assert run.states() == ["held"]
    assert problems_of(run) == [problem]
    assert run.recorded() == ["02-hotel-reception"]


def test_a_billing_error_holds_after_one_batch_so_the_cap_is_never_reached(
    tmp_path: Path,
) -> None:
    run = make_run(tmp_path, FakeBatches(outcome=lambda n, s: errored("billing_error")))

    with pytest.raises(RunStopped) as raised:
        run()

    assert "without an answer" not in str(raised.value)
    assert str(raised.value).startswith("01-pharmacy: errored (billing_error); it would fail again")
    assert len(run.server.creates()) == 1
    assert run.states() == ["held"]


def test_an_invalid_request_message_loses_its_control_characters(tmp_path: Path) -> None:
    server = FakeBatches(outcome=lambda n, s: errored("invalid_request_error", "bad\x1b[31m\n"))
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == "01-pharmacy: invalid_request_error: bad?[31m?" + HELD


def test_the_run_state_repr_never_shows_the_key(tmp_path: Path) -> None:
    # Minor 5: the dataclass repr included the request headers.
    state = runner_module._AnthropicRun(
        model=HAIKU,
        corpus_dir=tmp_path,
        model_dir=tmp_path,
        headers=runner_module.anthropic_headers(KEY),
        transport=FakeBatches(),
        sleep=Sleeps(),
        now=lambda: NOW,
        poll_seconds=60,
        echo=Echo(),
    )

    assert KEY not in repr(state)


def test_a_batch_of_another_model_is_never_adopted(tmp_path: Path) -> None:
    # Minor 7: a batch recorded by another model with the same size, created minutes before.
    server = FakeBatches(create_script=["lost-none"])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    other = server.register(list(ALL_SCENARIOS[:1]), created_at=iso(-timedelta(minutes=3)))
    write_state(run.runs / OPUS, batch_entry(other, ALL_SCENARIOS[:1], "done"))

    run()

    assert not any("adopted" in line for line in run.echo)
    assert len(server.creates()) == 2
    assert run.batch_ids() == ["msgbatch_02"]


# --- TASK-005c3b D1, D2: the creating state and the reconciliation -----------------------------


def test_the_creating_entry_is_saved_before_the_first_post(tmp_path: Path) -> None:
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    seen: list[dict[str, Any]] = []

    class Watching:
        def send(
            self,
            method: str,
            url: str,
            headers: dict[str, str],
            body: bytes | None,
            timeout: float,
        ) -> HttpResponse:
            if method == "POST":
                seen.append(run.state())
            return server.send(method, url, headers, body, timeout)

    run.sleeps.advance(-120)  # as Run.rewind: after the H6 wait, the create starts at NOW
    run_anthropic(
        HAIKU, run.corpus, run.runs, KEY, Watching(), sleep=run.sleeps, now=run.sleeps.now,
        echo=run.echo,
    )  # fmt: skip

    assert seen == [{"batches": [batch_entry(None, ALL_SCENARIOS[:2], "creating")]}]
    assert run.state()["batches"] == [batch_entry("msgbatch_01", ALL_SCENARIOS[:2], "done")]
    text = (run.model_dir / BATCH_STATE_FILE).read_bytes()
    assert text == (json.dumps(run.state(), indent=2, sort_keys=True) + "\n").encode()


def test_started_at_is_the_utc_instant_in_whole_seconds(tmp_path: Path) -> None:
    # TASK-005c3g G2: the API's clock agrees with this run's (the clock check passes).
    server = FakeBatches(date_header="Mon, 02 Nov 2026 10:15:03 GMT")
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    athens = timezone(timedelta(hours=2))

    run_anthropic(
        HAIKU, run.corpus, run.runs, KEY, run.server, sleep=run.sleeps,
        now=lambda: datetime(2026, 11, 2, 12, 15, 3, 999_999, tzinfo=athens), echo=run.echo,
    )  # fmt: skip

    assert run.state()["batches"][0]["started_at"] == "2026-11-02T10:15:03Z"


GATEWAY = [500, 502, 503, 504, 529]


@pytest.mark.parametrize("code", GATEWAY)
def test_a_gateway_error_after_the_batch_was_created_adopts_it(tmp_path: Path, code: int) -> None:
    # D2: "a 5xx means no batch" was wrong: a gateway can answer after the API accepted it.
    server = FakeBatches(create_script=[Created(HttpResponse(code, {}, b""))])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])

    run()

    assert len(server.creates()) == 1
    assert run.echo[1:3] == [  # after the H6 workspace wait (TASK-005c3h)
        waiting(120),
        "adopted batch msgbatch_01 created by an earlier create request",
    ]
    assert run.recorded() == list(ALL_SCENARIOS[:2])
    assert run.sleeps == [120, 120, 60]


@pytest.mark.parametrize("code", GATEWAY)
def test_a_gateway_error_without_a_batch_is_checked_once_then_retried(
    tmp_path: Path, code: int
) -> None:
    server = FakeBatches(create_script=[HttpResponse(code, {"retry-after": "30"}, b"")])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert len(server.creates()) == 2
    assert run.lists() == 2  # TASK-005c3h H6: the workspace, then the reconciliation
    assert run.sleeps == [120, 120, 30, 60]
    assert run.states() == ["done"]


# -- TASK-005c3c E2: every create answer that is neither a batch nor a definite rejection --

UNCLEAR = [301, 302, 307, 408, 501, 520, 522, 524, 599]


@pytest.mark.parametrize("code", UNCLEAR)
def test_an_unclear_status_after_the_batch_was_created_adopts_it(tmp_path: Path, code: int) -> None:
    # TASK-005c3b I1: these used to be marked failed at once, leaving the batch a paid orphan.
    location = {"location": "https://evil.example.com/"} if code < 400 else {}
    server = FakeBatches(create_script=[Created(HttpResponse(code, location, b""))])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])

    run()

    assert len(server.creates()) == 1
    assert run.echo[1:3] == [  # after the H6 workspace wait (TASK-005c3h)
        waiting(120),
        "adopted batch msgbatch_01 created by an earlier create request",
    ]
    assert run.recorded() == list(ALL_SCENARIOS[:2])
    assert server.list_times == [NOW, NOW + timedelta(seconds=120)]
    assert not any("evil" in call.url for call in server.calls)


@pytest.mark.parametrize("code", UNCLEAR)
def test_an_unclear_status_without_a_batch_is_looked_up_once_then_marked_failed(
    tmp_path: Path, code: int
) -> None:
    server = FakeBatches(create_script=[HttpResponse(code, {}, b"")])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == f"creating a batch failed: HTTP {code}"
    assert server.list_times == [NOW, NOW + timedelta(seconds=120)]  # H6 first
    assert run.sleeps == [120, 120]
    assert len(server.creates()) == 1
    assert run.state()["batches"] == [batch_entry(None, ALL_SCENARIOS[:1], "failed")]


@pytest.mark.parametrize("code", [400, 401, 403, 404, 409, 413, 422])
def test_a_definite_rejection_lists_nothing(tmp_path: Path, code: int) -> None:
    server = FakeBatches(create_script=[Created(HttpResponse(code, {}, b""))])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped, match=rf"^creating a batch failed: HTTP {code}$"):
        run()

    assert run.lists() == 1  # TASK-005c3h H6: the workspace list only, no reconciliation
    assert run.sleeps == [120]
    assert run.states() == ["failed"]


def test_four_gateway_errors_mark_the_create_failed(tmp_path: Path) -> None:
    server = FakeBatches(create_script=[HttpResponse(503, {}, b"")] * 4)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == "creating a batch failed 4 times; last: HTTP 503"
    assert run.lists() == 5  # TASK-005c3h H6: the workspace, then the four reconciliations
    assert run.states() == ["failed"]


# -- TASK-005c3c E3: one list call, at least 120 s after the later of the call and its answer --


def test_a_batch_missing_from_the_list_for_100_seconds_is_found(tmp_path: Path) -> None:
    server = FakeBatches(create_script=["lost"], lag=100)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert len(server.creates()) == 1
    assert server.list_times == [NOW, NOW + timedelta(seconds=120)]  # H6 first
    assert run.sleeps == [120, 120, 60]
    assert "adopted batch msgbatch_01 created by an earlier create request" in run.echo


def test_a_duplicated_create_with_a_late_twin_stops_the_run(tmp_path: Path) -> None:
    # TASK-005c3b I3: the first check saw only the first batch, adopted it, and the twin, still
    # hidden, became a paid orphan.
    server = FakeBatches(create_script=["duplicated"])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped, match=r"^2 recent batches could be the one created") as raised:
        run()

    assert "msgbatch_01, msgbatch_02;" in str(raised.value)
    assert run.lists() == 2  # TASK-005c3h H6: the workspace, then the one reconciliation
    assert run.states() == ["creating"]
    assert server.polled == {}


def test_a_slow_unclear_answer_is_looked_up_120_seconds_after_it_arrived(tmp_path: Path) -> None:
    # The 504 arrives 90 s after the request; the batch may have been created just before it.
    server = FakeBatches(
        create_script=[Created(HttpResponse(504, {}, b""))], create_seconds=90, lag=110
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert server.list_times == [NOW, NOW + timedelta(seconds=90 + 120)]  # H6 first
    assert run.sleeps[:2] == [120, 120]
    assert len(server.creates()) == 1
    assert "adopted batch msgbatch_01 created by an earlier create request" in run.echo


def test_a_cancelled_batch_is_never_adopted(tmp_path: Path) -> None:
    server = FakeBatches(create_script=["lost-none"])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    cancelled = server.register(list(ALL_SCENARIOS[:1]), created_at=iso(timedelta(seconds=1)))
    server.cancelled.add(cancelled)

    run()

    assert not any("adopted" in line for line in run.echo)
    assert len(server.creates()) == 2
    assert run.batch_ids() == ["msgbatch_02"]


def test_several_candidates_one_left_after_the_owner_cancels_is_adopted(tmp_path: Path) -> None:
    server = FakeBatches(
        create_script=["lost"],
        listing=[other_batch("msgbatch_twin", 1, iso(timedelta(seconds=9)))],  # the newer one
        extras_after_create=True,
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    with pytest.raises(RunStopped, match="2 recent batches could be the one"):
        run()
    server.listing[0]["cancel_initiated_at"] = iso(timedelta(minutes=1))  # the owner cancels it

    run()

    assert "adopted batch msgbatch_01 created by an earlier create request" in run.echo
    assert len(server.creates()) == 1
    assert run.recorded() == list(ALL_SCENARIOS[:1])


@pytest.mark.parametrize("value", [5, True, ["2026-11-02T10:00:30Z"]])
def test_an_unreadable_cancel_initiated_at_is_an_unexpected_batch_object(
    tmp_path: Path, value: object
) -> None:
    listed = other_batch("msgbatch_x", 1, CREATED_AT) | {"cancel_initiated_at": value}
    server = FakeBatches(
        create_script=["lost-none"], list_answer=ok({"data": [listed]}), extras_after_create=True
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped, match="listing batches failed: unexpected batch object"):
        run()

    assert run.states() == ["creating"]  # the reconciliation's listing, not the workspace's


def test_a_batch_without_cancel_initiated_at_is_not_cancelled(tmp_path: Path) -> None:
    # "Specified only if cancellation was initiated": a missing field is no cancel.
    listed = other_batch("msgbatch_x", 1, iso(timedelta(seconds=1)))
    server = FakeBatches(create_script=["lost-none"], listing=[listed], extras_after_create=True)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    server.batches["msgbatch_x"] = list(ALL_SCENARIOS[:1])  # so that it can be polled
    server.created_at["msgbatch_x"] = listed["created_at"]
    server.listing.clear()

    server.omit_cancel_field = True

    run()

    assert "adopted batch msgbatch_x created by an earlier create request" in run.echo


# -- the start of every run --


def test_an_interrupted_create_that_made_a_batch_is_adopted_by_the_next_run(
    tmp_path: Path,
) -> None:
    server = FakeBatches(create_script=["interrupt-created"])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    with pytest.raises(KeyboardInterrupt):
        run()

    run()

    # TASK-005c3d H3: a leftover entry waits until started_at + 60 s + 120 s.
    assert run.echo[1:3] == [  # after the first run's H6 workspace wait line
        waiting(180),
        "adopted batch msgbatch_01 created by an earlier create request",
    ]
    assert run.sleeps == [120, 180, 60]  # the first run's H6 wait, then the second run's
    assert len(server.creates()) == 1
    assert run.recorded() == list(ALL_SCENARIOS[:2])
    assert run.state()["batches"] == [batch_entry("msgbatch_01", ALL_SCENARIOS[:2], "done")]


def test_an_interrupted_create_without_a_batch_is_marked_failed_and_sent_again(
    tmp_path: Path,
) -> None:
    server = FakeBatches(create_script=["interrupt"])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    with pytest.raises(KeyboardInterrupt):
        run()

    run()

    assert run.echo[1:4] == [  # after the first run's H6 workspace wait line
        waiting(180),
        "no batch was created by the earlier create request (started 2026-11-02T10:00:00Z)",
        "batch msgbatch_01 created for 2 scenarios",
    ]
    assert run.sleeps == [120, 180, 60]  # the first run's H6 wait, then the second run's
    assert run.lists() == 2  # the first run's workspace list, the second run's reconciliation
    assert run.states() == ["failed", "done"]
    assert run.recorded() == list(ALL_SCENARIOS[:2])


def test_an_old_creating_entry_is_checked_once_without_waiting(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    old = batch_entry(None, ALL_SCENARIOS[:1], "creating", started_at="2026-11-02T09:00:00Z")
    write_state(run.model_dir, old)

    run()

    assert run.echo[0] == (
        "no batch was created by the earlier create request (started 2026-11-02T09:00:00Z)"
    )
    assert run.lists() == 1
    assert run.server.list_times == [NOW]  # E3: t0 + 120 s is past, so no sleep
    assert run.sleeps == [60]  # the poll only


def test_a_creating_entry_is_reconciled_even_when_nothing_is_pending(tmp_path: Path) -> None:
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    for name in ALL_SCENARIOS[:2]:
        scenario = load_scenario(run.corpus / name)
        write_answer(run.model_dir / name, scenario, HAIKU, anthropic_message(HAIKU))
    server.register(list(ALL_SCENARIOS[:2]))
    write_state(run.model_dir, batch_entry(None, ALL_SCENARIOS[:2], "creating"))

    run()

    assert run.echo[:2] == [
        waiting(180),
        "adopted batch msgbatch_01 created by an earlier create request",
    ]
    assert f"{ALL_SCENARIOS[0]}: already recorded" in run.echo
    assert run.server.creates() == []
    assert run.states() == ["done"]


def test_several_candidates_for_a_creating_entry_stop_the_run(tmp_path: Path) -> None:
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    server.register(list(ALL_SCENARIOS[:1]))
    server.register(list(ALL_SCENARIOS[:1]))
    write_state(run.model_dir, CREATING)
    before = (run.model_dir / BATCH_STATE_FILE).read_bytes()

    with pytest.raises(RunStopped, match=r"^2 recent batches could be the one created by an "):
        run()

    assert (run.model_dir / BATCH_STATE_FILE).read_bytes() == before
    assert run.server.creates() == []


def test_the_reconcile_lists_through_the_retries(tmp_path: Path) -> None:
    server = FakeBatches(
        create_script=["lost"],
        list_failures=[TransportError("TimeoutError: no response")],
        extras_after_create=True,  # the failure hits the reconciliation's list, not H6's
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert run.sleeps == [120, 120, 15, 60]  # TASK-005c3h H6: the workspace wait first
    assert "adopted batch msgbatch_01 created by an earlier create request" in run.echo


# --- TASK-005c3c E4: held batches and the release ----------------------------------------------

HELD_STOP = (
    "batch msgbatch_01 is held for the reviewer: batch msgbatch_01: no result for "
    "02-hotel-reception; after the reviewer's decision, run the same command with "
    "--release msgbatch_01"
)


def held_run(tmp_path: Path) -> Run:
    """A run whose first batch missed 02-hotel-reception's line, so the batch is held."""
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    line = json.dumps({"custom_id": "01-pharmacy", "result": succeeded()}) + "\n"
    server.raw_results["msgbatch_01"] = line.encode()
    with pytest.raises(RunStopped, match="is held: send batch-msgbatch_01"):
        run()
    server.calls.clear()
    run.echo.clear()
    return run


def test_a_held_batch_stops_the_next_run_before_any_call(tmp_path: Path) -> None:
    # TASK-005c3b I2: a plain re-run used to send 02-hotel-reception again, in a new paid batch.
    run = held_run(tmp_path)
    before = (run.model_dir / BATCH_STATE_FILE).read_bytes()

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == HELD_STOP
    assert run.server.calls == []
    assert run.echo == []
    assert (run.model_dir / BATCH_STATE_FILE).read_bytes() == before


def test_a_held_batch_is_checked_before_a_creating_entry_is_reconciled(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:2])
    held = batch_entry("msgbatch_01", ALL_SCENARIOS[:2], "held", problems=["p1", "p2"])
    write_state(run.model_dir, held, batch_entry(None, ["02-hotel-reception"], "creating"))

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == (
        "batch msgbatch_01 is held for the reviewer: p1; p2; after the reviewer's decision, run "
        "the same command with --release msgbatch_01"
    )
    assert run.server.calls == []


def test_a_released_batch_is_done_and_the_pending_scenarios_go_in_a_new_batch(
    tmp_path: Path,
) -> None:
    run = held_run(tmp_path)

    run(release="msgbatch_01")

    assert run.echo[0] == "batch msgbatch_01 released after review"
    assert run.echo[1] == "batch msgbatch_02 created for 1 scenarios"
    assert run.server.creates() == [["02-hotel-reception"]]
    assert run.recorded() == list(ALL_SCENARIOS[:2])
    assert run.states() == ["done", "done"]
    assert problems_of(run, 0) == ["batch msgbatch_01: no result for 02-hotel-reception"]
    assert problems_of(run, 1) == []


def test_a_release_stops_for_another_held_batch(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:2])
    write_state(
        run.model_dir,
        batch_entry("msgbatch_01", ["01-pharmacy"], "held", problems=["p1"]),
        batch_entry("msgbatch_02", ["02-hotel-reception"], "held", problems=["p2"]),
    )
    # TASK-005c3h H2: the reviewer emptied the kept results file ("send them all again").
    (run.model_dir / "batch-msgbatch_01.results.jsonl").write_bytes(b"")

    with pytest.raises(RunStopped, match=r"^batch msgbatch_02 is held for the reviewer: p2; "):
        run(release="msgbatch_01")

    assert run.echo == ["batch msgbatch_01 released after review"]
    assert run.states() == ["done", "held"]
    assert run.server.calls == []


@pytest.mark.parametrize("state", ["open", "done", "unknown"])
def test_only_a_held_batch_can_be_released(tmp_path: Path, state: str) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    entries = [] if state == "unknown" else [batch_entry("msgbatch_01", ALL_SCENARIOS[:1], state)]
    write_state(run.model_dir, *entries)
    before = (run.model_dir / BATCH_STATE_FILE).read_bytes()

    with pytest.raises(RunStopped) as raised:
        run(release="msgbatch_01")

    assert str(raised.value) == "msgbatch_01 is not a held batch of claude-haiku-5-5"
    assert (run.model_dir / BATCH_STATE_FILE).read_bytes() == before
    assert run.server.calls == []


def test_a_done_entry_keeps_its_problems_and_loads(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    released = batch_entry("msgbatch_91", ALL_SCENARIOS[:1], "done", problems=["p1"])
    write_state(run.model_dir, released)

    run()

    assert run.states() == ["done", "done"]
    assert problems_of(run, 0) == ["p1"]


def test_the_attempt_cap_counts_held_batches(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:2])
    done = [batch_entry(f"msgbatch_9{n}", ["02-hotel-reception"], "done") for n in range(3)]
    held = batch_entry("msgbatch_99", ["02-hotel-reception"], "held", problems=["p1"])
    write_state(run.model_dir, *done, held)
    (run.model_dir / "batch-msgbatch_99.results.jsonl").write_bytes(b"")  # H2: send it again

    with pytest.raises(RunStopped, match="02-hotel-reception was sent in 4 batches"):
        run(release="msgbatch_99")  # released, it still counts

    assert after_the_clock(run.server) == []
    # A held entry counts like open and done (it is checked first, so this is the rule itself).
    entry = runner_module._Batch  # the private entry type: no public way to build one
    held_entries = [
        entry(f"msgbatch_8{n}", ("02-hotel-reception",), STARTED_AT, CREATED_AT, "held", ("p",))
        for n in range(4)
    ]
    state = runner_module._AnthropicRun(
        model=HAIKU, corpus_dir=run.corpus, model_dir=run.model_dir,
        headers=runner_module.anthropic_headers(KEY), transport=FakeBatches(), sleep=Sleeps(),
        now=lambda: NOW, poll_seconds=60, echo=Echo(),
    )  # fmt: skip
    with pytest.raises(RunStopped, match="02-hotel-reception was sent in 4 batches"):
        state._next_attempt(["02-hotel-reception"], held_entries)


# --- TASK-005c3g G4: a released batch is first recorded from its kept results ---------------

KEPT = "batch-msgbatch_01.results.jsonl"


def result_line(name: str, result: object) -> dict[str, Any]:
    return {"custom_id": name, "result": result}


def held_with(tmp_path: Path, *lines: object) -> Run:
    """A held msgbatch_01 of 01-pharmacy and 02-hotel-reception, neither recorded, whose kept
    results file holds ``lines`` (objects as JSON, bytes as they are)."""
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:2])
    held = batch_entry("msgbatch_01", ALL_SCENARIOS[:2], "held", problems=["p1"])
    write_state(run.model_dir, held)
    data = b"".join(
        (line if isinstance(line, bytes) else json.dumps(line).encode()) + b"\n" for line in lines
    )
    (run.model_dir / KEPT).write_bytes(data)
    return run


def recorded_from(name: str) -> str:
    return f"{name}: recorded from the kept results of batch msgbatch_01"


def test_a_refused_answer_is_recorded_from_the_kept_results_on_release_with_no_new_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pauses: list[float]
) -> None:
    # The audit's p03: --release sent 01-pharmacy again in a new, paid batch.
    server = FakeBatches(outcome=lambda n, s: succeeded(truth_text(s)))
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    refused = RefusedReplace(10, "response.json")
    monkeypatch.setattr(os, "replace", refused)
    with pytest.raises(
        RunStopped, match=r"^01-pharmacy: could not be recorded \(PermissionError\)"
    ):
        run()
    monkeypatch.setattr(os, "replace", refused.real)
    assert (run.model_dir / "01-pharmacy" / "response.json.tmp").exists()
    server.calls.clear()
    run.echo.clear()

    run(release="msgbatch_01")

    assert server.creates() == []
    assert run.echo[:2] == [recorded_from("01-pharmacy"), "batch msgbatch_01 released after review"]
    lines = [json.loads(line) for line in (run.model_dir / KEPT).read_bytes().splitlines()]
    message = next(line for line in lines if line["custom_id"] == "01-pharmacy")["result"][
        "message"
    ]
    expected = json.dumps(message, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    assert (run.model_dir / "01-pharmacy" / "response.json").read_bytes() == expected.encode()
    assert run.recorded() == list(ALL_SCENARIOS[:2])
    assert run.states() == ["done"]
    assert sorted(run.model_dir.rglob("*.tmp")) == []


def test_a_release_records_each_scenario_with_one_succeeded_line_and_sends_only_the_others(
    tmp_path: Path,
) -> None:
    run = held_with(
        tmp_path, result_line("02-hotel-reception", succeeded(truth_text("02-hotel-reception")))
    )

    run(release="msgbatch_01")

    assert run.echo[:2] == [
        recorded_from("02-hotel-reception"),
        "batch msgbatch_01 released after review",
    ]
    assert run.server.creates() == [["01-pharmacy"]]
    assert run.recorded() == list(ALL_SCENARIOS[:2])
    assert run.states() == ["done", "done"]


@pytest.mark.parametrize(
    "lines",
    [
        [result_line("01-pharmacy", succeeded()), result_line("01-pharmacy", succeeded())],
        [result_line("01-pharmacy", errored("invalid_request_error"))],
        [result_line("01-pharmacy", {"type": "expired"})],
        [result_line("01-pharmacy", {"type": "succeeded", "message": "not an object"})],
        [result_line("01-pharmacy", {"type": "succeeded"})],
        [result_line("01-pharmacy", "no result object")],
        [b"{not json"],
        [],
    ],
    ids=["two-lines", "errored", "expired", "message-not-object", "no-message", "no-result",
         "not-json", "no-line"],
)  # fmt: skip
def test_a_scenario_without_exactly_one_succeeded_line_is_sent_again(
    tmp_path: Path, lines: list[object]
) -> None:
    run = held_with(tmp_path, result_line("02-hotel-reception", succeeded()), *lines)

    run(release="msgbatch_01")

    assert recorded_from("01-pharmacy") not in run.echo
    assert recorded_from("02-hotel-reception") in run.echo
    assert run.server.creates() == [["01-pharmacy"]]
    assert run.recorded() == list(ALL_SCENARIOS[:2])


def test_a_recording_that_fails_during_the_release_keeps_the_batch_held_and_sends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = held_with(
        tmp_path,
        result_line("01-pharmacy", succeeded()),
        result_line("02-hotel-reception", succeeded()),
    )
    failing_write_answer(monkeypatch, "02-hotel-reception", OSError(28, f"No space left {KEY}"))

    with pytest.raises(RunStopped) as raised:
        run(release="msgbatch_01")

    assert str(raised.value) == (
        "02-hotel-reception: could not be recorded (OSError); batch msgbatch_01 is still held: "
        "send this message to the reviewer"
    )
    assert run.states() == ["held"]
    assert run.server.calls == []
    assert run.recorded() == ["01-pharmacy"]
    # Released again after the reviewer's decision: 01-pharmacy is recorded, so it is skipped.
    monkeypatch.undo()
    run.echo.clear()
    run(release="msgbatch_01")
    assert run.echo[:2] == [
        recorded_from("02-hotel-reception"),
        "batch msgbatch_01 released after review",
    ]
    assert run.server.creates() == []


def kept_unreadable(run: Run, reason: str) -> str:
    """TASK-005c3h H2: the stop of a release whose kept results file cannot be read."""
    return (
        f"cannot read the kept results {run.model_dir / KEPT}: {reason}; batch msgbatch_01 is "
        "still held: send this message to the reviewer"
    )


def test_a_missing_kept_results_file_keeps_the_batch_held_and_sends_nothing(
    tmp_path: Path,
) -> None:
    # The review's #1: a missing file released the batch, and the run re-sent every scenario.
    run = held_with(tmp_path)
    (run.model_dir / KEPT).unlink()
    before = (run.model_dir / BATCH_STATE_FILE).read_bytes()

    with pytest.raises(RunStopped) as raised:
        run(release="msgbatch_01")

    # The reason is the OS's text for a missing file: only its place is checked (DoD 6).
    head, tail = kept_unreadable(run, "<reason>").split("<reason>")
    assert str(raised.value).startswith(head)
    assert str(raised.value).endswith(tail)
    assert run.states() == ["held"]
    assert (run.model_dir / BATCH_STATE_FILE).read_bytes() == before
    assert run.server.calls == []
    assert run.echo == []
    assert run.recorded() == []


def test_a_release_with_every_scenario_recorded_reads_nothing_and_says_nothing_about_it(
    tmp_path: Path,
) -> None:
    run = held_with(tmp_path)
    (run.model_dir / KEPT).unlink()
    for name in ALL_SCENARIOS[:2]:
        scenario = load_scenario(run.corpus / name)
        write_answer(run.model_dir / name, scenario, HAIKU, anthropic_message(HAIKU))

    run(release="msgbatch_01")

    assert run.echo == ["batch msgbatch_01 released after review", "nothing to run"]
    assert after_the_clock(run.server) == []


def refused_kept_reads(monkeypatch: pytest.MonkeyPatch, times: int) -> list[int]:
    """``Path.read_bytes`` of the kept results file raises ``PermissionError`` (WinError 5 or 32
    as Python reports it) for its first ``times`` reads; returns the count of refusals."""
    real_read = Path.read_bytes
    refused = [0]

    def read_bytes(self: Path) -> bytes:
        if self.name == KEPT and refused[0] < times:
            refused[0] += 1
            raise PermissionError(13, "Permission denied")
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    return refused


@pytest.mark.parametrize("times", [1, 9])
def test_a_refused_read_of_the_kept_results_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, times: int
) -> None:
    # The review's #1: the read had no retry while every write had one.
    run = held_with(
        tmp_path,
        result_line("01-pharmacy", succeeded()),
        result_line("02-hotel-reception", succeeded()),
    )
    refused = refused_kept_reads(monkeypatch, times)

    run(release="msgbatch_01")

    assert refused == [times]
    assert run.sleeps == [refused_pause(attempt) for attempt in range(times)]
    assert run.echo[:3] == [
        recorded_from("01-pharmacy"),
        recorded_from("02-hotel-reception"),
        "batch msgbatch_01 released after review",
    ]
    assert run.server.creates() == []
    assert run.recorded() == list(ALL_SCENARIOS[:2])


def test_ten_refused_reads_of_the_kept_results_keep_the_batch_held_and_send_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = held_with(tmp_path, result_line("01-pharmacy", succeeded()))
    refused_kept_reads(monkeypatch, 10)

    with pytest.raises(RunStopped) as raised:
        run(release="msgbatch_01")

    assert str(raised.value) == kept_unreadable(run, "Permission denied")
    assert run.sleeps == [refused_pause(attempt) for attempt in range(9)]
    assert run.states() == ["held"]
    assert run.server.calls == []
    assert run.recorded() == []


def pretty_results(*lines: dict[str, Any]) -> bytes:
    return json.dumps(list(lines), indent=2).encode()


@pytest.mark.parametrize(
    "data",
    [
        json.dumps(
            [result_line(name, succeeded()) for name in ALL_SCENARIOS[:2]]
        ).encode() + b"\n",
        pretty_results(*(result_line(name, succeeded()) for name in ALL_SCENARIOS[:2])),
        json.dumps({"results": [result_line("01-pharmacy", succeeded())]}).encode() + b"\r\n",
        b"{not json\n" + json.dumps(result_line("09-other", succeeded())).encode() + b"\n",
    ],
    ids=["array", "pretty-printed", "wrapper", "malformed-and-unknown-id"],
)  # fmt: skip
def test_a_kept_file_with_no_result_of_the_batch_keeps_it_held_and_sends_nothing(
    tmp_path: Path, data: bytes
) -> None:
    # TASK-005c3s S3-3: a file restored in another shape was read as "no results", and the
    # release sent every scenario again although their answers were in it.
    run = held_with(tmp_path)
    (run.model_dir / KEPT).write_bytes(data)
    before = (run.model_dir / BATCH_STATE_FILE).read_bytes()

    with pytest.raises(RunStopped) as raised:
        run(release="msgbatch_01")

    assert str(raised.value) == (
        f"{run.model_dir / KEPT} holds no result of batch msgbatch_01; batch msgbatch_01 is still "
        "held: send this message to the reviewer"
    )
    assert (run.model_dir / BATCH_STATE_FILE).read_bytes() == before
    assert run.server.calls == []
    assert run.recorded() == []


@pytest.mark.parametrize("data", [b"", b"\r\n", b"\n\n", b"\r\n \t\r\n"])
def test_a_kept_file_of_whitespace_only_releases_and_sends_every_pending_scenario_again(
    tmp_path: Path, data: bytes
) -> None:
    # The reviewer's explicit "send them all again" (runbook §8: New-Item writes 0 bytes).
    run = held_with(tmp_path)
    (run.model_dir / KEPT).write_bytes(data)

    run(release="msgbatch_01")

    assert run.echo[0] == "batch msgbatch_01 released after review"
    assert run.server.creates() == [list(ALL_SCENARIOS[:2])]
    assert run.recorded() == list(ALL_SCENARIOS[:2])


def test_the_original_kept_file_of_a_held_batch_still_records_its_good_lines(
    tmp_path: Path,
) -> None:
    # A malformed line and a duplicate are why such a batch is held: never a reason to refuse it.
    run = held_with(
        tmp_path,
        b"{not json",
        result_line("01-pharmacy", succeeded()),
        result_line("01-pharmacy", succeeded()),
        result_line("02-hotel-reception", succeeded()),
    )

    run(release="msgbatch_01")

    assert run.echo[:2] == [
        recorded_from("02-hotel-reception"),
        "batch msgbatch_01 released after review",
    ]
    assert run.server.creates() == [["01-pharmacy"]]
    assert run.recorded() == list(ALL_SCENARIOS[:2])


# --- TASK-005c3g G5: an answer that could not be saved is never sent again --------------------


def unsaved_stop(path: Path) -> str:
    return (
        f"01-pharmacy: an answer that could not be saved is kept in {path}; nothing was sent; "
        "send this message to the reviewer and do not delete or rename the file"
    )


def test_a_leftover_unsaved_answer_stops_the_run_before_any_request(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:2])
    kept = run.model_dir / "01-pharmacy" / "response.json.tmp"
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b'{"kept": "answer"}\n')

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == unsaved_stop(kept)
    # TASK-005c3h H4: only the free clock call comes before this check.
    assert [(call.method, call.url) for call in run.server.calls] == [("GET", CLOCK_URL)]
    assert kept.read_bytes() == b'{"kept": "answer"}\n'
    assert sorted(path.name for path in run.model_dir.rglob("*")) == [
        "01-pharmacy",
        "response.json.tmp",
    ]


def test_a_leftover_temporary_file_of_a_recorded_scenario_is_ignored(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:2])
    scenario = load_scenario(run.corpus / "01-pharmacy")
    write_answer(run.model_dir / "01-pharmacy", scenario, HAIKU, anthropic_message(HAIKU))
    (run.model_dir / "01-pharmacy" / "response.json.tmp").write_bytes(b"{}\n")

    run()

    assert run.server.creates() == [["02-hotel-reception"]]
    assert run.recorded() == list(ALL_SCENARIOS[:2])


def test_the_unsaved_answer_check_comes_after_the_release_records_what_it_can(
    tmp_path: Path,
) -> None:
    run = held_with(tmp_path, result_line("01-pharmacy", succeeded()))
    (run.model_dir / "01-pharmacy").mkdir()
    (run.model_dir / "01-pharmacy" / "response.json.tmp").write_bytes(b"{}\n")

    run(release="msgbatch_01")

    assert run.echo[0] == recorded_from("01-pharmacy")
    assert run.server.creates() == [["02-hotel-reception"]]
    assert sorted(run.model_dir.rglob("*.tmp")) == []


# --- TASK-005c3c E6: an open batch with nothing pending is resumed ----------------------------


def test_an_open_batch_with_nothing_pending_is_resumed_and_closed(tmp_path: Path) -> None:
    # TASK-005c3b M5: it used to stay open while the plan said "resumed by --yes".
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    for name in ALL_SCENARIOS[:2]:
        scenario = load_scenario(run.corpus / name)
        write_answer(run.model_dir / name, scenario, HAIKU, anthropic_message(HAIKU))
    batch_id = server.register(list(ALL_SCENARIOS[:2]))
    write_state(run.model_dir, batch_entry(batch_id, ALL_SCENARIOS[:2], "open"))

    run()

    assert run.echo[0] == "resuming batch msgbatch_01"
    assert run.echo[-3:] == [
        "01-pharmacy: already recorded",
        "02-hotel-reception: already recorded",
        "nothing to run",
    ]
    assert server.polled == {"msgbatch_01": 2}
    assert run.states() == ["done"]
    assert server.creates() == []


# --- TASK-005c3d H2: a closed adoption window ----------------------------------------------------


def waiting(seconds: int) -> str:
    return f"waiting {seconds} s before looking for a batch the create request may have made"


def test_a_leftover_entry_does_not_adopt_a_batch_created_days_later(tmp_path: Path) -> None:
    # U1: the window had no upper bound, so a same-size batch of another checkout was adopted.
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    server.register(list(ALL_SCENARIOS[:1]), created_at=iso(-timedelta(days=1)))
    leftover = batch_entry(None, ALL_SCENARIOS[:1], "creating", started_at="2026-10-30T10:00:00Z")
    write_state(run.model_dir, leftover)

    run()

    assert run.echo[0] == (
        "no batch was created by the earlier create request (started 2026-10-30T10:00:00Z)"
    )
    assert not any("adopted" in line for line in run.echo)
    assert run.states() == ["failed", "done"]
    assert run.batch_ids() == [None, "msgbatch_02"]


@pytest.mark.parametrize(
    ("minutes", "adopted"),
    [(9, True), (10, True), (11, False), (-9, True), (-10, True), (-11, False)],
)
def test_the_in_run_window_ends_ten_minutes_after_the_unclear_answer(
    tmp_path: Path, minutes: int, adopted: bool
) -> None:
    # t_end is the instant the unclear answer arrived (here NOW, as started_at).
    server = FakeBatches(create_script=["lost-none"], extras_after_create=True)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    edge = server.register(list(ALL_SCENARIOS[:1]), created_at=iso(timedelta(minutes=minutes)))

    run()

    assert (f"adopted batch {edge} created by an earlier create request" in run.echo) is adopted
    assert len(server.creates()) == (1 if adopted else 2)


def test_the_in_run_window_ends_after_a_slow_answer(tmp_path: Path) -> None:
    # The answer arrives 90 s after the request: the window ends 10 minutes after the answer.
    server = FakeBatches(
        create_script=[HttpResponse(504, {}, b"")], create_seconds=90, extras_after_create=True
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    server.register(list(ALL_SCENARIOS[:1]), created_at=iso(timedelta(seconds=90 + 599)))

    run()

    assert "adopted batch msgbatch_01 created by an earlier create request" in run.echo


@pytest.mark.parametrize(("seconds", "adopted"), [(660, True), (661, False)])
def test_the_leftover_window_ends_ten_minutes_after_the_call_timeout(
    tmp_path: Path, seconds: int, adopted: bool
) -> None:
    # A leftover entry has no known answer: t_end = started_at + CALL_TIMEOUT_SECONDS (60 s).
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    started = timedelta(hours=-1)
    server.register(list(ALL_SCENARIOS[:1]), created_at=iso(started + timedelta(seconds=seconds)))
    write_state(
        run.model_dir, batch_entry(None, ALL_SCENARIOS[:1], "creating", started_at=iso(started))
    )

    run()

    assert ("adopted batch msgbatch_01 created by an earlier create request" in run.echo) is adopted
    assert run.states() == (["done"] if adopted else ["failed", "done"])


# --- TASK-005c3d H3, H4, H5: the leftover t0, capped waits, the waiting line ----------------------


def test_a_leftover_entry_counts_from_the_end_of_its_call_timeout(tmp_path: Path) -> None:
    # R1: t0 was started_at, although the create call may have run for up to 60 s.
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    leftover = batch_entry(
        None, ALL_SCENARIOS[:1], "creating", started_at=iso(-timedelta(seconds=30))
    )
    write_state(run.model_dir, leftover)

    run()

    assert run.sleeps == [150, 60]  # the reconcile wait, then the poll of the new batch
    assert run.lists() == 1
    assert run.server.list_times == [NOW + timedelta(seconds=150)]
    assert run.echo[0] == waiting(150)


@pytest.mark.parametrize("minutes", [5, 10])
def test_a_leftover_started_at_in_the_near_future_is_capped_at_180_seconds(
    tmp_path: Path, minutes: int
) -> None:
    # U7: the wait had no cap; within the skew window it is capped, beyond it the run stops.
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    future = batch_entry(
        None, ALL_SCENARIOS[:1], "creating", started_at=iso(timedelta(minutes=minutes))
    )
    write_state(run.model_dir, future)

    run()

    assert run.sleeps[0] == 180
    assert run.echo[0] == waiting(180)
    assert run.lists() == 1


@pytest.mark.parametrize("state", ["creating", "done", "failed"])
def test_an_entry_started_more_than_ten_minutes_in_the_future_stops_after_the_clock_call_only(
    tmp_path: Path, state: str
) -> None:
    # TASK-005c3h H4: the clock call comes first (a slow laptop gets the clock stop), then this.
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    started_at = "2026-11-02T10:10:01Z"  # NOW + 10 min + 1 s
    write_state(
        run.model_dir, batch_entry("msgbatch_91", ["01-pharmacy"], state, started_at=started_at)
    )
    before = (run.model_dir / BATCH_STATE_FILE).read_bytes()

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == (
        "batches.json has an entry started at 2026-11-02T10:10:01Z, in the future: the clock or "
        "the file is wrong; send batches.json to the reviewer"
    )
    assert [(call.method, call.url) for call in run.server.calls] == [("GET", CLOCK_URL)]
    assert run.sleeps == []
    assert (run.model_dir / BATCH_STATE_FILE).read_bytes() == before


def test_the_future_check_comes_after_the_held_check(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:2])
    write_state(
        run.model_dir,
        batch_entry("msgbatch_01", ["01-pharmacy"], "held", problems=["p1"]),
        batch_entry(
            "msgbatch_02", ["02-hotel-reception"], "done", started_at="2026-11-03T10:00:00Z"
        ),
    )

    with pytest.raises(RunStopped, match=r"^batch msgbatch_01 is held for the reviewer: p1; "):
        run()


def test_the_waiting_line_rounds_the_seconds_actually_slept_up(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    run.sleeps.advance(0.5)
    leftover = batch_entry(
        None, ALL_SCENARIOS[:1], "creating", started_at=iso(-timedelta(seconds=30))
    )
    write_state(run.model_dir, leftover)

    run()

    assert run.sleeps[0] == pytest.approx(149.5)
    assert run.echo[0] == waiting(math.ceil(149.5))


def test_no_waiting_line_without_a_wait(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    write_state(
        run.model_dir,
        batch_entry(None, ALL_SCENARIOS[:1], "creating", started_at="2026-11-02T09:00:00Z"),
    )

    run()

    assert not any(line.startswith("waiting") for line in run.echo)


def test_an_in_run_reconcile_says_that_it_waits(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(create_script=["lost"]), ALL_SCENARIOS[:1])

    run()

    assert run.echo[:3] == [
        WORKSPACE_WAIT.format(seconds=120),  # TASK-005c3h H6: the two waits say different things
        waiting(120),
        "adopted batch msgbatch_01 created by an earlier create request",
    ]


# --- TASK-005c3d H6: the batch list is followed page by page --------------------------------------


def cancelled_batch(batch_id: str, total: int, created: str) -> dict[str, Any]:
    return other_batch(batch_id, total, created) | {"cancel_initiated_at": created}


def test_the_batch_behind_100_newer_cancelled_ones_is_found_on_page_two(tmp_path: Path) -> None:
    # U5: one page of 100 was read as the whole list.
    server = FakeBatches(
        create_script=["lost-none"],
        newest=[
            cancelled_batch(f"msgbatch_c{n:03d}", 1, iso(timedelta(seconds=30))) for n in range(100)
        ],
        extras_after_create=True,
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    server.register(list(ALL_SCENARIOS[:1]), created_at=iso(timedelta(seconds=5)))

    run()

    assert "adopted batch msgbatch_01 created by an earlier create request" in run.echo
    assert len(server.creates()) == 1
    list_urls = [call.url for call in server.calls if call.url.startswith(LIST_URL)]
    # TASK-005c3h H6: the workspace's one page, then the reconciliation's two.
    assert list_urls == [LIST_URL, LIST_URL, f"{LIST_URL}&after_id=msgbatch_c099"]


def test_the_paging_stops_at_the_first_page_that_ends_before_the_window(tmp_path: Path) -> None:
    recent = [other_batch(f"msgbatch_r{n:03d}", 3, iso(-timedelta(minutes=5))) for n in range(100)]
    old = [other_batch(f"msgbatch_o{n:03d}", 3, iso(-timedelta(minutes=20))) for n in range(150)]
    server = FakeBatches(create_script=["lost-none"], newest=recent + old, extras_after_create=True)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    # Page 1 ends at -5 min (inside the window), page 2 at -20 min: page 3 is never read
    # (after the one page of the H6 workspace list).
    assert run.pages() == 1 + 2
    assert len(server.creates()) == 2


def test_a_page_ending_exactly_at_the_window_is_followed(tmp_path: Path) -> None:
    edge = [other_batch(f"msgbatch_e{n:03d}", 3, iso(-timedelta(minutes=10))) for n in range(100)]
    server = FakeBatches(create_script=["lost-none"], newest=edge, extras_after_create=True)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    server.register(list(ALL_SCENARIOS[:1]), created_at=iso(-timedelta(minutes=10)))

    run()

    assert run.pages() == 1 + 2  # the H6 workspace page, then the reconciliation's two
    assert "adopted batch msgbatch_01 created by an earlier create request" in run.echo


@pytest.mark.parametrize(("count", "stops"), [(1000, False), (1001, True)])
def test_more_than_ten_pages_stop_the_run(tmp_path: Path, count: int, stops: bool) -> None:
    newer = [
        cancelled_batch(f"msgbatch_c{n:04d}", 1, iso(timedelta(seconds=1))) for n in range(count)
    ]
    server = FakeBatches(create_script=["lost-none"], newest=newer, extras_after_create=True)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    if stops:
        with pytest.raises(RunStopped) as raised:
            run()
        assert str(raised.value) == (
            "more than 1000 batches were created recently in this workspace; look in the Claude "
            "Console (Batches) and send this message to the reviewer"
        )
        assert run.states() == ["creating"]  # fail closed: never "none found"
        assert len(server.creates()) == 1
    else:
        run()
        assert len(server.creates()) == 2
    # 1000: the tenth page has has_more false; 1001: it has more, and no eleventh page is read
    # (after the one page of the H6 workspace list).
    assert run.pages() == 1 + 10


def list_page(data: list[dict[str, Any]], **fields: object) -> HttpResponse:
    return ok({"data": data, "has_more": True, "first_id": None, "last_id": None} | fields)


@pytest.mark.parametrize(
    "fields",
    [
        {"last_id": "msgbatch/../x"},
        {"last_id": "msgbatch_other"},
        {"last_id": None},
        {"last_id": 5},
        {"has_more": "yes"},
        {"has_more": None},
    ],
    ids=["not-an-id", "not-the-last", "null", "number", "has-more-text", "has-more-null"],
)
def test_an_unusable_page_cursor_stops_with_no_further_call(
    tmp_path: Path, fields: dict[str, object]
) -> None:
    listed = other_batch("msgbatch_x", 3, iso(timedelta(seconds=1)))
    server = FakeBatches(
        create_script=["lost-none"],
        list_pages=[list_page([listed], **fields)],
        extras_after_create=True,
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == "listing batches failed: unexpected list object"
    assert run.pages() == 1 + 1  # the H6 workspace page, then this one
    assert run.states() == ["creating"]


def test_an_empty_page_that_has_more_stops(tmp_path: Path) -> None:
    server = FakeBatches(
        create_script=["lost-none"], list_pages=[list_page([])], extras_after_create=True
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped, match=r"^listing batches failed: unexpected list object$"):
        run()

    assert run.pages() == 1 + 1  # the H6 workspace page, then this one
    assert run.states() == ["creating"]


def test_a_failed_second_page_stops_the_run(tmp_path: Path) -> None:
    listed = other_batch("msgbatch_x", 3, iso(timedelta(seconds=1)))
    second = HttpResponse(403, {}, b"")
    server = FakeBatches(
        create_script=["lost-none"],
        list_pages=[list_page([listed], last_id="msgbatch_x"), second],
        extras_after_create=True,
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped, match=r"^listing batches failed: HTTP 403$"):
        run()

    assert run.pages() == 1 + 2  # the H6 workspace page, then these two
    assert run.states() == ["creating"]


# --- TASK-005c3d H8, H9: invalid Unicode, and no result can wedge a batch ------------------------


def test_a_lone_surrogate_answer_is_recorded_as_malformed_and_the_run_goes_on(
    tmp_path: Path,
) -> None:
    # U2: the encode error escaped from write_answer: the batch stayed open, every run crashed.
    server = FakeBatches(
        outcome=lambda n, s: succeeded("\ud83d") if s == "01-pharmacy" else succeeded()
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])

    run()

    assert run.recorded() == list(ALL_SCENARIOS[:2])
    assert "01-pharmacy: recorded (malformed, invalid, recall 0.000)" in run.echo
    assert run.states() == ["done"]
    raw = (run.model_dir / "01-pharmacy" / "response.json").read_bytes()
    assert raw.isascii()
    assert b"\\ud83d" in raw


def test_the_recorded_line_shows_an_odd_stop_safely(tmp_path: Path) -> None:
    message = anthropic_message(HAIKU, stop="\x1b[2J")
    server = FakeBatches(outcome=lambda n, s: {"type": "succeeded", "message": message})
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert "01-pharmacy: recorded (?[2J, invalid, recall 0.000)" in run.echo


def failing_write_answer(monkeypatch: pytest.MonkeyPatch, name: str, error: BaseException) -> None:
    """``write_answer`` of the runner raises ``error`` for the scenario ``name`` only."""
    real = runner_module.write_answer

    def write(run_dir: Path, scenario: Any, model: str, response: object) -> Any:
        if run_dir.name == name:
            raise error
        return real(run_dir, scenario, model, response)

    monkeypatch.setattr(runner_module, "write_answer", write)


@pytest.mark.parametrize(
    ("error", "problem"),
    [
        (
            OSError(28, f"No space left on device {KEY}"),
            "02-hotel-reception: could not be recorded (OSError)",
        ),
        (
            UnicodeEncodeError("utf-8", "x", 0, 1, "bad"),
            "02-hotel-reception: could not be recorded (UnicodeEncodeError)",
        ),
        (RunStopped("a stop while recording"), "02-hotel-reception: a stop while recording"),
    ],
    ids=["oserror", "unicode", "runstopped"],
)
def test_a_result_that_cannot_be_recorded_holds_the_batch_and_the_others_are_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException, problem: str
) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:3])
    failing_write_answer(monkeypatch, "02-hotel-reception", error)

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == problem + HELD
    assert run.recorded() == ["01-pharmacy", "03-call-centre"]
    assert run.states() == ["held"]
    assert problems_of(run) == [problem]
    assert KEY.encode() not in (run.model_dir / BATCH_STATE_FILE).read_bytes()
    # The next run stops at the held batch before any call.
    run.server.calls.clear()
    with pytest.raises(RunStopped, match=r"^batch msgbatch_01 is held for the reviewer: "):
        run()
    assert run.server.calls == []


def test_an_interrupt_while_recording_is_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:2])
    failing_write_answer(monkeypatch, "02-hotel-reception", KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        run()

    assert run.states() == ["open"]  # the next run resumes the batch


def test_a_scenario_that_became_invalid_while_recording_is_named_once(tmp_path: Path) -> None:
    # TASK-005c3e F7: the problem read "01-pharmacy: 01-pharmacy: invalid scenario: …".
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])

    def break_the_scenario() -> None:
        (run.corpus / "01-pharmacy" / "scenario.json").write_text("not json", encoding="utf-8")

    server.on_first_poll = break_the_scenario

    with pytest.raises(RunStopped) as raised:
        run()

    problems = problems_of(run)
    assert len(problems) == 1
    assert problems[0].startswith("01-pharmacy: invalid scenario: ")
    assert problems[0].count("01-pharmacy: ") == 1
    assert str(raised.value) == problems[0] + HELD
    assert run.recorded() == ["02-hotel-reception"]


def test_a_record_that_cannot_be_completed_stops_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:2])
    scenario = load_scenario(run.corpus / "01-pharmacy")
    write_answer(run.model_dir / "01-pharmacy", scenario, HAIKU, anthropic_message(HAIKU))
    (run.model_dir / "01-pharmacy" / "comparison.json").unlink()

    def broken(*args: object) -> tuple[str, ...]:
        raise OSError(13, f"Permission denied {KEY}")

    monkeypatch.setattr(runner_module, "rescore_run", broken)

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == (
        # OSError(13, ...) is built as its subclass PermissionError: only the type is named.
        "01-pharmacy: the record could not be completed (PermissionError); send this message to "
        "the reviewer"
    )
    assert after_the_clock(run.server) == []


# --- TASK-005c3d H12: a held batch keeps at most 20 problems -------------------------------------


def malformed_results(count: int) -> bytes:
    return b"not json\n" * count


@pytest.mark.parametrize(("lines", "stored"), [(19, 20), (20, 21), (50, 21)])
def test_a_held_batch_stores_and_prints_at_most_20_problems(
    tmp_path: Path, lines: int, stored: int
) -> None:
    # R2: a 10 000-line damaged file gave 10 000 problems in batches.json and in one message.
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    server.raw_results["msgbatch_01"] = malformed_results(lines)

    with pytest.raises(RunStopped) as raised:
        run()

    problems = problems_of(run)
    total = lines + 1  # every line, and no result for 01-pharmacy
    assert len(problems) == stored
    assert problems[0] == "batch msgbatch_01: results line 1 is not a JSON object"
    if total > 20:
        assert problems[19] == "batch msgbatch_01: results line 20 is not a JSON object"
        assert problems[20] == (
            f"\N{HORIZONTAL ELLIPSIS} and {total - 20} more problems "
            "(see batch-msgbatch_01.results.jsonl)"
        )
    else:
        assert problems[-1] == "batch msgbatch_01: no result for 01-pharmacy"
    assert str(raised.value) == "; ".join(problems) + HELD
    # The held stop of the next run prints the same bounded list.
    with pytest.raises(RunStopped) as again:
        run()
    assert str(again.value).count("; batch msgbatch_01: results line") == min(lines, 20) - 1


@pytest.mark.parametrize(("count", "valid"), [(21, True), (22, False)])
def test_a_state_with_more_than_21_problems_is_invalid(
    tmp_path: Path, count: int, valid: bool
) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    problems = [f"p{n}" for n in range(count)]
    write_state(
        run.model_dir, batch_entry("msgbatch_01", ["01-pharmacy"], "held", problems=problems)
    )

    with pytest.raises(RunStopped) as raised:
        run()

    text = str(raised.value)
    assert text.endswith("is not a valid batch state; it was not changed") is not valid
    assert run.server.calls == []


# --- TASK-005c3d H13: audit_batches -------------------------------------------------------------


def audit(run: Run) -> None:
    runner_module.audit_batches(
        HAIKU, run.runs, KEY, run.server, sleep=run.sleeps, now=run.sleeps.now, echo=run.echo
    )


def audit_wait(seconds: int) -> str:
    return f"waiting {seconds} s before checking for batches that no run recorded"


def unrecorded(batch_id: str, count: int, created: str) -> str:
    return (
        f"warning: batch {batch_id} ({count} requests, created {created}) is recorded by no run: "
        "it may be a paid duplicate; send this warning to the reviewer, who decides whether to "
        "cancel it in the Claude Console (Batches)"
    )


def test_the_audit_does_nothing_without_an_entry(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    write_state(run.model_dir)

    audit(run)

    assert run.server.calls == []
    assert run.echo == []


@pytest.mark.parametrize(
    ("started", "slept"),
    [
        (timedelta(minutes=5), 300),
        (timedelta(0), 300),
        (-timedelta(seconds=299), 1),
        (-timedelta(minutes=5), None),
    ],
)
def test_the_audit_waits_until_five_minutes_after_the_last_create_at_most_300_s(
    tmp_path: Path, started: timedelta, slept: int | None
) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    first = batch_entry("msgbatch_91", ["01-pharmacy"], "done", started_at=iso(-timedelta(hours=1)))
    last = batch_entry("msgbatch_92", ["01-pharmacy"], "done", started_at=iso(started))
    write_state(run.model_dir, first, last)

    audit(run)

    assert run.sleeps == ([] if slept is None else [slept])
    assert run.echo == ([] if slept is None else [audit_wait(slept)])
    assert run.server.lists == 1


def test_the_audit_names_only_unrecorded_live_batches_since_the_first_create(
    tmp_path: Path,
) -> None:
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    started = timedelta(hours=-1)
    server.listing = [
        other_batch("msgbatch_new", 2, iso(-timedelta(minutes=1))),  # unrecorded: named
        cancelled_batch("msgbatch_cancelled", 1, iso(-timedelta(minutes=2))),
        other_batch("msgbatch_known", 1, iso(-timedelta(minutes=3))),
        other_batch("msgbatch_edge", 3, iso(started - timedelta(minutes=10))),  # named
        other_batch("msgbatch_older", 1, iso(started - timedelta(minutes=10, seconds=1))),
    ]
    write_state(
        run.model_dir,
        batch_entry("msgbatch_known", ["01-pharmacy"], "done", started_at=iso(started)),
    )
    before = (run.model_dir / BATCH_STATE_FILE).read_bytes()

    audit(run)

    assert run.echo == [
        unrecorded("msgbatch_new", 2, iso(-timedelta(minutes=1))),
        unrecorded("msgbatch_edge", 3, "2026-11-02T08:50:00Z"),
    ]
    assert (run.model_dir / BATCH_STATE_FILE).read_bytes() == before  # never changes a record


def test_the_audit_spares_the_candidates_of_any_models_pending_create(tmp_path: Path) -> None:
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    server.listing = [
        other_batch("msgbatch_haiku", 1, iso(timedelta(seconds=5))),
        other_batch("msgbatch_opus", 2, iso(timedelta(seconds=5))),
        other_batch("msgbatch_late", 1, iso(timedelta(minutes=12))),  # beyond the window: named
    ]
    write_state(run.model_dir, batch_entry(None, ["01-pharmacy"], "creating"))
    write_state(run.runs / OPUS, batch_entry(None, ALL_SCENARIOS[:2], "creating"))

    audit(run)

    assert run.echo[1:] == [unrecorded("msgbatch_late", 1, iso(timedelta(minutes=12)))]


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        (
            HttpResponse(401, {}, json.dumps({"error": {"type": "auth", "message": KEY}}).encode()),
            "listing batches failed: HTTP 401 (auth: [redacted])",
        ),
        (
            ok({"data": [{"id": "x"}], "has_more": False}),
            "listing batches failed: unexpected batch object",
        ),
    ],
    ids=["http-401", "bad-object"],
)
def test_a_failed_audit_listing_is_a_warning(
    tmp_path: Path, answer: HttpResponse, reason: str
) -> None:
    run = make_run(tmp_path, FakeBatches(list_answer=answer), ALL_SCENARIOS[:1])
    write_state(run.model_dir, batch_entry("msgbatch_91", ["01-pharmacy"], "done"))

    audit(run)

    assert run.echo[-1] == (
        f"warning: could not check for batches that no run recorded ({reason}): look in the "
        "Claude Console (Batches)"
    )


def test_an_unexpected_error_in_the_audit_is_a_warning_with_its_type_only(
    tmp_path: Path,
) -> None:
    class Broken:
        def send(
            self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
        ) -> HttpResponse:
            raise RuntimeError(f"text with {KEY}")

    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    write_state(run.model_dir, batch_entry("msgbatch_91", ["01-pharmacy"], "done"))
    runner_module.audit_batches(
        HAIKU, run.runs, KEY, Broken(), sleep=run.sleeps, now=run.sleeps.now, echo=run.echo
    )

    assert run.echo[-1] == (
        "warning: could not check for batches that no run recorded (RuntimeError): look in the "
        "Claude Console (Batches)"
    )


def test_an_invalid_state_of_another_model_is_a_warning(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    write_state(run.model_dir, batch_entry("msgbatch_91", ["01-pharmacy"], "done"))
    (run.runs / OPUS).mkdir()
    (run.runs / OPUS / BATCH_STATE_FILE).write_text("not json", encoding="utf-8")

    audit(run)

    assert run.echo[-1].startswith("warning: could not check for batches that no run recorded (")
    assert run.echo[-1].endswith(
        "is not a valid batch state; it was not changed): look in the Claude Console (Batches)"
    )


def test_an_interrupt_during_the_audit_listing_ends_the_audit(tmp_path: Path) -> None:
    class Interrupting:
        def send(
            self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
        ) -> HttpResponse:
            raise KeyboardInterrupt

    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    write_state(run.model_dir, batch_entry("msgbatch_91", ["01-pharmacy"], "done"))
    runner_module.audit_batches(
        HAIKU, run.runs, KEY, Interrupting(), sleep=run.sleeps, now=run.sleeps.now, echo=run.echo
    )

    assert run.echo[-1] == "audit interrupted: look in the Claude Console (Batches)"


# --- TASK-005c3e J2: created_at is an instant, or the batch object is unreadable --------------

UNREADABLE_CREATED_AT = ["yesterday", "\ud83d"]


def batch_object(created_at: str, **fields: object) -> HttpResponse:
    """A 200 with a batch object of one request whose ``created_at`` is the given text."""
    return ok(other_batch("msgbatch_01", 1, created_at) | fields)


@pytest.mark.parametrize("created_at", UNREADABLE_CREATED_AT, ids=["yesterday", "surrogate"])
def test_a_create_answer_with_an_unreadable_created_at_is_reconciled_and_adopted(
    tmp_path: Path, created_at: str
) -> None:
    # The API made the batch (listed with a readable created_at); its create answer is unreadable.
    server = FakeBatches(create_script=[Created(batch_object(created_at))])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert run.lists() == 2  # TASK-005c3h H6: the workspace, then the reconciliation
    assert "adopted batch msgbatch_01 created by an earlier create request" in run.echo
    assert len(server.creates()) == 1
    assert run.state()["batches"][0]["created_at"] == CREATED_AT
    assert run.states() == ["done"]


@pytest.mark.parametrize("created_at", UNREADABLE_CREATED_AT, ids=["yesterday", "surrogate"])
def test_a_create_answer_and_a_list_with_an_unreadable_created_at_stop_the_run(
    tmp_path: Path, created_at: str
) -> None:
    server = FakeBatches(
        create_script=[batch_object(created_at)],
        newest=[other_batch("msgbatch_01", 1, created_at)],
        extras_after_create=True,
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == "listing batches failed: unexpected batch object"
    assert run.lists() == 2  # TASK-005c3h H6: the workspace, then the reconciliation
    assert run.states() == ["creating"]  # batches.json still loads; the next run reconciles it
    assert runner_module.run_plan(HAIKU, run.corpus, run.runs).count("Unconfirmed create") == 1


@pytest.mark.parametrize("created_at", UNREADABLE_CREATED_AT, ids=["yesterday", "surrogate"])
def test_a_poll_with_an_unreadable_created_at_stops_the_run(
    tmp_path: Path, created_at: str
) -> None:
    server = FakeBatches(poll_failure=batch_object(created_at, processing_status="ended"))
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == "polling batch msgbatch_01 failed: unexpected batch object"
    assert run.states() == ["open"]


# --- TASK-005c3e J3: the audit window starts at the first create request of any model --------


def test_the_audit_lists_back_to_the_first_create_of_any_model(tmp_path: Path) -> None:
    # F1 (probe P6): a Haiku duplicate listed only after Haiku's own audit; Opus's audit, an hour
    # later, must reach back to it although Opus's own first create is later.
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    run.model = OPUS
    haiku_started = -timedelta(hours=1)
    write_state(
        run.runs / HAIKU,
        batch_entry("msgbatch_91", ["01-pharmacy"], "done", started_at=iso(haiku_started)),
    )
    write_state(run.model_dir, batch_entry("msgbatch_92", ["01-pharmacy"], "done"))
    server.listing = [
        other_batch("msgbatch_92", 1, iso(timedelta(seconds=5))),
        other_batch("msgbatch_dup", 1, iso(haiku_started + timedelta(seconds=320))),
        other_batch("msgbatch_91", 1, iso(haiku_started + timedelta(seconds=5))),
        other_batch("msgbatch_older", 1, iso(haiku_started - timedelta(minutes=10, seconds=1))),
    ]

    runner_module.audit_batches(
        OPUS, run.runs, KEY, server, sleep=run.sleeps, now=run.sleeps.now, echo=run.echo
    )

    assert run.echo[1:] == [unrecorded("msgbatch_dup", 1, "2026-11-02T09:05:20Z")]
    assert run.sleeps == [300]  # the wait still counts from this model's last create only


def test_the_wider_window_never_names_another_models_recorded_batch(tmp_path: Path) -> None:
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    opus_started = -timedelta(hours=2)
    write_state(
        run.runs / OPUS,
        batch_entry("msgbatch_91", ["01-pharmacy"], "done", started_at=iso(opus_started)),
    )
    write_state(run.model_dir, batch_entry("msgbatch_92", ["01-pharmacy"], "done"))
    server.listing = [
        other_batch("msgbatch_92", 1, iso(timedelta(seconds=5))),
        other_batch("msgbatch_91", 1, iso(opus_started + timedelta(seconds=5))),
    ]

    audit(run)

    assert run.echo == [audit_wait(300)]
    assert server.lists == 1


def test_no_audit_for_a_model_without_an_entry_while_another_model_has_some(
    tmp_path: Path,
) -> None:
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    write_state(run.runs / OPUS, batch_entry("msgbatch_91", ["01-pharmacy"], "done"))
    write_state(run.model_dir)

    audit(run)

    assert server.calls == []
    assert run.echo == []
    assert run.sleeps == []


def test_an_invalid_state_of_another_model_warns_before_any_wait_or_call(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    write_state(run.model_dir, batch_entry("msgbatch_91", ["01-pharmacy"], "done"))
    (run.runs / OPUS).mkdir()
    (run.runs / OPUS / BATCH_STATE_FILE).write_text("not json", encoding="utf-8")

    audit(run)

    assert run.server.calls == []
    assert run.sleeps == []
    assert len(run.echo) == 1
    assert run.echo[0].startswith("warning: could not check for batches that no run recorded (")


def test_an_invalid_acknowledgement_file_warns_before_any_wait_or_call(tmp_path: Path) -> None:
    # TASK-005c3h H11 review: the audit read the file only after its wait and list.
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    write_state(run.model_dir, batch_entry("msgbatch_91", ["01-pharmacy"], "done"))
    path = run.runs / "acknowledged-batches.txt"
    path.write_bytes(b"not/an id\n")

    audit(run)

    assert run.server.calls == []
    assert run.sleeps == []
    assert run.echo == [
        "warning: could not check for batches that no run recorded "
        f"({path} line 1 is not a batch id; it was not changed): look in the Claude Console "
        "(Batches)"
    ]


# --- TASK-005c3g G3: a refused replace in a run ------------------------------------------------


@pytest.fixture
def pauses(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """``time.sleep`` (the pauses of ``write_atomic``) recorded instead of slept."""
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    return slept


@pytest.mark.parametrize("name", ["response.json", "batches.json"])
@pytest.mark.parametrize("times", [1, 9])
def test_a_refused_replace_is_absorbed_and_the_run_completes_with_one_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pauses: list[float], times: int, name: str
) -> None:
    # The audit's p03: one refused replace held the batch, and --release paid for it twice.
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:3])
    refused = RefusedReplace(times, name)
    monkeypatch.setattr(os, "replace", refused)

    run()

    assert refused.refused == times
    assert len(pauses) == times
    assert run.server.creates() == [list(ALL_SCENARIOS[:3])]
    assert run.recorded() == list(ALL_SCENARIOS[:3])
    assert run.states() == ["done"]
    assert sorted(run.model_dir.rglob("*.tmp")) == []


def test_ten_refused_replaces_of_an_answer_hold_the_batch_and_keep_the_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pauses: list[float]
) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:3])
    monkeypatch.setattr(os, "replace", RefusedReplace(10, "response.json"))

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == "01-pharmacy: could not be recorded (PermissionError)" + HELD
    assert run.recorded() == list(ALL_SCENARIOS[1:3])
    assert run.states() == ["held"]
    kept = run.model_dir / "01-pharmacy" / "response.json.tmp"
    assert json.loads(kept.read_bytes()) == succeeded()["message"]


# --- TASK-005c3g G6: a plain OSError of urllib in a run ---------------------------------------


class Answer:
    """What ``urllib``'s opener returns for an ``HttpResponse`` of the fake server."""

    def __init__(self, response: HttpResponse) -> None:
        self.status = response.status
        self.headers = http.client.HTTPMessage()
        for name, value in response.headers.items():
            self.headers[name] = value
        self._body = response.body

    def read(self, amount: int = -1) -> bytes:
        return self._body if amount < 0 else self._body[:amount]

    def __enter__(self) -> "Answer":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


@dataclass
class RawOSErrorOpener:
    """An opener over ``FakeBatches`` for the real ``UrllibTransport``: the first ``method``
    call to ``url`` raises a plain ``OSError`` (Winsock 10052), before the server sees it, or,
    with ``after``, once the server has handled it (the answer was lost while it was read)."""

    server: FakeBatches
    method: str
    url: str
    after: bool
    raised: int = 0

    def open(self, fullurl: urllib.request.Request, data: None = None, timeout: float = 0) -> Any:
        method, url = fullurl.get_method(), fullurl.full_url
        hit = self.raised == 0 and method == self.method and url == self.url
        if hit and not self.after:
            self.raised += 1
            raise OSError(10052, "network reset")
        headers = {name.lower(): value for name, value in fullurl.header_items()}
        response = self.server.send(method, url, headers, cast(bytes | None, fullurl.data), timeout)
        if hit:
            self.raised += 1
            raise OSError(10052, "network reset")
        return Answer(response)


def run_over_urllib(run: Run, opener: RawOSErrorOpener) -> None:
    run_anthropic(
        HAIKU,
        run.corpus,
        run.runs,
        KEY,
        UrllibTransport(opener),
        sleep=run.sleeps,
        now=run.sleeps.now,
        echo=run.echo,
    )


def test_a_plain_os_error_on_a_poll_is_retried_once_and_the_run_continues(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:2])
    opener = RawOSErrorOpener(run.server, "GET", f"{BATCHES_URL}/msgbatch_01", after=False)

    run_over_urllib(run, opener)

    assert opener.raised == 1
    assert run.sleeps == [120, 15, 60]  # TASK-005c3h H6: the workspace wait first
    assert run.server.creates() == [list(ALL_SCENARIOS[:2])]
    assert run.recorded() == list(ALL_SCENARIOS[:2])


def test_a_plain_os_error_after_a_create_is_an_unclear_create_and_adopts_its_batch(
    tmp_path: Path,
) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:2])
    opener = RawOSErrorOpener(run.server, "POST", BATCHES_URL, after=True)

    run_over_urllib(run, opener)

    assert opener.raised == 1
    assert run.server.creates() == [list(ALL_SCENARIOS[:2])]
    assert "adopted batch msgbatch_01 created by an earlier create request" in run.echo
    assert run.recorded() == list(ALL_SCENARIOS[:2])
    assert run.states() == ["done"]


# --- TASK-005c3g G2: the laptop clock against the API's ---------------------------------------
# The fake's Date header and created_at follow the API's clock (the Sleeps clock); the runner's
# now() is that clock plus Run.skew.


def clock_stop(seconds: int) -> str:
    return (
        f"the laptop clock differs from the API's by {seconds} s (more than 120 s): sync the "
        "Windows clock (Settings, Time & language, Date & time, Sync now) and run again; nothing "
        "was sent"
    )


NO_DATE = (
    "the API answer has no readable Date header, so the laptop clock cannot be checked; send this "
    "message to the reviewer"
)


def with_one_request_file(run: Run) -> bytes:
    """01-pharmacy's request.json, as an earlier run wrote it; returns its bytes."""
    payload = request_payload(render_test_a(load_scenario(run.corpus / "01-pharmacy")), HAIKU)
    write_request(run.model_dir / "01-pharmacy", payload)
    return (run.model_dir / "01-pharmacy" / "request.json").read_bytes()


def files_under(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


@pytest.mark.parametrize(
    ("skew", "seconds"),
    [
        (timedelta(minutes=15), 900),
        (timedelta(minutes=-15), 900),
        (timedelta(minutes=120), 7200),
        (timedelta(seconds=121), 121),
        (timedelta(seconds=-121), 121),
    ],
    ids=["+15min", "-15min", "+120min", "+121s", "-121s"],
)
def test_a_wrong_laptop_clock_stops_after_one_free_call_and_writes_nothing(
    tmp_path: Path, skew: timedelta, seconds: int
) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:2])
    run.skew = skew
    request = with_one_request_file(run)

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == clock_stop(seconds)
    assert [(call.method, call.url) for call in run.server.calls] == [("GET", CLOCK_URL)]
    assert run.server.creates() == []
    assert files_under(run.runs) == [f"{HAIKU}/01-pharmacy/request.json"]
    assert (run.model_dir / "01-pharmacy" / "request.json").read_bytes() == request
    assert run.sleeps == []


@pytest.mark.parametrize("seconds", [119, -119, 120, -120])
def test_a_laptop_clock_within_two_minutes_proceeds(tmp_path: Path, seconds: int) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:2])
    run.skew = timedelta(seconds=seconds)

    run()

    assert run.server.calls[0].url == CLOCK_URL
    assert run.server.creates() == [list(ALL_SCENARIOS[:2])]
    assert run.recorded() == list(ALL_SCENARIOS[:2])


def test_the_skew_is_measured_when_the_clock_answer_arrived(tmp_path: Path) -> None:
    # The answer takes 200 s and the laptop is 30 s fast: at arrival the laptop says 10:03:50
    # and the header 10:03:20 (30 s, passes); a reading of the laptop clock taken before the call
    # (10:00:30) would differ from the header by 170 s and stop.
    run = make_run(tmp_path, FakeBatches(clock_seconds=200), ALL_SCENARIOS[:1])
    run.skew = timedelta(seconds=30)
    run.rewind = False  # the run starts at NOW (no H6 wait: the clock call took 200 s)

    run()

    assert run.server.clock_checks == [NOW + timedelta(seconds=200)]
    assert run.server.creates() == [["01-pharmacy"]]


@pytest.mark.parametrize(
    "date",
    [None, "", "yesterday", "Mon, 02 Nov 2026 25:00:00 GMT", "Mon, 02 Nov 2026 10:00:00 -0000"],
    ids=["missing", "empty", "words", "hour-25", "naive"],
)
def test_an_answer_without_a_readable_date_stops_the_run(tmp_path: Path, date: str | None) -> None:
    run = make_run(tmp_path, FakeBatches(date_header=date), ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == NO_DATE
    assert [call.url for call in run.server.calls] == [CLOCK_URL]


def test_a_refused_clock_call_stops_with_the_listing_message_redacted(tmp_path: Path) -> None:
    answer = HttpResponse(
        401,
        {},
        json.dumps(
            {"error": {"type": "authentication_error", "message": f"bad key {KEY}"}}
        ).encode(),
    )
    run = make_run(tmp_path, FakeBatches(clock_answers=[answer]), ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == (
        "listing batches failed: HTTP 401 (authentication_error: bad key [redacted])"
    )
    assert [call.url for call in run.server.calls] == [CLOCK_URL]


def test_a_transient_clock_answer_is_retried(tmp_path: Path) -> None:
    server = FakeBatches(clock_answers=[HttpResponse(503, {}, b"")])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert run.sleeps[0] == 15
    assert [call.url for call in server.calls[:2]] == [CLOCK_URL, CLOCK_URL]
    assert run.recorded() == ["01-pharmacy"]


def test_the_clock_is_checked_before_any_other_request_and_before_a_record_is_completed(
    tmp_path: Path,
) -> None:
    # A creating entry (reconcile), an open batch to poll and a half-written record: none is
    # touched before the clock is checked.
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:2])
    run.skew = timedelta(minutes=15)
    scenario = load_scenario(run.corpus / "01-pharmacy")
    write_answer(run.model_dir / "01-pharmacy", scenario, HAIKU, anthropic_message(HAIKU))
    (run.model_dir / "01-pharmacy" / "candidate.txt").unlink()
    write_state(run.model_dir, batch_entry(None, ["02-hotel-reception"], "creating"))
    before = files_under(run.runs)

    with pytest.raises(RunStopped, match=r"^the laptop clock differs from the API's by 900 s"):
        run()

    assert [call.url for call in run.server.calls] == [CLOCK_URL]
    assert files_under(run.runs) == before
    assert run.echo == []


def test_the_clock_check_comes_before_the_future_check_and_after_the_held_check(
    tmp_path: Path,
) -> None:
    # TASK-005c3h H4: release, then held, then the clock call, then the future check.
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    run.skew = timedelta(minutes=15)
    later = batch_entry(None, ["01-pharmacy"], "failed", started_at="2026-11-02T10:40:00Z")
    write_state(run.model_dir, later)

    with pytest.raises(RunStopped) as raised:
        run()  # 10:40 is in the future, but the laptop clock is checked first

    assert str(raised.value) == clock_stop(900)
    assert [(call.method, call.url) for call in run.server.calls] == [("GET", CLOCK_URL)]
    held = batch_entry("msgbatch_01", ["01-pharmacy"], "held", problems=["p1"])
    write_state(run.model_dir, held, later)
    run.server.calls.clear()
    with pytest.raises(RunStopped, match=r"^batch msgbatch_01 is held for the reviewer: p1; "):
        run()
    assert run.server.calls == []


def test_a_slow_laptop_clock_gets_the_clock_stop_and_not_the_future_stop(tmp_path: Path) -> None:
    # The review's #12: the future check ran first, so a slow laptop was told to send
    # batches.json to the reviewer instead of syncing its clock. An entry stamped by the API's
    # clock 4 minutes ago is 11 minutes ahead of a laptop 15 minutes slow (5 minutes ago would
    # be exactly at the 10-minute edge, which the future check allows).
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    run.skew = -timedelta(minutes=15)
    stamped = batch_entry(None, ["01-pharmacy"], "failed", started_at=iso(-timedelta(minutes=4)))
    write_state(run.model_dir, stamped)

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == clock_stop(900)
    assert [(call.method, call.url) for call in run.server.calls] == [("GET", CLOCK_URL)]


def test_the_audits_lost_create_at_a_fast_clock_cannot_happen_any_more(tmp_path: Path) -> None:
    # The audit's p02: at +15 min the reconcile window missed the batch of a lost create, a
    # second batch was created and the audit named nothing. Now the run stops first.
    server = FakeBatches(create_script=["lost"], stamp_with_clock=True)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    run.skew = timedelta(minutes=15)

    with pytest.raises(RunStopped, match=r"^the laptop clock differs from the API's by 900 s"):
        run()

    assert server.creates() == []
    assert server.batches == {}


def test_the_audit_looks_back_to_ten_minutes_before_the_earliest_created_at(
    tmp_path: Path,
) -> None:
    # A batch recorded while the laptop clock was 30 minutes fast: its created_at (API time) is
    # 30 minutes before its started_at (laptop time).
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    created = iso(timedelta(minutes=-30))
    write_state(run.model_dir, batch_entry("msgbatch_91", ["01-pharmacy"], "done", created))
    server.listing = [
        other_batch("msgbatch_dup", 1, iso(timedelta(minutes=-15))),
        other_batch("msgbatch_91", 1, created),
        other_batch("msgbatch_edge", 1, iso(timedelta(minutes=-40))),
        other_batch("msgbatch_older", 1, iso(timedelta(minutes=-40, seconds=-1))),
    ]

    audit(run)

    assert run.echo[1:] == [
        unrecorded("msgbatch_dup", 1, "2026-11-02T09:45:00Z"),
        unrecorded("msgbatch_edge", 1, "2026-11-02T09:20:00Z"),
    ]


# --- TASK-005c3h H3: a release can be repeated --------------------------------------------------

ALREADY_RELEASED = "batch msgbatch_01 was already released"


def test_a_release_stopped_by_the_clock_is_completed_by_the_same_command(tmp_path: Path) -> None:
    # The review's #5: the clock stop says "run again", but the same --release then stopped with
    # "msgbatch_01 is not a held batch".
    run = held_with(tmp_path, result_line("02-hotel-reception", succeeded()))
    run.server.register(list(ALL_SCENARIOS[:2]))  # the held batch: the next one is msgbatch_02
    run.skew = timedelta(minutes=15)
    with pytest.raises(RunStopped) as raised:
        run(release="msgbatch_01")
    assert str(raised.value) == clock_stop(900)
    assert run.states() == ["done"]
    assert run.echo == [
        recorded_from("02-hotel-reception"),
        "batch msgbatch_01 released after review",
    ]
    run.skew = timedelta(0)  # the owner synced the clock
    run.echo.clear()

    run(release="msgbatch_01")

    assert run.echo[0] == ALREADY_RELEASED
    assert run.server.creates() == [["01-pharmacy"]]
    assert run.recorded() == list(ALL_SCENARIOS[:2])
    assert run.states() == ["done", "done"]
    assert problems_of(run, 0) == ["p1"]
    assert problems_of(run, 1) == []


def test_a_release_stopped_by_an_unsaved_answer_is_completed_by_the_same_command(
    tmp_path: Path,
) -> None:
    run = held_with(tmp_path, result_line("02-hotel-reception", succeeded()))
    kept = run.model_dir / "01-pharmacy" / "response.json.tmp"
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"{}\n")
    with pytest.raises(RunStopped) as raised:
        run(release="msgbatch_01")
    assert str(raised.value) == unsaved_stop(kept)
    assert run.states() == ["done"]
    kept.unlink()  # the reviewer decided that the file holds no answer
    run.echo.clear()

    run(release="msgbatch_01")

    assert run.echo[0] == ALREADY_RELEASED
    assert run.server.creates() == [["01-pharmacy"]]
    assert run.recorded() == list(ALL_SCENARIOS[:2])


@pytest.mark.parametrize("problems", [[], ["p1"]])
def test_a_release_of_another_batch_id_still_stops(tmp_path: Path, problems: list[str]) -> None:
    # Only a done entry with problems (held, then released) is "already released".
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    write_state(
        run.model_dir, batch_entry("msgbatch_01", ALL_SCENARIOS[:1], "done", problems=problems)
    )

    with pytest.raises(RunStopped) as raised:
        run(release="msgbatch_02" if problems else "msgbatch_01")

    expected = "msgbatch_02" if problems else "msgbatch_01"
    assert str(raised.value) == f"{expected} is not a held batch of claude-haiku-5-5"
    assert run.server.calls == []


# --- TASK-005c3h H5: a leftover answer file of the open batch -----------------------------------


def replaced_line(name: str, batch_id: str = "msgbatch_01") -> str:
    return (
        f"{name}: a leftover response.json.tmp will be replaced by the answer of batch {batch_id}"
    )


class InterruptedReplace:
    """``os.replace`` interrupted (Ctrl+C) once, for the first replace of a ``response.json``:
    ``write_answer`` keeps the complete, synced temporary file (``keep``)."""

    def __init__(self) -> None:
        self.real = os.replace
        self.done = False

    def __call__(self, source: object, destination: object) -> None:
        if not self.done and Path(str(destination)).name == "response.json":
            self.done = True
            raise KeyboardInterrupt
        self.real(cast(Any, source), cast(Any, destination))


def test_an_interrupted_recording_is_resumed_by_the_same_command_with_no_new_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The review's #2: "interrupted: run the same command again to resume" then stopped on the
    # leftover response.json.tmp of a scenario whose results are simply downloaded again.
    server = FakeBatches(outcome=lambda n, s: succeeded(truth_text(s)))
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    monkeypatch.setattr(os, "replace", InterruptedReplace())
    with pytest.raises(KeyboardInterrupt):
        run()
    monkeypatch.setattr(os, "replace", InterruptedReplace().real)
    kept = run.model_dir / "01-pharmacy" / "response.json.tmp"
    assert kept.is_file()
    assert run.states() == ["open"]
    run.echo.clear()

    run()

    assert run.echo[:2] == [replaced_line("01-pharmacy"), "resuming batch msgbatch_01"]
    assert server.creates() == [list(ALL_SCENARIOS[:2])]
    assert run.recorded() == list(ALL_SCENARIOS[:2])
    assert run.states() == ["done"]
    assert sorted(run.model_dir.rglob("*.tmp")) == []
    message = json.loads(server.results("msgbatch_01").splitlines()[-1])["result"]["message"]
    expected = json.dumps(message, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    assert (run.model_dir / "01-pharmacy" / "response.json").read_bytes() == expected.encode()


def test_a_truncated_leftover_of_the_open_batch_is_replaced_by_the_downloaded_answer(
    tmp_path: Path,
) -> None:
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    batch_id = server.register(list(ALL_SCENARIOS[:2]))
    write_state(run.model_dir, batch_entry(batch_id, ALL_SCENARIOS[:2], "open"))
    kept = run.model_dir / "02-hotel-reception" / "response.json.tmp"
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b'{\n  "content": [\n    {"te')

    run()

    assert run.echo[:2] == [replaced_line("02-hotel-reception"), "resuming batch msgbatch_01"]
    assert server.creates() == []
    assert run.recorded() == list(ALL_SCENARIOS[:2])
    assert sorted(run.model_dir.rglob("*.tmp")) == []
    assert (
        json.loads((run.model_dir / "02-hotel-reception" / "response.json").read_bytes())
        == (succeeded()["message"])
    )


@pytest.mark.parametrize("state", ["done", "failed", "creating"])
def test_a_leftover_of_a_scenario_of_no_open_batch_still_stops(tmp_path: Path, state: str) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:2])
    write_state(run.model_dir, batch_entry("msgbatch_01", ALL_SCENARIOS[:2], state))
    kept = run.model_dir / "01-pharmacy" / "response.json.tmp"
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"{}\n")

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == unsaved_stop(kept)
    assert [(call.method, call.url) for call in run.server.calls] == [("GET", CLOCK_URL)]
    assert kept.read_bytes() == b"{}\n"


def test_only_the_open_batchs_scenarios_are_exempt(tmp_path: Path) -> None:
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    batch_id = server.register(["02-hotel-reception"])
    write_state(run.model_dir, batch_entry(batch_id, ["02-hotel-reception"], "open"))
    for name in ALL_SCENARIOS[:2]:
        (run.model_dir / name).mkdir(parents=True)
        (run.model_dir / name / "response.json.tmp").write_bytes(b"{}\n")

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == unsaved_stop(run.model_dir / "01-pharmacy" / "response.json.tmp")
    assert after_the_clock(server) == []


# --- TASK-005c3h H6: the workspace before a model's first batch -------------------------------

WORKSPACE_WAIT = (
    "waiting {seconds} s before checking the workspace for batches that no run recorded"
)
ACKNOWLEDGED = "acknowledged-batches.txt"


def workspace_stop(run: Run, *ids: str) -> str:
    """The stop of the workspace check, naming ``ids`` (at most 5, then ``and <n> more``)."""
    shown = ", ".join(ids[:5]) + (f" and {len(ids) - 5} more" if len(ids) > 5 else "")
    noun = "batch" if len(ids) == 1 else "batches"
    return (
        f"the workspace has {noun} {shown} that no batches.json in {run.runs} records and "
        "acknowledged-batches.txt does not name; nothing was sent: if the records of an earlier "
        "run were lost, restore them before running again; otherwise send this message to the "
        "reviewer"
    )


def test_a_first_run_waits_lists_the_workspace_and_then_creates(tmp_path: Path) -> None:
    # The review's #21: lost local state re-sent and paid every scenario.
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    run.rewind = False  # the run starts at NOW

    run()

    assert run.echo[:2] == [
        WORKSPACE_WAIT.format(seconds=120),
        "batch msgbatch_01 created for 2 scenarios",
    ]
    assert run.sleeps[0] == 120
    assert server.list_times == [NOW + timedelta(seconds=120)]
    assert [call.url for call in server.calls[:3]] == [CLOCK_URL, LIST_URL, BATCHES_URL]
    assert server.creates() == [list(ALL_SCENARIOS[:2])]
    assert run.recorded() == list(ALL_SCENARIOS[:2])


def test_the_workspace_is_listed_120_seconds_after_the_run_started(tmp_path: Path) -> None:
    # TASK-005c3s S3-1: the clock call took 30 s, so the wait is the rest of the 120 s.
    server = FakeBatches(clock_seconds=30)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    run.rewind = False  # the run starts at NOW

    run()

    assert run.sleeps[0] == 90
    assert run.echo[0] == WORKSPACE_WAIT.format(seconds=90)
    assert server.list_times == [NOW + timedelta(seconds=120)]


def test_a_batch_created_before_the_run_and_listed_late_is_seen(tmp_path: Path) -> None:
    # S3-1: created 60 s before this run started, missing from the list for 90 s: a list at the
    # start would not show it; the list 120 s after the start does.
    server = FakeBatches(lag=90)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    run.rewind = False  # the run starts at NOW
    lost = server.register(list(ALL_SCENARIOS[:2]), created_at=iso(-timedelta(seconds=60)))
    server.posted.add(lost)
    server.posted_at[lost] = NOW - timedelta(seconds=60)

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == workspace_stop(run, lost)
    assert server.creates() == []
    assert not (run.model_dir / BATCH_STATE_FILE).exists()


def test_an_unknown_batch_stops_before_any_create_and_changes_no_state(tmp_path: Path) -> None:
    server = FakeBatches(newest=[other_batch("msgbatch_lost", 2, iso(-timedelta(days=1)))])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    write_state(run.model_dir)  # a batches.json without entries: still a first batch
    before = (run.model_dir / BATCH_STATE_FILE).read_bytes()

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == workspace_stop(run, "msgbatch_lost")
    assert server.creates() == []
    assert (run.model_dir / BATCH_STATE_FILE).read_bytes() == before
    assert files_under(run.runs) == [f"{HAIKU}/batches.json"]


def test_the_stop_names_at_most_five_batches_newest_first(tmp_path: Path) -> None:
    newest = [other_batch(f"msgbatch_u{n}", 1, iso(-timedelta(hours=n))) for n in range(7)]
    server = FakeBatches(newest=[*newest[:3], cancelled_batch("msgbatch_c", 1, iso(timedelta(0)))])
    server.listing = newest[3:]
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == workspace_stop(run, *(f"msgbatch_u{n}" for n in range(7)))
    assert "msgbatch_u4, and" not in str(raised.value)
    assert str(raised.value).startswith(
        "the workspace has batches msgbatch_u0, msgbatch_u1, msgbatch_u2, msgbatch_u3, "
        "msgbatch_u4 and 2 more that no batches.json"
    )


def test_a_cancelled_batch_does_not_stop_the_first_run(tmp_path: Path) -> None:
    server = FakeBatches(newest=[cancelled_batch("msgbatch_c", 2, iso(-timedelta(hours=1)))])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])

    run()

    assert server.creates() == [list(ALL_SCENARIOS[:2])]


def test_another_models_recorded_batch_does_not_stop_the_first_run(tmp_path: Path) -> None:
    # The normal order: Haiku, then Gemini, then Opus: Haiku's batches are in its batches.json.
    server = FakeBatches(newest=[other_batch("msgbatch_77", 2, iso(-timedelta(hours=1)))])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    run.model = OPUS
    server.outcome = lambda n, s: succeeded(model=OPUS)
    write_state(run.runs / HAIKU, batch_entry("msgbatch_77", ALL_SCENARIOS[:2], "done"))

    run()

    assert server.creates() == [list(ALL_SCENARIOS[:2])]


def acknowledge(run: Run, data: bytes) -> Path:
    path = run.runs / ACKNOWLEDGED
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_an_acknowledged_batch_neither_stops_the_first_run_nor_is_named_by_the_audit(
    tmp_path: Path,
) -> None:
    # The review's #22: a batch the audit named once was named again forever.
    server = FakeBatches(newest=[other_batch("msgbatch_dup", 2, iso(-timedelta(minutes=5)))])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    acknowledge(run, b"msgbatch_dup  # decided on 2026-11-08: a paid duplicate\n")

    run()
    run.echo.clear()
    audit(run)

    assert server.creates() == [list(ALL_SCENARIOS[:2])]
    assert not any(line.startswith("warning:") for line in run.echo)


def test_a_model_with_entries_has_no_workspace_wait_or_list(tmp_path: Path) -> None:
    server = FakeBatches(newest=[other_batch("msgbatch_lost", 1, iso(-timedelta(days=1)))])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    write_answer(
        run.model_dir / "01-pharmacy",
        load_scenario(run.corpus / "01-pharmacy"),
        HAIKU,
        anthropic_message(HAIKU),
    )
    write_state(run.model_dir, batch_entry("msgbatch_91", ["01-pharmacy"], "done"))

    run()

    assert run.sleeps == [60]  # the poll of the new batch only
    assert run.lists() == 0
    assert not any(line.startswith("waiting") for line in run.echo)
    assert server.creates() == [["02-hotel-reception"]]


def test_a_workspace_of_more_than_ten_pages_stops_the_first_run(tmp_path: Path) -> None:
    newer = [
        cancelled_batch(f"msgbatch_c{n:04d}", 1, iso(-timedelta(hours=1))) for n in range(1001)
    ]
    server = FakeBatches(newest=newer)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == (
        "the workspace lists more than 1000 batches: this check cannot see them all; nothing was "
        "sent; send this message to the reviewer"
    )
    assert run.pages() == 10
    assert server.creates() == []


# --- TASK-005c3h H6: acknowledged-batches.txt ---------------------------------------------------


def unknown_two(tmp_path: Path) -> Run:
    """A first run whose workspace holds two unknown, live batches."""
    server = FakeBatches(
        newest=[
            other_batch("msgbatch_x", 2, iso(-timedelta(hours=2))),
            other_batch("msgbatch_y", 2, iso(-timedelta(hours=3))),
        ]
    )
    return make_run(tmp_path, server, ALL_SCENARIOS[:2])


def test_an_acknowledgement_file_with_a_bom_and_crlf_is_accepted(tmp_path: Path) -> None:
    # TASK-005c3s S3-2: PowerShell 5.1 writes a BOM (Set-Content -Encoding UTF8, Out-File, >)
    # and CRLF (every way).
    run = unknown_two(tmp_path)
    acknowledge(
        run,
        b"\xef\xbb\xbfmsgbatch_x  # reason one\r\n# a comment\r\n\r\n  \r\nmsgbatch_y\t# two\r\n",
    )

    run()

    assert run.server.creates() == [list(ALL_SCENARIOS[:2])]


def test_an_acknowledgement_file_written_by_add_content_ascii_is_accepted(tmp_path: Path) -> None:
    # The runbook's command, run twice (no BOM, CRLF: TASK-005c3s probe P4).
    run = unknown_two(tmp_path)
    acknowledge(run, b"msgbatch_x  # reason\r\nmsgbatch_y  # reason\r\n")

    run()

    assert run.server.creates() == [list(ALL_SCENARIOS[:2])]


def test_an_acknowledgement_file_that_is_not_utf8_stops(tmp_path: Path) -> None:
    run = unknown_two(tmp_path)
    # A plain 5.1 window's ">": UTF-16 LE with its byte order mark (FF FE).
    data = "\N{ZERO WIDTH NO-BREAK SPACE}msgbatch_x  # reason\r\n".encode("utf-16-le")
    assert data[:2] == b"\xff\xfe"
    path = acknowledge(run, data)

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == (
        f"{path} is not UTF-8: write it with the command in the runbook; it was not changed"
    )
    assert path.read_bytes() == data
    assert run.server.creates() == []


@pytest.mark.parametrize(
    ("text", "number"),
    [
        ("msgbatch_x\nnot/an id\n", 2),
        ("msgbatch_x # ok\n\n# fine\nmsgbatch_y;rm\n", 4),
        # H11 review: a bare CR inside a line (no PowerShell command writes one) would have
        # hidden the next id in the comment of the first.
        ("msgbatch_x  # reason\rmsgbatch_y\n", 1),
        ("\N{NO-BREAK SPACE}msgbatch_x\N{LINE SEPARATOR}\n", 1),
    ],
)
def test_an_acknowledgement_line_that_is_no_batch_id_stops_without_its_text(
    tmp_path: Path, text: str, number: int
) -> None:
    run = unknown_two(tmp_path)
    path = acknowledge(run, text.encode("utf-8"))

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == f"{path} line {number} is not a batch id; it was not changed"
    assert run.server.creates() == []


def test_an_unreadable_acknowledgement_file_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = unknown_two(tmp_path)
    path = acknowledge(run, b"msgbatch_x\nmsgbatch_y\n")
    real_read = Path.read_bytes

    def read_bytes(self: Path) -> bytes:
        if self.name == ACKNOWLEDGED:
            raise PermissionError(13, "Permission denied")
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == f"cannot read {path}: Permission denied"
    assert run.sleeps[-9:] == [refused_pause(attempt) for attempt in range(9)]
    assert run.server.creates() == []


# --- TASK-005c3h H6: two ended candidates (TASK-005c3s S2-1, S3-7) -----------------------------


def test_two_ended_candidates_are_resolved_by_acknowledging_the_newer_one(tmp_path: Path) -> None:
    # The sweep's P3: both batches of a duplicated create ended before the owner could cancel
    # one ("Batches may be canceled any time before processing ends"), so every run stopped.
    server = FakeBatches(create_script=["duplicated"], polls_in_progress=-1)  # always ended
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    with pytest.raises(RunStopped) as raised:
        run()
    assert str(raised.value) == (
        "2 recent batches could be the one created by an earlier create request: msgbatch_01, "
        "msgbatch_02; in the Claude Console (Batches) cancel all but the oldest, then run the "
        "same command again: it adopts the one left; a batch that has already ended cannot be "
        "cancelled: then send this message to the reviewer"
    )
    assert run.states() == ["creating"]
    acknowledge(run, b"msgbatch_02  # the reviewer: a paid duplicate of msgbatch_01\r\n")
    run.echo.clear()

    run()

    assert "adopted batch msgbatch_01 created by an earlier create request" in run.echo
    assert len(server.creates()) == 1
    assert run.recorded() == ["01-pharmacy"]
    assert run.states() == ["done"]
    assert run.batch_ids() == ["msgbatch_01"]
    run.echo.clear()
    audit(run)
    assert not any(line.startswith("warning:") for line in run.echo)


# --- TASK-005c3h H6: the kept results before every create (TASK-005c3s S3-8) ---------------


def lost_answer_stop(name: str, batch_id: str = "msgbatch_01") -> str:
    return (
        f"{name}: its answer is in batch-{batch_id}.results.jsonl but its records are missing: "
        "restore them from git before running again; nothing was sent"
    )


def test_a_lost_scenario_directory_stops_before_any_create(tmp_path: Path) -> None:
    # The sweep's P8: the scenario was sent again in a new batch although its answer was kept.
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    run()
    shutil.rmtree(run.model_dir / ALL_SCENARIOS[0])
    server.calls.clear()

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == lost_answer_stop("01-pharmacy")
    assert server.creates() == []
    assert run.states() == ["done"]


def test_a_scenario_a_release_left_pending_on_purpose_is_sent(tmp_path: Path) -> None:
    run = held_with(
        tmp_path,
        result_line("01-pharmacy", errored("billing_error")),
        result_line("02-hotel-reception", succeeded()),
    )
    run.server.register(list(ALL_SCENARIOS[:2]))
    run(release="msgbatch_01")
    assert run.server.creates() == [["01-pharmacy"]]


def test_an_unreadable_kept_file_of_a_done_batch_stops_before_any_create(tmp_path: Path) -> None:
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    run()
    shutil.rmtree(run.model_dir / ALL_SCENARIOS[0])
    kept = run.model_dir / KEPT
    kept.unlink()

    with pytest.raises(RunStopped) as raised:
        run()

    head = f"cannot read the kept results {kept}: "
    tail = "; nothing was sent: restore it from git or send this message to the reviewer"
    assert str(raised.value).startswith(head)
    assert str(raised.value).endswith(tail)
    assert server.creates() == [list(ALL_SCENARIOS[:2])]


# --- TASK-005c3h H7: each request checked against its pinned SHA256 --------------------------


def changed_csv(corpus: Path, name: str) -> None:
    """One bit of the last character before the final line break of the scenario's first CSV
    file (as ``test_a_changed_byte_in_a_csv_changes_the_pinned_payload``)."""
    csv = sorted((corpus / name).glob("*.csv"))[0]
    data = bytearray(csv.read_bytes())
    data[-2] ^= 1
    csv.write_bytes(bytes(data))


def request_differs(name: str, model: str) -> str:
    pinned = runner_module.PAYLOAD_SHA256[(model, name)]
    return (
        f"{name}: the request for {model} differs from the one the harness pins (SHA256 "
        f"{pinned}); nothing was sent"
    )


def test_a_changed_request_stops_before_the_create_and_writes_no_request(tmp_path: Path) -> None:
    # The review's #6, #19, #47: the pins were checked once, at the start, on the prompt and
    # the schema only; 28 of the 30 inputs only in tests.
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:3])
    changed_csv(run.corpus, "02-hotel-reception")

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == request_differs("02-hotel-reception", HAIKU)
    assert after_the_workspace(server) == []  # only the free clock call and workspace list
    assert sorted(run.runs.rglob("request.json")) == []
    assert not (run.model_dir / BATCH_STATE_FILE).exists()


def test_a_request_without_a_pin_stops_before_the_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pins = dict(runner_module.PAYLOAD_SHA256)
    del pins[(HAIKU, "01-pharmacy")]
    monkeypatch.setattr(runner_module, "PAYLOAD_SHA256", pins)
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == (
        "01-pharmacy: no request is pinned for claude-haiku-5-5; nothing was sent"
    )
    assert after_the_workspace(server) == []
    assert sorted(run.runs.rglob("request.json")) == []


@pytest.mark.parametrize("model", ["claude-haiku-5-5", "claude-opus-5-5", "gemini-3.8-flash"])
def test_the_real_corpus_passes_the_request_check_for_every_model(
    tmp_path: Path, model: str
) -> None:
    plan = runner_module.run_plan(model, CORPUS_PATH, tmp_path / "runs")

    assert "Pending (10): " in plan


# --- TASK-005c3h H8: small fixes ------------------------------------------------------------


def test_a_csv_that_cannot_be_read_is_named_and_not_called_the_instruction_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The review's #10: any OSError of the rendering was "cannot read the instruction file".
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    csv = sorted((run.corpus / "01-pharmacy").glob("*.csv"))[0]
    real_read = Path.read_bytes

    def read_bytes(self: Path) -> bytes:
        if self == csv:
            raise PermissionError(13, "Permission denied", str(self))
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == f"01-pharmacy: cannot read {csv}: Permission denied"
    assert run.server.creates() == []


def failing_derivation(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """``write_answer`` saves ``response.json`` of scenario ``name``, then writing its
    ``candidate.txt`` fails (a full disk): the answer is recorded, its record is not complete."""
    real = records_module.write_atomic

    def write(path: Path, data: bytes, **kwargs: Any) -> None:
        if path.parent.name == name and path.name == "candidate.txt":
            raise OSError(28, f"No space left on device {KEY}")
        real(path, data, **kwargs)

    monkeypatch.setattr(records_module, "write_atomic", write)


def test_a_recorded_answer_whose_record_fails_says_so_and_the_release_completes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The review's #3 and TASK-005c3s S3-6: it said "could not be recorded" although
    # response.json was saved; a held batch is completed only by the release.
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    with monkeypatch.context() as patch:
        failing_derivation(patch, "02-hotel-reception")
        with pytest.raises(RunStopped) as raised:
            run()

    assert str(raised.value) == (
        "02-hotel-reception: recorded, but its record could not be completed (OSError); it is "
        "completed when the batch is released" + HELD
    )
    assert run.states() == ["held"]
    assert run.recorded() == list(ALL_SCENARIOS[:2])
    run.echo.clear()

    run(release="msgbatch_01")

    assert run.echo[:2] == [
        "batch msgbatch_01 released after review",
        "02-hotel-reception: record completed (candidate.txt, comparison.json)",
    ]
    assert server.creates() == [list(ALL_SCENARIOS[:2])]


def test_an_answer_that_was_not_saved_is_still_could_not_be_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pauses: list[float]
) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    monkeypatch.setattr(os, "replace", RefusedReplace(10, "response.json"))

    with pytest.raises(
        RunStopped, match=r"^01-pharmacy: could not be recorded \(PermissionError\)"
    ):
        run()


def test_a_recorded_answer_that_cannot_be_stat_ed_stops_the_run_before_any_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # TASK-005c3s S1-1: on 3.14 the scenario counted as pending (or recorded) depending on the
    # OS; on 3.12 the error was raised. Now presence is decided by our code: the error stops.
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    scenario = load_scenario(run.corpus / "01-pharmacy")
    write_answer(run.model_dir / "01-pharmacy", scenario, HAIKU, anthropic_message(HAIKU))
    target = run.model_dir / "01-pharmacy" / "response.json"
    real = os.stat

    def stat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        if Path(path) == target:
            raise PermissionError(13, "Permission denied", str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat)

    with pytest.raises(PermissionError):
        run()

    assert server.creates() == []
