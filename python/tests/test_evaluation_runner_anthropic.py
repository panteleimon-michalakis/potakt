"""The Anthropic Test A runner (TASK-005c3 F2, F3): Message Batches through a fake server.

``FakeBatches`` plays the Message Batches API: it creates numbered batches, answers polls with
``in_progress`` and then ``ended``, lists them and serves each batch's JSONL results, built from an
``outcome`` function of (batch number, scenario id). No test opens a connection. The clock of a
run is its ``Sleeps``: ``now()`` moves only by the seconds slept (TASK-005c3b D2).
"""

import json
import math
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from runner_helpers import (
    ALL_SCENARIOS,
    KEY,
    NOW,
    STARTED_AT,
    Echo,
    Sleeps,
    anthropic_message,
    batch_entry,
    corpus_copy,
    files_containing,
    truth_text,
    write_state,
)
from typer.testing import CliRunner

import potakt.evaluation.runner as runner_module
from potakt import __version__
from potakt.cli import app
from potakt.evaluation import load_scenario, summarize, write_answer
from potakt.evaluation.runner import (
    ANTHROPIC_VERSION,
    BATCH_STATE_FILE,
    BATCHES_URL,
    HEARTBEAT_POLLS,
    run_anthropic,
)
from potakt.evaluation.transport import HttpResponse, RunStopped, TransportError

HAIKU = "claude-haiku-5-5"
OPUS = "claude-opus-5-5"
CREATED_AT = "2026-11-02T10:00:05.123456Z"
LIST_URL = f"{BATCHES_URL}?limit=100"

Outcome = Callable[[int, str], dict[str, Any]]


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
    # TransportError; an HttpResponse is returned as is (no batch); Created(response) registers
    # the batch, then returns the response.
    create_script: list[CreateStep] = field(default_factory=list[CreateStep])
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
        self.calls.append(Call(method, url, headers, body, timeout))
        if method == "POST" and url == BATCHES_URL:
            return self._create(body)
        if method == "GET" and url.startswith(LIST_URL):
            if self.list_failures:
                raise self.list_failures.pop(0)
            self.lists += 1
            now = self.now()
            self.list_times.append(now)
            if self.list_pages:
                return self.list_pages.pop(0)
            if self.list_answer is not None:
                return self.list_answer
            data = (
                self.newest
                + [
                    self.batch(batch_id)
                    for batch_id in reversed(self.batches)
                    if batch_id not in self.posted
                    or now
                    >= self.posted_at[batch_id]
                    + timedelta(seconds=self.lag + self.extra_lag.get(batch_id, 0.0))
                ]
                + self.listing
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

    @property
    def model_dir(self) -> Path:
        return self.runs / self.model

    def __call__(self, release: str | None = None) -> None:
        run_anthropic(
            self.model,
            self.corpus,
            self.runs,
            KEY,
            self.server,
            sleep=self.sleeps,
            now=self.sleeps.now,
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
    assert [call.method for call in server.calls] == ["POST", "GET", "GET", "GET"]
    assert run.sleeps == [60]
    assert run.echo[0] == "batch msgbatch_01 created for 10 scenarios"
    assert run.echo[1] == (
        "batch msgbatch_01: in_progress "
        "(processing 10, succeeded 0, errored 0, expired 0, canceled 0)"
    )
    assert run.echo[2] == (
        "batch msgbatch_01: ended (processing 0, succeeded 10, errored 0, expired 0, canceled 0)"
    )
    assert run.echo[3] == "01-pharmacy: recorded (end_turn, valid, recall 1.000)"
    assert run.echo[4] == "02-hotel-reception: recorded (end_turn, invalid, recall 0.000)"
    assert len(run.echo) == 13


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

    assert run.sleeps == [60, 60, 60]
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

    assert run.server.calls == []
    assert run.echo == ["nothing to run"]


# --- resume and pending ----------------------------------------------------------------------


def test_an_open_batch_is_resumed_without_a_create_call(tmp_path: Path) -> None:
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:3])
    batch_id = server.register(list(ALL_SCENARIOS[:3]))
    write_state(run.model_dir, batch_entry(batch_id, ALL_SCENARIOS[:3], "open"))

    run()

    assert [call.method for call in server.calls] == ["GET", "GET", "GET"]
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
    assert run.sleeps == [60, 15, 60]
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
    assert run.sleeps == [60, 15, 60, 45, 60, 135, 60]
    assert run.recorded() == ["01-pharmacy"]


def test_the_attempt_cap_holds_across_invocations(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:2])
    batches = [batch_entry(f"msgbatch_9{n}", ["02-hotel-reception"], "done") for n in range(4)]
    write_state(run.model_dir, *batches)

    with pytest.raises(RunStopped, match="02-hotel-reception was sent in 4 batches"):
        run()

    assert run.server.calls == []


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
    assert run.sleeps == []
    assert run.lists() == 0


def test_a_create_answered_with_429_is_simply_retried(tmp_path: Path) -> None:
    server = FakeBatches(create_script=[HttpResponse(429, {"retry-after": "20"}, b"")])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert len(server.creates()) == 2
    assert run.lists() == 0
    assert run.sleeps == [20, 60]
    assert run.states() == ["done"]


def test_four_429_answers_mark_the_create_failed(tmp_path: Path) -> None:
    server = FakeBatches(create_script=[HttpResponse(429, {}, b"")] * 4)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == "creating a batch failed 4 times; last: HTTP 429"
    assert run.sleeps == [15, 45, 135]
    assert run.lists() == 0
    assert run.states() == ["failed"]


def test_a_poll_failure_stops_the_run_and_keeps_the_batch_open(tmp_path: Path) -> None:
    server = FakeBatches(poll_failure=TransportError("TimeoutError: no response received"))
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped, match="the request failed 4 times"):
        run()

    assert run.state()["batches"][0]["state"] == "open"
    assert run.sleeps == [15, 45, 135]


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
    assert run.sleeps == [120, 60]
    assert run.lists() == 1
    assert run.recorded() == list(ALL_SCENARIOS[:3])


def test_a_batch_created_exactly_ten_minutes_before_is_a_candidate(tmp_path: Path) -> None:
    server = FakeBatches(create_script=["lost-none"])
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
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert len(server.creates()) == 2
    # E3: one list call at started + 120 s, then the retry delay and the poll.
    assert run.sleeps == [120, 15, 60]
    assert run.lists() == 1
    assert server.list_times == [NOW + timedelta(seconds=120)]
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
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == (
        "2 recent batches could be the one created by an earlier create request: msgbatch_01, "
        "msgbatch_twin; in the Claude Console (Batches) cancel all but the oldest, then run the "
        "same command again: it adopts the one left"
    )
    assert run.state()["batches"] == [batch_entry(None, ALL_SCENARIOS[:1], "creating")]
    assert len(server.creates()) == 1


def test_four_lost_creates_stop_the_run_and_mark_the_create_failed(tmp_path: Path) -> None:
    server = FakeBatches(create_script=["lost-none"] * 4)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped, match=r"^creating a batch failed 4 times; last: TimeoutError"):
        run()

    assert len(server.creates()) == 4
    assert run.sleeps == [120, 15, 120, 45, 120, 135, 120]
    assert run.lists() == 4
    assert run.states() == ["failed"]


def test_a_create_answered_200_without_a_batch_object_is_treated_as_lost(tmp_path: Path) -> None:
    server = FakeBatches(create_script=[HttpResponse(200, {}, b"<html>")])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert len(server.creates()) == 2
    assert run.lists() == 1
    assert run.sleeps == [120, 15, 60]


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
    server = FakeBatches(create_script=["lost-none"], list_answer=listing)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == message
    assert run.states() == ["creating"]  # the next run reconciles it again


def test_a_fractional_second_beyond_microseconds_is_read(tmp_path: Path) -> None:
    server = FakeBatches(create_script=["lost-none"])
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

    assert run.server.calls == []


def test_a_scenario_id_that_is_no_custom_id_stops(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    (run.corpus / "01 pharmacy").mkdir()
    (run.corpus / "01 pharmacy" / "scenario.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RunStopped, match="01 pharmacy: not a valid custom_id"):
        run()

    assert run.server.calls == []


def test_a_different_existing_request_stops(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
    (run.model_dir / ALL_SCENARIOS[0]).mkdir(parents=True)
    (run.model_dir / ALL_SCENARIOS[0] / "request.json").write_bytes(b"{}\n")

    with pytest.raises(RunStopped, match="exists with a different request"):
        run()

    assert run.server.calls == []


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
    assert run.server.calls == []


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
        def __call__(self, seconds: float) -> None:
            raise KeyboardInterrupt

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


@pytest.mark.parametrize("kind", ["\x1b[2J", "a" * 65, 7])
def test_an_odd_error_type_is_not_echoed(tmp_path: Path, kind: object) -> None:
    # Minor 3: the error type of a result line was echoed as the server sent it.
    def outcome(number: int, scenario: str) -> dict[str, Any]:
        if number > 1:
            return succeeded()
        return {"type": "errored", "error": {"type": "error", "error": {"type": kind}}}

    run = make_run(tmp_path, FakeBatches(outcome=outcome), ALL_SCENARIOS[:1])

    run()

    assert "01-pharmacy: errored (unknown error); pending" in run.echo


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

    run_anthropic(
        HAIKU, run.corpus, run.runs, KEY, Watching(), sleep=run.sleeps, now=run.sleeps.now,
        echo=run.echo,
    )  # fmt: skip

    assert seen == [{"batches": [batch_entry(None, ALL_SCENARIOS[:2], "creating")]}]
    assert run.state()["batches"] == [batch_entry("msgbatch_01", ALL_SCENARIOS[:2], "done")]
    text = (run.model_dir / BATCH_STATE_FILE).read_bytes()
    assert text == (json.dumps(run.state(), indent=2, sort_keys=True) + "\n").encode()


def test_started_at_is_the_utc_instant_in_whole_seconds(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeBatches(), ALL_SCENARIOS[:1])
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
    assert run.echo[:2] == [
        waiting(120),
        "adopted batch msgbatch_01 created by an earlier create request",
    ]
    assert run.recorded() == list(ALL_SCENARIOS[:2])
    assert run.sleeps == [120, 60]


@pytest.mark.parametrize("code", GATEWAY)
def test_a_gateway_error_without_a_batch_is_checked_once_then_retried(
    tmp_path: Path, code: int
) -> None:
    server = FakeBatches(create_script=[HttpResponse(code, {"retry-after": "30"}, b"")])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert len(server.creates()) == 2
    assert run.lists() == 1
    assert run.sleeps == [120, 30, 60]
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
    assert run.echo[:2] == [
        waiting(120),
        "adopted batch msgbatch_01 created by an earlier create request",
    ]
    assert run.recorded() == list(ALL_SCENARIOS[:2])
    assert server.list_times == [NOW + timedelta(seconds=120)]
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
    assert server.list_times == [NOW + timedelta(seconds=120)]
    assert run.sleeps == [120]
    assert len(server.creates()) == 1
    assert run.state()["batches"] == [batch_entry(None, ALL_SCENARIOS[:1], "failed")]


@pytest.mark.parametrize("code", [400, 401, 403, 404, 409, 413, 422])
def test_a_definite_rejection_lists_nothing(tmp_path: Path, code: int) -> None:
    server = FakeBatches(create_script=[Created(HttpResponse(code, {}, b""))])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped, match=rf"^creating a batch failed: HTTP {code}$"):
        run()

    assert run.lists() == 0
    assert run.sleeps == []
    assert run.states() == ["failed"]


def test_four_gateway_errors_mark_the_create_failed(tmp_path: Path) -> None:
    server = FakeBatches(create_script=[HttpResponse(503, {}, b"")] * 4)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == "creating a batch failed 4 times; last: HTTP 503"
    assert run.lists() == 4
    assert run.states() == ["failed"]


# -- TASK-005c3c E3: one list call, at least 120 s after the later of the call and its answer --


def test_a_batch_missing_from_the_list_for_100_seconds_is_found(tmp_path: Path) -> None:
    server = FakeBatches(create_script=["lost"], lag=100)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert len(server.creates()) == 1
    assert server.list_times == [NOW + timedelta(seconds=120)]
    assert run.sleeps == [120, 60]
    assert "adopted batch msgbatch_01 created by an earlier create request" in run.echo


def test_a_duplicated_create_with_a_late_twin_stops_the_run(tmp_path: Path) -> None:
    # TASK-005c3b I3: the first check saw only the first batch, adopted it, and the twin, still
    # hidden, became a paid orphan.
    server = FakeBatches(create_script=["duplicated"])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped, match=r"^2 recent batches could be the one created") as raised:
        run()

    assert "msgbatch_01, msgbatch_02;" in str(raised.value)
    assert run.lists() == 1
    assert run.states() == ["creating"]
    assert server.polled == {}


def test_a_slow_unclear_answer_is_looked_up_120_seconds_after_it_arrived(tmp_path: Path) -> None:
    # The 504 arrives 90 s after the request; the batch may have been created just before it.
    server = FakeBatches(
        create_script=[Created(HttpResponse(504, {}, b""))], create_seconds=90, lag=110
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert server.list_times == [NOW + timedelta(seconds=90 + 120)]
    assert run.sleeps[0] == 120
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
    server = FakeBatches(create_script=["lost-none"], list_answer=ok({"data": [listed]}))
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped, match="listing batches failed: unexpected batch object"):
        run()


def test_a_batch_without_cancel_initiated_at_is_not_cancelled(tmp_path: Path) -> None:
    # "Specified only if cancellation was initiated": a missing field is no cancel.
    listed = other_batch("msgbatch_x", 1, iso(timedelta(seconds=1)))
    server = FakeBatches(create_script=["lost-none"], listing=[listed])
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
    assert run.echo[:2] == [
        waiting(180),
        "adopted batch msgbatch_01 created by an earlier create request",
    ]
    assert run.sleeps == [180, 60]
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

    assert run.echo[:3] == [
        waiting(180),
        "no batch was created by the earlier create request (started 2026-11-02T10:00:00Z)",
        "batch msgbatch_01 created for 2 scenarios",
    ]
    assert run.sleeps == [180, 60]
    assert run.lists() == 1
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
        create_script=["lost"], list_failures=[TransportError("TimeoutError: no response")]
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert run.sleeps == [120, 15, 60]
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

    with pytest.raises(RunStopped, match="02-hotel-reception was sent in 4 batches"):
        run(release="msgbatch_99")  # released, it still counts

    assert run.server.calls == []
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
    server = FakeBatches(create_script=["lost-none"])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    edge = server.register(list(ALL_SCENARIOS[:1]), created_at=iso(timedelta(minutes=minutes)))

    run()

    assert (f"adopted batch {edge} created by an earlier create request" in run.echo) is adopted
    assert len(server.creates()) == (1 if adopted else 2)


def test_the_in_run_window_ends_after_a_slow_answer(tmp_path: Path) -> None:
    # The answer arrives 90 s after the request: the window ends 10 minutes after the answer.
    server = FakeBatches(create_script=[HttpResponse(504, {}, b"")], create_seconds=90)
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
def test_an_entry_started_more_than_ten_minutes_in_the_future_stops_before_any_call(
    tmp_path: Path, state: str
) -> None:
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
    assert run.server.calls == []
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

    assert run.echo[:2] == [
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
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    server.register(list(ALL_SCENARIOS[:1]), created_at=iso(timedelta(seconds=5)))

    run()

    assert "adopted batch msgbatch_01 created by an earlier create request" in run.echo
    assert len(server.creates()) == 1
    list_urls = [call.url for call in server.calls if call.url.startswith(LIST_URL)]
    assert list_urls == [LIST_URL, f"{LIST_URL}&after_id=msgbatch_c099"]


def test_the_paging_stops_at_the_first_page_that_ends_before_the_window(tmp_path: Path) -> None:
    recent = [other_batch(f"msgbatch_r{n:03d}", 3, iso(-timedelta(minutes=5))) for n in range(100)]
    old = [other_batch(f"msgbatch_o{n:03d}", 3, iso(-timedelta(minutes=20))) for n in range(150)]
    server = FakeBatches(create_script=["lost-none"], newest=recent + old)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    # Page 1 ends at -5 min (inside the window), page 2 at -20 min: page 3 is never read.
    assert run.pages() == 2
    assert len(server.creates()) == 2


def test_a_page_ending_exactly_at_the_window_is_followed(tmp_path: Path) -> None:
    edge = [other_batch(f"msgbatch_e{n:03d}", 3, iso(-timedelta(minutes=10))) for n in range(100)]
    server = FakeBatches(create_script=["lost-none"], newest=edge)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    server.register(list(ALL_SCENARIOS[:1]), created_at=iso(-timedelta(minutes=10)))

    run()

    assert run.pages() == 2
    assert "adopted batch msgbatch_01 created by an earlier create request" in run.echo


@pytest.mark.parametrize(("count", "stops"), [(1000, False), (1001, True)])
def test_more_than_ten_pages_stop_the_run(tmp_path: Path, count: int, stops: bool) -> None:
    newer = [
        cancelled_batch(f"msgbatch_c{n:04d}", 1, iso(timedelta(seconds=1))) for n in range(count)
    ]
    server = FakeBatches(create_script=["lost-none"], newest=newer)
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
    # 1000: the tenth page has has_more false; 1001: it has more, and no eleventh page is read.
    assert run.pages() == 10


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
    server = FakeBatches(create_script=["lost-none"], list_pages=[list_page([listed], **fields)])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == "listing batches failed: unexpected list object"
    assert run.pages() == 1
    assert run.states() == ["creating"]


def test_an_empty_page_that_has_more_stops(tmp_path: Path) -> None:
    server = FakeBatches(create_script=["lost-none"], list_pages=[list_page([])])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped, match=r"^listing batches failed: unexpected list object$"):
        run()

    assert run.pages() == 1


def test_a_failed_second_page_stops_the_run(tmp_path: Path) -> None:
    listed = other_batch("msgbatch_x", 3, iso(timedelta(seconds=1)))
    second = HttpResponse(403, {}, b"")
    server = FakeBatches(
        create_script=["lost-none"],
        list_pages=[list_page([listed], last_id="msgbatch_x"), second],
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped, match=r"^listing batches failed: HTTP 403$"):
        run()

    assert run.pages() == 2
    assert run.states() == ["creating"]
