"""The Anthropic Test A runner (TASK-005c3 F2, F3): Message Batches through a fake server.

``FakeBatches`` plays the Message Batches API: it creates numbered batches, answers polls with
``in_progress`` and then ``ended``, lists them and serves each batch's JSONL results, built from an
``outcome`` function of (batch number, scenario id). No test opens a connection. The clock of a
run is its ``Sleeps``: ``now()`` moves only by the seconds slept (TASK-005c3b D2).
"""

import json
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
DO_NOT_RUN = "; do not run again: send this message to the reviewer"  # D6

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
    # registers the batch, then raises KeyboardInterrupt; an HttpResponse is returned as is (no
    # batch); Created(response) registers the batch, then returns the response.
    create_script: list[CreateStep] = field(default_factory=list[CreateStep])
    listing: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    # The batches created by create calls are missing from the first ``hidden_lists`` list calls.
    hidden_lists: int = 0
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
        if method == "GET" and url == LIST_URL:
            if self.list_failures:
                raise self.list_failures.pop(0)
            self.lists += 1
            if self.list_answer is not None:
                return self.list_answer
            hidden = self.posted if self.lists <= self.hidden_lists else set[str]()
            data = [
                self.batch(batch_id)
                for batch_id in reversed(self.batches)
                if batch_id not in hidden
            ] + self.listing
            return ok({"data": data, "has_more": False, "first_id": None, "last_id": None})
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
        batch_id = self.register(scenarios)
        self.posted.add(batch_id)
        if isinstance(step, Created):
            return step.response
        if step == "lost":
            raise TransportError("TimeoutError: no response received")
        if step == "interrupt-created":
            raise KeyboardInterrupt
        return ok(self.batch(batch_id))

    def creates(self) -> list[list[str]]:
        """The scenarios of every create call, in order."""
        return [
            [r["custom_id"] for r in json.loads(call.body or b"{}")["requests"]]
            for call in self.calls
            if call.method == "POST"
        ]


def ok(document: object) -> HttpResponse:
    return HttpResponse(200, {"content-type": "application/json"}, json.dumps(document).encode())


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

    def __call__(self) -> None:
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
        )

    def state(self) -> dict[str, Any]:
        return json.loads((self.model_dir / BATCH_STATE_FILE).read_bytes())

    def states(self) -> list[str]:
        return [batch["state"] for batch in self.state()["batches"]]

    def lists(self) -> int:
        return sum(call.url == LIST_URL for call in self.server.calls)

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
        assert call.headers == {
            "x-api-key": KEY,
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


TO_THE_REVIEWER = "; send batch-msgbatch_01.results.jsonl to the reviewer before running again"


def test_a_scenario_missing_from_the_results_stops_the_run(tmp_path: Path) -> None:
    # D3: it used to stay pending silently, and the next batch would pay for it again.
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:3])
    first = json.dumps({"custom_id": ALL_SCENARIOS[0], "result": succeeded()}) + "\n"
    server.raw_results["msgbatch_01"] = first.encode()

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == (
        "batch msgbatch_01: no result for 02-hotel-reception; "
        "batch msgbatch_01: no result for 03-call-centre" + TO_THE_REVIEWER
    )
    assert run.server.creates() == [list(ALL_SCENARIOS[:3])]
    assert run.recorded() == [ALL_SCENARIOS[0]]
    assert run.states() == ["done"]
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

    assert str(raised.value) == (
        "02-hotel-reception: invalid_request_error: output_config: unknown field" + DO_NOT_RUN
    )
    assert run.recorded() == ["01-pharmacy", "03-call-centre"]
    assert not (run.model_dir / "02-hotel-reception" / "response.json").exists()
    assert (run.model_dir / "02-hotel-reception" / "request.json").is_file()
    assert run.state()["batches"][0]["state"] == "done"
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
        "; batch msgbatch_01: no result for 02-hotel-reception" + TO_THE_REVIEWER
    )
    raw = run.model_dir / "batch-msgbatch_01.results.jsonl"
    assert raw.read_bytes() == server.raw_results["msgbatch_01"]
    assert run.recorded() == ["01-pharmacy"]


def test_a_duplicate_results_line_stops(tmp_path: Path) -> None:
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    line = json.dumps({"custom_id": "01-pharmacy", "result": succeeded()}).encode()
    server.raw_results["msgbatch_01"] = line + b"\n" + line + b"\n"

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == (
        "batch msgbatch_01: results line 2 names 01-pharmacy twice" + TO_THE_REVIEWER
    )
    assert run.recorded() == ["01-pharmacy"]


def test_an_invalid_request_error_and_a_missing_result_keep_both_endings(tmp_path: Path) -> None:
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    line = {"custom_id": "01-pharmacy", "result": errored("invalid_request_error", "bad")}
    server.raw_results["msgbatch_01"] = (json.dumps(line) + "\n").encode()

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == (
        "batch msgbatch_01: no result for 02-hotel-reception; "
        "01-pharmacy: invalid_request_error: bad" + DO_NOT_RUN + TO_THE_REVIEWER
    )


def test_a_results_url_on_another_host_is_refused(tmp_path: Path) -> None:
    server = FakeBatches(results_base="https://evil.example.com/v1/messages/batches")
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped, match=r"refused: evil\.example\.com is not an API host"):
        run()

    assert not any("evil" in call.url for call in server.calls)
    assert run.recorded() == []


@pytest.mark.parametrize("code", [400, 401, 403, 404, 413])
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
        }
    ]
    assert run.sleeps == [10, 60]
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


def test_a_lost_create_with_no_candidate_is_checked_twice_then_retried(tmp_path: Path) -> None:
    server = FakeBatches(
        create_script=["lost-none"],
        listing=[other_batch("msgbatch_old", 1, iso(-timedelta(minutes=30)))],
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert len(server.creates()) == 2
    # check 1 at started + 10 s, check 2 at started + 70 s, then the retry delay and the poll.
    assert run.sleeps == [10, 60, 15, 60]
    assert run.lists() == 2
    assert [batch["id"] for batch in run.state()["batches"]] == ["msgbatch_01"]
    # started_at is the instant just before the create call that created the batch (the retry).
    assert run.state()["batches"][0]["started_at"] == "2026-11-02T10:01:25Z"


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
    assert run.sleeps == [10, 60, 15, 10, 60, 45, 10, 60, 135, 10, 60]
    assert run.states() == ["failed"]


def test_a_create_answered_200_without_a_batch_object_is_treated_as_lost(tmp_path: Path) -> None:
    server = FakeBatches(create_script=[HttpResponse(200, {}, b"<html>")])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert len(server.creates()) == 2
    assert run.lists() == 2
    assert run.sleeps == [10, 60, 15, 60]


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


def test_a_succeeded_result_without_a_message_stops(tmp_path: Path) -> None:
    server = FakeBatches(outcome=lambda n, s: {"type": "succeeded"})
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped, match="01-pharmacy: a succeeded result without a message"):
        run()

    assert run.recorded() == []


def state_text(*entries: dict[str, Any]) -> str:
    return json.dumps({"batches": list(entries)})


OPEN = batch_entry("msgbatch_01", ["01-pharmacy"], "open")
CREATING = batch_entry(None, ["01-pharmacy"], "creating")


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
        f"01-pharmacy: invalid_request_error: {'x' * 290}[redacted]" + DO_NOT_RUN
    )


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

    assert str(raised.value) == "01-pharmacy: invalid_request_error: bad?[31m?" + DO_NOT_RUN


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
    assert run.echo[0] == "adopted batch msgbatch_01 created by an earlier create request"
    assert run.recorded() == list(ALL_SCENARIOS[:2])
    assert run.sleeps == [10, 60]


@pytest.mark.parametrize("code", GATEWAY)
def test_a_gateway_error_without_a_batch_is_checked_twice_then_retried(
    tmp_path: Path, code: int
) -> None:
    server = FakeBatches(create_script=[HttpResponse(code, {"retry-after": "30"}, b"")])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert len(server.creates()) == 2
    assert run.lists() == 2
    assert run.sleeps == [10, 60, 30, 60]
    assert run.states() == ["done"]


def test_four_gateway_errors_mark_the_create_failed(tmp_path: Path) -> None:
    server = FakeBatches(create_script=[HttpResponse(503, {}, b"")] * 4)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == "creating a batch failed 4 times; last: HTTP 503"
    assert run.lists() == 8
    assert run.states() == ["failed"]


def test_a_batch_missing_from_the_first_list_is_found_by_the_second(tmp_path: Path) -> None:
    server = FakeBatches(create_script=["lost"], hidden_lists=1)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert len(server.creates()) == 1
    assert run.lists() == 2
    assert run.sleeps == [10, 60, 60]
    assert "adopted batch msgbatch_01 created by an earlier create request" in run.echo


def test_a_slow_create_answer_waits_a_full_minute_before_the_second_check(
    tmp_path: Path,
) -> None:
    # The 504 arrives 65 s after the request: the batch may have been created just before, so
    # the second list call comes at least 60 s after the first one.
    server = FakeBatches(
        create_script=[Created(HttpResponse(504, {}, b""))], hidden_lists=1, create_seconds=65
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert run.sleeps[0] == 60
    assert run.lists() == 2
    assert len(server.creates()) == 1


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

    assert run.echo[0] == "adopted batch msgbatch_01 created by an earlier create request"
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

    assert run.echo[0] == (
        "no batch was created by the earlier create request (started 2026-11-02T10:00:00Z)"
    )
    assert run.echo[1] == "batch msgbatch_01 created for 2 scenarios"
    assert run.sleeps == [10, 60, 60]
    assert run.lists() == 2
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

    assert run.echo[0] == "adopted batch msgbatch_01 created by an earlier create request"
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

    assert run.sleeps == [10, 15, 60]
    assert "adopted batch msgbatch_01 created by an earlier create request" in run.echo
