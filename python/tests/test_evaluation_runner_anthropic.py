"""The Anthropic Test A runner (TASK-005c3 F2, F3): Message Batches through a fake server.

``FakeBatches`` plays the Message Batches API: it creates numbered batches, answers polls with
``in_progress`` and then ``ended``, lists them and serves each batch's JSONL results, built from an
``outcome`` function of (batch number, scenario id). No test opens a connection.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from runner_helpers import (
    ALL_SCENARIOS,
    KEY,
    NOW,
    Echo,
    Sleeps,
    anthropic_message,
    corpus_copy,
    files_containing,
    truth_text,
)
from typer.testing import CliRunner

from potakt import __version__
from potakt.cli import app
from potakt.evaluation import load_scenario, summarize, write_answer
from potakt.evaluation.runner import (
    ANTHROPIC_VERSION,
    BATCH_STATE_FILE,
    BATCHES_URL,
    run_anthropic,
)
from potakt.evaluation.transport import HttpResponse, RunStopped, TransportError

HAIKU = "claude-haiku-5-5"
OPUS = "claude-opus-5-5"
CREATED_AT = "2026-11-02T10:00:05.123456Z"

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


@dataclass
class FakeBatches:
    """The Message Batches API, scripted."""

    outcome: Outcome = all_succeed
    polls_in_progress: int = 1
    # Consumed by the create calls, in order, before normal creates: "lost" registers the batch
    # and raises TransportError (the response was lost); "lost-none" raises without creating it;
    # an HttpResponse is returned as is.
    create_script: list[str | HttpResponse] = field(default_factory=list[str | HttpResponse])
    listing: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
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

    def register(self, scenarios: list[str], created_at: str = CREATED_AT) -> str:
        batch_id = f"msgbatch_{len(self.batches) + 1:02d}"
        self.batches[batch_id] = scenarios
        self.created_at[batch_id] = created_at
        return batch_id

    def number(self, batch_id: str) -> int:
        return list(self.batches).index(batch_id) + 1

    def batch(self, batch_id: str) -> dict[str, Any]:
        scenarios = self.batches[batch_id]
        ended = self.polled.get(batch_id, 0) > self.polls_in_progress
        counts = dict.fromkeys(("processing", "succeeded", "errored", "canceled", "expired"), 0)
        if ended:
            for scenario in scenarios:
                counts[self.outcome(self.number(batch_id), scenario)["type"]] += 1
        else:
            counts["processing"] = len(scenarios)
        return {
            "id": batch_id,
            "type": "message_batch",
            "processing_status": "ended" if ended else "in_progress",
            "request_counts": counts,
            "created_at": self.created_at[batch_id],
            "ended_at": None,
            "expires_at": "2026-11-03T10:00:05Z",
            "cancel_initiated_at": None,
            "results_url": f"{self.results_base}/{batch_id}/results" if ended else None,
        }

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
        if method == "GET" and url == f"{BATCHES_URL}?limit=20":
            if self.list_answer is not None:
                return self.list_answer
            data = [self.batch(batch_id) for batch_id in reversed(self.batches)] + self.listing
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
        if isinstance(step, HttpResponse):
            return step
        if step == "lost-none":
            raise TransportError("TimeoutError: no response received")
        batch_id = self.register(scenarios)
        if step == "lost":
            raise TransportError("TimeoutError: no response received")
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
            now=lambda: NOW,
            poll_seconds=60,
            echo=self.echo,
        )

    def state(self) -> dict[str, Any]:
        return json.loads((self.model_dir / BATCH_STATE_FILE).read_bytes())

    def batch_ids(self) -> list[str]:
        """The batch ids in ``batches.json``; none if the file was never written."""
        if not (self.model_dir / BATCH_STATE_FILE).exists():
            return []
        return [batch["id"] for batch in self.state()["batches"]]

    def recorded(self) -> list[str]:
        return sorted(p.parent.name for p in self.model_dir.glob("*/response.json"))


def make_run(tmp_path: Path, server: FakeBatches, names: tuple[str, ...] = ALL_SCENARIOS) -> Run:
    return Run(corpus_copy(tmp_path, names), tmp_path / "runs", server, Sleeps(), Echo())


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
    run.model_dir.mkdir(parents=True)
    state = {
        "batches": [
            {
                "id": batch_id,
                "scenarios": list(ALL_SCENARIOS[:3]),
                "created_at": CREATED_AT,
                "state": "open",
            }
        ]
    }
    (run.model_dir / BATCH_STATE_FILE).write_text(json.dumps(state), encoding="utf-8")

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


def test_a_scenario_missing_from_the_results_stays_pending(tmp_path: Path) -> None:
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    first = json.dumps({"custom_id": ALL_SCENARIOS[0], "result": succeeded()}) + "\n"
    server.raw_results["msgbatch_01"] = first.encode()

    run()

    assert run.server.creates() == [list(ALL_SCENARIOS[:2]), [ALL_SCENARIOS[1]]]
    assert run.recorded() == list(ALL_SCENARIOS[:2])


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
    run.model_dir.mkdir(parents=True)
    batches = [
        {
            "id": f"msgbatch_9{n}",
            "scenarios": ["02-hotel-reception"],
            "created_at": CREATED_AT,
            "state": "done",
        }
        for n in range(4)
    ]
    (run.model_dir / BATCH_STATE_FILE).write_text(
        json.dumps({"batches": batches}), encoding="utf-8"
    )

    with pytest.raises(RunStopped, match="02-hotel-reception was sent in 4 batches"):
        run()

    assert run.server.calls == []


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
        "02-hotel-reception: invalid_request_error: output_config: unknown field"
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

    with pytest.raises(RunStopped, match=r"^batch msgbatch_01: results line 2 "):
        run()

    raw = run.model_dir / "batch-msgbatch_01.results.jsonl"
    assert raw.read_bytes() == server.raw_results["msgbatch_01"]
    assert run.recorded() == ["01-pharmacy"]


def test_a_duplicate_results_line_stops(tmp_path: Path) -> None:
    server = FakeBatches()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    line = json.dumps({"custom_id": "01-pharmacy", "result": succeeded()}).encode()
    server.raw_results["msgbatch_01"] = line + b"\n" + line + b"\n"

    with pytest.raises(RunStopped, match=r"results line 2 .*01-pharmacy.* twice"):
        run()

    assert run.recorded() == ["01-pharmacy"]


def test_a_results_url_on_another_host_is_refused(tmp_path: Path) -> None:
    server = FakeBatches(results_base="https://evil.example.com/v1/messages/batches")
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped, match=r"refused: evil\.example\.com is not an API host"):
        run()

    assert not any("evil" in call.url for call in server.calls)
    assert run.recorded() == []


def test_a_create_refused_with_400_stops_and_records_no_batch(tmp_path: Path) -> None:
    body = json.dumps(
        {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}
    )
    server = FakeBatches(create_script=[HttpResponse(400, {}, body.encode())])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == "creating a batch failed: HTTP 400 (invalid_request_error: bad)"
    assert run.batch_ids() == []
    assert run.sleeps == []


def test_a_create_answered_with_429_is_simply_retried(tmp_path: Path) -> None:
    server = FakeBatches(create_script=[HttpResponse(429, {"retry-after": "20"}, b"")])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert len(server.creates()) == 2
    assert not any(call.url.endswith("?limit=20") for call in server.calls)
    assert run.sleeps == [20, 60]
    assert len(run.state()["batches"]) == 1


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
    assert "adopted batch msgbatch_01 created by the interrupted request" in run.echo
    assert run.state()["batches"] == [
        {
            "id": "msgbatch_01",
            "scenarios": list(ALL_SCENARIOS[:3]),
            "created_at": CREATED_AT,
            "state": "done",
        }
    ]
    assert run.sleeps == [10, 60]
    assert run.recorded() == list(ALL_SCENARIOS[:3])


def test_a_batch_created_exactly_ten_minutes_before_is_a_candidate(tmp_path: Path) -> None:
    server = FakeBatches(create_script=["lost-none"])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    # Created by the lost request at the edge of the window (the fake's ids are only names).
    server.batches["msgbatch_edge"] = list(ALL_SCENARIOS[:1])
    server.created_at["msgbatch_edge"] = iso(-timedelta(minutes=10))

    run()

    assert "adopted batch msgbatch_edge created by the interrupted request" in run.echo
    assert len(server.creates()) == 1


def test_a_lost_create_with_no_candidate_is_retried(tmp_path: Path) -> None:
    server = FakeBatches(
        create_script=["lost-none"],
        listing=[other_batch("msgbatch_old", 1, iso(-timedelta(minutes=30)))],
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert len(server.creates()) == 2
    assert run.sleeps == [10, 15, 60]
    assert [batch["id"] for batch in run.state()["batches"]] == ["msgbatch_01"]


def test_a_batch_already_in_the_state_is_no_candidate(tmp_path: Path) -> None:
    server = FakeBatches(create_script=["lost-none"])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    known = server.register(
        ["01-pharmacy", "02-hotel-reception"], created_at=iso(-timedelta(minutes=2))
    )
    run.model_dir.mkdir(parents=True)
    state = {
        "batches": [
            {
                "id": known,
                "scenarios": list(ALL_SCENARIOS[:2]),
                "created_at": CREATED_AT,
                "state": "done",
            }
        ]
    }
    (run.model_dir / BATCH_STATE_FILE).write_text(json.dumps(state), encoding="utf-8")

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
        "the create response was lost and 2 recent batches could be it: msgbatch_01, "
        "msgbatch_twin; cancel the extra ones in the Claude Console, then run again"
    )
    assert run.batch_ids() == []
    assert len(server.creates()) == 1


def test_four_lost_creates_stop_the_run(tmp_path: Path) -> None:
    server = FakeBatches(create_script=["lost-none"] * 4)
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped, match=r"^creating a batch failed 4 times; last: TimeoutError"):
        run()

    assert len(server.creates()) == 4
    assert run.sleeps == [10, 15, 10, 45, 10, 135, 10]


def test_a_create_answered_200_without_a_batch_object_is_treated_as_lost(tmp_path: Path) -> None:
    server = FakeBatches(create_script=[HttpResponse(200, {}, b"<html>")])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert len(server.creates()) == 2
    assert any(call.url.endswith("?limit=20") for call in server.calls)
    assert run.sleeps == [10, 15, 60]


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


def test_a_fractional_second_beyond_microseconds_is_read(tmp_path: Path) -> None:
    server = FakeBatches(create_script=["lost-none"])
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    server.batches["msgbatch_ns"] = list(ALL_SCENARIOS[:1])
    server.created_at["msgbatch_ns"] = "2026-11-02T12:00:05.123456789+02:00"

    run()

    assert "adopted batch msgbatch_ns created by the interrupted request" in run.echo


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
    entry = {"id": batch_id, "scenarios": list(ALL_SCENARIOS[:2]), "created_at": CREATED_AT}
    state = {"batches": [entry | {"state": "open"}]}
    (run.model_dir / BATCH_STATE_FILE).write_text(json.dumps(state), encoding="utf-8")

    run()

    assert f"{ALL_SCENARIOS[0]}: already recorded" in run.echo
    assert run.recorded() == list(ALL_SCENARIOS[:2])


def test_a_succeeded_result_without_a_message_stops(tmp_path: Path) -> None:
    server = FakeBatches(outcome=lambda n, s: {"type": "succeeded"})
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped, match="01-pharmacy: a succeeded result without a message"):
        run()

    assert run.recorded() == []


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '{"batches": {}}',
        '{"batches": [{"id": "../x", "scenarios": [], "created_at": "", "state": "open"}]}',
        '{"batches": [{"id": "b", "scenarios": [1], "created_at": "", "state": "open"}]}',
        '{"batches": [{"id": "b", "scenarios": [], "created_at": "", "state": "lost"}]}',
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
