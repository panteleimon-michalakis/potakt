"""The Gemini Test A runner (TASK-005c3 F4): one ``generateContent`` call per pending scenario,
through a fake transport. No test opens a connection."""

import functools
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from runner_helpers import (
    ALL_SCENARIOS,
    EMPTY,
    KEY,
    REPAIR_CORPUS,
    REPAIRED,
    Echo,
    RefusedReplace,
    Sleeps,
    corpus_copy,
    expected_repair,
    files_containing,
    gemini_response,
    truth_text,
    write_first_run,
)
from typer.testing import CliRunner

import potakt.evaluation.records as records_module
import potakt.evaluation.runner as runner_module
from potakt.cli import app
from potakt.evaluation import CORPUS_PATH, load_scenario, render_payload, write_answer
from potakt.evaluation.runner import GEMINI_URL, USER_AGENT, run_gemini
from potakt.evaluation.transport import HttpResponse, RunStopped, TransportError

GEMINI = "gemini-3.8-flash"
ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
)


def ok(document: object) -> HttpResponse:
    return HttpResponse(200, {"content-type": "application/json"}, json.dumps(document).encode())


def error(status: int, state: str, text: str, delay: str | None = None) -> HttpResponse:
    details = []
    if delay is not None:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay})
    body = {"error": {"code": status, "message": text, "status": state, "details": details}}
    return HttpResponse(status, {}, json.dumps(body).encode())


@dataclass
class Call:
    url: str
    headers: dict[str, str]
    body: bytes
    timeout: float


@dataclass
class FakeGemini:
    """Answers each call from ``script`` (per scenario, in order), else with a junk answer."""

    script: dict[str, list[HttpResponse | TransportError]] = field(
        default_factory=dict[str, list[HttpResponse | TransportError]]
    )
    calls: list[Call] = field(default_factory=list[Call])
    # ADR-0008 D5: ``POST <scenario>`` for each call and ``now HH:MM:SS`` for each clock read.
    events: list[str] = field(default_factory=list[str])

    def send(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
    ) -> HttpResponse:
        assert method == "POST"
        assert body is not None
        self.calls.append(Call(url, headers, body, timeout))
        name = self.scenario_of(body)
        self.events.append(f"POST {name}")
        steps = self.script.get(name, [])
        if steps:
            step = steps.pop(0)
            if isinstance(step, TransportError):
                raise step
            return step
        return ok(gemini_response())

    @staticmethod
    def scenario_of(body: bytes) -> str:
        """The scenario of a request: the one whose ``request.md`` its user message contains."""
        text = json.loads(body)["contents"][0]["parts"][0]["text"]
        return next(name for name, request in requests().items() if request in text)

    def sent(self) -> list[str]:
        return [self.scenario_of(call.body) for call in self.calls]


@functools.cache
def requests() -> dict[str, str]:
    """Each scenario's ``request.md``, read when a call is first matched (never at import)."""
    return {
        name: (CORPUS_PATH / name / "request.md").read_text(encoding="utf-8").strip()
        for name in ALL_SCENARIOS
    }


@dataclass
class Run:
    corpus: Path
    runs: Path
    server: FakeGemini
    sleeps: Sleeps
    echo: Echo

    @property
    def model_dir(self) -> Path:
        return self.runs / GEMINI

    def now(self) -> datetime:
        """The runner's clock (``Sleeps``), each read logged in the server's events."""
        moment = self.sleeps.now()
        self.server.events.append(f"now {moment:%H:%M:%S}")
        return moment

    def __call__(self) -> None:
        run_gemini(
            GEMINI,
            self.corpus,
            self.runs,
            KEY,
            self.server,
            sleep=self.sleeps,
            now=self.now,
            echo=self.echo,
        )

    def recorded(self) -> list[str]:
        return sorted(p.parent.name for p in self.model_dir.glob("*/response.json"))


def make_run(tmp_path: Path, server: FakeGemini, names: tuple[str, ...] = ALL_SCENARIOS) -> Run:
    return Run(corpus_copy(tmp_path, names), tmp_path / "runs", server, Sleeps(), Echo())


def test_the_endpoint_is_generate_content_of_the_model() -> None:
    assert GEMINI_URL.format(model=GEMINI) == ENDPOINT


def test_the_happy_path_records_all_ten_scenarios_one_at_a_time(tmp_path: Path) -> None:
    server = FakeGemini(script={"01-pharmacy": [ok(gemini_response(truth_text("01-pharmacy")))]})
    run = make_run(tmp_path, server)

    run()

    assert run.recorded() == list(ALL_SCENARIOS)
    assert server.sent() == list(ALL_SCENARIOS)
    assert run.sleeps == []
    assert run.echo[0] == "01-pharmacy: recorded (STOP, valid, recall 1.000)"
    assert run.echo[1] == "02-hotel-reception: recorded (STOP, invalid, recall 0.000)"
    assert len(run.echo) == 10
    for call, name in zip(server.calls, ALL_SCENARIOS, strict=True):
        assert call.url == ENDPOINT
        assert call.headers == {
            "x-goog-api-key": KEY,
            "content-type": "application/json",
            "user-agent": USER_AGENT,
        }
        assert call.timeout == 600.0
        assert call.body == (run.model_dir / name / "request.json").read_bytes()
    recorded = json.loads((run.model_dir / "02-hotel-reception" / "response.json").read_bytes())
    assert recorded == gemini_response()


def test_the_key_is_only_in_the_header(tmp_path: Path) -> None:
    server = FakeGemini(script={"02-hotel-reception": [error(429, "RESOURCE_EXHAUSTED", "q")]})
    run = make_run(tmp_path, server, ALL_SCENARIOS[:3])

    run()

    assert files_containing(tmp_path, KEY) == []
    assert all(KEY not in line for line in run.echo)
    for call in server.calls:
        assert KEY not in call.url
        assert "?" not in call.url
        assert KEY.encode() not in call.body
        assert [name for name, value in call.headers.items() if value == KEY] == ["x-goog-api-key"]


def test_a_429_with_a_retry_delay_then_200(tmp_path: Path) -> None:
    server = FakeGemini(
        script={"01-pharmacy": [error(429, "RESOURCE_EXHAUSTED", "Quota exceeded", "30s")]}
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert run.sleeps == [30]
    assert server.sent() == ["01-pharmacy", "01-pharmacy"]
    assert run.recorded() == ["01-pharmacy"]


def test_four_503_stop_the_run_and_keep_earlier_records(tmp_path: Path) -> None:
    unavailable = error(503, "UNAVAILABLE", "The model is overloaded.")
    server = FakeGemini(script={"02-hotel-reception": [unavailable] * 4})
    run = make_run(tmp_path, server, ALL_SCENARIOS[:3])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == (
        "02-hotel-reception: the request failed 4 times; "
        "last: HTTP 503 (UNAVAILABLE: The model is overloaded.)"
    )
    assert run.recorded() == ["01-pharmacy"]
    assert run.sleeps == [15, 45, 135]
    assert server.sent() == ["01-pharmacy"] + ["02-hotel-reception"] * 4


def test_another_status_stops_at_once(tmp_path: Path) -> None:
    server = FakeGemini(script={"01-pharmacy": [error(400, "INVALID_ARGUMENT", "bad field")]})
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == "01-pharmacy: HTTP 400 (INVALID_ARGUMENT: bad field)"
    assert run.recorded() == []
    assert server.sent() == ["01-pharmacy"]


def test_a_blocked_prompt_is_recorded_as_an_invalid_answer(tmp_path: Path) -> None:
    blocked = {"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {}}
    server = FakeGemini(script={"01-pharmacy": [ok(blocked)]})
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert run.echo == ["01-pharmacy: recorded (blocked:SAFETY, invalid, recall 0.000)"]
    assert json.loads((run.model_dir / "01-pharmacy" / "response.json").read_bytes()) == blocked


@pytest.mark.parametrize("body", [b"<html>busy</html>", b"[1, 2]", b""])
def test_a_200_without_a_json_object_is_retried(tmp_path: Path, body: bytes) -> None:
    server = FakeGemini(script={"01-pharmacy": [HttpResponse(200, {}, body)]})
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert run.sleeps == [15]
    assert run.recorded() == ["01-pharmacy"]


def test_four_200_without_a_json_object_stop_the_run(tmp_path: Path) -> None:
    server = FakeGemini(script={"01-pharmacy": [HttpResponse(200, {}, b"<html>")] * 4})
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == (
        "01-pharmacy: the request failed 4 times; last: HTTP 200 without a JSON object"
    )
    assert run.recorded() == []


def test_a_transport_error_is_retried(tmp_path: Path) -> None:
    server = FakeGemini(script={"01-pharmacy": [TransportError("TimeoutError: no response")]})
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    run()

    assert run.sleeps == [15]
    assert run.recorded() == ["01-pharmacy"]


def test_a_rerun_sends_only_the_pending_scenarios(tmp_path: Path) -> None:
    server = FakeGemini()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:4])
    for name in ("01-pharmacy", "03-call-centre"):
        scenario = load_scenario(run.corpus / name)
        write_answer(run.model_dir / name, scenario, GEMINI, gemini_response())

    run()

    assert server.sent() == ["02-hotel-reception", "04-restaurant"]
    assert run.recorded() == list(ALL_SCENARIOS[:4])


def test_a_half_written_record_is_completed_and_not_sent_again(tmp_path: Path) -> None:
    server = FakeGemini()
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    scenario = load_scenario(run.corpus / "01-pharmacy")
    write_answer(run.model_dir / "01-pharmacy", scenario, GEMINI, gemini_response())
    (run.model_dir / "01-pharmacy" / "comparison.json").unlink()

    run()

    assert run.echo[0] == "01-pharmacy: record completed (comparison.json)"
    assert server.sent() == ["02-hotel-reception"]


def test_nothing_pending_prints_nothing_to_run(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeGemini(), ALL_SCENARIOS[:1])
    run()
    run.server.calls.clear()
    run.echo.clear()

    run()

    assert run.server.calls == []
    assert run.echo == ["nothing to run"]


def test_only_gemini_runs_on_generate_content(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"claude-haiku-5-5 is not a Gemini model"):
        run_gemini(
            "claude-haiku-5-5",
            tmp_path,
            tmp_path,
            KEY,
            FakeGemini(),
            sleep=Sleeps(),
            now=Sleeps().now,
            echo=Echo(),
        )


def test_the_stop_message_is_redacted_of_the_key(tmp_path: Path) -> None:
    server = FakeGemini(script={"01-pharmacy": [error(400, "INVALID_ARGUMENT", f"key {KEY}")]})
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == "01-pharmacy: HTTP 400 (INVALID_ARGUMENT: key [redacted])"


# --- TASK-005c3d H8, H9 ----------------------------------------------------------------------


def test_a_lone_surrogate_answer_is_recorded_as_malformed(tmp_path: Path) -> None:
    server = FakeGemini(script={"01-pharmacy": [ok(gemini_response("\ud83d"))]})
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])

    run()

    assert run.echo[0] == "01-pharmacy: recorded (malformed, invalid, recall 0.000)"
    assert run.recorded() == list(ALL_SCENARIOS[:2])


def test_an_answer_that_cannot_be_recorded_stops_with_its_type_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_run(tmp_path, FakeGemini(), ALL_SCENARIOS[:3])
    real = runner_module.write_answer

    def write(run_dir: Path, scenario: Any, model: str, response: object) -> Any:
        if run_dir.name == "02-hotel-reception":
            raise OSError(28, f"No space left on device {KEY}")
        return real(run_dir, scenario, model, response)

    monkeypatch.setattr(runner_module, "write_answer", write)

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == (
        "02-hotel-reception: could not be recorded (OSError); send this message to the reviewer"
    )
    assert run.recorded() == ["01-pharmacy"]
    assert run.server.sent() == ["01-pharmacy", "02-hotel-reception"]


def test_a_saved_answer_whose_record_fails_says_so_and_the_next_run_completes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # TASK-005c3h H8 (the review's #3): response.json was saved, then candidate.txt failed.
    run = make_run(tmp_path, FakeGemini(), ALL_SCENARIOS[:2])
    real = records_module.write_atomic

    def write(path: Path, data: bytes, **kwargs: Any) -> None:
        if path.parent.name == "01-pharmacy" and path.name == "candidate.txt":
            raise OSError(28, f"No space left on device {KEY}")
        real(path, data, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(records_module, "write_atomic", write)
        with pytest.raises(RunStopped) as raised:
            run()

    assert str(raised.value) == (
        "01-pharmacy: recorded, but its record could not be completed (OSError); the next run "
        "completes it; send this message to the reviewer"
    )
    run.echo.clear()

    run()

    assert run.echo[0] == "01-pharmacy: record completed (candidate.txt, comparison.json)"
    assert run.server.sent() == ["01-pharmacy", "02-hotel-reception"]
    assert run.recorded() == list(ALL_SCENARIOS[:2])


def test_an_interrupt_while_recording_is_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_run(tmp_path, FakeGemini(), ALL_SCENARIOS[:1])

    def write(*args: object) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(runner_module, "write_answer", write)

    with pytest.raises(KeyboardInterrupt):
        run()


# --- TASK-005c3g G3: a refused replace ------------------------------------------------------


@pytest.mark.parametrize("times", [1, 9])
def test_a_refused_replace_of_an_answer_is_absorbed_and_each_scenario_sent_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, times: int
) -> None:
    # The audit's p03: the first answer stayed in response.json.tmp and the next run sent it again.
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    refused = RefusedReplace(times, "response.json")
    monkeypatch.setattr(os, "replace", refused)
    run = make_run(tmp_path, FakeGemini(), ALL_SCENARIOS[:2])

    run()

    assert refused.refused == times
    assert len(slept) == times
    assert run.server.sent() == list(ALL_SCENARIOS[:2])
    assert run.recorded() == list(ALL_SCENARIOS[:2])
    assert sorted(run.model_dir.rglob("*.tmp")) == []


def test_ten_refused_replaces_keep_the_answer_and_the_next_run_stops_on_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The audit's p03: the next run sent 01-pharmacy again and the first answer was lost.
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    refused = RefusedReplace(10, "response.json")
    monkeypatch.setattr(os, "replace", refused)
    run = make_run(tmp_path, FakeGemini(), ALL_SCENARIOS[:2])
    with pytest.raises(RunStopped, match=r"^01-pharmacy: could not be recorded"):
        run()
    monkeypatch.setattr(os, "replace", refused.real)
    kept = run.model_dir / "01-pharmacy" / "response.json.tmp"
    assert json.loads(kept.read_bytes()) == gemini_response()
    run.server.calls.clear()

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == (
        f"01-pharmacy: an answer that could not be saved is kept in {kept}; nothing was sent; "
        "send this message to the reviewer and do not delete or rename the file"
    )
    assert run.server.calls == []
    assert json.loads(kept.read_bytes()) == gemini_response()


def test_a_leftover_unsaved_answer_of_a_later_scenario_stops_before_the_first_request(
    tmp_path: Path,
) -> None:
    run = make_run(tmp_path, FakeGemini(), ALL_SCENARIOS[:3])
    kept = run.model_dir / "03-call-centre" / "response.json.tmp"
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"{}\n")

    with pytest.raises(RunStopped, match=r"^03-call-centre: an answer that could not be saved"):
        run()

    assert run.server.calls == []
    assert not (run.model_dir / "01-pharmacy").exists()


def test_a_leftover_temporary_file_of_a_recorded_scenario_is_ignored(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeGemini(), ALL_SCENARIOS[:2])
    scenario = load_scenario(run.corpus / "01-pharmacy")
    write_answer(run.model_dir / "01-pharmacy", scenario, GEMINI, gemini_response())
    (run.model_dir / "01-pharmacy" / "response.json.tmp").write_bytes(b"{}\n")

    run()

    assert run.server.sent() == ["02-hotel-reception"]
    assert run.recorded() == list(ALL_SCENARIOS[:2])


# --- TASK-005c3h H7: every request checked against its pin before the first one is sent ------


def changed_csv(corpus: Path, name: str) -> None:
    csv = sorted((corpus / name).glob("*.csv"))[0]
    data = bytearray(csv.read_bytes())
    data[-2] ^= 1
    csv.write_bytes(bytes(data))


@pytest.mark.parametrize("name", ["01-pharmacy", "03-call-centre"])
def test_a_changed_request_stops_before_the_first_request(tmp_path: Path, name: str) -> None:
    # TASK-005c3s S3-4: Gemini renders one scenario at a time, so a check at the third scenario
    # alone would stop after two requests were sent.
    run = make_run(tmp_path, FakeGemini(), ALL_SCENARIOS[:3])
    changed_csv(run.corpus, name)

    with pytest.raises(RunStopped) as raised:
        run()

    pinned = runner_module.PAYLOAD_SHA256[(GEMINI, name)]
    assert str(raised.value) == (
        f"{name}: the request for gemini-3.8-flash differs from the one the harness pins (SHA256 "
        f"{pinned}); nothing was sent"
    )
    assert run.server.calls == []
    assert sorted(run.runs.rglob("request.json")) == []


def test_a_request_without_a_pin_stops_before_the_first_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pins = dict(runner_module.PAYLOAD_SHA256)
    del pins[(GEMINI, "02-hotel-reception")]
    monkeypatch.setattr(runner_module, "PAYLOAD_SHA256", pins)
    run = make_run(tmp_path, FakeGemini(), ALL_SCENARIOS[:2])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == (
        "02-hotel-reception: no request is pinned for gemini-3.8-flash; nothing was sent"
    )
    assert run.server.calls == []
    assert sorted(run.runs.rglob("request.json")) == []


def test_each_request_is_checked_again_when_it_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The pre-check passed; the second request changed before it was built: no file, no call
    # for it (the first one was sent and recorded).
    run = make_run(tmp_path, FakeGemini(), ALL_SCENARIOS[:2])
    real_send = run.server.send

    def send(*args: Any) -> HttpResponse:
        response = real_send(*args)
        changed_csv(run.corpus, "02-hotel-reception")
        return response

    monkeypatch.setattr(run.server, "send", send)

    with pytest.raises(RunStopped, match=r"^02-hotel-reception: the request for gemini-3\.8"):
        run()

    assert run.server.sent() == ["01-pharmacy"]
    assert run.recorded() == ["01-pharmacy"]
    assert not (run.model_dir / "02-hotel-reception" / "request.json").exists()


# --- ADR-0008 D5: received.json, the instant each answer arrived --------------------------------


def received(run: Run, name: str) -> bytes:
    return (run.model_dir / name / "received.json").read_bytes()


def received_at(instant: str) -> bytes:
    return b'{\n  "received_at": "' + instant.encode() + b'"\n}\n'


def test_each_answer_records_when_it_was_received(tmp_path: Path) -> None:
    server = FakeGemini(
        script={"02-hotel-reception": [error(429, "RESOURCE_EXHAUSTED", "Quota exceeded", "30s")]}
    )
    run = make_run(tmp_path, server, ALL_SCENARIOS[:3])

    run()

    assert received(run, "01-pharmacy") == received_at("2026-11-02T10:00:00Z")
    assert received(run, "02-hotel-reception") == received_at("2026-11-02T10:00:30Z")
    assert received(run, "03-call-centre") == received_at("2026-11-02T10:00:30Z")
    assert server.events == [
        "POST 01-pharmacy",
        "now 10:00:00",
        "POST 02-hotel-reception",
        "POST 02-hotel-reception",
        "now 10:00:30",
        "POST 03-call-centre",
        "now 10:00:30",
    ]
    for name in ALL_SCENARIOS[:3]:
        assert sorted(path.name for path in (run.model_dir / name).iterdir()) == [
            "candidate.txt",
            "comparison.json",
            "received.json",
            "request.json",
            "response.json",
        ]


def test_a_failed_call_writes_no_received_json(tmp_path: Path) -> None:
    server = FakeGemini(script={"01-pharmacy": [error(400, "INVALID_ARGUMENT", "bad field")]})
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])

    with pytest.raises(RunStopped, match=r"^01-pharmacy: HTTP 400"):
        run()

    assert not (run.model_dir / "01-pharmacy" / "received.json").exists()
    assert server.events == ["POST 01-pharmacy"]


def test_received_json_comes_before_the_answer_and_the_next_attempt_replaces_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_run(tmp_path, FakeGemini(), ALL_SCENARIOS[:1])
    seen: list[bytes] = []

    def write(run_dir: Path, scenario: Any, model: str, response: object) -> Any:
        seen.append((run_dir / "received.json").read_bytes())
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as patch:
        patch.setattr(runner_module, "write_answer", write)
        with pytest.raises(RunStopped) as raised:
            run()

    assert seen == [received_at("2026-11-02T10:00:00Z")]
    assert str(raised.value) == (
        "01-pharmacy: could not be recorded (OSError); send this message to the reviewer"
    )
    assert not (run.model_dir / "01-pharmacy" / "response.json").exists()
    run.sleeps.advance(3600)

    run()

    assert run.recorded() == ["01-pharmacy"]
    assert received(run, "01-pharmacy") == received_at("2026-11-02T11:00:00Z")


def test_a_refused_received_json_stops_before_the_answer_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setattr(os, "replace", RefusedReplace(10, "received.json"))
    run = make_run(tmp_path, FakeGemini(), ALL_SCENARIOS[:2])

    with pytest.raises(RunStopped) as raised:
        run()

    assert str(raised.value) == (
        "01-pharmacy: could not be recorded (PermissionError); send this message to the reviewer"
    )
    assert run.recorded() == []
    assert run.server.sent() == ["01-pharmacy"]


def test_rescore_leaves_received_json_byte_for_byte(tmp_path: Path) -> None:
    server = FakeGemini(script={"01-pharmacy": [ok(gemini_response(truth_text("01-pharmacy")))]})
    run = make_run(tmp_path, server, ALL_SCENARIOS[:2])
    run()
    before = {name: received(run, name) for name in ALL_SCENARIOS[:2]}

    rescored = CliRunner().invoke(
        app, ["eval", "rescore", str(run.model_dir), "--corpus", str(run.corpus)]
    )

    assert rescored.exit_code == 0, rescored.stderr
    assert {name: received(run, name) for name in ALL_SCENARIOS[:2]} == before


def test_a_gemini_repair_run_sends_the_repair_requests_and_records_them(tmp_path: Path) -> None:
    # TASK-005e5 decision 6 (ADR-0008 D7): the first request plus the answer as the model's turn.
    server = FakeGemini(script={REPAIRED[2]: [ok(gemini_response(truth_text(REPAIRED[2])))]})
    run = make_run(tmp_path, server, REPAIR_CORPUS)
    first_dir = write_first_run(run.runs, run.corpus, GEMINI)
    repair_dir = run.runs / "repair" / GEMINI

    run_gemini(
        GEMINI,
        run.corpus,
        run.runs,
        KEY,
        server,
        sleep=run.sleeps,
        now=run.now,
        echo=run.echo,
        kind="repair",
    )

    assert server.sent() == list(REPAIRED)
    for call, name in zip(server.calls, REPAIRED, strict=True):
        expected = render_payload(expected_repair(run.corpus, first_dir, GEMINI, name))
        assert call.body == expected.encode("utf-8")
        assert (repair_dir / name / "request.json").read_bytes() == call.body
        roles = [turn["role"] for turn in json.loads(call.body)["contents"]]
        assert roles == ["user", "model", "user"]
        assert sorted(path.name for path in (repair_dir / name).iterdir()) == [
            "candidate.txt",
            "comparison.json",
            "received.json",
            "request.json",
            "response.json",
        ]
    assert run.echo[-1] == "05-supermarket-checkouts: recorded (STOP, valid, recall 1.000)"
    assert not (repair_dir / EMPTY).exists()
    assert sorted(path.name for path in first_dir.iterdir()) == list(REPAIR_CORPUS)


def test_a_gemini_scenario_without_an_answer_is_sent_again_by_every_run(tmp_path: Path) -> None:
    # TASK-005e5 decision 4: ADR-0008 D6 applies to the Claude batches only; Gemini has none.
    unavailable = error(503, "UNAVAILABLE", "The model is overloaded.")
    server = FakeGemini(script={"01-pharmacy": [unavailable] * 24})
    run = make_run(tmp_path, server, ALL_SCENARIOS[:1])
    for _ in range(6):  # six runs of four failed attempts each: more than D6's five batches
        with pytest.raises(RunStopped, match="the request failed 4 times"):
            run()

    run()

    assert server.sent() == ["01-pharmacy"] * 25
    assert run.recorded() == ["01-pharmacy"]
    assert not (run.model_dir / "01-pharmacy" / "unanswered.json").exists()
