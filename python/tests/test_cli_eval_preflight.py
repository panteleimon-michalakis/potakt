"""``potakt eval preflight`` (TASK-005c3e J7): free, read-only checks of the key and the model.

No test reads a real key or opens a connection: the fake key comes from ``monkeypatch.setenv``
and every call goes to ``FakeApi`` through the ``make_transport`` factory. A check fails only on a
definite contradiction; a field the API does not report (missing, or of an unexpected type) is
printed as ``not reported``.
"""

import email.utils
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from runner_helpers import KEY, Sleeps, files_containing
from test_evaluation_runner_anthropic import other_batch
from typer.testing import CliRunner, Result

import potakt.cli as cli
import potakt.evaluation.runner as runner_module
from potakt.cli import app
from potakt.evaluation.runner import (
    BATCHES_URL,
    RUNS_PATH,
    anthropic_headers,
    gemini_headers,
)
from potakt.evaluation.transport import HttpResponse, TransportError

HAIKU = "claude-haiku-5-5"
OPUS = "claude-opus-5-5"
GEMINI = "gemini-3.8-flash"
ANTHROPIC_VARIABLE = "POTAKT_ANTHROPIC_API_KEY"
GEMINI_VARIABLE = "POTAKT_GEMINI_API_KEY"
MODELS = "https://api.anthropic.com/v1/models"
GEMINI_MODELS = "https://generativelanguage.googleapis.com/v1beta/models"
LIST_ONE = f"{BATCHES_URL}?limit=1"
PASSED = "preflight passed: no prompt was sent and nothing was charged"

runner = CliRunner()


def ok(document: object) -> HttpResponse:
    return HttpResponse(200, {"content-type": "application/json"}, json.dumps(document).encode())


def error_answer(status: int, kind: str, message: str) -> HttpResponse:
    body = {"type": "error", "error": {"type": kind, "message": message}}
    return HttpResponse(status, {}, json.dumps(body).encode())


def claude_model(model: str = HAIKU, **changes: object) -> dict[str, Any]:
    """A ``GET /v1/models/{id}`` answer as the documented example, with real numbers."""
    levels = {name: {"supported": True} for name in ("low", "medium", "high", "xhigh", "max")}
    document: dict[str, Any] = {
        "type": "model",
        "id": model,
        "display_name": "Claude Haiku 5.5" if model == HAIKU else "Claude Opus 5.5",
        "created_at": "2026-10-07T00:00:00Z",
        "lifecycle": "active",
        "max_input_tokens": 1_000_000,
        "max_tokens": 128_000,
        "capabilities": {
            "batch": {"supported": True},
            "effort": {"supported": True, **levels},
        },
    }
    return document | changes


def list_page(count: int = 1) -> dict[str, Any]:
    data = [other_batch(f"msgbatch_0{n}", 1, "2026-11-02T10:00:05Z") for n in range(count)]
    return {
        "data": data,
        "has_more": count > 0,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
    }


def gemini_model(**changes: object) -> dict[str, Any]:
    document: dict[str, Any] = {
        "name": f"models/{GEMINI}",
        "baseModelId": GEMINI,
        "version": "3.8",
        "displayName": "Gemini 3.8 Flash",
        "inputTokenLimit": 1_048_576,
        "outputTokenLimit": 65_536,
        "supportedGenerationMethods": ["generateContent", "countTokens"],
    }
    return document | changes


@dataclass
class Call:
    method: str
    url: str
    headers: dict[str, str]
    body: bytes | None
    timeout: float


@dataclass
class FakeApi:
    """Answers each call with the next of ``answers`` (a response, or an exception to raise:
    ``TransportError`` or ``KeyboardInterrupt``); a call beyond them fails the test."""

    answers: list[HttpResponse | BaseException]
    calls: list[Call] = field(default_factory=list[Call])

    def send(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
    ) -> HttpResponse:
        self.calls.append(Call(method, url, dict(headers), body, timeout))
        assert self.answers, f"unexpected call {method} {url}"
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> Sleeps:
    sleeps = Sleeps()
    monkeypatch.setattr(cli.time, "sleep", sleeps)
    monkeypatch.setattr(cli, "make_transport", lambda: FakeApi([]))
    return sleeps


def preflight(
    monkeypatch: pytest.MonkeyPatch, model: str, *answers: HttpResponse | BaseException
) -> tuple[Result, FakeApi]:
    variable = GEMINI_VARIABLE if model == GEMINI else ANTHROPIC_VARIABLE
    monkeypatch.setenv(variable, KEY)
    server = FakeApi(list(answers))
    monkeypatch.setattr(cli, "make_transport", lambda: server)
    return runner.invoke(app, ["eval", "preflight", "--model", model]), server


# --- the checks that pass --------------------------------------------------------------------


def test_a_haiku_preflight_passes_with_exactly_its_two_free_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, server = preflight(monkeypatch, HAIKU, ok(claude_model()), ok(list_page()))

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        "key: accepted by api.anthropic.com",
        "clock: not reported",  # TASK-005c3g G2: this fake sends no Date header
        "model: claude-haiku-5-5 (Claude Haiku 5.5), created 2026-10-07T00:00:00Z, lifecycle "
        "active",
        "batch: supported",
        "effort high: supported",
        "max_tokens: 128000 (the run asks for 64000)",
        "batch list: readable (1 batch on the first page)",
        PASSED,
    ]
    assert result.stderr == ""
    headers = anthropic_headers(KEY)
    assert headers["authorization"] == f"Bearer {KEY}"
    assert server.calls == [
        Call("GET", f"{MODELS}/{HAIKU}", headers, None, 60.0),
        Call("GET", LIST_ONE, headers, None, 60.0),
    ]


def test_an_opus_preflight_compares_with_its_own_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, server = preflight(monkeypatch, OPUS, ok(claude_model(OPUS)), ok(list_page(0)))

    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert "max_tokens: 128000 (the run asks for 64000)" in lines
    assert "batch list: readable (0 batches on the first page)" in lines
    assert server.calls[0].url == f"{MODELS}/{OPUS}"


def test_a_gemini_preflight_passes_with_exactly_its_one_free_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, server = preflight(monkeypatch, GEMINI, ok(gemini_model()))

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        "key: accepted by generativelanguage.googleapis.com",
        "model: models/gemini-3.8-flash (Gemini 3.8 Flash), version 3.8",
        "generateContent: supported",
        "outputTokenLimit: 65536 (the run asks for 64000)",
        PASSED,
    ]
    headers = gemini_headers(KEY)
    assert headers["x-goog-api-key"] == KEY
    assert server.calls == [Call("GET", f"{GEMINI_MODELS}/{GEMINI}", headers, None, 60.0)]
    assert "key=" not in server.calls[0].url


# --- the checks that fail (definite contradictions only) ---------------------------------------

CLAUDE_FAILS = [
    (
        [error_answer(401, "authentication_error", f"invalid key {KEY}")],
        0,
        "model metadata: HTTP 401 (authentication_error: invalid key [redacted])",
    ),
    (
        [error_answer(404, "not_found_error", "model: claude-haiku-5-5")],
        0,
        "model metadata: HTTP 404 (not_found_error: model: claude-haiku-5-5)",
    ),
    ([HttpResponse(200, {}, b"<html>")], 0, "model metadata: HTTP 200 without a JSON object"),
    ([ok([claude_model()])], 0, "model metadata: HTTP 200 without a JSON object"),
    (
        [ok({k: v for k, v in claude_model().items() if k != "id"})],
        1,
        "model: no id in the answer",
    ),
    (
        [ok(claude_model(id=OPUS))],
        1,
        "model: the answer names claude-opus-5-5, not claude-haiku-5-5",
    ),
    (
        [ok(claude_model(id=5))],
        1,
        "model: the answer names an id that is not a string, not claude-haiku-5-5",
    ),
    ([ok(claude_model(id=""))], 1, "model: the answer names an empty id, not claude-haiku-5-5"),
    (
        [ok(claude_model(id=f"other\n{KEY}"))],
        1,
        "model: the answer names other?[redacted], not claude-haiku-5-5",
    ),
    (
        [ok(claude_model(capabilities={"batch": {"supported": False}}))],
        2,
        "batch: not supported by this model",
    ),
    (
        [ok(claude_model(capabilities={"effort": {"high": {"supported": False}}}))],
        3,
        "effort high: not supported by this model (the run sends effort high)",
    ),
    (
        [ok(claude_model(max_tokens=16_000))],
        4,
        "max_tokens: the model allows 16000, the run asks for 64000",
    ),
    (
        [ok(claude_model(max_tokens=63_999))],
        4,
        "max_tokens: the model allows 63999, the run asks for 64000",
    ),
    # The list page is read as _list_back_to reads one (TASK-005c3d H6).
    (
        [ok(claude_model()), error_answer(403, "permission_error", "no")],
        5,
        "batch list: HTTP 403 (permission_error: no)",
    ),
    (
        [ok(claude_model()), HttpResponse(200, {}, b"no")],
        5,
        "batch list: HTTP 200 without a JSON object",
    ),
    (
        [ok(claude_model()), ok({"data": "none", "has_more": False})],
        5,
        "batch list: unexpected list object",
    ),
    (
        [ok(claude_model()), ok({"data": [{"id": "x"}], "has_more": False})],
        5,
        "batch list: unexpected batch object",
    ),
    (
        [ok(claude_model()), ok({"data": [other_batch("msgbatch_01", 1, "\ud83d")]})],
        5,
        "batch list: unexpected batch object",
    ),
    ([ok(claude_model()), ok({"data": []})], 5, "batch list: unexpected list object"),
]


@pytest.mark.parametrize(("answers", "lines", "reason"), CLAUDE_FAILS)
def test_a_claude_check_fails_only_on_a_definite_contradiction(
    monkeypatch: pytest.MonkeyPatch, answers: list[HttpResponse], lines: int, reason: str
) -> None:
    result, server = preflight(monkeypatch, HAIKU, *answers)

    assert result.exit_code == 1
    assert result.stderr == f"preflight failed: {reason}\n"
    # The earlier lines stay on stdout; TASK-005c3g G2: "clock: not reported" follows the key line.
    assert len(result.stdout.splitlines()) == lines + (1 if lines else 0)
    assert PASSED not in result.stdout
    assert len(server.calls) == len(answers)  # nothing after the failed check
    assert KEY not in result.output


GEMINI_FAILS = [
    (
        HttpResponse(
            400,
            {},
            json.dumps(
                {"error": {"status": "INVALID_ARGUMENT", "message": f"API key not valid {KEY}"}}
            ).encode(),
        ),
        0,
        "model metadata: HTTP 400 (INVALID_ARGUMENT: API key not valid [redacted])",
    ),
    (HttpResponse(200, {}, b"[]"), 0, "model metadata: HTTP 200 without a JSON object"),
    (
        ok(gemini_model(name="models/gemini-2.0-flash")),
        1,
        "model: the answer names models/gemini-2.0-flash, not models/gemini-3.8-flash",
    ),
    (
        ok(gemini_model(name=GEMINI)),
        1,
        "model: the answer names gemini-3.8-flash, not models/gemini-3.8-flash",
    ),
    (
        ok(gemini_model(supportedGenerationMethods=["countTokens", "embedContent"])),
        2,
        "generateContent: not in supportedGenerationMethods",
    ),
    (
        ok(gemini_model(supportedGenerationMethods=[])),
        2,
        "generateContent: not in supportedGenerationMethods",
    ),
    (
        ok(gemini_model(outputTokenLimit=8192)),
        3,
        "outputTokenLimit: the model allows 8192, the run asks for 64000",
    ),
    (
        ok(gemini_model(outputTokenLimit=63_999)),
        3,
        "outputTokenLimit: the model allows 63999, the run asks for 64000",
    ),
]


@pytest.mark.parametrize(("answer", "lines", "reason"), GEMINI_FAILS)
def test_a_gemini_check_fails_only_on_a_definite_contradiction(
    monkeypatch: pytest.MonkeyPatch, answer: HttpResponse, lines: int, reason: str
) -> None:
    result, server = preflight(monkeypatch, GEMINI, answer)

    assert result.exit_code == 1
    assert result.stderr == f"preflight failed: {reason}\n"
    assert len(result.stdout.splitlines()) == lines
    assert len(server.calls) == 1
    assert KEY not in result.output


def test_an_output_limit_equal_to_the_runs_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    claude, _ = preflight(monkeypatch, HAIKU, ok(claude_model(max_tokens=64_000)), ok(list_page()))
    gemini, _ = preflight(monkeypatch, GEMINI, ok(gemini_model(outputTokenLimit=64_000)))

    assert claude.exit_code == 0, claude.output
    assert "max_tokens: 64000 (the run asks for 64000)" in claude.stdout.splitlines()
    assert gemini.exit_code == 0, gemini.output
    assert "outputTokenLimit: 64000 (the run asks for 64000)" in gemini.stdout.splitlines()


# --- not reported: a missing field, or one of an unexpected type, never fails -------------------


def test_a_claude_answer_with_only_its_id_reports_nothing_else_and_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = preflight(monkeypatch, HAIKU, ok({"id": HAIKU}), ok(list_page()))

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        "key: accepted by api.anthropic.com",
        "clock: not reported",
        "model: claude-haiku-5-5 (not reported), created not reported, lifecycle not reported",
        "batch: not reported",
        "effort high: not reported",
        "max_tokens: not reported (the run asks for 64000)",
        "batch list: readable (1 batch on the first page)",
        PASSED,
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"max_tokens": 0},  # the documented example's placeholder
        {"max_tokens": -1},
        {"max_tokens": True},
        {"max_tokens": "128000"},
        {"max_tokens": 1.5},
        {"max_tokens": None},
        {"capabilities": None},
        {"capabilities": {"batch": None, "effort": None}},
        {"capabilities": {"batch": {"supported": "yes"}, "effort": {"high": None}}},
        {"capabilities": {"batch": {}, "effort": {"high": {"supported": 0}}}},
        {"display_name": 5, "created_at": None, "lifecycle": ["active"]},
        {"display_name": ""},
    ],
)
def test_an_odd_claude_field_is_not_reported_and_does_not_fail(
    monkeypatch: pytest.MonkeyPatch, changes: dict[str, object]
) -> None:
    result, _ = preflight(monkeypatch, HAIKU, ok(claude_model() | changes), ok(list_page()))

    assert result.exit_code == 0, result.output
    assert "not reported" in result.stdout
    assert result.stdout.endswith(PASSED + "\n")


def test_a_gemini_answer_without_fields_reports_nothing_and_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = preflight(monkeypatch, GEMINI, ok({}))

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        "key: accepted by generativelanguage.googleapis.com",
        "model: not reported (not reported), version not reported",
        "generateContent: not reported",
        "outputTokenLimit: not reported (the run asks for 64000)",
        PASSED,
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"name": 5, "displayName": None, "version": 3.8},
        {"supportedGenerationMethods": "generateContent"},
        {"supportedGenerationMethods": None},
        {"outputTokenLimit": 0},
        {"outputTokenLimit": "65536"},
        {"outputTokenLimit": False},
    ],
)
def test_an_odd_gemini_field_is_not_reported_and_does_not_fail(
    monkeypatch: pytest.MonkeyPatch, changes: dict[str, object]
) -> None:
    result, _ = preflight(monkeypatch, GEMINI, ok(gemini_model(**changes)))

    assert result.exit_code == 0, result.output
    assert "not reported" in result.stdout
    assert result.stdout.endswith(PASSED + "\n")


# --- the key, the files, the server strings --------------------------------------------------


def test_server_strings_are_shown_safely_and_the_key_never(monkeypatch: pytest.MonkeyPatch) -> None:
    odd = claude_model(display_name=f"Claude\x1b[2J {KEY}", lifecycle="active\nsecond line")
    result, _ = preflight(monkeypatch, HAIKU, ok(odd), ok(list_page()))

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[2] == (
        "model: claude-haiku-5-5 (Claude?[2J [redacted]), created 2026-10-07T00:00:00Z, "
        "lifecycle active?second line"
    )
    assert KEY not in result.output


@pytest.mark.parametrize("model", [HAIKU, GEMINI])
def test_the_preflight_writes_nothing_and_never_leaks_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.chdir(tmp_path)
    runs_existed = RUNS_PATH.exists()
    before = sorted(os.listdir(tmp_path))
    answers = [ok(claude_model()), ok(list_page())] if model == HAIKU else [ok(gemini_model())]

    result, _ = preflight(monkeypatch, model, *answers)

    assert result.exit_code == 0, result.output
    assert sorted(os.listdir(tmp_path)) == before == []
    assert RUNS_PATH.exists() == runs_existed
    assert files_containing(tmp_path, KEY) == []
    assert KEY not in result.output


def test_a_request_that_fails_after_its_retries_stops_with_exit_1(
    monkeypatch: pytest.MonkeyPatch, no_sleep: Sleeps
) -> None:
    busy = error_answer(529, "overloaded_error", "busy")
    timeout = TransportError("TimeoutError: no response received")
    result, server = preflight(monkeypatch, HAIKU, ok(claude_model()), busy, busy, busy, timeout)

    assert result.exit_code == 1
    assert result.stderr == (
        "the request failed 4 times; last: TimeoutError: no response received\n"
    )
    assert len(server.calls) == 5
    assert no_sleep == [15, 45, 135]
    assert PASSED not in result.stdout


def test_a_retried_status_then_success_passes(
    monkeypatch: pytest.MonkeyPatch, no_sleep: Sleeps
) -> None:
    result, _ = preflight(monkeypatch, GEMINI, HttpResponse(503, {}, b""), ok(gemini_model()))

    assert result.exit_code == 0, result.output
    assert no_sleep == [15]


@pytest.mark.parametrize(
    ("model", "variable"), [(HAIKU, ANTHROPIC_VARIABLE), (GEMINI, GEMINI_VARIABLE)]
)
def test_a_missing_key_exits_1_with_no_call(
    monkeypatch: pytest.MonkeyPatch, model: str, variable: str
) -> None:
    server = FakeApi([])
    monkeypatch.setattr(cli, "make_transport", lambda: server)

    result = runner.invoke(app, ["eval", "preflight", "--model", model])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == f"{variable} is not set in this shell\n"
    assert server.calls == []


def test_a_gemini_key_to_the_anthropic_host_is_refused_with_no_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # H7: whatever builds the headers, the Gemini key header never reaches api.anthropic.com.
    monkeypatch.setattr(runner_module, "anthropic_headers", runner_module.gemini_headers)

    result, server = preflight(monkeypatch, HAIKU, ok(claude_model()))

    assert result.exit_code == 1
    assert result.stderr == "refused: api.anthropic.com is not the API host of this key\n"
    assert server.calls == []
    assert KEY not in result.output


@pytest.mark.parametrize("model", [HAIKU, GEMINI])
def test_a_value_error_of_the_transport_is_shown_by_its_type_only(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    # TASK-005c3g G10 (n1): only a refusal of check_url is a stop naming the host; any other
    # ValueError's text could hold anything (it used to be printed, redacted).
    class Raising:
        def send(
            self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
        ) -> HttpResponse:
            raise ValueError(f"boom {headers} {KEY}")

    variable = GEMINI_VARIABLE if model == GEMINI else ANTHROPIC_VARIABLE
    monkeypatch.setenv(variable, KEY)
    monkeypatch.setattr(cli, "make_transport", Raising)

    result = runner.invoke(app, ["eval", "preflight", "--model", model])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == (
        "eval preflight stopped by an unexpected error (ValueError): send this message to the "
        "reviewer\n"
    )
    assert "boom" not in result.output


@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "gpt-5", ""])
def test_an_unregistered_model_is_a_usage_error(model: str) -> None:
    result = runner.invoke(app, ["eval", "preflight", "--model", model])

    assert result.exit_code == 2
    assert result.stdout == ""


def test_the_model_is_required() -> None:
    result = runner.invoke(app, ["eval", "preflight"])

    assert result.exit_code == 2


def test_an_unexpected_error_exits_1_with_its_type_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ANTHROPIC_VARIABLE, KEY)

    class Broken:
        def send(
            self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
        ) -> HttpResponse:
            raise RuntimeError(f"text with {KEY}")

    monkeypatch.setattr(cli, "make_transport", Broken)

    result = runner.invoke(app, ["eval", "preflight", "--model", HAIKU])

    assert result.exit_code == 1
    assert result.stderr == (
        "eval preflight stopped by an unexpected error (RuntimeError): send this message to the "
        "reviewer\n"
    )
    assert KEY not in result.output
    assert "Traceback" not in result.output


def test_an_interruption_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    result, _ = preflight(monkeypatch, HAIKU, ok(claude_model()), KeyboardInterrupt())

    assert result.exit_code == 1
    assert result.stdout.splitlines()[0] == "key: accepted by api.anthropic.com"
    assert result.stderr == (
        "interrupted: the preflight sent no prompt and recorded nothing; run it again\n"
    )


# --- the help --------------------------------------------------------------------------------


def help_text(*command: str) -> str:
    result = runner.invoke(app, [*command, "--help"], terminal_width=200)
    assert result.exit_code == 0
    return " ".join(result.stdout.replace("\N{BOX DRAWINGS LIGHT VERTICAL}", " ").split())


def test_the_preflight_help_states_the_calls_and_the_exit_codes() -> None:
    text = help_text("eval", "preflight")

    assert "free, read-only calls" in text
    assert "no prompt is sent, nothing is written and nothing is charged" in text
    assert "GET /v1/models/<id> and one page of the batch list" in text
    assert "GET /v1beta/models/<id>" in text
    assert "POTAKT_ANTHROPIC_API_KEY" in text
    assert "POTAKT_GEMINI_API_KEY" in text
    assert "a field the API does not report is shown as not reported" in text
    # TASK-005c3g G2
    assert "a laptop clock more than 2 minutes from the API's (the Date header)" in text
    assert (
        "Exit code 0: every check passed; 1: a failed check (preflight failed: <check>: <reason> "
        "on stderr), a missing or invalid key, a request that failed after its retries, an "
        "unexpected error (its type only, never its text) or an interruption; 2: usage error "
        "(also an unregistered model id)"
    ) in text


def test_the_eval_group_names_the_commands_that_send_requests() -> None:
    text = help_text("eval")

    assert "eval run calls the model APIs" in text
    assert "eval preflight makes free, read-only calls with the same keys (no prompt)" in text
    assert "the other commands are offline: no model calls, no credentials" in text


# --- TASK-005c3g G2: the laptop clock against the API's ---------------------------------------

API_TIME = datetime(2026, 11, 7, 15, 20, 0, tzinfo=UTC)


def dated(response: HttpResponse, date: str | None) -> HttpResponse:
    """``response`` with a ``date`` header (none if ``None``)."""
    headers = dict(response.headers)
    if date is not None:
        headers["date"] = date
    return HttpResponse(response.status, headers, response.body)


API_DATE = email.utils.format_datetime(API_TIME, usegmt=True)  # Sat, 07 Nov 2026 15:20:00 GMT


@pytest.mark.parametrize(
    ("seconds", "shown"), [(0, 0), (3, 3), (-3, 3), (119.4, 120), (120, 120), (-120, 120)]
)
def test_the_preflight_prints_how_close_the_laptop_clock_is(
    monkeypatch: pytest.MonkeyPatch, seconds: float, shown: int
) -> None:
    monkeypatch.setattr(cli, "utc_now", lambda: API_TIME + timedelta(seconds=seconds))

    result, server = preflight(
        monkeypatch, HAIKU, dated(ok(claude_model()), API_DATE), ok(list_page())
    )

    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert lines[:2] == [
        "key: accepted by api.anthropic.com",
        f"clock: within {shown} s of api.anthropic.com",
    ]
    assert len(server.calls) == 2


@pytest.mark.parametrize(("seconds", "shown"), [(121, 121), (-121, 121), (900, 900)])
def test_a_wrong_laptop_clock_fails_the_preflight_after_one_call(
    monkeypatch: pytest.MonkeyPatch, seconds: int, shown: int
) -> None:
    monkeypatch.setattr(cli, "utc_now", lambda: API_TIME + timedelta(seconds=seconds))

    result, server = preflight(
        monkeypatch, HAIKU, dated(ok(claude_model()), API_DATE), ok(list_page())
    )

    assert result.exit_code == 1
    assert result.stdout == "key: accepted by api.anthropic.com\n"
    assert result.stderr == (
        f"preflight failed: clock: the laptop clock differs from the API's by {shown} s (more "
        "than 120 s): sync the Windows clock (Settings, Time & language, Date & time, Sync now) "
        "and run again; nothing was sent\n"
    )
    assert len(server.calls) == 1


@pytest.mark.parametrize(
    "date",
    [None, "", "yesterday", "Sat, 07 Nov 2026 25:00:00 GMT", "Sat, 07 Nov 2026 15:20:00 -0000"],
)
def test_a_missing_or_unreadable_date_is_not_reported_by_the_preflight(
    monkeypatch: pytest.MonkeyPatch, date: str | None
) -> None:
    monkeypatch.setattr(cli, "utc_now", lambda: API_TIME + timedelta(hours=5))

    result, _ = preflight(monkeypatch, HAIKU, dated(ok(claude_model()), date), ok(list_page()))

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[1] == "clock: not reported"


def test_the_gemini_preflight_has_no_clock_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "utc_now", lambda: API_TIME + timedelta(hours=5))

    result, _ = preflight(monkeypatch, GEMINI, dated(ok(gemini_model()), API_DATE))

    assert result.exit_code == 0, result.output
    assert not any(line.startswith("clock:") for line in result.stdout.splitlines())
