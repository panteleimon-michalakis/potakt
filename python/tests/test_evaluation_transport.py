"""The HTTPS transport of Test A (TASK-005c3 F1): the only module that may reach the network.

No test opens a connection: ``UrllibTransport`` gets a fake opener, or the real opener with a fake
HTTPS handler placed in front of the real one; ``send_with_retries`` gets a scripted transport and
a recording ``sleep``.
"""

import http.client
import io
import json
import ssl
import urllib.error
import urllib.request
import urllib.response
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from potakt.evaluation.transport import (
    ALLOWED_HOSTS,
    HttpResponse,
    RunStopped,
    TransportError,
    UrllibTransport,
    base_delay,
    build_opener,
    describe_response,
    send_with_retries,
)

URL = "https://api.anthropic.com/v1/messages/batches"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
)
KEY = "potakt-test-key"
HEADERS = {"x-api-key": KEY, "content-type": "application/json"}


def message(**headers: str) -> http.client.HTTPMessage:
    found = http.client.HTTPMessage()
    for name, value in headers.items():
        found[name.replace("_", "-")] = value
    return found


# --- UrllibTransport with a fake opener ------------------------------------------------------


class FakeResponse:
    def __init__(self, status: int, headers: http.client.HTTPMessage, body: bytes) -> None:
        self.status = status
        self.headers = headers
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class FakeOpener:
    """Records each request and answers with ``outcome`` (a response or an exception)."""

    def __init__(self, outcome: FakeResponse | BaseException) -> None:
        self.outcome = outcome
        self.calls: list[tuple[urllib.request.Request, float]] = []

    def open(
        self, fullurl: urllib.request.Request, data: None = None, timeout: float = 0.0
    ) -> FakeResponse:
        self.calls.append((fullurl, timeout))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def test_the_request_carries_method_url_headers_body_and_timeout() -> None:
    opener = FakeOpener(FakeResponse(200, message(Content_Type="application/json"), b"{}"))

    response = UrllibTransport(opener).send("POST", URL, HEADERS, b'{"a": 1}', 60.0)

    assert response == HttpResponse(200, {"content-type": "application/json"}, b"{}")
    [(request, timeout)] = opener.calls
    assert request.get_method() == "POST"
    assert request.full_url == URL
    assert {name.lower(): value for name, value in request.header_items()} == HEADERS
    assert request.data == b'{"a": 1}'
    assert timeout == 60.0


def test_a_get_without_a_body() -> None:
    opener = FakeOpener(FakeResponse(200, message(), b"[]"))

    UrllibTransport(opener).send("GET", URL + "?limit=20", HEADERS, None, 30.0)

    [(request, _)] = opener.calls
    assert request.get_method() == "GET"
    assert request.data is None


def test_response_header_names_are_lower_cased() -> None:
    opener = FakeOpener(FakeResponse(200, message(Retry_After="7", X_Request_Id="r1"), b""))

    response = UrllibTransport(opener).send("GET", URL, HEADERS, None, 1.0)

    assert response.headers == {"retry-after": "7", "x-request-id": "r1"}


def test_an_http_error_is_a_response_with_its_status_headers_and_body() -> None:
    body = b'{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}'
    error = urllib.error.HTTPError(
        URL, 529, "Overloaded", message(Retry_After="3"), io.BytesIO(body)
    )

    response = UrllibTransport(FakeOpener(error)).send("POST", URL, HEADERS, b"{}", 60.0)

    assert response == HttpResponse(529, {"retry-after": "3"}, body)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (urllib.error.URLError(ConnectionRefusedError(111, "Connection refused")), "URLError"),
        (urllib.error.URLError("unknown host for https://x?key=secret"), "URLError"),
        (TimeoutError("timed out"), "TimeoutError"),
        (ConnectionResetError(104, "Connection reset by peer"), "ConnectionResetError"),
        (http.client.RemoteDisconnected("Remote end closed connection"), "RemoteDisconnected"),
        (http.client.IncompleteRead(b"partial"), "IncompleteRead"),
        (ssl.SSLError(1, "unexpected eof"), "SSLError"),
    ],
)
def test_a_connection_failure_is_a_transport_error_without_details(
    error: BaseException, expected: str
) -> None:
    with pytest.raises(TransportError) as raised:
        UrllibTransport(FakeOpener(error)).send("POST", URL, HEADERS, b"{}", 60.0)

    text = str(raised.value)
    assert text.startswith(expected)
    assert "secret" not in text
    assert KEY not in text
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__


def test_the_transport_error_message_names_the_reason() -> None:
    error = urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))

    with pytest.raises(TransportError, match=r"^URLError: Connection refused$"):
        UrllibTransport(FakeOpener(error)).send("GET", URL, HEADERS, None, 1.0)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (urllib.error.URLError(TimeoutError()), "URLError: TimeoutError"),
        (TimeoutError("timed out"), "TimeoutError: no response received"),
        (urllib.error.URLError("text reason"), "URLError: no response received"),
    ],
)
def test_a_reason_without_an_os_message_names_its_type(error: BaseException, expected: str) -> None:
    with pytest.raises(TransportError) as raised:
        UrllibTransport(FakeOpener(error)).send("GET", URL, HEADERS, None, 1.0)

    assert str(raised.value) == expected


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            b'{"type": "error", "error": {"type": "api_error", "message": "boom"}}',
            "HTTP 500 (api_error: boom)",
        ),
        (
            b'{"error": {"code": 500, "message": "boom", "status": "INTERNAL"}}',
            "HTTP 500 (INTERNAL: boom)",
        ),
        (b'{"error": {"type": "api_error"}}', "HTTP 500 (api_error)"),
        (b'{"error": {"message": "boom"}}', "HTTP 500"),
        (b'{"error": "boom"}', "HTTP 500"),
        (b"[1, 2]", "HTTP 500"),
        (b"", "HTTP 500"),
        (
            b'{"error": {"type": "api_error", "message": "' + b"x" * 400 + b'"}}',
            "HTTP 500 (api_error: " + "x" * 300 + ")",
        ),
    ],
)
def test_describe_response_reads_only_the_error_type_and_message(
    body: bytes, expected: str
) -> None:
    assert describe_response(HttpResponse(500, {"x-api-key": KEY}, body)) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://api.anthropic.com:99999/x",
        "http://api.anthropic.com/v1/messages/batches",
        "ftp://api.anthropic.com/x",
        "file:///etc/passwd",
        "https://example.com/v1/messages/batches",
        "https://api.anthropic.com.example.com/x",
        "https://api.anthropic.com:8443/x",
        "https://user@api.anthropic.com/x",
        "https:///x",
    ],
)
def test_a_url_outside_the_allow_list_is_refused_before_the_opener_is_called(url: str) -> None:
    opener = FakeOpener(FakeResponse(200, message(), b""))

    with pytest.raises(ValueError, match="refused"):
        UrllibTransport(opener).send("GET", url, HEADERS, None, 1.0)

    assert opener.calls == []


def test_the_allowed_hosts_are_exactly_the_two_apis() -> None:
    assert sorted(ALLOWED_HOSTS) == ["api.anthropic.com", "generativelanguage.googleapis.com"]


def test_both_api_hosts_are_accepted() -> None:
    for url in (URL, GEMINI_URL):
        opener = FakeOpener(FakeResponse(200, message(), b"{}"))
        assert UrllibTransport(opener).send("POST", url, HEADERS, b"{}", 1.0).status == 200


# --- the real opener: no proxy, verified TLS, no redirects ------------------------------------


class FakeHttps(urllib.request.BaseHandler):
    """Answers ``https_open`` in front of the real ``HTTPSHandler`` (order 500), after any proxy
    handler (order 100) has had its turn, so no socket is ever opened."""

    handler_order = 450

    def __init__(self, status: int, headers: http.client.HTTPMessage, body: bytes = b"") -> None:
        self.status = status
        self.headers = headers
        self.body = body
        self.requests: list[tuple[str, str, str | None]] = []

    def https_open(self, request: urllib.request.Request) -> urllib.response.addinfourl:
        self.requests.append((request.type, request.host, request.full_url))
        response = urllib.response.addinfourl(
            io.BytesIO(self.body), self.headers, request.full_url, code=self.status
        )
        setattr(response, "msg", "fake")  # noqa: B010 (read by HTTPErrorProcessor; not in the stubs)
        return response


def handlers_of(opener: object) -> list[urllib.request.BaseHandler]:
    """The handlers of an ``OpenerDirector`` (an attribute the type stubs do not declare)."""
    return getattr(opener, "handlers")  # noqa: B009


@contextmanager
def real_opener(fake: FakeHttps) -> Iterator[urllib.request.OpenerDirector]:
    opener = build_opener()
    opener.add_handler(fake)
    yield opener


def test_the_opener_follows_no_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeHttps(302, message(Location="https://example.com/steal"))

    with real_opener(fake) as opener:
        response = UrllibTransport(opener).send("GET", URL, HEADERS, None, 1.0)

    assert response.status == 302
    assert response.headers["location"] == "https://example.com/steal"
    assert fake.requests == [("https", "api.anthropic.com", URL)]


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_no_redirect_status_is_followed(status: int, method: str) -> None:
    # urllib's own handler follows each of these for a GET (and 301-303 for a POST, as a GET).
    fake = FakeHttps(status, message(Location="https://api.anthropic.com/elsewhere"))
    body = b"{}" if method == "POST" else None

    with real_opener(fake) as opener:
        response = UrllibTransport(opener).send(method, URL, HEADERS, body, 1.0)

    assert response.status == status
    assert len(fake.requests) == 1


def test_the_opener_uses_no_proxy_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://proxy.invalid:8080")
    fake = FakeHttps(200, message(), b"{}")

    with real_opener(fake) as opener:
        response = UrllibTransport(opener).send("GET", URL, HEADERS, None, 1.0)

    assert response == HttpResponse(200, {}, b"{}")
    assert fake.requests == [("https", "api.anthropic.com", URL)]
    # ProxyHandler({}) has no *_open method, so the opener keeps no proxy handler at all.
    proxies = [h for h in handlers_of(build_opener()) if isinstance(h, urllib.request.ProxyHandler)]
    assert proxies == []


def test_the_opener_verifies_tls() -> None:
    handlers = [
        h for h in handlers_of(build_opener()) if isinstance(h, urllib.request.HTTPSHandler)
    ]

    assert len(handlers) == 1
    context: Any = getattr(handlers[0], "_context")  # noqa: B009 (a private attribute of urllib)
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname


def test_the_default_transport_uses_the_hardened_opener() -> None:
    opener: Any = getattr(UrllibTransport(), "_opener")  # noqa: B009
    kinds = {type(handler) for handler in handlers_of(opener)}

    assert urllib.request.HTTPRedirectHandler not in kinds
    assert any(issubclass(kind, urllib.request.HTTPRedirectHandler) for kind in kinds)


# --- send_with_retries -----------------------------------------------------------------------


class Scripted:
    """A transport that answers from a script of responses and ``TransportError``s."""

    def __init__(self, *script: HttpResponse | TransportError) -> None:
        self.script = list(script)
        self.calls: list[tuple[str, str, dict[str, str], bytes | None, float]] = []

    def send(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
    ) -> HttpResponse:
        self.calls.append((method, url, headers, body, timeout))
        step = self.script.pop(0)
        if isinstance(step, TransportError):
            raise step
        return step


def anthropic_error(status: int, kind: str, text: str, **headers: str) -> HttpResponse:
    body = json.dumps({"type": "error", "error": {"type": kind, "message": text}})
    return HttpResponse(status, dict(headers), body.encode())


def gemini_error(status: int, state: str, text: str, delay: str | None = None) -> HttpResponse:
    details: list[dict[str, str]] = []
    if delay is not None:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay})
    body = {"error": {"code": status, "message": text, "status": state, "details": details}}
    return HttpResponse(status, {}, json.dumps(body).encode())


OK = HttpResponse(200, {}, b"{}")


def run(transport: Scripted, url: str = URL) -> tuple[HttpResponse, list[float]]:
    sleeps: list[float] = []
    response = send_with_retries(transport, "POST", url, HEADERS, b"{}", 60.0, sleep=sleeps.append)
    return response, sleeps


def test_the_base_delays_are_15_45_135() -> None:
    assert [base_delay(attempt) for attempt in (2, 3, 4)] == [15, 45, 135]


def test_a_529_then_200_retries_once_after_15_seconds() -> None:
    transport = Scripted(anthropic_error(529, "overloaded_error", "Overloaded"), OK)

    response, sleeps = run(transport)

    assert response == OK
    assert sleeps == [15]
    assert len(transport.calls) == 2
    assert transport.calls[0] == transport.calls[1] == ("POST", URL, HEADERS, b"{}", 60.0)


def test_a_transport_error_then_200_is_retried() -> None:
    transport = Scripted(TransportError("TimeoutError: no response"), OK)

    response, sleeps = run(transport)

    assert response == OK
    assert sleeps == [15]


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 529])
def test_every_transient_status_is_retried(status: int) -> None:
    response, sleeps = run(Scripted(HttpResponse(status, {}, b""), OK))

    assert response == OK
    assert sleeps == [15]


def test_the_delays_grow_15_45_135() -> None:
    error = HttpResponse(500, {}, b"")

    response, sleeps = run(Scripted(error, error, error, OK))

    assert response == OK
    assert sleeps == [15, 45, 135]


def test_a_retry_after_header_above_the_base_is_honoured() -> None:
    response, sleeps = run(Scripted(HttpResponse(429, {"retry-after": "70"}, b""), OK))

    assert response == OK
    assert sleeps == [70]


def test_a_retry_after_below_the_base_keeps_the_base() -> None:
    _, sleeps = run(Scripted(HttpResponse(429, {"retry-after": "3"}, b""), OK))

    assert sleeps == [15]


def test_a_gemini_retry_delay_is_honoured() -> None:
    _, sleeps = run(Scripted(gemini_error(429, "RESOURCE_EXHAUSTED", "Quota", "30s"), OK))

    assert sleeps == [30]


def test_a_fractional_gemini_retry_delay_is_rounded_up() -> None:
    _, sleeps = run(Scripted(gemini_error(429, "RESOURCE_EXHAUSTED", "Quota", "31.2s"), OK))

    assert sleeps == [32]


@pytest.mark.parametrize(
    "response",
    [
        HttpResponse(429, {"retry-after": "10000"}, b""),
        gemini_error(429, "RESOURCE_EXHAUSTED", "Quota", "9000s"),
    ],
    ids=["retry-after", "retryDelay"],
)
def test_a_hint_above_300_seconds_is_capped(response: HttpResponse) -> None:
    _, sleeps = run(Scripted(response, OK))

    assert sleeps == [300]


@pytest.mark.parametrize(
    "response",
    [
        HttpResponse(429, {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, b""),
        HttpResponse(429, {"retry-after": "-5"}, b""),
        gemini_error(429, "RESOURCE_EXHAUSTED", "Quota", "soon"),
        HttpResponse(503, {}, b"not json"),
        HttpResponse(503, {}, b'{"error": {"details": "no list"}}'),
        HttpResponse(503, {}, b'{"error": {"details": [1, {"retryDelay": "5s"}]}}'),
    ],
)
def test_an_unusable_hint_is_ignored(response: HttpResponse) -> None:
    _, sleeps = run(Scripted(response, OK))

    assert sleeps == [15]


def test_four_503_stop_the_run_with_the_error_type_and_message() -> None:
    error = anthropic_error(503, "api_error", "Internal server error")

    with pytest.raises(RunStopped) as raised:
        run(Scripted(error, error, error, error))

    assert str(raised.value) == (
        "the request failed 4 times; last: HTTP 503 (api_error: Internal server error)"
    )


def test_four_gemini_errors_name_the_status() -> None:
    error = gemini_error(503, "UNAVAILABLE", "The model is overloaded.")
    sleeps: list[float] = []

    with pytest.raises(RunStopped) as raised:
        send_with_retries(
            Scripted(error, error, error, error),
            "POST",
            GEMINI_URL,
            HEADERS,
            b"{}",
            600.0,
            sleep=sleeps.append,
        )

    assert str(raised.value) == (
        "the request failed 4 times; last: HTTP 503 (UNAVAILABLE: The model is overloaded.)"
    )
    assert sleeps == [15, 45, 135]


def test_four_transport_errors_stop_the_run_naming_the_last() -> None:
    errors = [TransportError("TimeoutError: no response received") for _ in range(4)]

    with pytest.raises(RunStopped) as raised:
        run(Scripted(*errors))

    assert str(raised.value) == (
        "the request failed 4 times; last: TimeoutError: no response received"
    )


def test_a_failure_without_an_error_body_names_only_the_status() -> None:
    error = HttpResponse(502, {}, b"<html>Bad gateway</html>")

    with pytest.raises(RunStopped, match=r"; last: HTTP 502$"):
        run(Scripted(error, error, error, error))


def test_the_stop_message_never_contains_a_header_value() -> None:
    error = anthropic_error(429, "rate_limit_error", f"key {KEY} is limited")

    with pytest.raises(RunStopped) as raised:
        run(Scripted(error, error, error, error))

    assert KEY not in str(raised.value)
    assert "[redacted]" in str(raised.value)


def test_a_400_is_returned_at_once() -> None:
    error = anthropic_error(400, "invalid_request_error", "bad")
    transport = Scripted(error)

    response, sleeps = run(transport)

    assert response == error
    assert sleeps == []
    assert len(transport.calls) == 1


def test_a_302_is_returned_not_followed() -> None:
    redirect = HttpResponse(302, {"location": "https://example.com/"}, b"")
    transport = Scripted(redirect)

    response, sleeps = run(transport)

    assert response == redirect
    assert sleeps == []
    assert len(transport.calls) == 1


def test_send_with_retries_refuses_a_foreign_host_before_any_attempt() -> None:
    transport = Scripted(OK)

    with pytest.raises(ValueError, match="refused"):
        run(transport, "https://example.com/results")

    assert transport.calls == []
