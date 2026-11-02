"""The HTTPS transport of Test A (ADR-0007 §7 «Failures before an answer», TASK-005c3 F1).

The only module of ``potakt`` that may import ``urllib``, ``http``, ``ssl`` or ``socket``
(``tests/test_no_network.py``). Every request goes to one of ``ALLOWED_HOSTS`` over verified TLS,
through no proxy, and no redirect is ever followed, so the API key in a header can reach only the
API it belongs to. Messages of ``TransportError`` and ``RunStopped`` never contain a header, a
request body or a URL query.
"""

import http.client
import json
import math
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from email.message import Message
from typing import Any, Protocol, cast

ALLOWED_HOSTS = frozenset({"api.anthropic.com", "generativelanguage.googleapis.com"})
RETRIED_STATUSES = frozenset({429, 500, 502, 503, 504, 529})
ATTEMPTS = 4
MAX_DELAY_SECONDS = 300
MAX_RESPONSE_BYTES = 64 * 1024 * 1024  # TASK-005c3b D4: a larger body counts as no response
_BASE_SECONDS = 15
_GROWTH = 3
_RETRY_INFO = "google.rpc.RetryInfo"
_DURATION = re.compile(r"(\d+(?:\.\d+)?)s")
_ERROR_TEXT_LIMIT = 300
_REDACTED = "[redacted]"
KEY_HEADERS = frozenset({"x-api-key", "x-goog-api-key"})
# Connection-level failures: no HTTP response arrived (``HTTPError`` is caught before these).
_NETWORK_ERRORS = (
    urllib.error.URLError,
    TimeoutError,
    ConnectionError,
    http.client.HTTPException,
    ssl.SSLError,
)


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """An HTTP response of any status; header names are lower-cased."""

    status: int
    headers: dict[str, str]
    body: bytes


class TransportError(Exception):
    """No HTTP response arrived: a connection error or a timeout."""


class RunStopped(Exception):
    """The run stops and records no candidate; the message is safe to print."""


class Transport(Protocol):
    def send(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
    ) -> HttpResponse: ...


class Opener(Protocol):
    """The part of ``urllib.request.OpenerDirector`` the transport uses."""

    def open(
        self, fullurl: urllib.request.Request, data: None = None, timeout: float = ...
    ) -> Any: ...


def check_url(url: str) -> None:
    """Raise ``ValueError`` unless ``url`` is ``https://`` on one of ``ALLOWED_HOSTS``, with no
    explicit port and no user information. The message names the host only."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https":
        raise ValueError(f"refused: only https:// URLs are sent (got {parts.scheme or 'none'}://)")
    try:
        port = parts.port
    except ValueError:
        port = -1
    host = parts.hostname or ""
    if host not in ALLOWED_HOSTS or port is not None or parts.username or parts.password:
        raise ValueError(f"refused: {host or 'no host'} is not an API host of the experiment")


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: the 3xx reaches the caller as an ``HTTPError``.

    ``urllib`` would copy the request headers, the API key among them, to the new location.
    """

    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None


def build_opener() -> urllib.request.OpenerDirector:
    """An opener with no proxy (``HTTP(S)_PROXY`` ignored), verified TLS and no redirects."""
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        _RefuseRedirects(),
    )


def _headers(message: Message) -> dict[str, str]:
    return {name.lower(): value for name, value in message.items()}


def _describe(error: BaseException) -> str:
    """``type(error).__name__`` and a short reason that cannot contain a URL or a header."""
    cause: object = error.reason if isinstance(error, urllib.error.URLError) else error
    if isinstance(cause, OSError) and isinstance(cause.strerror, str) and cause.strerror:
        reason = cause.strerror
    elif isinstance(cause, BaseException) and cause is not error:
        reason = type(cause).__name__
    else:
        reason = "no response received"
    return f"{type(error).__name__}: {reason}"


class UrllibTransport:
    """``Transport`` over ``urllib`` with the hardened opener of ``build_opener``."""

    def __init__(self, opener: Opener | None = None) -> None:
        self._opener: Opener = opener if opener is not None else build_opener()

    def send(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
    ) -> HttpResponse:
        """Send one request; any HTTP status is a response, no response is a ``TransportError``.

        A body larger than ``MAX_RESPONSE_BYTES`` (success or error) is no response either: at
        most one byte more than the cap is ever read. Raises ``ValueError`` before sending if
        ``check_url`` refuses the URL.
        """
        check_url(url)
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            try:
                with self._opener.open(request, timeout=timeout) as response:
                    status = cast(int, response.status)
                    return HttpResponse(status, _headers(response.headers), _capped(response))
            except urllib.error.HTTPError as error:
                try:
                    return HttpResponse(error.code, _headers(error.headers), _capped(error))
                finally:
                    error.close()
        except _NETWORK_ERRORS as error:
            raise TransportError(_describe(error)) from None


def _capped(response: Any) -> bytes:
    """The body, read up to ``MAX_RESPONSE_BYTES`` + 1 bytes; a larger one is no response."""
    data = cast(bytes, response.read(MAX_RESPONSE_BYTES + 1))
    if len(data) > MAX_RESPONSE_BYTES:
        raise TransportError(f"response larger than {MAX_RESPONSE_BYTES // (1024 * 1024)} MiB")
    return data


def base_delay(attempt: int) -> int:
    """The base delay in seconds before attempt ``attempt`` (2, 3, 4 → 15, 45, 135)."""
    return _BASE_SECONDS * _GROWTH ** (attempt - 2)


def _json_error(body: bytes) -> dict[str, object] | None:
    """The ``error`` object of an Anthropic or a Gemini error body, if there is one."""
    try:
        document: object = json.loads(body)
    except (ValueError, RecursionError):
        return None
    if not isinstance(document, dict):
        return None
    error = cast(dict[str, object], document).get("error")
    return cast(dict[str, object], error) if isinstance(error, dict) else None


def server_hint(response: HttpResponse) -> int | None:
    """Seconds the server asks to wait: an integer ``retry-after`` header, or a Gemini
    ``RetryInfo.retryDelay`` such as ``"30s"`` (rounded up); the larger if both. ``None`` if
    neither is usable."""
    hints: list[int] = []
    header = response.headers.get("retry-after", "").strip()
    if header.isascii() and header.isdigit():
        hints.append(int(header))
    error = _json_error(response.body)
    details = error.get("details") if error is not None else None
    if isinstance(details, list):
        for item in cast(list[object], details):
            if not isinstance(item, dict):
                continue
            detail = cast(dict[str, object], item)
            kind, delay = detail.get("@type"), detail.get("retryDelay")
            if isinstance(kind, str) and kind.endswith(_RETRY_INFO) and isinstance(delay, str):
                matched = _DURATION.fullmatch(delay)
                if matched:
                    hints.append(math.ceil(float(matched.group(1))))
    return max(hints) if hints else None


def retry_delay(attempt: int, hint: int | None) -> int:
    """The delay before attempt ``attempt``: the larger of the base and the hint, capped."""
    return min(max(base_delay(attempt), hint or 0), MAX_DELAY_SECONDS)


def secret_values(headers: Mapping[str, str]) -> list[str]:
    """The values of the headers that carry an API key (``KEY_HEADERS``)."""
    return [value for name, value in headers.items() if name.lower() in KEY_HEADERS]


def redact(text: str, secrets: Mapping[str, str] | list[str]) -> str:
    """``text`` with every given value (header values, keys) replaced by ``[redacted]``."""
    values = secrets.values() if isinstance(secrets, Mapping) else secrets
    for value in sorted({value for value in values if value}, key=len, reverse=True):
        text = text.replace(value, _REDACTED)
    return text


def server_text(text: str, secrets: list[str] | None = None, limit: int = _ERROR_TEXT_LIMIT) -> str:
    """A string from a response, safe to print: ``secrets`` redacted first, then cut to
    ``limit`` characters, every non-printable character (control, escape, newline) as ``?``."""
    shortened = redact(text, secrets or [])[:limit]
    return "".join(char if char.isprintable() else "?" for char in shortened)


def describe_response(response: HttpResponse, secrets: list[str] | None = None) -> str:
    """``HTTP <status>``, plus ``(<type>: <message>)`` from an API error body: Anthropic's
    ``error.type`` and ``error.message``, Gemini's ``error.status`` and ``error.message``.
    Nothing else from the body; both go through ``server_text`` (``secrets`` redacted before the
    message is cut to 300 characters)."""
    text = f"HTTP {response.status}"
    error = _json_error(response.body)
    if error is None:
        return text
    kind = error.get("type") if isinstance(error.get("type"), str) else error.get("status")
    detail = error.get("message")
    if isinstance(kind, str) and isinstance(detail, str):
        return f"{text} ({server_text(kind, secrets)}: {server_text(detail, secrets)})"
    if isinstance(kind, str):
        return f"{text} ({server_text(kind, secrets)})"
    return text


def send_with_retries(
    transport: Transport,
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout: float,
    *,
    sleep: Callable[[float], object],
    attempts: int = ATTEMPTS,
) -> HttpResponse:
    """Send with the retries of ADR-0007 §7 «Failures before an answer».

    A ``TransportError`` and the statuses of ``RETRIED_STATUSES`` are retried, up to
    ``attempts`` in total, sleeping ``retry_delay`` before each retry. Any other status is
    returned at once (a 3xx too: it is never followed). After the last failed attempt,
    ``RunStopped`` names the last status and its API error, or the last ``TransportError``;
    the values of the key headers (``KEY_HEADERS``) are redacted from it. ``check_url`` runs
    before the first attempt.
    """
    check_url(url)
    secrets = secret_values(headers)
    last = ""
    for attempt in range(1, attempts + 1):
        hint: int | None = None
        try:
            response = transport.send(method, url, headers, body, timeout)
        except TransportError as error:
            last = str(error)
        else:
            if response.status not in RETRIED_STATUSES:
                return response
            last = describe_response(response, secrets)
            hint = server_hint(response)
        if attempt < attempts:
            sleep(retry_delay(attempt + 1, hint))
    message = f"the request failed {attempts} times; last: {last}"
    raise RunStopped(redact(message, secrets))
