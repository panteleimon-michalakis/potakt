"""TASK-005c3b D8: fault injection into the batch runner (Hypothesis).

A fake Message Batches API with a fake clock answers ``run_anthropic`` for a three-scenario
corpus. Hypothesis draws a script of faults (lost and failed create answers, interrupts, failed
polls and list calls), how long a new batch stays missing from the list, and each scenario's
result in each batch. A driver plays the owner: it runs the same call again after every stop or
interrupt and, when the run names several candidate batches, cancels all but the oldest. The
invariants are the money-safety rules: every scenario is recorded once, and no scenario is paid
for in two batches that are not cancelled.

What the fake does not model (its limits): an interrupt during a list call, a results download
or the recording of an answer; time passing during a call; a list lag above 30 seconds; a
cancelled batch that had already processed some requests (a cancelled batch counts as unpaid).
A batch the runner never polls still counts as paid: on the real API it runs anyway.
"""

import json
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, note, settings
from hypothesis import strategies as st
from runner_helpers import ALL_SCENARIOS, KEY, NOW, Echo, Sleeps, corpus_copy, files_containing
from test_evaluation_runner_anthropic import errored, succeeded

import potakt.evaluation.runner as runner_module
import potakt.evaluation.test_a as test_a_module
from potakt.evaluation.runner import BATCH_STATE_FILE, BATCHES_URL, run_anthropic
from potakt.evaluation.transport import HttpResponse, RunStopped, TransportError

HAIKU = "claude-haiku-5-5"
SCENARIOS = ALL_SCENARIOS[:3]
LIST_URL = f"{BATCHES_URL}?limit=100"
INVOCATIONS = 30
CREATE_FAULTS = (
    "ok",
    "lost-created",  # the batch is created, then the response is lost
    "lost-not-created",
    "503-created",  # a gateway answers 503 after the API accepted the request
    "503-not-created",
    "429",
    "interrupt-before",  # KeyboardInterrupt before the server sees the call
    "interrupt-after",  # the batch is created, then KeyboardInterrupt
    "duplicated",  # a proxy retried the request: two batches, then the response is lost
)
POLL_FAULTS = ("ok", "TransportError", "interrupt")
LIST_FAULTS = ("ok", "TransportError")
RESULTS = ("succeeded", "errored", "expired")


@dataclass
class Script:
    """The faults of one example, consumed in call order ("ok" once a list is used up)."""

    creates: list[str]
    polls: list[str]
    lists: list[str]
    lag: int  # seconds a new batch is missing from the list
    results: dict[str, list[str]]  # per scenario, its result in each batch that contains it


scripts = st.builds(
    Script,
    creates=st.lists(st.sampled_from(CREATE_FAULTS), max_size=8),
    polls=st.lists(st.sampled_from(POLL_FAULTS), max_size=6),
    lists=st.lists(st.sampled_from(LIST_FAULTS), max_size=6),
    lag=st.sampled_from((0, 30)),
    results=st.fixed_dictionaries(
        {name: st.lists(st.sampled_from(RESULTS), max_size=4) for name in SCENARIOS}
    ),
)


@dataclass
class Batch:
    id: str
    scenarios: list[str]
    created: float  # clock seconds after NOW
    results: dict[str, str]
    cancelled: bool = False
    polls: int = 0


@dataclass
class FakeApi:
    """The Message Batches API under ``script``; ``clock`` is the run's sleep and clock."""

    script: Script
    clock: Sleeps
    batches: dict[str, Batch] = field(default_factory=dict[str, Batch])

    def seconds(self) -> float:
        return (self.clock.now() - NOW).total_seconds()

    def create(self, scenarios: list[str]) -> Batch:
        results = {
            name: self.script.results[name].pop(0) if self.script.results[name] else "succeeded"
            for name in scenarios
        }
        batch = Batch(f"msgbatch_{len(self.batches) + 1:03d}", scenarios, self.seconds(), results)
        self.batches[batch.id] = batch
        return batch

    def cancel(self, batch_id: str) -> None:
        """The owner cancels a batch in the Console: its unprocessed requests are canceled."""
        self.batches[batch_id].cancelled = True

    def document(self, batch: Batch) -> dict[str, Any]:
        ended = batch.cancelled or batch.polls > 1
        counts: dict[str, int] = dict.fromkeys(
            ("processing", "succeeded", "errored", "canceled", "expired"), 0
        )
        if batch.cancelled:
            counts["canceled"] = len(batch.scenarios)
        elif ended:
            for result in batch.results.values():
                counts[result] += 1
        else:
            counts["processing"] = len(batch.scenarios)
        created = NOW + timedelta(seconds=batch.created)
        return {
            "id": batch.id,
            "type": "message_batch",
            "processing_status": "ended" if ended else "in_progress",
            "request_counts": counts,
            "created_at": created.isoformat().replace("+00:00", "Z"),
            "cancel_initiated_at": "2026-11-02T12:00:00Z" if batch.cancelled else None,
            "results_url": f"{BATCHES_URL}/{batch.id}/results" if ended else None,
        }

    def results_file(self, batch: Batch) -> bytes:
        lines: list[str] = []
        for name in batch.scenarios:
            kind = "canceled" if batch.cancelled else batch.results[name]
            result: dict[str, Any] = {
                "succeeded": lambda: succeeded(),
                "errored": lambda: errored("overloaded_error", "Overloaded"),
                "expired": lambda: {"type": "expired"},
                "canceled": lambda: {"type": "canceled"},
            }[kind]()
            lines.append(json.dumps({"custom_id": name, "result": result}))
        return ("\n".join(lines) + "\n").encode("utf-8")

    def send(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
    ) -> HttpResponse:
        if method == "POST":
            assert url == BATCHES_URL
            return self._create(json.loads(body or b"{}"))
        if url == LIST_URL:
            if (self.script.lists.pop(0) if self.script.lists else "ok") == "TransportError":
                raise TransportError("TimeoutError: no response received")
            visible = [
                batch
                for batch in reversed(self.batches.values())
                if batch.created + self.script.lag <= self.seconds()
            ]
            data = [self.document(batch) for batch in visible[:100]]
            return ok({"data": data, "has_more": False, "first_id": None, "last_id": None})
        if url.endswith("/results"):
            return HttpResponse(200, {}, self.results_file(self.batches[url.split("/")[-2]]))
        fault = self.script.polls.pop(0) if self.script.polls else "ok"
        if fault == "TransportError":
            raise TransportError("TimeoutError: no response received")
        if fault == "interrupt":
            raise KeyboardInterrupt
        batch = self.batches[url.rsplit("/", 1)[1]]
        batch.polls += 1
        return ok(self.document(batch))

    def _create(self, document: dict[str, Any]) -> HttpResponse:
        scenarios = [request["custom_id"] for request in document["requests"]]
        fault = self.script.creates.pop(0) if self.script.creates else "ok"
        if fault == "interrupt-before":
            raise KeyboardInterrupt
        if fault == "lost-not-created":
            raise TransportError("TimeoutError: no response received")
        if fault == "503-not-created":
            return HttpResponse(503, {}, b"")
        if fault == "429":
            return HttpResponse(429, {}, b"")
        batch = self.create(scenarios)
        if fault == "duplicated":
            self.create(scenarios)
            raise TransportError("TimeoutError: no response received")
        if fault == "lost-created":
            raise TransportError("TimeoutError: no response received")
        if fault == "503-created":
            return HttpResponse(503, {}, b"")
        if fault == "interrupt-after":
            raise KeyboardInterrupt
        return ok(self.document(batch))


def ok(document: object) -> HttpResponse:
    return HttpResponse(200, {"content-type": "application/json"}, json.dumps(document).encode())


@pytest.fixture(scope="module", autouse=True)
def schema_rendered_once() -> Iterator[None]:
    """Speed only: ``render_json_schema`` is pure and takes ~80 ms per request (Pydantic), three
    quarters of this test's time; here it is computed once. The bytes sent are unchanged, and the
    payload goldens and the rehearsal (D7) check them with the real function."""
    schema = test_a_module.render_json_schema()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(test_a_module, "render_json_schema", lambda: schema)
        yield


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return corpus_copy(tmp_path_factory.mktemp("faults"), SCENARIOS)


@pytest.fixture(scope="module")
def runs_base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("faults-runs")


@settings(max_examples=150, deadline=None)
@given(script=scripts)
def test_no_fault_sequence_pays_twice_or_loses_a_record(
    corpus: Path, runs_base: Path, script: Script
) -> None:
    note(f"script: {script!r}")
    runs = Path(tempfile.mkdtemp(dir=runs_base))
    model_dir = runs / HAIKU
    clock = Sleeps()
    api = FakeApi(script, clock)
    echo = Echo()
    capped = returned = False
    for _ in range(INVOCATIONS):
        try:
            run_anthropic(
                HAIKU, corpus, runs, KEY, api, sleep=clock, now=clock.now, echo=echo
            )  # fmt: skip
        except RunStopped as error:
            text = str(error)
            echo(f"STOPPED: {text}")
            if "without an answer; it is not sent again" in text:
                capped = True  # the 4-batch cap: the example ends here
                break
            if "could be the one created by an earlier create request" in text:
                named = text.split(": ", 1)[1].split(";", 1)[0].split(", ")
                oldest = min(named, key=lambda batch_id: (api.batches[batch_id].created, batch_id))
                for batch_id in named:
                    if batch_id != oldest:
                        api.cancel(batch_id)  # the owner, in the Console
        except KeyboardInterrupt:
            echo("INTERRUPTED")
        else:
            returned = True
            break
    note("\n".join(echo))

    recorded = [name for name in SCENARIOS if (model_dir / name / "response.json").is_file()]
    # 1. Every scenario recorded exactly once (write_answer never replaces a record), unless the
    #    batch cap was reached.
    assert capped or returned, f"no normal end after {INVOCATIONS} invocations"
    if not capped:
        assert recorded == list(SCENARIOS), f"not recorded: {set(SCENARIOS) - set(recorded)}"
    # 2. No double payment: per scenario, one live (not cancelled) batch succeeded at most.
    for name in SCENARIOS:
        paid = [
            batch.id
            for batch in api.batches.values()
            if not batch.cancelled and batch.results.get(name) == "succeeded"
        ]
        assert len(paid) <= 1, f"{name} paid in batches {paid}"
    # 3. batches.json loads, and a normal end leaves no creating entry.
    states = [batch.state for batch in runner_module._load_state(model_dir)]
    if returned:
        assert "creating" not in states, f"creating entry left: {states}"
    assert (model_dir / BATCH_STATE_FILE).is_file() or not api.batches
    # 4. The fake key is in no file.
    assert files_containing(runs, KEY) == []
