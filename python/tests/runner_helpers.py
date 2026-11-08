"""Shared pieces of the Test A runner tests (TASK-005c3 F2-F5): a small corpus copy, model
answers, a recording ``sleep`` with its clock, the ``batches.json`` entries and the check that the
fake key was written nowhere."""

import json
import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from potakt.evaluation import CORPUS_PATH

KEY = "potakt-test-key"  # the only key value any test uses (CLAUDE.md golden rule 3)
NOW = datetime(2026, 11, 2, 10, 0, 0, tzinfo=UTC)
ALL_SCENARIOS = tuple(
    sorted(
        (path.name for path in CORPUS_PATH.iterdir() if (path / "scenario.json").is_file()),
        key=lambda name: name.encode("utf-8"),
    )
)
JUNK = "Sorry, I cannot help with that."


def corpus_copy(tmp_path: Path, names: tuple[str, ...] = ALL_SCENARIOS) -> Path:
    """A corpus with the named real scenarios (all ten by default)."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name in names:
        shutil.copytree(CORPUS_PATH / name, corpus / name)
    return corpus


def truth_text(name: str) -> str:
    """The truth of a corpus scenario as an answer text: a valid candidate with recall 1."""
    return (CORPUS_PATH / name / "truth.json").read_text(encoding="utf-8")


def anthropic_message(model: str, text: str = JUNK, stop: str = "end_turn") -> dict[str, Any]:
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 1000, "output_tokens": 200},
    }


def gemini_response(text: str = JUNK, finish: str = "STOP") -> dict[str, Any]:
    return {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": finish}
        ],
        "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 200},
    }


class Sleeps(list[float]):
    """A recording ``sleep`` and its clock: ``now()`` is ``NOW`` plus every second slept, plus
    the seconds a fake server spent answering (``advance``)."""

    offset: float = 0.0

    def __call__(self, seconds: float) -> None:
        self.append(seconds)

    def advance(self, seconds: float) -> None:
        self.offset += seconds

    def now(self) -> datetime:
        return NOW + timedelta(seconds=sum(self) + self.offset)


class Echo(list[str]):
    """A recording ``echo``."""

    def __call__(self, line: str) -> None:
        self.append(line)


STARTED_AT = "2026-11-02T10:00:00Z"  # NOW as ``started_at`` (RFC 3339, whole seconds, Z)


def batch_entry(
    batch_id: str | None,
    scenarios: list[str] | tuple[str, ...],
    state: str,
    created_at: str | None = "2026-11-02T10:00:05.123456Z",
    started_at: str = STARTED_AT,
    problems: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """One entry of ``batches.json`` (TASK-005c3b D1, TASK-005c3c E4); ``creating`` and
    ``failed`` have no id and no ``created_at``; ``problems`` is empty unless the batch was
    held."""
    if state in ("creating", "failed"):
        batch_id, created_at = None, None
    return {
        "id": batch_id,
        "scenarios": list(scenarios),
        "started_at": started_at,
        "created_at": created_at,
        "state": state,
        "problems": list(problems),
    }


def write_state(model_dir: Path, *entries: dict[str, Any]) -> None:
    """Write ``batches.json`` with these entries, as the runner would. A ``done`` entry without
    a kept ``batch-<id>.results.jsonl`` gets an empty one: the runner keeps the results of every
    batch before it marks it done, and TASK-005c3h H6 (S3-8) reads them before every create."""
    model_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"batches": list(entries)}, indent=2, sort_keys=True) + "\n"
    (model_dir / "batches.json").write_bytes(text.encode("utf-8"))
    for entry in entries:
        kept = model_dir / f"batch-{entry['id']}.results.jsonl"
        if entry["state"] == "done" and not kept.exists():
            kept.write_bytes(b"")


class RefusedReplace:
    """``os.replace`` as Windows answers while another program holds the file (TASK-005c3g G3):
    ``PermissionError`` for the first ``times`` replaces whose destination is named ``name`` (any
    destination if ``name`` is ``None``), then the real replace. Our code decides, not the OS:
    the same on every OS."""

    def __init__(self, times: int, name: str | None = None) -> None:
        self.times = times
        self.name = name
        self.refused = 0
        self.real = os.replace

    def __call__(self, source: object, destination: object) -> None:
        if (self.name is None or Path(str(destination)).name == self.name) and (
            self.refused < self.times
        ):
            self.refused += 1
            raise PermissionError(13, "Permission denied")
        self.real(cast(Any, source), cast(Any, destination))


def files_containing(root: Path, needle: str) -> list[str]:
    """The files under ``root`` whose bytes contain ``needle`` (UTF-8)."""
    data = needle.encode("utf-8")
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and data in path.read_bytes()
    )
