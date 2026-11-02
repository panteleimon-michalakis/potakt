"""Shared pieces of the Test A runner tests (TASK-005c3 F2-F5): a small corpus copy, model
answers, a recording ``sleep`` and the check that the fake key was written nowhere."""

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    """A recording ``sleep``."""

    def __call__(self, seconds: float) -> None:
        self.append(seconds)


class Echo(list[str]):
    """A recording ``echo``."""

    def __call__(self, line: str) -> None:
        self.append(line)


def files_containing(root: Path, needle: str) -> list[str]:
    """The files under ``root`` whose bytes contain ``needle`` (UTF-8)."""
    data = needle.encode("utf-8")
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and data in path.read_bytes()
    )
