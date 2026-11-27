"""The draw-to-draw variation of the strong model (ADR-0008 D8; TASK-005e5 decision 9):
``potakt eval draws``.

Draw 1 is the first ``claude-opus-5-5`` run (``<runs>/claude-opus-5-5``), draws 2 and 3 are its
repeats (``<runs>/repeat-2/claude-opus-5-5``, ``<runs>/repeat-3/claude-opus-5-5``), all three
sent with the same pinned requests. For each corpus scenario the report gives its recall in each
draw (an invalid or unanswered candidate is 0), their arithmetic mean, minimum and maximum, and
for each draw its micro recall. It never changes the gate. Offline: no model call, no
credentials.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from potakt.evaluation.runner import REPEAT_DIRS, run_directory
from potakt.evaluation.summary import fraction, ratio, shown, summarize, unscored
from potakt.evaluation.test_a import STRONG_MODEL

__all__ = [
    "DRAW_DIRECTORIES",
    "Draw",
    "DrawsReport",
    "ScenarioDraws",
    "draws_report",
    "scenario_draws",
]

# The directories of the three draws, relative to the runs directory, as the report names them:
# fixed strings with "/", never a Path, so the text is the same on every operating system.
DRAW_DIRECTORIES = (
    STRONG_MODEL,
    f"{REPEAT_DIRS[0]}/{STRONG_MODEL}",
    f"{REPEAT_DIRS[1]}/{STRONG_MODEL}",
)
NOT_COMPLETE = "draw {number} is not complete (missing: {names})"


@dataclass(frozen=True, slots=True)
class Draw:
    """One draw: its number (1-3), its directory (``DRAW_DIRECTORIES``) and its micro counts."""

    number: int
    directory: str
    matched: int
    truth: int

    @property
    def recall(self) -> float | None:
        return ratio(self.matched, self.truth)

    def to_dict(self) -> dict[str, Any]:
        return {
            "draw": self.number,
            "directory": self.directory,
            "matched": self.matched,
            "truth": self.truth,
            "recall": self.recall,
        }


@dataclass(frozen=True, slots=True)
class ScenarioDraws:
    """One scenario's recall in each draw and their arithmetic mean, minimum and maximum
    (``None`` when a draw's truth count is 0)."""

    scenario: str
    recalls: tuple[float | None, ...]
    mean: float | None
    min: float | None
    max: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "recalls": list(self.recalls),
            "mean": self.mean,
            "min": self.min,
            "max": self.max,
        }

    def to_text(self) -> str:
        """The scenario's line in the text form (without its indent)."""
        recalls = ", ".join(shown(value) for value in self.recalls)
        return (
            f"{self.scenario}: {recalls}; mean {shown(self.mean)}, min {shown(self.min)}, "
            f"max {shown(self.max)}"
        )


def scenario_draws(scenario: str, counts: Sequence[tuple[int, int]]) -> ScenarioDraws:
    """The draws of one scenario from its (matched, truth) count in each draw. Each recall is
    ``matched / truth``; the mean is the arithmetic mean of the recalls (their sum over their
    number, not Σ matched / Σ truth), computed exactly and rounded once, so it never falls
    outside [minimum, maximum] (a float sum of 0.1 three times, over 3, is above 0.1)."""
    recalls = tuple(ratio(matched, truth) for matched, truth in counts)
    if not counts or any(truth == 0 for _, truth in counts):
        return ScenarioDraws(scenario, recalls, None, None, None)
    exact = [Fraction(matched, truth) for matched, truth in counts]
    mean = float(sum(exact, Fraction(0)) / len(exact))
    return ScenarioDraws(scenario, recalls, mean, float(min(exact)), float(max(exact)))


@dataclass(frozen=True, slots=True)
class DrawsReport:
    """The three draws and every corpus scenario's recalls in them (ADR-0008 D8, never the
    gate); ``to_text`` and ``to_json`` are deterministic."""

    model: str
    draws: tuple[Draw, ...]
    scenarios: tuple[ScenarioDraws, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "draws": [draw.to_dict() for draw in self.draws],
            "scenarios": [row.to_dict() for row in self.scenarios],
        }

    def to_json(self) -> str:
        """Indented UTF-8 JSON with sorted keys and a trailing ``\\n`` (like the summary's)."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"

    def to_text(self) -> str:
        """One fact per line, ending with ``\\n``."""
        lines = [
            f"Draws: {self.model} (ADR-0008 D8, not the gate)",
            *(
                f"Draw {draw.number} ({draw.directory}): "
                f"recall {fraction(draw.matched, draw.truth)}"
                for draw in self.draws
            ),
            "Per scenario (recall in draws 1, 2, 3; mean, min, max):",
            *(f"  {row.to_text()}" for row in self.scenarios),
        ]
        return "\n".join(lines) + "\n"


def draws_report(runs_dir: Path, corpus_dir: Path) -> DrawsReport:
    """The draws report of ``claude-opus-5-5`` (TASK-005e5 decision 9). ``ValueError`` if
    ``runs_dir`` or ``corpus_dir`` is not a directory (the summary's words), or naming the first
    draw that is not complete (``NOT_COMPLETE``; a missing directory misses every corpus
    scenario)."""
    for directory in (corpus_dir, runs_dir):
        if not directory.is_dir():
            raise ValueError(f"{directory} is not a directory")
    kinds = (None, *REPEAT_DIRS)
    directories = [run_directory(runs_dir, STRONG_MODEL, kind) for kind in kinds]
    for number, directory in enumerate(directories, start=1):
        missing = unscored(directory, corpus_dir)
        if missing:
            raise ValueError(NOT_COMPLETE.format(number=number, names=", ".join(missing)))
    summaries = [summarize(directory, corpus_dir) for directory in directories]
    draws = tuple(
        Draw(number, name, summary.matched_count, summary.truth_count)
        for number, (name, summary) in enumerate(
            zip(DRAW_DIRECTORIES, summaries, strict=True), start=1
        )
    )
    rows = [{row.scenario: row for row in summary.rows} for summary in summaries]
    scenarios = tuple(
        scenario_draws(
            name, [(found[name].matched_count, found[name].truth_count) for found in rows]
        )
        for name in rows[0]  # the corpus scenarios, in UTF-8 byte order (every draw is complete)
    )
    return DrawsReport(STRONG_MODEL, draws, scenarios)
