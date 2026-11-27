"""The recall after one repair (ADR-0008 D7; TASK-005e5 decision 7): ``potakt eval repairs``.

It reads a model's first run (``<runs>/<model dir>``) and its repair run
(``<runs>/repair/<model dir>``). Each scored first-run scenario whose candidate is invalid is one
invalid candidate; its repair outcome is, checked in this order: ``not sent`` (no answer or an
empty answer, ``repair_need``), ``without answer`` (the repair run gave up on it: ADR-0008 D6),
``valid`` or ``invalid`` (its repair is scored) and otherwise ``missing``. The recall after one
repair adds the matched count of every repair that is valid to the first run's matched count; it
never changes the gate. Offline: no model call, no credentials.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from potakt.evaluation.records import UNANSWERED_FILE
from potakt.evaluation.repair import RepairNeed, repair_need
from potakt.evaluation.runner import REPAIR_DIR, run_directory
from potakt.evaluation.summary import ScenarioRow, fraction, ratio, summarize, unscored
from potakt.evaluation.test_a import check_registered

__all__ = ["Outcome", "RepairCandidate", "RepairsReport", "repairs_report"]

Outcome = Literal["valid", "invalid", "without answer", "not sent", "missing"]
NO_COMPLETE_RUN = "{model} has no complete run (missing: {names})"


@dataclass(frozen=True, slots=True)
class RepairCandidate:
    """One invalid candidate of the first run and what its repair turn gave.

    ``reason`` is set for ``not sent`` only (``empty answer`` or ``no answer``);
    ``repair_issue_class`` for ``invalid`` only; ``matched_count`` is the repair's matched
    count for a scored repair (``valid`` or ``invalid``), ``None`` otherwise. ``truth_count`` is
    the scenario's, as the first run scored it."""

    scenario: str
    first_issue_class: str
    outcome: Outcome
    repair_issue_class: str | None
    reason: str | None
    matched_count: int | None
    truth_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "first_issue_class": self.first_issue_class,
            "outcome": self.outcome,
            "repair_issue_class": self.repair_issue_class,
            "reason": self.reason,
            "matched_count": self.matched_count,
            "truth_count": self.truth_count,
        }

    def to_text(self) -> str:
        """The candidate's line in the text form (without its indent)."""
        first = f"{self.scenario}: first invalid ({self.first_issue_class})"
        if self.outcome == "valid":
            return f"{first}, repair valid, matched {self.matched_count} of {self.truth_count}"
        if self.outcome == "invalid":
            return f"{first}, repair invalid ({self.repair_issue_class})"
        if self.outcome == "without answer":
            return f"{first}, repair without an answer (ADR-0008 D6)"
        if self.outcome == "not sent":
            reason = "empty answer" if self.reason == "empty answer" else "no answer, ADR-0008 D6"
            return f"{first}, not sent ({reason})"
        return f"{first}, missing"


@dataclass(frozen=True, slots=True)
class RepairsReport:
    """A model's first run and its repair run (ADR-0008 D7, never the gate); ``to_text`` and
    ``to_json`` are deterministic. ``matched`` and ``truth`` are the first run's over its scored
    scenarios; ``candidates`` holds every invalid candidate, in UTF-8 byte order."""

    model: str
    scored: int
    valid: int
    matched: int
    truth: int
    candidates: tuple[RepairCandidate, ...]

    def _with(self, outcome: Outcome) -> list[RepairCandidate]:
        return [candidate for candidate in self.candidates if candidate.outcome == outcome]

    @property
    def made_valid(self) -> tuple[tuple[str, int], ...]:
        """(scenario, matched count) of each candidate that one repair made valid."""
        return tuple((c.scenario, c.matched_count or 0) for c in self._with("valid"))

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(candidate.scenario for candidate in self._with("missing"))

    @property
    def complete(self) -> bool:
        """No invalid candidate is ``missing``."""
        return not self.missing

    @property
    def recorded(self) -> int:
        """The repairs that are scored: the outcomes ``valid`` and ``invalid``."""
        return len(self._with("valid")) + len(self._with("invalid"))

    @property
    def matched_after_repair(self) -> int:
        """The first run's matched count plus that of every repair that is valid."""
        return self.matched + sum(matched for _, matched in self.made_valid)

    def to_dict(self) -> dict[str, Any]:
        def counts(matched: int) -> dict[str, Any]:
            return {"matched": matched, "truth": self.truth, "recall": ratio(matched, self.truth)}

        return {
            "model": self.model,
            "scored": self.scored,
            "valid": self.valid,
            "invalid": len(self.candidates),
            "recorded": self.recorded,
            "without_answer": len(self._with("without answer")),
            "not_sent": len(self._with("not sent")),
            "missing": list(self.missing),
            "complete": self.complete,
            "recall": counts(self.matched),
            "recall_after_repair": counts(self.matched_after_repair),
            "made_valid": [
                {"scenario": name, "matched_count": matched} for name, matched in self.made_valid
            ],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }

    def to_json(self) -> str:
        """Indented UTF-8 JSON with sorted keys and a trailing ``\\n`` (like the summary's)."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"

    def to_text(self) -> str:
        """One fact per line, ending with ``\\n``."""
        complete = "yes" if self.complete else f"no (missing: {', '.join(self.missing)})"
        made_valid = len(self.made_valid)
        after = (
            f"Recall after one repair (not the gate): "
            f"{fraction(self.matched_after_repair, self.truth)}; {made_valid} of "
            f"{len(self.candidates)} invalid candidates valid after one repair"
        )
        if made_valid:
            after += f" ({', '.join(name for name, _ in self.made_valid)})"
        lines = [
            f"Repairs: {self.model} (ADR-0008 D7, not the gate)",
            f"First run: {self.scored} scored, {self.valid} valid, {len(self.candidates)} invalid",
            f"Repair run: {self.recorded} recorded, {len(self._with('without answer'))} without an "
            f"answer, {len(self._with('not sent'))} not sent, {len(self.missing)} missing",
            f"Complete: {complete}",
            f"Recall: {fraction(self.matched, self.truth)}",
            after,
            "Per invalid candidate:",
            *(f"  {candidate.to_text()}" for candidate in self.candidates),
        ]
        if not self.candidates:
            lines.append("  none")
        return "\n".join(lines) + "\n"


def repairs_report(model: str, runs_dir: Path, corpus_dir: Path) -> RepairsReport:
    """The repairs report of ``model`` (TASK-005e5 decision 7). ``UnknownModelError`` for an
    unregistered model; ``ValueError`` if ``runs_dir`` or ``corpus_dir`` is not a directory
    (the summary's words) or if the first run is not complete (``NO_COMPLETE_RUN``, the names as
    the order rule lists them); ``RunStopped`` from ``repair_need`` for a first-run record that
    cannot be read."""
    check_registered(model)
    for directory in (corpus_dir, runs_dir):
        if not directory.is_dir():
            raise ValueError(f"{directory} is not a directory")
    first_dir = run_directory(runs_dir, model, None)
    missing = unscored(first_dir, corpus_dir)
    if missing:
        raise ValueError(NO_COMPLETE_RUN.format(model=model, names=", ".join(missing)))
    first = summarize(first_dir, corpus_dir)
    repair_dir = run_directory(runs_dir, model, REPAIR_DIR)
    repaired: dict[str, ScenarioRow] = {}
    if repair_dir.is_dir():
        repaired = {row.scenario: row for row in summarize(repair_dir, corpus_dir).rows}
    candidates = tuple(
        _candidate(model, first_dir, repair_dir, row, repaired.get(row.scenario))
        for row in first.rows
        if not row.valid
    )
    return RepairsReport(
        model=model,
        scored=first.scenarios_run,
        valid=first.valid,
        matched=first.matched_count,
        truth=first.truth_count,
        candidates=candidates,
    )


def _candidate(
    model: str, first_dir: Path, repair_dir: Path, first: ScenarioRow, repair: ScenarioRow | None
) -> RepairCandidate:
    """The repair outcome of one invalid first candidate, checked in the task's order."""
    need = repair_need(model, first_dir / first.scenario)
    base = RepairCandidate(
        scenario=first.scenario,
        first_issue_class=first.issue_class or "",  # set on every invalid row
        outcome="missing",
        repair_issue_class=None,
        reason=None,
        matched_count=None,
        truth_count=first.truth_count,
    )
    if need is not None and not isinstance(need, RepairNeed):
        return replace(base, outcome="not sent", reason=need)
    if (repair_dir / first.scenario / UNANSWERED_FILE).is_file():
        return replace(base, outcome="without answer")
    if repair is None:
        return base
    if repair.valid:
        return replace(base, outcome="valid", matched_count=repair.matched_count)
    return replace(
        base,
        outcome="invalid",
        repair_issue_class=repair.issue_class,
        matched_count=repair.matched_count,
    )
