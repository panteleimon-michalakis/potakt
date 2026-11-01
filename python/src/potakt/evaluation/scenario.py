"""Phase 0 validation scenarios: metadata and the corpus loader (ADR-0007 §2).

A scenario directory holds ``request.md``, its CSV files, ``truth.json`` (the ground-truth DM),
``solution.json`` (a reference solution) and ``scenario.json`` (the metadata below). The loader
reuses the ``potakt.model`` loaders and reports every problem as an issue, never as a traceback.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, ValidationError, model_validator

from potakt.model import (
    DecisionModel,
    DecisionModelError,
    ProblemInstance,
    Solution,
    ValidationIssue,
    load_decision_model,
    load_instance,
    load_solution,
    validate_data,
    validate_solution,
)
from potakt.model.common import StrictModel, raise_errors
from potakt.model.data import LocalDateTime
from potakt.model.validation import issues_from_pydantic

REQUEST_FILE = "request.md"
METADATA_FILE = "scenario.json"
TRUTH_FILE = "truth.json"
SOLUTION_FILE = "solution.json"
CSV_SUFFIX = ".csv"  # compared by our code, case-sensitively, on every platform
ROOT = "/"

ScenarioId = Annotated[str, Field(pattern=r"^\d{2}-[a-z0-9]+(-[a-z0-9]+)*$")]
CsvFileName = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.csv$")]
Text = Annotated[str, Field(min_length=1)]
_Report = Callable[[str, str], None]  # report(path, message) of one scenario.json issue


class ReferenceScore(StrictModel):
    """The checker score of the reference solution under the truth (best found)."""

    hard: int
    soft: int


class Disruption(StrictModel):
    """A one-person absence ``[start, end)`` used for the re-planning measure of Test B."""

    resource_id: Text
    start: LocalDateTime
    end: LocalDateTime

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.end <= self.start:
            raise_errors("Disruption", [(("end",), "end must be after start")])
        return self


class UnsupportedRequest(StrictModel):
    """A request the catalog cannot express: not encoded in the truth, not part of recall."""

    quote: Text
    reason: Text


class Scenario(StrictModel):
    """The metadata of one scenario (``scenario.json``)."""

    id: ScenarioId
    title: Text
    language: Literal["el", "en"]
    domain: Text
    difficulty: Literal["easy", "medium", "hard"]
    style: Literal["email", "chat", "notes", "voice-transcript", "memo"]
    csv_files: list[CsvFileName] = Field(min_length=1)
    reference_score: ReferenceScore
    disruption: Disruption
    unsupported: list[UnsupportedRequest]
    notes: Text | None = None


@dataclass(frozen=True, slots=True)
class ScenarioIssue:
    """One problem in a scenario: the file, a JSON pointer into it (``/`` for the whole file) and a
    message."""

    file: str
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.file}#{self.path}: {self.message}"


class ScenarioError(Exception):
    """Raised when a scenario cannot be loaded; carries every issue found."""

    def __init__(self, issues: Sequence[ScenarioIssue]) -> None:
        self.issues: tuple[ScenarioIssue, ...] = tuple(issues)
        super().__init__("\n".join(str(issue) for issue in self.issues))


@dataclass(frozen=True, slots=True)
class LoadedScenario:
    """A scenario whose every file loaded and validated."""

    directory: Path
    scenario: Scenario
    request: str
    truth: DecisionModel
    instance: ProblemInstance
    solution: Solution


def load_scenario(directory: Path) -> LoadedScenario:
    """Load and validate one scenario directory; raise ``ScenarioError`` with every issue found.

    Checks: the metadata is valid, its ``id`` is the directory name and its ``csv_files`` are
    exactly the CSV files present; the request is not empty; the truth passes structural and
    data-level validation and has the scenario's language; the solution is valid for the truth;
    the disruption names a resource of the data and lies inside the horizon.
    """
    if not directory.is_dir():
        raise ScenarioError([ScenarioIssue(directory.name, ROOT, "not a scenario directory")])
    issues: list[ScenarioIssue] = []
    scenario = _load_metadata(directory, issues)
    request = _load_request(directory, issues)
    truth, instance, data_ok = _load_truth(directory, issues)
    solution = _load_solution(directory, truth, instance, data_ok, issues)
    if scenario is not None:
        issues += _consistency_issues(scenario, directory, truth, instance)
    if issues:
        raise ScenarioError(issues)
    assert scenario is not None and request is not None  # no issues: every part loaded
    assert truth is not None and instance is not None and solution is not None
    return LoadedScenario(directory, scenario, request, truth, instance, solution)


# --- files -----------------------------------------------------------------------------------


def _read(directory: Path, name: str, issues: list[ScenarioIssue]) -> str | None:
    path = directory / name
    if not path.exists():
        issues.append(ScenarioIssue(name, ROOT, "file not found"))
        return None
    try:
        return path.read_text(encoding="utf-8-sig")  # universal newlines: CRLF reads as \n
    except UnicodeDecodeError:
        issues.append(ScenarioIssue(name, ROOT, "not valid UTF-8"))
    except OSError as error:
        issues.append(ScenarioIssue(name, ROOT, f"cannot read: {error.strerror or error}"))
    return None


def _model_issues(name: str, found: Sequence[ValidationIssue]) -> list[ScenarioIssue]:
    return [ScenarioIssue(name, issue.path, issue.message) for issue in found]


def _parse_metadata(text: str) -> Scenario | list[ValidationIssue]:
    try:
        document: object = json.loads(text)
    except json.JSONDecodeError as error:
        message = f"invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}"
        return [ValidationIssue(ROOT, message)]
    except (RecursionError, ValueError):
        return [ValidationIssue(ROOT, "invalid JSON: nested too deeply or a number too large")]
    try:
        return Scenario.model_validate_json(text)
    except ValidationError as error:
        return issues_from_pydantic(error, document)


def _load_metadata(directory: Path, issues: list[ScenarioIssue]) -> Scenario | None:
    text = _read(directory, METADATA_FILE, issues)
    if text is None:
        return None
    parsed = _parse_metadata(text)
    if isinstance(parsed, Scenario):
        return parsed
    issues += _model_issues(METADATA_FILE, parsed)
    return None


def _load_request(directory: Path, issues: list[ScenarioIssue]) -> str | None:
    text = _read(directory, REQUEST_FILE, issues)
    if text is not None and not text.strip():
        issues.append(ScenarioIssue(REQUEST_FILE, ROOT, "the request is empty"))
        return None
    return text


def _load_truth(
    directory: Path, issues: list[ScenarioIssue]
) -> tuple[DecisionModel | None, ProblemInstance | None, bool]:
    """The truth DM and its instance; the flag says whether data-level validation passed."""
    if not (directory / TRUTH_FILE).exists():
        issues.append(ScenarioIssue(TRUTH_FILE, ROOT, "file not found"))
        return None, None, False
    try:
        truth = load_decision_model(directory / TRUTH_FILE)
        instance = load_instance(truth, directory)
    except DecisionModelError as error:
        issues += _model_issues(TRUTH_FILE, error.issues)
        return None, None, False
    found = validate_data(truth, instance)
    issues += _model_issues(TRUTH_FILE, found)
    return truth, instance, not found


def _load_solution(
    directory: Path,
    truth: DecisionModel | None,
    instance: ProblemInstance | None,
    data_ok: bool,
    issues: list[ScenarioIssue],
) -> Solution | None:
    if not (directory / SOLUTION_FILE).exists():
        issues.append(ScenarioIssue(SOLUTION_FILE, ROOT, "file not found"))
        return None
    try:
        solution = load_solution(directory / SOLUTION_FILE)
    except DecisionModelError as error:
        issues += _model_issues(SOLUTION_FILE, error.issues)
        return None
    if truth is not None and instance is not None and data_ok:
        # Solution ids are only meaningful against a valid model and data.
        issues += _model_issues(SOLUTION_FILE, validate_solution(truth, instance, solution))
    return solution


# --- consistency between the files -----------------------------------------------------------


def _consistency_issues(
    scenario: Scenario,
    directory: Path,
    truth: DecisionModel | None,
    instance: ProblemInstance | None,
) -> list[ScenarioIssue]:
    found: list[ScenarioIssue] = []

    def report(path: str, message: str) -> None:
        found.append(ScenarioIssue(METADATA_FILE, path, message))

    if scenario.id != directory.name:
        report("/id", f"id {scenario.id!r} must equal the directory name {directory.name!r}")
    _csv_file_issues(scenario.csv_files, directory, report)
    if truth is not None and scenario.language != truth.language:
        report(
            "/language",
            f"language {scenario.language!r} differs from the truth language {truth.language!r}",
        )
    if truth is not None and instance is not None:
        _disruption_issues(scenario.disruption, truth, instance, report)
    return found


def _csv_file_issues(listed: list[str], directory: Path, report: _Report) -> None:
    seen: set[str] = set()
    for index, name in enumerate(listed):
        if name in seen:
            report(f"/csv_files/{index}", f"{name} is listed twice")
        seen.add(name)
    present = {p.name for p in directory.iterdir() if p.is_file() and p.name.endswith(CSV_SUFFIX)}
    missing, unlisted = sorted(seen - present), sorted(present - seen)
    parts: list[str] = []
    if missing:
        parts.append(f"listed but not present: {', '.join(missing)}")
    if unlisted:
        parts.append(f"present but not listed: {', '.join(unlisted)}")
    if parts:
        report("/csv_files", "; ".join(parts))


def _disruption_issues(
    disruption: Disruption, truth: DecisionModel, instance: ProblemInstance, report: _Report
) -> None:
    if disruption.resource_id not in {resource.id for resource in instance.resources}:
        report("/disruption/resource_id", f"unknown resource id {disruption.resource_id!r}")
    first = datetime.combine(truth.horizon.start, time())
    after = datetime.combine(truth.horizon.end + timedelta(days=1), time())
    if disruption.start < first:
        report(
            "/disruption/start",
            f"{_shown(disruption.start)} is before the horizon start {_shown(first)}",
        )
    if disruption.end > after:
        report(
            "/disruption/end", f"{_shown(disruption.end)} is after the horizon end {_shown(after)}"
        )


def _shown(moment: datetime) -> str:
    return moment.isoformat(timespec="minutes")
