"""The summary of one model's recorded Test A runs (ADR-0007 §7 «Records», §5.3, §5.7).

A model's runs live in ``<runs>/<model dir>/<scenario id>/`` with ``comparison.json`` (the
``Comparison`` of the candidate) and ``response.json`` (the raw API response). The summary
micro-averages the scores over the scored scenarios, breaks recall down by language, template and
hardness (of the truth constraint), counts the diagnostic categories, sums the token usage of every
run and prices it. Nothing is silently skipped: every record that cannot be used is in ``notes``.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from potakt.evaluation.compare import CATEGORIES, Category
from potakt.evaluation.scenario import METADATA_FILE, Scenario

# The corpus of ADR-0007 §2, resolved from the repository root (src layout, like INSTRUCTION_PATH).
CORPUS_PATH = Path(__file__).resolve().parents[4] / "docs" / "validation" / "scenarios"
COMPARISON_FILE = "comparison.json"
RESPONSE_FILE = "response.json"
# USD per million (input, output) tokens, from the providers' official pricing pages on
# 2026-11-02 (Anthropic: Haiku's tier for prompts up to 100,000 tokens; Gemini: free tier).
PRICES_USD_PER_MILLION: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-haiku-5-5": (0.1, 0.5),
    "gemini-3.8-flash": (0.0, 0.0),
}
_OUTSIDE_DIRECTORY_ALPHABET = re.compile(r"[^A-Za-z0-9._-]")
_MILLION = 1_000_000


def model_directory_name(model: str) -> str:
    """The directory of a model's runs: every character outside ``A-Z a-z 0-9 . _ -`` is ``-``."""
    return _OUTSIDE_DIRECTORY_ALPHABET.sub("-", model)


# --- the records -----------------------------------------------------------------------------


class _Pair(BaseModel):
    model_config = ConfigDict(extra="ignore")

    template: str
    hardness: Literal["hard", "soft"]


class _Missed(_Pair):
    category: Category


class _Record(BaseModel):
    """The fields of a ``comparison.json`` (``Comparison.to_json``) that the summary uses."""

    model_config = ConfigDict(extra="ignore")

    valid: bool
    truth_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    merged_truth_count: int = Field(ge=0)
    merged_matched_count: int = Field(ge=0)
    matched: list[_Pair]
    unmatched_truth: list[_Missed]

    @model_validator(mode="after")
    def _check_counts(self) -> Self:
        if len(self.matched) + len(self.unmatched_truth) != self.truth_count:
            raise ValueError("matched + unmatched_truth must equal truth_count")
        return self


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _usage(response: object) -> tuple[int, int] | None:
    """(input, output) tokens of an Anthropic or a Gemini response; ``None`` if unknown.

    Anthropic: ``usage.input_tokens`` and ``usage.output_tokens`` (thinking included). Gemini:
    ``usageMetadata.promptTokenCount``, and ``candidatesTokenCount`` plus ``thoughtsTokenCount``
    (if present) as output.
    """
    if not isinstance(response, dict):
        return None
    document = cast(dict[str, object], response)
    usage, metadata = document.get("usage"), document.get("usageMetadata")
    if isinstance(usage, dict):
        anthropic = cast(dict[str, object], usage)
        tokens = _count(anthropic.get("input_tokens")), _count(anthropic.get("output_tokens"))
        if tokens[0] is not None and tokens[1] is not None:
            return tokens[0], tokens[1]
    if isinstance(metadata, dict):
        gemini = cast(dict[str, object], metadata)
        prompt = _count(gemini.get("promptTokenCount"))
        output = _count(gemini.get("candidatesTokenCount"))
        thoughts = _count(gemini.get("thoughtsTokenCount", 0))
        if prompt is not None and output is not None and thoughts is not None:
            return prompt, output + thoughts
    return None


# --- the summary -----------------------------------------------------------------------------


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _shown(ratio: float | None) -> str:
    return "n/a" if ratio is None else f"{ratio:.3f}"


@dataclass(frozen=True, slots=True)
class Summary:
    """One model's Test A results; ``to_json`` and ``to_text`` are deterministic.

    Scores count the scored scenarios (those with a usable ``comparison.json`` of a corpus
    scenario); usage counts every run directory. Groups map a name to (matched, truth).
    """

    model: str
    scenarios_run: int
    valid: int
    truth_count: int
    candidate_count: int
    matched_count: int
    merged_truth_count: int
    merged_matched_count: int
    by_language: dict[str, tuple[int, int]]
    by_template: dict[str, tuple[int, int]]
    by_hardness: dict[str, tuple[int, int]]
    diagnostics: dict[Category, int]
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    not_scored: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def recall(self) -> float | None:
        return _ratio(self.matched_count, self.truth_count)

    @property
    def merged_recall(self) -> float | None:
        return _ratio(self.merged_matched_count, self.merged_truth_count)

    @property
    def precision(self) -> float | None:
        return _ratio(self.matched_count, self.candidate_count)

    def to_dict(self) -> dict[str, Any]:
        def groups(found: dict[str, tuple[int, int]]) -> dict[str, dict[str, Any]]:
            return {
                name: {"matched": matched, "truth": truth, "recall": _ratio(matched, truth)}
                for name, (matched, truth) in sorted(found.items())
            }

        return {
            "model": self.model,
            "scenarios_run": self.scenarios_run,
            "valid": self.valid,
            "truth_count": self.truth_count,
            "candidate_count": self.candidate_count,
            "matched_count": self.matched_count,
            "recall": self.recall,
            "precision": self.precision,
            "merged_truth_count": self.merged_truth_count,
            "merged_matched_count": self.merged_matched_count,
            "merged_recall": self.merged_recall,
            "recall_by_language": groups(self.by_language),
            "recall_by_template": groups(self.by_template),
            "recall_by_hardness": groups(self.by_hardness),
            "diagnostics": dict(self.diagnostics),
            "usage": {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens},
            "cost_usd": self.cost_usd,
            "not_scored": list(self.not_scored),
            "notes": list(self.notes),
        }

    def to_json(self) -> str:
        """Indented UTF-8 JSON with sorted keys and a trailing ``\\n`` (like the comparison's)."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"

    def to_text(self) -> str:
        """A short report, one fact per line, ending with ``\\n``."""

        def fraction(matched: int, truth: int) -> str:
            return f"{_shown(_ratio(matched, truth))} ({matched}/{truth})"

        def groups(found: dict[str, tuple[int, int]]) -> str:
            shown = [f"{name} {fraction(*counts)}" for name, counts in sorted(found.items())]
            return ", ".join(shown) or "none"

        cost = (
            f"${self.cost_usd:.4f} (prices pinned on 2026-11-02)"
            if self.cost_usd is not None
            else f"cost unknown (no pinned price for {self.model})"
        )
        lines = [
            f"Test A summary: {self.model}",
            f"Scenarios: {self.scenarios_run} scored, {self.valid} valid",
            f"Recall: {fraction(self.matched_count, self.truth_count)}",
            f"Merged recall: {fraction(self.merged_matched_count, self.merged_truth_count)}",
            f"Precision: {fraction(self.matched_count, self.candidate_count)}",
            f"Recall by language: {groups(self.by_language)}",
            f"Recall by hardness: {groups(self.by_hardness)}",
            "Recall by template:",
            *(f"  {name} {fraction(*counts)}" for name, counts in sorted(self.by_template.items())),
            "Diagnostics: "
            + ", ".join(f"{category} {self.diagnostics[category]}" for category in CATEGORIES),
            f"Usage: {self.input_tokens} input tokens, {self.output_tokens} output tokens",
            f"Cost: {cost}",
        ]
        if self.not_scored:
            lines.append(f"Not scored: {', '.join(self.not_scored)}")
        lines += [f"Note: {note}" for note in self.notes]
        return "\n".join(lines) + "\n"


def summarize(runs_dir: Path, corpus_dir: Path) -> Summary:
    """Summarize the runs of one model, ``runs_dir`` (named after the model, ADR-0007 §7).

    Each subdirectory is one scenario run, named by its scenario id; its language comes from the
    corpus. Raises ``ValueError`` if ``runs_dir`` is not a directory.
    """
    if not runs_dir.is_dir():
        raise ValueError(f"{runs_dir} is not a directory")
    runs = sorted(
        (path for path in runs_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name.encode("utf-8"),
    )
    totals: Counter[str] = Counter()
    languages: dict[str, Counter[str]] = {}
    templates: dict[str, Counter[str]] = {}
    hardness: dict[str, Counter[str]] = {}
    diagnostics: dict[Category, int] = dict.fromkeys(CATEGORIES, 0)
    not_scored: list[str] = []
    notes: list[str] = []
    for run in runs:
        name = run.name
        record, problem = _read_record(run, corpus_dir)
        if record is None:
            not_scored.append(name)
            notes.append(f"{name}: {problem}; not scored")
        else:
            record_scored, language = record
            totals["scored"] += 1
            totals["valid"] += record_scored.valid
            totals["truth"] += record_scored.truth_count
            totals["candidate"] += record_scored.candidate_count
            totals["matched"] += len(record_scored.matched)
            totals["merged_truth"] += record_scored.merged_truth_count
            totals["merged_matched"] += record_scored.merged_matched_count
            _tally(languages, language, len(record_scored.matched), record_scored.truth_count)
            for pair in record_scored.matched:
                _tally(templates, pair.template, 1, 1)
                _tally(hardness, pair.hardness, 1, 1)
            for missed in record_scored.unmatched_truth:
                _tally(templates, missed.template, 0, 1)
                _tally(hardness, missed.hardness, 0, 1)
                diagnostics[missed.category] += 1
        tokens, usage_problem = _read_usage(run)
        if tokens is None:
            notes.append(f"{name}: {usage_problem}; usage counted as 0")
        else:
            totals["input"] += tokens[0]
            totals["output"] += tokens[1]
    return Summary(
        model=runs_dir.name,
        scenarios_run=totals["scored"],
        valid=totals["valid"],
        truth_count=totals["truth"],
        candidate_count=totals["candidate"],
        matched_count=totals["matched"],
        merged_truth_count=totals["merged_truth"],
        merged_matched_count=totals["merged_matched"],
        by_language=_pairs(languages),
        by_template=_pairs(templates),
        by_hardness=_pairs(hardness),
        diagnostics=diagnostics,
        input_tokens=totals["input"],
        output_tokens=totals["output"],
        cost_usd=_cost(runs_dir.name, totals["input"], totals["output"]),
        not_scored=tuple(not_scored),
        notes=tuple(notes),
    )


def _read_record(run: Path, corpus_dir: Path) -> tuple[tuple[_Record, str] | None, str]:
    """The comparison of one run and its scenario language, or ``None`` and the problem."""
    try:
        metadata = (corpus_dir / run.name / METADATA_FILE).read_text(encoding="utf-8-sig")
        language = Scenario.model_validate_json(metadata).language
    except (OSError, ValidationError):
        return None, "not a scenario of the corpus"
    path = run / COMPARISON_FILE
    if not path.is_file():
        return None, f"{COMPARISON_FILE} not found"
    try:
        record = _Record.model_validate_json(path.read_bytes())
    except ValidationError:
        return None, f"{COMPARISON_FILE} is not a valid comparison"
    return (record, language), ""


def _read_usage(run: Path) -> tuple[tuple[int, int] | None, str]:
    path = run / RESPONSE_FILE
    if not path.is_file():
        return None, f"{RESPONSE_FILE} not found"
    try:
        response: object = json.loads(path.read_bytes())
    except (ValueError, RecursionError):
        return None, f"{RESPONSE_FILE} is not valid JSON"
    tokens = _usage(response)
    return (tokens, "") if tokens is not None else (None, f"usage unknown in {RESPONSE_FILE}")


def _tally(groups: dict[str, Counter[str]], name: str, matched: int, truth: int) -> None:
    counts = groups.setdefault(name, Counter())
    counts["matched"] += matched
    counts["truth"] += truth


def _pairs(groups: dict[str, Counter[str]]) -> dict[str, tuple[int, int]]:
    return {name: (counts["matched"], counts["truth"]) for name, counts in groups.items()}


def _cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """USD at the pinned prices, rounded to micro-dollars; ``None`` for a model without a price.

    ``model`` is the runs directory name, so a priced model id is looked up by its directory name.
    """
    prices = {model_directory_name(name): price for name, price in PRICES_USD_PER_MILLION.items()}
    if model not in prices:
        return None
    price_in, price_out = prices[model]
    return round((input_tokens * price_in + output_tokens * price_out) / _MILLION, 6)
