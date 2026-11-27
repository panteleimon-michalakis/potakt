"""The summary of one model's recorded Test A runs (ADR-0007 §7 «Records», §5.3, §5.7 and the
amendment «Summary»).

A model's runs live in ``<runs>/<model dir>/<scenario id>/`` with ``comparison.json`` (the
``Comparison`` of the candidate) and ``response.json`` (the raw API response). The summary
micro-averages the scores over the scored scenarios, breaks recall down by language, template and
hardness (of the truth constraint), counts the diagnostic categories and how the answers stopped,
sums the token usage of every run and prices each run, and gives one row per scored scenario
(ADR-0008 D3(a)) with the candidate's assumptions next to the scenario's unsupported requests
(D12, read from ``candidate.txt``). It names every corpus scenario without a
scored run and is complete only when there is none. Nothing is silently skipped: every record
that cannot be used is in ``notes``.

Next to the gate, never changing it, it reports the recall after look-alike ids (ADR-0008 D9,
second point): each invalid candidate whose first failing stage is data-level validation has its
unknown ids replaced by their one look-alike data id (``replace_lookalike_ids``) and is
evaluated again; the matched count of each candidate this makes valid replaces its 0.
"""

import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from potakt.evaluation.compare import CATEGORIES, Category, InvalidStage
from potakt.evaluation.records import (
    CANDIDATE_FILE,
    COMPARISON_FILE,
    REGISTERED_MODELS,
    RESPONSE_FILE,
    answer_of,
)
from potakt.evaluation.scenario import METADATA_FILE, Scenario, ScenarioError, load_scenario
from potakt.evaluation.test_a import evaluate_candidate, replace_lookalike_ids
from potakt.evaluation.transport import server_text
from potakt.model import MAX_NESTING, json_nesting_exceeds

# The corpus of ADR-0007 §2, resolved from the repository root (src layout, like INSTRUCTION_PATH).
CORPUS_PATH = Path(__file__).resolve().parents[4] / "docs" / "validation" / "scenarios"
# USD per million (input, output) tokens, from the providers' official pricing pages on
# 2026-11-02: the Anthropic Batch API prices (ADR-0007 §7 amendment «Transport»), Gemini's free
# tier. A run whose input tokens exceed a long-prompt threshold uses that model's second pair
# (Haiku: prompts over 100,000 tokens).
PRICES_USD_PER_MILLION: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.05, 0.25),
    "gemini-3.8-flash": (0.0, 0.0),
}
LONG_PROMPT_PRICES_USD_PER_MILLION: dict[str, tuple[int, tuple[float, float]]] = {
    "claude-haiku-5-5": (100_000, (0.25, 1.25)),
}
# The class of an invalid candidate's first issue, in ADR-0008 D3(a)'s words. ``None``: the answer
# did not end normally, so it never reached validation.
ISSUE_CLASSES: dict[InvalidStage | None, str] = {
    "json": "JSON or fence",
    "schema": "schema",
    "rule": "cross-field rule",
    "data_file": "data file",
    "data": "data-level id or attribute",
    None: "stop reason",
}
# ADR-0007 §5.4: the gate of Test A; ADR-0008 D3(c): the bootstrap of the micro recall.
GATE_RECALL = 0.70
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261109
# ADR-0008 D12: an assumption is model output, shown printable and cut to this many characters.
ASSUMPTION_TEXT_LIMIT = 1000
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
    invalid_stage: InvalidStage | None  # no default: the key must be present

    @model_validator(mode="after")
    def _check_counts(self) -> Self:
        if len(self.matched) + len(self.unmatched_truth) != self.truth_count:
            raise ValueError("matched + unmatched_truth must equal truth_count")
        if self.valid and self.invalid_stage is not None:
            raise ValueError("a valid candidate has no invalid_stage")
        return self


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _usage(response: object) -> tuple[int, int, int | None] | None:
    """(input, output, thinking) tokens of an Anthropic or a Gemini response; ``None`` if the
    input or the output is unknown, and thinking ``None`` if only the thinking is.

    Anthropic: ``usage.input_tokens`` and ``usage.output_tokens`` (thinking included), and
    ``usage.output_tokens_details.thinking_tokens`` as thinking when it is a count not larger
    than the output. Gemini: ``usageMetadata.promptTokenCount``, and ``candidatesTokenCount``
    plus ``thoughtsTokenCount`` (0 if absent) as output, the latter also as thinking.
    """
    if not isinstance(response, dict):
        return None
    document = cast(dict[str, object], response)
    usage, metadata = document.get("usage"), document.get("usageMetadata")
    if isinstance(usage, dict):
        anthropic = cast(dict[str, object], usage)
        tokens = _count(anthropic.get("input_tokens")), _count(anthropic.get("output_tokens"))
        if tokens[0] is not None and tokens[1] is not None:
            return tokens[0], tokens[1], _anthropic_thinking(anthropic, tokens[1])
    if isinstance(metadata, dict):
        gemini = cast(dict[str, object], metadata)
        prompt = _count(gemini.get("promptTokenCount"))
        output = _count(gemini.get("candidatesTokenCount"))
        thoughts = _count(gemini.get("thoughtsTokenCount", 0))
        if prompt is not None and output is not None and thoughts is not None:
            return prompt, output + thoughts, thoughts
    return None


def _anthropic_thinking(usage: dict[str, object], output: int) -> int | None:
    """``output_tokens_details.thinking_tokens`` if it is a count not larger than ``output``
    (the inclusive total), else ``None``."""
    details = usage.get("output_tokens_details")
    if not isinstance(details, dict):
        return None
    thinking = _count(cast(dict[str, object], details).get("thinking_tokens"))
    return thinking if thinking is not None and thinking <= output else None


# --- the summary -----------------------------------------------------------------------------


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _shown(ratio: float | None) -> str:
    return "n/a" if ratio is None else f"{ratio:.3f}"


@dataclass(frozen=True, slots=True)
class ScenarioRow:
    """One scored scenario (ADR-0008 D3(a)), with its unsupported requests and the candidate's
    assumptions (D12).

    ``issue_class`` is ``None`` for a valid candidate. ``stop`` is the answer's stop exactly as
    the server sent it, ``None`` when the model is not registered or ``response.json`` cannot be
    used. Tokens are ``None`` when the run's usage (or only its thinking) is unknown; ``cost_usd``
    is ``None`` when the usage is unknown or the model has no price, and is rounded on its own.
    ``unsupported`` holds the scenario's ``unsupported[].quote`` in order; ``assumptions`` the
    candidate's assumption texts, printable and cut (``_read_assumptions``).
    """

    scenario: str
    language: str
    difficulty: str
    truth_count: int
    valid: bool
    issue_class: str | None
    stop: str | None
    output_tokens: int | None
    thinking_tokens: int | None
    cost_usd: float | None
    matched_count: int
    candidate_count: int
    unsupported: tuple[str, ...]
    assumptions: tuple[str, ...]

    @property
    def recall(self) -> float | None:
        return _ratio(self.matched_count, self.truth_count)

    @property
    def precision(self) -> float | None:
        return _ratio(self.matched_count, self.candidate_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "language": self.language,
            "difficulty": self.difficulty,
            "truth_count": self.truth_count,
            "valid": self.valid,
            "issue_class": self.issue_class,
            "stop": self.stop,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "cost_usd": self.cost_usd,
            "matched_count": self.matched_count,
            "candidate_count": self.candidate_count,
            "recall": self.recall,
            "precision": self.precision,
            "unsupported": list(self.unsupported),
            "assumptions": list(self.assumptions),
        }

    def to_text(self) -> str:
        """The row's line in the summary text (without its indent)."""
        validity = "valid" if self.valid else f"invalid ({self.issue_class})"
        stop = "n/a" if self.stop is None else server_text(self.stop)
        cost = "cost n/a" if self.cost_usd is None else f"${self.cost_usd:.4f}"
        return (
            f"{self.scenario}: {self.language} {self.difficulty}, truth {self.truth_count}, "
            f"{validity}, stop {stop}, output {_number(self.output_tokens)} "
            f"(thinking {_number(self.thinking_tokens)}), {cost}, matched {self.matched_count}, "
            f"recall {_shown(self.recall)}, precision {_shown(self.precision)}"
        )


def _number(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def bootstrap_interval(
    counts: list[tuple[int, int]],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float] | None:
    """The 95 % percentile bootstrap interval of the micro recall (ADR-0008 D3(c)).

    ``counts`` holds one (matched, truth) pair per scenario; ``None`` if it is empty or a truth
    is 0. Each of ``resamples`` resamples draws ``n = len(counts)`` pairs with replacement and
    computes Σ matched / Σ truth; the interval is the nearest-rank 2.5th and 97.5th percentiles
    of the sorted ratios (0-based positions ``ceil(0.025 * resamples) - 1`` and
    ``ceil(0.975 * resamples) - 1``).

    A draw is ``int(rng.random() * n)`` with ``rng = random.Random(seed)``, never the module's
    generator. Only ``random()`` is used because it is the one part of ``random`` whose sequence
    Python keeps the same across versions: "The generator's random() method will continue to
    produce the same sequence when the compatible seeder is given the same seed" (Python docs,
    «Notes on Reproducibility»); ``randrange``, ``choice`` and ``choices`` may change.
    """
    if not counts or any(truth == 0 for _, truth in counts):
        return None
    rng = random.Random(seed)
    n = len(counts)
    ratios: list[float] = []
    for _ in range(resamples):
        matched = truth = 0
        for _ in range(n):
            drawn_matched, drawn_truth = counts[int(rng.random() * n)]
            matched += drawn_matched
            truth += drawn_truth
        ratios.append(matched / truth)
    ratios.sort()
    return (
        ratios[math.ceil(0.025 * resamples) - 1],
        ratios[math.ceil(0.975 * resamples) - 1],
    )


def _contains_gate(interval: tuple[float, float]) -> bool:
    """``lower <= GATE_RECALL <= upper`` (ADR-0008 D3: then the report says so)."""
    return interval[0] <= GATE_RECALL <= interval[1]


@dataclass(frozen=True, slots=True)
class Summary:
    """One model's Test A results; ``to_json`` and ``to_text`` are deterministic.

    Scores count the scored scenarios (those with a usable ``comparison.json`` of a corpus
    scenario); usage, thinking and stops count every run directory; ``thinking_unknown`` names
    the runs whose usage is known but whose thinking is not. Groups map a name to (matched, truth).
    ``missing`` names the corpus scenarios without a scored run. ``lookalike_valid`` holds
    (scenario, matched count) of each scored invalid candidate that replacing its look-alike ids
    makes valid (ADR-0008 D9), in row order.
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
    thinking_tokens: int
    thinking_unknown: tuple[str, ...]
    cost_usd: float | None
    not_scored: tuple[str, ...]
    notes: tuple[str, ...]
    missing: tuple[str, ...]
    stops: dict[str, int]
    rows: tuple[ScenarioRow, ...]
    recall_interval: tuple[float, float] | None
    lookalike_valid: tuple[tuple[str, int], ...]

    @property
    def valid_counts(self) -> tuple[int, int]:
        """(matched, truth) over the valid candidates only (ADR-0008 D3(b), not the gate)."""
        valid = [row for row in self.rows if row.valid]
        return sum(row.matched_count for row in valid), sum(row.truth_count for row in valid)

    @property
    def lookalike_counts(self) -> tuple[int, int]:
        """(matched, truth) after look-alike ids are replaced (ADR-0008 D9, not the gate): the
        matched count of every candidate that this makes valid is added."""
        made_valid = sum(matched for _, matched in self.lookalike_valid)
        return self.matched_count + made_valid, self.truth_count

    @property
    def complete(self) -> bool:
        """Every corpus scenario is scored and every run is (ADR-0007 §7 amendment «Summary»)."""
        return not self.missing and not self.not_scored

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

        valid_matched, valid_truth = self.valid_counts
        lookalike_matched, lookalike_truth = self.lookalike_counts
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
            "thinking_tokens": self.thinking_tokens,
            "thinking_unknown": list(self.thinking_unknown),
            "cost_usd": self.cost_usd,
            "not_scored": list(self.not_scored),
            "notes": list(self.notes),
            "missing": list(self.missing),
            "complete": self.complete,
            "stops": dict(sorted(self.stops.items())),
            "scenarios": [row.to_dict() for row in self.rows],
            "recall_valid_only": {
                "matched": valid_matched,
                "truth": valid_truth,
                "recall": _ratio(valid_matched, valid_truth),
            },
            "recall_lookalike": {
                "matched": lookalike_matched,
                "truth": lookalike_truth,
                "recall": _ratio(lookalike_matched, lookalike_truth),
                "invalid": self.scenarios_run - self.valid,
                "made_valid": [
                    {"scenario": name, "matched_count": matched}
                    for name, matched in self.lookalike_valid
                ],
            },
            "recall_interval": None
            if self.recall_interval is None
            else {
                "lower": self.recall_interval[0],
                "upper": self.recall_interval[1],
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "contains_gate": _contains_gate(self.recall_interval),
            },
        }

    def to_json(self) -> str:
        """Indented UTF-8 JSON with sorted keys and a trailing ``\\n`` (like the comparison's)."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"

    def _interval_lines(self) -> list[str]:
        """The interval line and, only if the interval contains the gate, the gate note."""
        if self.recall_interval is None:
            return ["Recall 95% interval: n/a"]
        lower, upper = self.recall_interval
        scenarios = len(self.rows)
        lines = [
            f"Recall 95% interval: {lower:.3f} to {upper:.3f} (bootstrap over {scenarios}"
            f" scenarios, {BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED})"
        ]
        if _contains_gate(self.recall_interval):
            lines.append(
                f"Gate note: the interval contains {GATE_RECALL:.2f}; the point value decides,"
                f" and it lies within the uncertainty of a {scenarios}-scenario corpus"
            )
        return lines

    def _lookalike_line(self) -> str:
        """ADR-0008 D9: the recall after look-alike ids and the candidates this made valid."""
        matched, truth = self.lookalike_counts
        made_valid = len(self.lookalike_valid)
        line = (
            f"Recall after look-alike ids (not the gate): {_shown(_ratio(matched, truth))} "
            f"({matched}/{truth}); {made_valid} of {self.scenarios_run - self.valid} invalid "
            "candidates made valid"
        )
        if made_valid:
            line += f" ({', '.join(name for name, _ in self.lookalike_valid)})"
        return line

    def _assumption_lines(self) -> list[str]:
        """ADR-0008 D12: each row with an unsupported quote, its quotes and the candidate's
        assumptions; no line at all without such a row."""
        shown = [row for row in self.rows if row.unsupported]
        if not shown:
            return []
        lines = ["Unsupported requests and assumptions:"]
        for row in shown:
            lines.append(f"  {row.scenario}")
            lines += [f"    unsupported: {quote}" for quote in row.unsupported]
            lines += [f"    assumption: {text}" for text in row.assumptions] or [
                "    assumption: none"
            ]
        return lines

    def to_text(self) -> str:
        """A short report, one fact per line, ending with ``\\n``."""

        def fraction(matched: int, truth: int) -> str:
            return f"{_shown(_ratio(matched, truth))} ({matched}/{truth})"

        def groups(found: dict[str, tuple[int, int]]) -> str:
            shown = [f"{name} {fraction(*counts)}" for name, counts in sorted(found.items())]
            return ", ".join(shown) or "none"

        cost = (
            f"${self.cost_usd:.4f} (batch prices pinned on 2026-11-02)"
            if self.cost_usd is not None
            else f"cost unknown (no pinned price for {self.model})"
        )
        if self.complete:
            complete = "yes"
        elif self.missing:
            complete = f"no (missing: {', '.join(self.missing)})"
        else:
            complete = f"no (not scored: {', '.join(self.not_scored)})"
        # TASK-005c3h H8 (the review's #9): a stop is a server string, shown through server_text.
        stops = ", ".join(
            f"{server_text(name)} {count}" for name, count in sorted(self.stops.items())
        )
        unknown_thinking = (
            f" (unknown for {', '.join(self.thinking_unknown)})" if self.thinking_unknown else ""
        )
        lines = [
            f"Test A summary: {self.model}",
            f"Scenarios: {self.scenarios_run} scored, {self.valid} valid",
            f"Complete: {complete}",
            f"Recall: {fraction(self.matched_count, self.truth_count)}",
            *self._interval_lines(),
            f"Recall over valid candidates (not the gate): {fraction(*self.valid_counts)}",
            self._lookalike_line(),
            f"Merged recall: {fraction(self.merged_matched_count, self.merged_truth_count)}",
            f"Precision: {fraction(self.matched_count, self.candidate_count)}",
            f"Recall by language: {groups(self.by_language)}",
            f"Recall by hardness: {groups(self.by_hardness)}",
            "Recall by template:",
            *(f"  {name} {fraction(*counts)}" for name, counts in sorted(self.by_template.items())),
            "Diagnostics: "
            + ", ".join(f"{category} {self.diagnostics[category]}" for category in CATEGORIES),
            f"Stops: {stops or 'none'}",
            f"Usage: {self.input_tokens} input tokens, {self.output_tokens} output tokens",
            f"Thinking: {self.thinking_tokens} of the output tokens{unknown_thinking}",
            f"Cost: {cost}",
            "Per scenario:",
            *(f"  {row.to_text()}" for row in self.rows),
            *self._assumption_lines(),
        ]
        if self.not_scored:
            lines.append(f"Not scored: {', '.join(self.not_scored)}")
        lines += [f"Note: {note}" for note in self.notes]
        return "\n".join(lines) + "\n"


def summarize(runs_dir: Path, corpus_dir: Path) -> Summary:
    """Summarize the runs of one model, ``runs_dir`` (named after the model, ADR-0007 §7).

    Each subdirectory is one scenario run, named by its scenario id; its language comes from the
    corpus. The corpus scenarios are the subdirectories of ``corpus_dir`` with a
    ``scenario.json``. The model is the name of the resolved ``runs_dir`` (so ``.`` works).
    Raises ``ValueError`` if ``runs_dir`` or ``corpus_dir`` is not a directory.
    """
    for directory in (runs_dir, corpus_dir):
        if not directory.is_dir():
            raise ValueError(f"{directory} is not a directory")
    model = runs_dir.resolve().name
    registered = model in REGISTERED_MODELS
    totals: Counter[str] = Counter()
    languages: dict[str, Counter[str]] = {}
    templates: dict[str, Counter[str]] = {}
    hardness: dict[str, Counter[str]] = {}
    diagnostics: dict[Category, int] = dict.fromkeys(CATEGORIES, 0)
    stops: Counter[str] = Counter()
    scored: set[str] = set()
    not_scored: list[str] = []
    notes: list[str] = []
    thinking_unknown: list[str] = []
    rows: list[ScenarioRow] = []
    lookalike_valid: list[tuple[str, int]] = []
    cost_micro: float | None = 0.0 if _prices(model) else None
    for run in _directories(runs_dir):
        name = run.name
        record, problem = _read_record(run, corpus_dir)
        if record is None:
            not_scored.append(name)
            notes.append(f"{name}: {problem}; not scored")
        else:
            record_scored, scenario = record
            scored.add(name)
            totals["scored"] += 1
            totals["valid"] += record_scored.valid
            totals["truth"] += record_scored.truth_count
            totals["candidate"] += record_scored.candidate_count
            totals["matched"] += len(record_scored.matched)
            totals["merged_truth"] += record_scored.merged_truth_count
            totals["merged_matched"] += record_scored.merged_matched_count
            _tally(
                languages,
                scenario.language,
                len(record_scored.matched),
                record_scored.truth_count,
            )
            for pair in record_scored.matched:
                _tally(templates, pair.template, 1, 1)
                _tally(hardness, pair.hardness, 1, 1)
            for missed in record_scored.unmatched_truth:
                _tally(templates, missed.template, 0, 1)
                _tally(hardness, missed.hardness, 0, 1)
                diagnostics[missed.category] += 1
        response, response_problem = _read_response(run)
        tokens = None if response_problem else _usage(response)
        if response_problem:
            notes.append(f"{name}: {response_problem}; usage counted as 0")
        elif tokens is None:
            notes.append(f"{name}: usage unknown in {RESPONSE_FILE}; usage counted as 0")
        run_cost: float | None = None
        if tokens is not None:
            input_tokens, output_tokens, thinking_tokens = tokens
            totals["input"] += input_tokens
            totals["output"] += output_tokens
            if thinking_tokens is None:
                thinking_unknown.append(name)
            else:
                totals["thinking"] += thinking_tokens
            if cost_micro is not None:
                run_cost = _run_cost_micro(model, input_tokens, output_tokens)
                cost_micro += run_cost
        # One answer_of call gives both the counted stop and the row's stop.
        stop = answer_of(model, response).stop if registered and not response_problem else None
        if stop is not None:
            stops[stop] += 1
        if record is not None:
            found, assumptions_problem = _read_assumptions(run)
            if assumptions_problem:
                notes.append(f"{name}: {assumptions_problem}; assumptions not shown")
            rows.append(_row(name, *record, stop, tokens, run_cost, found))
            if not record[0].valid and record[0].invalid_stage == "data":
                made_valid, lookalike_problem = _lookalike_matched(run, corpus_dir)
                if lookalike_problem:
                    notes.append(f"{name}: {lookalike_problem}; look-alike ids not replaced")
                if made_valid is not None:
                    lookalike_valid.append((name, made_valid))
    if not registered:
        notes.append(f"{model}: not a registered model; stops not counted")
    missing = tuple(
        corpus.name for corpus in _directories(corpus_dir) if (corpus / METADATA_FILE).is_file()
    )
    return Summary(
        model=model,
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
        thinking_tokens=totals["thinking"],
        thinking_unknown=tuple(thinking_unknown),
        cost_usd=None if cost_micro is None else round(cost_micro / _MILLION, 6),
        not_scored=tuple(not_scored),
        notes=tuple(notes),
        missing=tuple(name for name in missing if name not in scored),
        stops=dict(stops),
        rows=tuple(rows),
        # ADR-0008 D3(c) and its amendment: the scored scenarios, in the order of the rows.
        recall_interval=bootstrap_interval([(row.matched_count, row.truth_count) for row in rows]),
        lookalike_valid=tuple(lookalike_valid),
    )


# TASK-005e5 decision 2: the problem of a look-alike step whose files cannot be copied.
COPY_FAILED = "the scenario's files cannot be copied"


def _lookalike_matched(run: Path, corpus_dir: Path) -> tuple[int | None, str]:
    """ADR-0008 D9: the matched count of one run's data-level invalid candidate once its
    look-alike ids are replaced, if that makes it valid, and ``""``; or ``None`` and the problem.

    A ``candidate.txt`` that is missing or not UTF-8 gives nothing and no problem (the
    assumptions already note a missing one); a scenario that cannot be loaded gives the problem.
    TASK-005e5 decision 2: an ``OSError`` anywhere in the step (loading the scenario, replacing
    the ids, evaluating the changed candidate: each copies or reads the scenario's files, and a
    full disk refuses the copy) gives ``COPY_FAILED``, never the error's text.
    """
    try:
        text = (run / CANDIDATE_FILE).read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None, ""
    try:
        try:
            scenario = load_scenario(corpus_dir / run.name)
        except ScenarioError:
            return None, "the scenario cannot be loaded"
        replaced = replace_lookalike_ids(scenario, text)
        if replaced is None:
            return None, ""
        comparison = evaluate_candidate(scenario, replaced)
    except OSError:
        return None, COPY_FAILED
    return (comparison.matched_count if comparison.valid else None), ""


def _row(
    name: str,
    record: _Record,
    scenario: Scenario,
    stop: str | None,
    tokens: tuple[int, int, int | None] | None,
    run_cost_micro: float | None,
    assumptions: tuple[str, ...],
) -> ScenarioRow:
    """The row of one scored run; ``run_cost_micro`` is ``None`` without usage or a price."""
    return ScenarioRow(
        scenario=name,
        language=scenario.language,
        difficulty=scenario.difficulty,
        truth_count=record.truth_count,
        valid=record.valid,
        issue_class=None if record.valid else ISSUE_CLASSES[record.invalid_stage],
        stop=stop,
        output_tokens=None if tokens is None else tokens[1],
        thinking_tokens=None if tokens is None else tokens[2],
        cost_usd=None if run_cost_micro is None else round(run_cost_micro / _MILLION, 6),
        matched_count=len(record.matched),
        candidate_count=record.candidate_count,
        unsupported=tuple(request.quote for request in scenario.unsupported),
        assumptions=assumptions,
    )


def _directories(parent: Path) -> list[Path]:
    """The subdirectories of ``parent`` in UTF-8 byte order of their names."""
    return sorted(
        (path for path in parent.iterdir() if path.is_dir()),
        key=lambda path: path.name.encode("utf-8"),
    )


def _read_record(run: Path, corpus_dir: Path) -> tuple[tuple[_Record, Scenario] | None, str]:
    """The comparison of one run and its corpus scenario, or ``None`` and the problem."""
    try:
        metadata = (corpus_dir / run.name / METADATA_FILE).read_text(encoding="utf-8-sig")
        scenario = Scenario.model_validate_json(metadata)
    except (OSError, ValidationError):
        return None, "not a scenario of the corpus"
    path = run / COMPARISON_FILE
    if not path.is_file():
        return None, f"{COMPARISON_FILE} not found"
    try:
        data = path.read_bytes()
    except OSError:
        return None, f"{COMPARISON_FILE} cannot be read"
    try:
        record = _Record.model_validate_json(data)
    except ValidationError:
        return None, f"{COMPARISON_FILE} is not a valid comparison"
    return (record, scenario), ""


# The problems of ``_json_text``.
_UNREADABLE = "cannot be read"
_NOT_JSON = "is not valid JSON"
_TOO_DEEP = f"is nested more than {MAX_NESTING} levels deep"


def _json_text(path: Path) -> tuple[str | None, str]:
    """The text of a JSON record before ``json.loads``, and ``""``; or ``None`` and the problem.

    The bytes are read (an ``OSError`` is ``_UNREADABLE``), decoded as strict UTF-8 (else
    ``_NOT_JSON``) and checked with ``json_nesting_exceeds`` on the raw text (else
    ``_TOO_DEEP``): an explicit limit decided by our code, so no reader depends on the
    interpreter's recursion limit (CLAUDE.md DoD 6).
    """
    try:
        data = path.read_bytes()
    except OSError:
        return None, _UNREADABLE
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None, _NOT_JSON
    if json_nesting_exceeds(text):
        return None, _TOO_DEEP
    return text, ""


def _read_response(run: Path) -> tuple[object, str]:
    """The parsed ``response.json`` of one run and ``""``, or ``None`` and the problem."""
    path = run / RESPONSE_FILE
    if not path.is_file():
        return None, f"{RESPONSE_FILE} not found"
    text, problem = _json_text(path)
    if text is None:
        return None, f"{RESPONSE_FILE} {problem}"
    try:
        return json.loads(text), ""
    except ValueError:
        return None, f"{RESPONSE_FILE} {_NOT_JSON}"


def _read_assumptions(run: Path) -> tuple[tuple[str, ...], str]:
    """The assumption texts of one run's ``candidate.txt`` and ``""``, or ``()`` and the problem.

    The candidate is read as JSON whatever its validity: each item of a top-level
    ``assumptions`` list that is an object with a string ``text`` counts, in order, through
    ``server_text`` (model output: printable, cut to ``ASSUMPTION_TEXT_LIMIT``, and a lone
    surrogate becomes ``?``). A missing file or one that cannot be read is a problem; anything
    else (not UTF-8, nested too deep, not JSON, no such list) gives no assumption and no problem.
    """
    path = run / CANDIDATE_FILE
    if not path.is_file():
        return (), f"{CANDIDATE_FILE} not found"
    text, problem = _json_text(path)
    if problem == _UNREADABLE:
        return (), f"{CANDIDATE_FILE} {_UNREADABLE}"
    if text is None:
        return (), ""
    try:
        candidate: object = json.loads(text)
    except ValueError:
        return (), ""
    if not isinstance(candidate, dict):
        return (), ""
    items = cast(dict[str, object], candidate).get("assumptions")
    if not isinstance(items, list):
        return (), ""
    texts: list[str] = []
    for item in cast(list[object], items):
        if isinstance(item, dict):
            text = cast(dict[str, object], item).get("text")
            if isinstance(text, str):
                texts.append(server_text(text, limit=ASSUMPTION_TEXT_LIMIT))
    return tuple(texts), ""


def _tally(groups: dict[str, Counter[str]], name: str, matched: int, truth: int) -> None:
    counts = groups.setdefault(name, Counter())
    counts["matched"] += matched
    counts["truth"] += truth


def _pairs(groups: dict[str, Counter[str]]) -> dict[str, tuple[int, int]]:
    return {name: (counts["matched"], counts["truth"]) for name, counts in groups.items()}


def _prices(model: str) -> tuple[float, float] | None:
    """The pinned prices of ``model``, or ``None`` for a model without a price.

    ``model`` is a runs directory name, so each priced model id is compared by its directory
    name (``model_directory_name``).
    """
    prices = {model_directory_name(name): price for name, price in PRICES_USD_PER_MILLION.items()}
    return prices.get(model)


def _run_cost_micro(model: str, input_tokens: int, output_tokens: int) -> float:
    """The cost of one run in micro-dollars (USD * 1e6) at its own price tier; the model is priced.

    The total is divided by a million and rounded to micro-dollars once, in ``summarize``.
    """
    price_in, price_out = cast(tuple[float, float], _prices(model))
    long_prompt = {
        model_directory_name(name): tier
        for name, tier in LONG_PROMPT_PRICES_USD_PER_MILLION.items()
    }.get(model)
    if long_prompt is not None and input_tokens > long_prompt[0]:
        price_in, price_out = long_prompt[1]
    return input_tokens * price_in + output_tokens * price_out
