"""The Test A metric (ADR-0007 §5): constraint denotations and the comparison of two models.

Two constraints match iff their denotations are equal: the same template, hardness, scope
extension on the scenario data and normalized parameters (the table in ADR-0007 §5.2); for the
templates in ``FOLDED_SCOPE_TEMPLATES`` the scope is folded into the parameters, which describe the
rule's effect. Matching is one-to-one; recall and precision leave ``coverage`` out; the
informational recall with absences merged (§5.7) is reported next to them. This module is the
reference implementation of the metric; it needs no LLM, no network and no solver.
"""

import json
from collections import Counter, defaultdict, deque
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from potakt.checker import evaluators
from potakt.checker.context import Facts, period_spans, to_minutes
from potakt.model import (
    Apart,
    Assignment,
    AttributeRef,
    Balance,
    Constraint,
    DecisionModel,
    DecisionModelError,
    MaxConsecutiveDays,
    MaxPerPeriod,
    MinPerPeriod,
    MinRest,
    Preference,
    ProblemInstance,
    RequiresAttribute,
    Solution,
    Stability,
    Together,
    Unavailable,
    ValidationIssue,
    load_instance,
    parse_decision_model,
    validate_data,
)
from potakt.model.common import AttributeValue, Hardness
from potakt.model.scope import in_scope

Category = Literal["missing", "wrong_hardness", "wrong_parameters", "wrong_scope"]
CATEGORIES: tuple[Category, ...] = ("missing", "wrong_hardness", "wrong_parameters", "wrong_scope")
UNSCORED_TEMPLATES = frozenset({"coverage"})  # mandatory and structural (ADR-0007 §5.1)
# Templates denoted by their effect, with the scope folded into the parameters (ADR-0007 §5.2):
# what assignments they forbid, or which pair they bind on which tasks.
FOLDED_SCOPE_TEMPLATES = frozenset({"requires_attribute", "unavailable", "together", "apart"})

Params = tuple[tuple[str, Hashable], ...]


@dataclass(frozen=True, slots=True)
class Denotation:
    """What a constraint means on the scenario data (ADR-0007 §5.2); equal denotations match.

    ``resources`` and ``tasks`` are empty for ``FOLDED_SCOPE_TEMPLATES``: their scope is part of
    ``params``, so a different scope is a parameter difference there.
    """

    template: str
    hardness: Hardness
    resources: frozenset[str]  # in-scope resource ids
    tasks: frozenset[str]  # in-scope task ids
    params: Params  # normalized parameters, by name


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A DM that passed structural and data-level validation, with its loaded data."""

    dm: DecisionModel
    instance: ProblemInstance


@dataclass(frozen=True, slots=True)
class InvalidModel:
    """A candidate that failed validation: every issue found."""

    issues: tuple[ValidationIssue, ...]


def load_model(text: str, directory: Path) -> LoadedModel | InvalidModel:
    """Parse DM JSON text and validate it against the CSV files of ``directory``.

    Structural, CSV and data-level problems give an ``InvalidModel``; nothing raises.
    """
    try:
        dm = parse_decision_model(text)
        instance = load_instance(dm, directory)
    except DecisionModelError as error:
        return InvalidModel(error.issues)
    issues = validate_data(dm, instance)
    if issues:
        return InvalidModel(tuple(issues))
    return LoadedModel(dm, instance)


# --- denotations -----------------------------------------------------------------------------


def scope_extension(
    constraint: Constraint, dm: DecisionModel, instance: ProblemInstance
) -> tuple[frozenset[str], frozenset[str]]:
    """The in-scope resource ids and task ids of ``constraint`` on the data (ADR-0007 §5.2)."""
    zone = dm.zone
    resources = frozenset(
        resource.id
        for resource in instance.resources
        if in_scope(constraint.scope.resources, {**resource.attributes, "id": resource.id})
    )
    tasks = frozenset(
        task.id
        for task in instance.tasks
        if in_scope(
            constraint.scope.tasks,
            {**task.attributes, **task.derived_attributes(zone), "id": task.id},
        )
    )
    return resources, tasks


def denote(constraint: Constraint, dm: DecisionModel, instance: ProblemInstance) -> Denotation:
    """The denotation of ``constraint`` of ``dm`` on its loaded ``instance``.

    The inputs must be valid (``validate_data`` reports no issues).
    """
    resources, tasks = scope_extension(constraint, dm, instance)
    params = _params(constraint, dm, instance, resources, tasks)
    if constraint.template in FOLDED_SCOPE_TEMPLATES:
        resources = tasks = frozenset[str]()
    return Denotation(constraint.template, constraint.hardness, resources, tasks, params)


def _params(
    rule: Constraint,
    dm: DecisionModel,
    instance: ProblemInstance,
    resources: frozenset[str],
    tasks: frozenset[str],
) -> Params:
    """The normalized parameters of one rule (ADR-0007 §5.2), sorted by name."""
    match rule:
        case RequiresAttribute() | Unavailable():
            return (("forbidden", forbidden_pairs(rule, dm, instance)),)
        case MinRest():
            return (("minutes", to_minutes(rule.params.hours)),)
        case MaxPerPeriod() | MinPerPeriod():
            p = rule.params
            limits = frozenset(
                (resource.id, effective_limit(rule, resource.attributes))
                for resource in instance.resources
                if resource.id in resources
            )
            return (("limits", limits), ("measure", p.measure), ("periods", _periods(rule, dm)))
        case MaxConsecutiveDays():
            return (("days", rule.params.days),)
        case Preference():
            return (("kind", rule.params.kind),)
        case Balance():
            return (("measure", rule.params.measure),)
        case Together() | Apart():
            # Inert (no task) unless both members are in scope (the checker's together / apart).
            pair = frozenset(rule.params.resource_ids)
            return (("resource_ids", pair), ("tasks", tasks if pair <= resources else None))
        case Stability():
            baseline = frozenset(
                (entry.task_id, entry.resource_id)
                for entry in rule.params.baseline
                if entry.task_id in tasks and entry.resource_id in resources
            )
            return (("baseline", baseline),)
        case _:  # coverage, no_overlap
            return ()


def forbidden_pairs(
    rule: RequiresAttribute | Unavailable, dm: DecisionModel, instance: ProblemInstance
) -> frozenset[tuple[str | None, str]]:
    """The ``(resource, task)`` pairs the rule forbids among its in-scope resources and tasks.

    The checker itself decides, on a plan in which every resource works every task: each such
    rule has one violation per forbidden assignment, so the semantics cannot drift.
    """
    everything = Solution(
        schema_version="0.1",
        model_id=dm.id,
        model_version=dm.version,
        assignments=tuple(
            Assignment(task_id=task.id, resource_id=resource.id)
            for task in instance.tasks
            for resource in instance.resources
        ),
    )
    context = Facts(dm, instance, everything).view(rule.scope)
    found = (
        evaluators.requires_attribute(rule, context)
        if isinstance(rule, RequiresAttribute)
        else evaluators.unavailable(rule, context)
    )
    return frozenset(
        (violation.resource_id, task_id) for violation in found for task_id in violation.task_ids
    )


def effective_limit(
    rule: MaxPerPeriod | MinPerPeriod, attributes: dict[str, AttributeValue]
) -> int:
    """The effective limit of one resource: tasks for ``count``, minutes for ``hours``."""
    limit = rule.params.limit
    if isinstance(limit, AttributeRef):
        value = attributes.get(limit.attribute)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"no numeric {limit.attribute!r}; run validate_data first")
        limit = value
    return int(limit) if rule.params.measure == "count" else to_minutes(limit)


def _periods(rule: MaxPerPeriod | MinPerPeriod, dm: DecisionModel) -> tuple[tuple[date, date], ...]:
    """The periods the rule checks on the DM's horizon, as (first, last) dates clipped to it.

    Tasks start inside the horizon, so equal clipped periods mean the same grouping and the same
    number of periods (``min_per_period`` checks empty ones too): ``week`` and ``horizon`` match on
    a horizon of exactly one ISO week, while a longer horizon does not match a shorter one.
    """
    horizon = dm.horizon
    return tuple(
        (max(span.first, horizon.start), min(span.last, horizon.end))
        for span in period_spans(rule.params.period, horizon)
    )


# --- comparison ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MatchedPair:
    truth_id: str
    candidate_id: str
    template: str
    hardness: Hardness

    def to_dict(self) -> dict[str, Any]:
        return {
            "truth_id": self.truth_id,
            "candidate_id": self.candidate_id,
            "template": self.template,
            "hardness": self.hardness,
        }


@dataclass(frozen=True, slots=True)
class UnmatchedTruth:
    """A truth constraint that was not recovered, with its diagnostic category (ADR-0007 §5.5)."""

    truth_id: str
    template: str
    hardness: Hardness
    category: Category
    closest_candidate_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "truth_id": self.truth_id,
            "template": self.template,
            "hardness": self.hardness,
            "category": self.category,
            "closest_candidate_id": self.closest_candidate_id,
        }


@dataclass(frozen=True, slots=True)
class UnmatchedCandidate:
    candidate_id: str
    template: str
    hardness: Hardness

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "template": self.template,
            "hardness": self.hardness,
        }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


@dataclass(frozen=True, slots=True)
class Comparison:
    """The result of comparing a candidate DM with the truth on one scenario.

    Counts leave ``coverage`` out; an invalid candidate counts no constraints. The tuples are
    sorted by id, so equal inputs always give byte-identical JSON. The ``merged_*`` counts are
    the informational recall with absences merged (ADR-0007 §5.7), which never decides the gate.
    """

    valid: bool
    issues: tuple[ValidationIssue, ...]
    truth_count: int
    candidate_count: int
    matched: tuple[MatchedPair, ...]
    unmatched_truth: tuple[UnmatchedTruth, ...]
    unmatched_candidate: tuple[UnmatchedCandidate, ...]
    merged_truth_count: int
    merged_matched_count: int

    @property
    def matched_count(self) -> int:
        return len(self.matched)

    @property
    def recall(self) -> float | None:
        """Matched / truth constraints; ``None`` if the truth has none."""
        return _ratio(self.matched_count, self.truth_count)

    @property
    def merged_recall(self) -> float | None:
        """The recall with absences merged (ADR-0007 §5.7); ``None`` if the truth has none."""
        return _ratio(self.merged_matched_count, self.merged_truth_count)

    @property
    def precision(self) -> float | None:
        """Matched / candidate constraints; ``None`` if the candidate has none."""
        return _ratio(self.matched_count, self.candidate_count)

    def categories(self) -> dict[Category, int]:
        """How many unmatched truth constraints fall in each diagnostic category."""
        counts = dict.fromkeys(CATEGORIES, 0)
        for missed in self.unmatched_truth:
            counts[missed.category] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "issues": [{"path": issue.path, "message": issue.message} for issue in self.issues],
            "truth_count": self.truth_count,
            "candidate_count": self.candidate_count,
            "matched_count": self.matched_count,
            "recall": self.recall,
            "precision": self.precision,
            "merged_truth_count": self.merged_truth_count,
            "merged_matched_count": self.merged_matched_count,
            "merged_recall": self.merged_recall,
            "categories": self.categories(),
            "matched": [pair.to_dict() for pair in self.matched],
            "unmatched_truth": [missed.to_dict() for missed in self.unmatched_truth],
            "unmatched_candidate": [extra.to_dict() for extra in self.unmatched_candidate],
        }

    def to_json(self) -> str:
        """Indented UTF-8 JSON with sorted keys and a trailing ``\\n`` (like the checker's)."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class _Scored:
    rule: Constraint
    denotation: Denotation


def _scored(model: LoadedModel) -> list[_Scored]:
    return [
        _Scored(rule, denote(rule, model.dm, model.instance))
        for rule in model.dm.constraints
        if rule.template not in UNSCORED_TEMPLATES
    ]


def compare_models(truth: LoadedModel, candidate: LoadedModel | InvalidModel) -> Comparison:
    """Compare a candidate with the truth of the same scenario (ADR-0007 §5)."""
    expected = _scored(truth)
    merged_truth = _absences_merged(expected)
    if isinstance(candidate, InvalidModel):
        missed = [_unmatched(item, "missing", None) for item in expected]
        return Comparison(
            valid=False,
            issues=candidate.issues,
            truth_count=len(expected),
            candidate_count=0,
            matched=(),
            unmatched_truth=_by_id(missed),
            unmatched_candidate=(),
            merged_truth_count=len(merged_truth),
            merged_matched_count=0,
        )
    found = _scored(candidate)
    pairs = _match(expected, found)
    used_truth = {i for i, _ in pairs}
    used_candidate = {j for _, j in pairs}
    left_truth = [i for i in range(len(expected)) if i not in used_truth]
    left_candidate = [j for j in range(len(found)) if j not in used_candidate]
    closest = _diagnose(expected, found, left_truth, left_candidate)
    matched = [
        MatchedPair(
            expected[i].rule.id,
            found[j].rule.id,
            expected[i].rule.template,
            expected[i].rule.hardness,
        )
        for i, j in pairs
    ]
    unmatched = [
        _unmatched(expected[i], *closest[i])
        if i in closest
        else _unmatched(expected[i], "missing", None)
        for i in left_truth
    ]
    extra = [
        UnmatchedCandidate(found[j].rule.id, found[j].rule.template, found[j].rule.hardness)
        for j in left_candidate
    ]
    return Comparison(
        valid=True,
        issues=(),
        truth_count=len(expected),
        candidate_count=len(found),
        matched=tuple(sorted(matched, key=lambda m: (m.truth_id, m.candidate_id))),
        unmatched_truth=_by_id(unmatched),
        unmatched_candidate=tuple(sorted(extra, key=lambda e: e.candidate_id)),
        merged_truth_count=len(merged_truth),
        merged_matched_count=(merged_truth & _absences_merged(found)).total(),
    )


def _absences_merged(items: list[_Scored]) -> Counter[Denotation]:
    """The scored denotations with every ``unavailable`` rule of one hardness merged into one
    rule forbidding the union of their pairs (ADR-0007 §5.7), as a multiset."""
    merged: Counter[Denotation] = Counter()
    forbidden: dict[Hardness, set[Hashable]] = defaultdict(set)
    for item in items:
        denotation = item.denotation
        if denotation.template == "unavailable":
            pairs = dict(denotation.params)["forbidden"]
            assert isinstance(pairs, frozenset)
            forbidden[denotation.hardness] |= pairs
        else:
            merged[denotation] += 1
    for hardness, pairs in forbidden.items():
        params: Params = (("forbidden", frozenset(pairs)),)
        merged[Denotation("unavailable", hardness, frozenset(), frozenset(), params)] += 1
    return merged


def _unmatched(item: _Scored, category: Category, closest_id: str | None) -> UnmatchedTruth:
    return UnmatchedTruth(
        item.rule.id, item.rule.template, item.rule.hardness, category, closest_id
    )


def _by_id(missed: Sequence[UnmatchedTruth]) -> tuple[UnmatchedTruth, ...]:
    return tuple(sorted(missed, key=lambda m: m.truth_id))


def _match(expected: list[_Scored], found: list[_Scored]) -> list[tuple[int, int]]:
    """One-to-one pairs of equal denotations, formed in DM order (a multiset intersection)."""
    available: dict[Denotation, deque[int]] = defaultdict(deque)
    for j, item in enumerate(found):
        available[item.denotation].append(j)
    pairs: list[tuple[int, int]] = []
    for i, item in enumerate(expected):
        queue = available.get(item.denotation)
        if queue:
            pairs.append((i, queue.popleft()))
    return pairs


# Components of a denotation that can differ, least significant first (ADR-0007 §5.5).
_COMPONENTS = ("hardness", "parameters", "scope")
_CATEGORY_OF: dict[str, Category] = {
    "hardness": "wrong_hardness",
    "parameters": "wrong_parameters",
    "scope": "wrong_scope",
}


def _differences(a: Denotation, b: Denotation) -> list[int]:
    """The ranks (indexes into ``_COMPONENTS``) of the components in which two denotations
    of the same template differ."""
    differs = (
        a.hardness != b.hardness,
        a.params != b.params,
        a.resources != b.resources or a.tasks != b.tasks,
    )
    return [rank for rank, different in enumerate(differs) if different]


def _diagnose(
    expected: list[_Scored], found: list[_Scored], left_truth: list[int], left_candidate: list[int]
) -> dict[int, tuple[Category, str]]:
    """Pair unmatched truth constraints one-to-one with unmatched candidates of the same template,
    closest first; the category names the most significant difference."""
    options: list[tuple[tuple[int, int, int, int], Category]] = []
    for i in left_truth:
        for j in left_candidate:
            a, b = expected[i].denotation, found[j].denotation
            if a.template != b.template:
                continue
            ranks = _differences(a, b)  # never empty: equal denotations were matched
            category = _CATEGORY_OF[_COMPONENTS[max(ranks)]]
            options.append(((len(ranks), max(ranks), i, j), category))
    result: dict[int, tuple[Category, str]] = {}
    taken: set[int] = set()
    for (_, _, i, j), category in sorted(options):
        if i not in result and j not in taken:
            result[i] = (category, found[j].rule.id)
            taken.add(j)
    return result
