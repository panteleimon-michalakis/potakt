"""Equal effect under the truth's hard pair rules (ADR-0008 D10) and the other-template
diagnostic (ADR-0008 D11).

The reviewer's probes B5-1 (06) and B5-2 (04) are permanent tests here, with their reverse cases,
a small synthetic scenario for ``apart`` and ``together``, and a property that proves, with the
checker's own evaluators, that normalized-equal rules have equal effect on every plan that keeps
the truth's hard pair rules. The probe C2-1 (01) is the D11 case.
"""

import copy
import json
from collections.abc import Callable
from functools import cache
from pathlib import Path
from random import Random
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from potakt.checker import EVALUATORS
from potakt.checker.context import Facts
from potakt.checker.scoring import rewards
from potakt.evaluation import (
    CORPUS_PATH,
    Comparison,
    LoadedModel,
    LoadedScenario,
    compare_models,
    denote,
    evaluate_candidate,
    load_model,
    load_scenario,
)
from potakt.evaluation.compare import (
    _same_effect,  # pyright: ignore[reportPrivateUsage]
    _scored,  # pyright: ignore[reportPrivateUsage]
    effect_pairs,
    hard_forbidden_pairs,
    normalize,
)
from potakt.model import (
    Apart,
    Assignment,
    Constraint,
    DecisionModel,
    Preference,
    ProblemInstance,
    Solution,
    Stability,
    Together,
    parse_decision_model,
)

PAIR_RULES = Path(__file__).resolve().parent / "fixtures" / "compare" / "pair-rules"
PAIR_RULE_TEMPLATES = ("stability", "together", "apart")


@cache
def scenario(name: str) -> LoadedScenario:
    return load_scenario(CORPUS_PATH / name)


def truth_document(name: str) -> dict[str, Any]:
    text = (CORPUS_PATH / name / "truth.json").read_text(encoding="utf-8")
    return json.loads(text)


def constraint_of(document: dict[str, Any], rule_id: str) -> dict[str, Any]:
    return next(c for c in document["constraints"] if c["id"] == rule_id)


def evaluate(name: str, document: dict[str, Any]) -> Comparison:
    return evaluate_candidate(scenario(name), json.dumps(document, ensure_ascii=False))


def truth_of(name: str) -> LoadedModel:
    loaded = scenario(name)
    return LoadedModel(loaded.truth, loaded.instance)


def forbidden_of(name: str) -> frozenset[tuple[str, str]]:
    return hard_forbidden_pairs(truth_of(name))


def unmatched(result: Comparison) -> list[tuple[str, str]]:
    return [(missed.truth_id, missed.category) for missed in result.unmatched_truth]


def without_pairs(
    document: dict[str, Any], rule_id: str, keep: Callable[[str, str], bool]
) -> dict[str, Any]:
    """``document`` with the baseline of ``rule_id`` reduced to the entries ``keep`` accepts."""
    changed = copy.deepcopy(document)
    rule = constraint_of(changed, rule_id)
    rule["params"]["baseline"] = [
        entry
        for entry in rule["params"]["baseline"]
        if keep(entry["resource_id"], entry["task_id"])
    ]
    return changed


def with_task_ids(document: dict[str, Any], rule_id: str, task_ids: list[str]) -> dict[str, Any]:
    """``document`` with ``rule_id`` scoped to exactly ``task_ids``."""
    changed = copy.deepcopy(document)
    rule = constraint_of(changed, rule_id)
    rule.setdefault("scope", {})["tasks"] = [{"attribute": "id", "op": "in", "value": task_ids}]
    return changed


# --- B5-1: 06 keep-the-circulated-draft --------------------------------------------------------

S06 = "06-ambulance-crews"
DRAFT = "keep-the-circulated-draft"
# The three draft pairs the truth's hard absences forbid (TASK-005r1 B5-1). The ids are Greek,
# quoted verbatim; ruff confuses some Greek capitals with Latin ones (RUF001).
ABSENCE_PAIRS = {("Ε04", "13-Α-Δ"), ("Ε04", "14-Α-Ο"), ("Ε11", "09-Α-Δ")}  # noqa: RUF001


def draft_pairs(document: dict[str, Any]) -> set[tuple[str, str]]:
    baseline = constraint_of(document, DRAFT)["params"]["baseline"]
    return {(entry["resource_id"], entry["task_id"]) for entry in baseline}


def test_the_hard_absences_of_06_forbid_exactly_three_draft_pairs() -> None:
    assert draft_pairs(truth_document(S06)) & forbidden_of(S06) == ABSENCE_PAIRS


def test_b5_1_the_draft_without_the_pairs_the_absences_forbid_matches() -> None:
    candidate = without_pairs(truth_document(S06), DRAFT, lambda r, t: (r, t) not in ABSENCE_PAIRS)

    result = evaluate(S06, candidate)

    assert len(draft_pairs(candidate)) == 53
    assert result.valid
    assert (result.matched_count, result.truth_count, result.recall) == (11, 11, 1.0)
    assert DRAFT in {pair.truth_id for pair in result.matched}
    assert result.merged_matched_count == result.merged_truth_count  # D10 also when merged


def test_b5_1_reverse_one_pair_fewer_that_no_hard_rule_forbids_is_still_wrong_parameters() -> None:
    document = truth_document(S06)
    kept = min(draft_pairs(document) - ABSENCE_PAIRS)
    candidate = without_pairs(document, DRAFT, lambda r, t: (r, t) != kept)

    result = evaluate(S06, candidate)

    assert result.valid
    assert result.recall == 10 / 11
    assert unmatched(result) == [(DRAFT, "wrong_parameters")]


def test_the_truth_s_hard_pairs_decide_not_the_candidate_s() -> None:
    # The candidate drops the truth's two absences and adds a hard absence of its own that forbids
    # one more draft pair. Under the truth's F its draft equals the truth's; under its own F (or
    # under the candidate's F for both) it would not.
    document = truth_document(S06)
    extra_resource, extra_task = min(draft_pairs(document) - ABSENCE_PAIRS)
    candidate = without_pairs(document, DRAFT, lambda r, t: (r, t) not in ABSENCE_PAIRS)
    candidate["constraints"] = [
        c for c in candidate["constraints"] if c["template"] != "unavailable"
    ]
    task = next(t for t in scenario(S06).instance.tasks if t.id == extra_task)
    window = {
        "resource_id": extra_resource,
        "start": task.start.isoformat(timespec="minutes"),
        "end": task.end.isoformat(timespec="minutes"),
    }
    candidate["constraints"].append(
        {
            "id": "candidate-absence",
            "template": "unavailable",
            "hardness": "hard",
            "rationale": "x",
            "provenance": {"kind": "default"},
            "params": {"windows": [window]},
        }
    )

    result = evaluate(S06, candidate)

    assert result.valid
    assert DRAFT in {pair.truth_id for pair in result.matched}
    assert {truth_id for truth_id, _ in unmatched(result)} == {
        "karampetsos-sick-leave",
        "mouratidou-seminar",
    }


# --- B5-2: 04 ollie-with-grace ----------------------------------------------------------------

S04 = "04-restaurant"
OLLIE = "ollie-with-grace"
FLOOR = [{"attribute": "area", "op": "eq", "value": "floor"}]


def floor_task_ids() -> list[str]:
    return [t.id for t in scenario(S04).instance.tasks if t.attributes.get("area") == "floor"]


def test_b5_2_together_scoped_to_the_floor_matches() -> None:
    candidate = truth_document(S04)
    constraint_of(candidate, OLLIE)["scope"] = {"tasks": FLOOR}

    result = evaluate(S04, candidate)

    assert result.valid
    assert (result.matched_count, result.truth_count, result.recall) == (12, 12, 1.0)
    assert OLLIE in {pair.truth_id for pair in result.matched}
    assert result.merged_matched_count == result.merged_truth_count


def test_b5_2_reverse_one_floor_task_fewer_that_both_can_work_is_wrong_parameters() -> None:
    forbidden = forbidden_of(S04)
    floor = floor_task_ids()
    workable = [t for t in floor if ("F1", t) not in forbidden and ("F2", t) not in forbidden]
    candidate = with_task_ids(truth_document(S04), OLLIE, [t for t in floor if t != workable[0]])

    result = evaluate(S04, candidate)

    assert result.valid
    assert result.recall == 11 / 12
    assert unmatched(result) == [(OLLIE, "wrong_parameters")]


# --- the synthetic pair-rules scenario --------------------------------------------------------

APART, TOGETHER = "a-apart-from-b", "a-with-c"


def pair_rules_truth() -> dict[str, Any]:
    return json.loads((PAIR_RULES / "truth.json").read_text(encoding="utf-8"))


def compare_pair_rules(rule_id: str, task_ids: list[str]) -> Comparison:
    truth = pair_rules_truth()
    loaded = load_model(json.dumps(truth), PAIR_RULES)
    assert isinstance(loaded, LoadedModel)
    candidate = with_task_ids(truth, rule_id, task_ids)
    return compare_models(loaded, load_model(json.dumps(candidate), PAIR_RULES))


@pytest.mark.parametrize(
    ("rule_id", "task_ids", "matched"),
    [
        # apart: t2 is forbidden to a only, t3 to both: dropping them changes no violation.
        (APART, ["t1", "t4"], True),
        (APART, ["t1", "t3", "t4"], True),
        (APART, ["t1", "t2", "t4"], True),
        # t4 is forbidden to neither, t1 only to c (not a member).
        (APART, ["t1", "t2", "t3"], False),
        (APART, ["t2", "t3", "t4"], False),
        # together: only t3 is forbidden to both a and c.
        (TOGETHER, ["t1", "t2", "t4"], True),
        # t1 is forbidden to c only, t2 to a only: the other member can still work it alone.
        (TOGETHER, ["t2", "t4"], False),
        (TOGETHER, ["t1", "t4"], False),
        (TOGETHER, ["t1", "t2", "t3"], False),
    ],
)
def test_pair_rules_match_modulo_the_tasks_the_hard_rules_make_irrelevant(
    rule_id: str, task_ids: list[str], matched: bool
) -> None:
    result = compare_pair_rules(rule_id, task_ids)

    assert result.valid
    assert (rule_id in {pair.truth_id for pair in result.matched}) is matched
    if not matched:
        assert unmatched(result) == [(rule_id, "wrong_parameters")]


def test_normalize_drops_the_pairs_and_tasks_of_the_rules_definition() -> None:
    truth = load_model((PAIR_RULES / "truth.json").read_text(encoding="utf-8"), PAIR_RULES)
    assert isinstance(truth, LoadedModel)
    forbidden = hard_forbidden_pairs(truth)
    rules = {rule.id: rule for rule in truth.dm.constraints}

    apart = normalize(denote(rules[APART], truth.dm, truth.instance), forbidden)
    together = normalize(denote(rules[TOGETHER], truth.dm, truth.instance), forbidden)
    skills = denote(rules["skills"], truth.dm, truth.instance)

    assert forbidden == {("c", "t1"), ("a", "t2"), ("a", "t3"), ("b", "t3"), ("c", "t3")}
    assert dict(apart.params)["tasks"] == {"t1", "t4"}
    assert dict(together.params)["tasks"] == {"t1", "t2", "t4"}
    assert normalize(skills, forbidden) == skills  # other templates are unchanged


def test_an_inert_pair_rule_is_unchanged() -> None:
    truth = load_model((PAIR_RULES / "truth.json").read_text(encoding="utf-8"), PAIR_RULES)
    assert isinstance(truth, LoadedModel)
    document = pair_rules_truth()
    constraint_of(document, APART)["scope"] = {
        "resources": [{"attribute": "id", "op": "ne", "value": "b"}]
    }
    dm = parse_decision_model(json.dumps(document))
    rule = next(r for r in dm.constraints if r.id == APART)
    inert = denote(rule, dm, truth.instance)

    assert dict(inert.params)["tasks"] is None
    assert normalize(inert, hard_forbidden_pairs(truth)) == inert


# --- corpus invariants ------------------------------------------------------------------------

SCENARIOS = sorted(path.name for path in CORPUS_PATH.iterdir() if (path / "truth.json").is_file())


@pytest.mark.parametrize("name", SCENARIOS)
def test_every_corpus_truth_still_matches_itself(name: str) -> None:
    truth = truth_of(name)

    result = compare_models(truth, truth)

    assert (result.recall, result.precision, result.merged_recall) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize("name", SCENARIOS)
def test_normalize_is_idempotent_and_leaves_no_corpus_pair_rule_vacuous(name: str) -> None:
    truth = truth_of(name)
    forbidden = hard_forbidden_pairs(truth)

    for rule in truth.dm.constraints:
        once = normalize(denote(rule, truth.dm, truth.instance), forbidden)
        assert normalize(once, forbidden) == once, rule.id
        params = dict(once.params)
        if rule.template == "stability":
            assert params["baseline"], f"{name} {rule.id}: empty baseline"
        elif rule.template in ("together", "apart") and params["tasks"] is not None:
            assert params["tasks"], f"{name} {rule.id}: no task left"


# --- equal effect, proved with the checker -----------------------------------------------------


def plan(dm: DecisionModel, pairs: list[tuple[str, str]]) -> Solution:
    return Solution(
        schema_version="0.1",
        model_id=dm.id,
        model_version=dm.version,
        assignments=tuple(Assignment(task_id=t, resource_id=r) for r, t in pairs),
    )


def total(
    rule: Constraint, dm: DecisionModel, instance: ProblemInstance, solution: Solution
) -> int:
    context = Facts(dm, instance, solution).view(rule.scope)
    return sum(v.amount for v in EVALUATORS[rule.template](rule, context))


def replaced(name: str, rule_id: str, document: dict[str, Any]) -> tuple[DecisionModel, Constraint]:
    """The truth DM of ``name`` with ``rule_id`` replaced by ``document``, and that rule."""
    truth = truth_document(name)
    truth["constraints"] = [document if c["id"] == rule_id else c for c in truth["constraints"]]
    dm = parse_decision_model(json.dumps(truth, ensure_ascii=False))
    return dm, next(rule for rule in dm.constraints if rule.id == rule_id)


def variant(
    name: str, rule: Constraint, forbidden: frozenset[tuple[str, str]], rng: Random
) -> tuple[DecisionModel, Constraint]:
    """A random candidate rule whose normalized denotation equals ``rule``'s (ADR-0008 D10)."""
    document = copy.deepcopy(constraint_of(truth_document(name), rule.id))
    instance = scenario(name).instance
    if isinstance(rule, Stability):
        baseline = [(e.resource_id, e.task_id) for e in rule.params.baseline]
        kept = [p for p in baseline if p not in forbidden or rng.random() < 0.5]
        added = [p for p in sorted(forbidden) if p not in baseline and rng.random() < 0.02]
        document["params"]["baseline"] = [
            {"task_id": t, "resource_id": r} for r, t in rng.sample(kept + added, len(kept + added))
        ]
    else:
        assert isinstance(rule, Together | Apart)
        members = rule.params.resource_ids
        both = all if isinstance(rule, Together) else any
        dropped = [t.id for t in instance.tasks if both((r, t.id) in forbidden for r in members)]
        removed = {t for t in dropped if rng.random() < 0.7}
        kept = [t.id for t in instance.tasks if t.id not in removed]
        document.setdefault("scope", {})["tasks"] = [{"attribute": "id", "op": "in", "value": kept}]
    return replaced(name, rule.id, document)


EFFECT_SCENARIOS = {
    S06: (DRAFT, "shared-car"),
    S04: (OLLIE, "dan-and-marco-apart"),
}


# One drawn seed per example: drawing every coin flip from Hypothesis (``st.randoms``) makes the
# smallest example thousands of draws long and fails the ``large_base_example`` health check.
SEEDS = st.integers(min_value=0, max_value=2**32 - 1).map(Random)


@pytest.mark.parametrize("name", EFFECT_SCENARIOS)
@settings(max_examples=200, deadline=None)
@given(rng=SEEDS)
def test_normalized_equal_rules_have_equal_effect_on_every_plan_that_keeps_the_hard_rules(
    name: str, rng: Random
) -> None:
    truth = truth_of(name)
    forbidden = hard_forbidden_pairs(truth)
    allowed = [
        (r.id, t.id)
        for t in truth.instance.tasks
        for r in truth.instance.resources
        if (r.id, t.id) not in forbidden
    ]
    drawn = plan(truth.dm, [pair for pair in allowed if rng.random() < 0.3])
    empty = plan(truth.dm, [])
    rules = {rule.id: rule for rule in truth.dm.constraints}

    for rule_id in EFFECT_SCENARIOS[name]:
        rule = rules[rule_id]
        dm, other = variant(name, rule, forbidden, rng)
        expected = normalize(denote(rule, truth.dm, truth.instance), forbidden)
        assert normalize(denote(other, dm, truth.instance), forbidden) == expected
        if isinstance(rule, Stability):
            # Every plan's penalty shifts by the same constant (the empty plan gives it).
            shift = total(rule, truth.dm, truth.instance, empty) - total(
                other, dm, truth.instance, empty
            )
            assert (
                total(rule, truth.dm, truth.instance, drawn)
                - total(other, dm, truth.instance, drawn)
                == shift
            )
        else:
            assert total(rule, truth.dm, truth.instance, drawn) == total(
                other, dm, truth.instance, drawn
            )


# --- D11: another template with the same effect (C2-1) -----------------------------------------

S01 = "01-pharmacy"
STELIOS = "stelios-avoids-afternoons"
SOFT_OFF = "stelios-afternoons-soft-off"
AFTERNOON = "απόγευμα"  # the `shift` value of the afternoon tasks, quoted verbatim


def afternoon_windows() -> list[dict[str, str]]:
    """One window for b2 per afternoon shift of 01 (its two tasks share their times)."""
    times = sorted(
        {
            (t.start.isoformat(timespec="minutes"), t.end.isoformat(timespec="minutes"))
            for t in scenario(S01).instance.tasks
            if t.attributes.get("shift") == AFTERNOON
        }
    )
    return [{"resource_id": "b2", "start": start, "end": end} for start, end in times]


def soft_off(**fields: Any) -> dict[str, Any]:
    """C2-1's candidate rule: a soft ``unavailable`` for b2 on exactly the afternoon shifts."""
    return {
        "id": SOFT_OFF,
        "template": "unavailable",
        "hardness": "soft",
        "weight": 3,
        "rationale": "x",
        # The request's Greek words, quoted verbatim; ruff confuses Greek letters with Latin ones.
        "provenance": {"kind": "utterance", "quote": "να μην πιάνει απογεύματα"},  # noqa: RUF001
        "params": {"windows": afternoon_windows()},
        **fields,
    }


def replacing_stelios(rule: dict[str, Any]) -> dict[str, Any]:
    document = truth_document(S01)
    document["constraints"] = [rule if c["id"] == STELIOS else c for c in document["constraints"]]
    return document


def test_c2_1_a_soft_unavailable_with_the_same_effect_is_reported_as_another_template() -> None:
    candidate = replacing_stelios(soft_off())
    loaded = load_model(json.dumps(candidate, ensure_ascii=False), CORPUS_PATH / S01)
    assert isinstance(loaded, LoadedModel)
    rule = next(r for r in loaded.dm.constraints if r.id == SOFT_OFF)
    stelios = next(r for r in scenario(S01).truth.constraints if r.id == STELIOS)

    result = evaluate(S01, candidate)

    assert len(afternoon_windows()) == 3
    assert (
        effect_pairs(rule, loaded.dm, loaded.instance)
        == effect_pairs(stelios, scenario(S01).truth, scenario(S01).instance)
        != frozenset()
    )
    assert result.valid
    assert (result.matched_count, result.truth_count, result.recall) == (4, 5, 0.8)
    assert result.precision == 0.8
    assert [(m.truth_id, m.category, m.closest_candidate_id) for m in result.unmatched_truth] == [
        (STELIOS, "other_template_same_effect", SOFT_OFF)
    ]
    assert result.categories()["other_template_same_effect"] == 1
    assert [extra.candidate_id for extra in result.unmatched_candidate] == [SOFT_OFF]


@pytest.mark.parametrize(
    "rule",
    [
        pytest.param(soft_off(hardness="hard", weight=None), id="another hardness"),
        pytest.param(
            soft_off(scope={"tasks": [{"attribute": "id", "op": "ne", "value": "17-apog-farm"}]}),
            id="one pair fewer",
        ),
    ],
)
def test_c2_1_another_hardness_or_effect_stays_missing(rule: dict[str, Any]) -> None:
    rule = {key: value for key, value in rule.items() if value is not None}

    result = evaluate(S01, replacing_stelios(rule))

    assert result.valid
    assert result.recall == 0.8
    assert [(m.truth_id, m.category, m.closest_candidate_id) for m in result.unmatched_truth] == [
        (STELIOS, "missing", None)
    ]


def test_one_candidate_gets_only_the_first_of_two_equal_truth_constraints() -> None:
    # A copy of Stelios's preference after it in DM order, with an id that sorts before it.
    document = truth_document(S01)
    copy_of_stelios = {**constraint_of(document, STELIOS), "id": "a-stelios-copy"}
    document["constraints"].append(copy_of_stelios)
    truth = load_model(json.dumps(document, ensure_ascii=False), CORPUS_PATH / S01)
    assert isinstance(truth, LoadedModel)
    candidate = load_model(
        json.dumps(replacing_stelios(soft_off()), ensure_ascii=False), CORPUS_PATH / S01
    )

    result = compare_models(truth, candidate)

    assert result.recall == 4 / 6
    assert [(m.truth_id, m.category, m.closest_candidate_id) for m in result.unmatched_truth] == [
        ("a-stelios-copy", "missing", None),
        (STELIOS, "other_template_same_effect", SOFT_OFF),
    ]


def test_an_empty_effect_never_qualifies() -> None:
    # On the plan "every resource works every task" a together has no violation, nor has a
    # stability (every baseline pair is worked): equal, but empty, effects.
    document = pair_rules_truth()
    stability = {
        "id": "keep-a-on-t4",
        "template": "stability",
        "hardness": "soft",
        "weight": 1,
        "rationale": "x",
        "provenance": {"kind": "default"},
        "params": {"baseline": [{"task_id": "t4", "resource_id": "a"}]},
    }
    document["constraints"] = [
        stability if c["id"] == TOGETHER else c for c in document["constraints"]
    ]
    truth = load_model((PAIR_RULES / "truth.json").read_text(encoding="utf-8"), PAIR_RULES)
    candidate = load_model(json.dumps(document), PAIR_RULES)
    assert isinstance(truth, LoadedModel) and isinstance(candidate, LoadedModel)
    rules = {rule.id: rule for rule in [*truth.dm.constraints, *candidate.dm.constraints]}

    result = compare_models(truth, candidate)

    assert effect_pairs(rules[TOGETHER], truth.dm, truth.instance) == frozenset()
    assert effect_pairs(rules["keep-a-on-t4"], candidate.dm, candidate.instance) == frozenset()
    assert [(m.truth_id, m.category, m.closest_candidate_id) for m in result.unmatched_truth] == [
        (TOGETHER, "missing", None)
    ]


def test_the_pairing_never_takes_a_candidate_of_the_same_template() -> None:
    # Through compare_models a free candidate of the same template cannot exist: diagnostics
    # pair it with the missing truth constraint first. The condition is tested on the helper.
    truth = truth_of(S01)
    scored = _scored(truth, frozenset())  # pyright: ignore[reportPrivateUsage]
    index = next(i for i, item in enumerate(scored) if item.rule.id == STELIOS)

    paired = _same_effect(  # pyright: ignore[reportPrivateUsage]
        scored, scored, [index], [index], truth, truth
    )

    assert paired == {}


# --- D11: a reward is not a penalty --------------------------------------------------------------


def stelios_rewarding(document: dict[str, Any]) -> dict[str, Any]:
    """``document`` with Stelios's preference turned from ``avoid`` into ``prefer``."""
    changed = copy.deepcopy(document)
    constraint_of(changed, STELIOS)["params"]["kind"] = "prefer"
    return changed


@pytest.mark.parametrize(
    ("truth_doc", "candidate_doc", "unmatched_id"),
    [
        pytest.param(
            stelios_rewarding(truth_document(S01)),
            replacing_stelios(soft_off()),
            STELIOS,
            id="truth prefers",
        ),
        pytest.param(
            replacing_stelios(soft_off()),
            stelios_rewarding(truth_document(S01)),
            SOFT_OFF,
            id="candidate prefers",
        ),
    ],
)
def test_a_rewarded_pair_set_is_not_the_effect_of_a_soft_unavailable_either_way(
    truth_doc: dict[str, Any], candidate_doc: dict[str, Any], unmatched_id: str
) -> None:
    # Same pairs, opposite direction: scoring turns the `prefer` amount into a reward, so the two
    # rules do not even agree on the sign of their effect (ADR-0008 D11: "penalized or forbidden").
    truth = load_model(json.dumps(truth_doc, ensure_ascii=False), CORPUS_PATH / S01)
    candidate = load_model(json.dumps(candidate_doc, ensure_ascii=False), CORPUS_PATH / S01)
    assert isinstance(truth, LoadedModel) and isinstance(candidate, LoadedModel)

    result = compare_models(truth, candidate)

    assert result.valid
    assert (result.matched_count, result.truth_count, result.recall) == (4, 5, 0.8)
    assert [(m.truth_id, m.category, m.closest_candidate_id) for m in result.unmatched_truth] == [
        (unmatched_id, "missing", None)
    ]
    assert result.categories()["other_template_same_effect"] == 0


def corpus_preferences() -> list[tuple[str, Preference, LoadedModel]]:
    truths = [(name, truth_of(name)) for name in SCENARIOS]
    return [
        (name, rule, truth)
        for name, truth in truths
        for rule in truth.dm.constraints
        if isinstance(rule, Preference)
    ]


def test_in_the_corpus_a_prefer_preference_has_an_empty_effect_and_an_avoid_one_does_not() -> None:
    found = {"prefer": 0, "avoid": 0}
    for name, rule, truth in corpus_preferences():
        effect = effect_pairs(rule, truth.dm, truth.instance)
        found[rule.params.kind] += 1
        assert (effect != frozenset()) is (rule.params.kind == "avoid"), f"{name} {rule.id}"

    assert found == {"prefer": 4, "avoid": 6}


def test_rewards_is_true_only_for_a_prefer_preference() -> None:
    other_templates: dict[str, Constraint] = {}
    for name in SCENARIOS:
        for rule in truth_of(name).dm.constraints:
            if not isinstance(rule, Preference):
                other_templates.setdefault(rule.template, rule)
    avoids = [rule for _, rule, _ in corpus_preferences() if rule.params.kind == "avoid"]
    prefers = [rule for _, rule, _ in corpus_preferences() if rule.params.kind == "prefer"]

    assert prefers and avoids
    assert all(rewards(rule) for rule in prefers)
    assert not any(rewards(rule) for rule in avoids)
    assert len(other_templates) >= 5  # coverage, unavailable, requires_attribute, ... all present
    assert not any(rewards(rule) for rule in other_templates.values())


# --- conditions of D10 and D11 that a reviewer's mutants found unpinned -------------------------


def compare_documents(
    name: str, truth_doc: dict[str, Any], candidate_doc: dict[str, Any]
) -> Comparison:
    directory = CORPUS_PATH / name
    truth = load_model(json.dumps(truth_doc, ensure_ascii=False), directory)
    candidate = load_model(json.dumps(candidate_doc, ensure_ascii=False), directory)
    assert isinstance(truth, LoadedModel) and isinstance(candidate, LoadedModel)
    return compare_models(truth, candidate)


def diagnosed(result: Comparison) -> list[tuple[str, str, str | None]]:
    return [(m.truth_id, m.category, m.closest_candidate_id) for m in result.unmatched_truth]


SICK_LEAVE = "karampetsos-sick-leave"
# The two draft pairs only the sick leave forbids (the other absences forbid the third, see above).
SICK_PAIRS = {("Ε04", "13-Α-Δ"), ("Ε04", "14-Α-Ο")}  # noqa: RUF001


def test_only_the_hard_absences_make_a_draft_pair_irrelevant() -> None:
    # The truth's sick leave is soft here, so F (ADR-0008 D10) loses the 20 pairs it forbade, among
    # them the two draft pairs the candidate omits; no hard rule forbids them any more.
    truth_doc = truth_document(S06)
    constraint_of(truth_doc, SICK_LEAVE).update(hardness="soft", weight=5)
    candidate_doc = without_pairs(truth_doc, DRAFT, lambda r, t: (r, t) not in SICK_PAIRS)
    softened = load_model(json.dumps(truth_doc, ensure_ascii=False), CORPUS_PATH / S06)
    assert isinstance(softened, LoadedModel)
    assert len(forbidden_of(S06)) == 127
    assert len(hard_forbidden_pairs(softened)) == 107
    assert draft_pairs(truth_doc) & (forbidden_of(S06) - hard_forbidden_pairs(softened)) == (
        SICK_PAIRS
    )

    result = compare_documents(S06, truth_doc, candidate_doc)

    assert result.valid
    assert diagnosed(result) == [(DRAFT, "wrong_parameters", DRAFT)]


def test_a_candidate_diagnosed_for_one_truth_rule_is_not_free_for_another() -> None:
    # Both soft absences are `unavailable`: the three-window candidate is the closest of the
    # two-window truth rule, so it cannot also be Stelios's same-effect encoding.
    truth_doc = truth_document(S01)
    truth_doc["constraints"].append(
        soft_off(id="t1-two-windows", params={"windows": afternoon_windows()[:2]})
    )
    candidate_doc = replacing_stelios(soft_off(id="c-three-windows"))

    result = compare_documents(S01, truth_doc, candidate_doc)

    assert diagnosed(result) == [
        (STELIOS, "missing", None),
        ("t1-two-windows", "wrong_parameters", "c-three-windows"),
    ]


MORNING = "πρωί"  # the `shift` value of the morning tasks, quoted verbatim


def test_a_truth_rule_diagnosed_by_template_is_not_reconsidered_for_d11() -> None:
    # Stelios's rule is matched with a preference for the wrong shift (wrong_scope); the soft
    # absence with exactly his effect must not take it over.
    candidate_doc = truth_document(S01)
    stelios = constraint_of(candidate_doc, STELIOS)
    stelios["id"] = "stelios-avoids-mornings"
    stelios["scope"]["tasks"][0]["value"] = MORNING
    candidate_doc["constraints"].append(soft_off(id="c-three-windows"))

    result = compare_documents(S01, truth_document(S01), candidate_doc)

    assert diagnosed(result) == [(STELIOS, "wrong_scope", "stelios-avoids-mornings")]


def test_d11_takes_the_first_equal_effect_candidate_in_candidate_order() -> None:
    candidate_doc = replacing_stelios(soft_off(id="c-first"))
    candidate_doc["constraints"].append(soft_off(id="c-second"))

    result = compare_documents(S01, truth_document(S01), candidate_doc)

    assert diagnosed(result) == [(STELIOS, "other_template_same_effect", "c-first")]
    assert [extra.candidate_id for extra in result.unmatched_candidate] == ["c-first", "c-second"]


def test_the_effect_of_a_rule_is_its_violated_pairs_on_the_everything_plan() -> None:
    truth = load_model((PAIR_RULES / "truth.json").read_text(encoding="utf-8"), PAIR_RULES)
    assert isinstance(truth, LoadedModel)
    rules = {rule.id: rule for rule in truth.dm.constraints}

    skills = effect_pairs(rules["skills"], truth.dm, truth.instance)

    assert skills == hard_forbidden_pairs(truth)
    assert effect_pairs(rules[APART], truth.dm, truth.instance) == frozenset()  # no resource id
