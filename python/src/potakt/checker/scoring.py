"""Scoring a solution against a Decision Model (ADR-0005 §5)."""

from potakt.checker.context import Facts, RuleContext
from potakt.checker.evaluators import EVALUATORS
from potakt.checker.result import CheckResult, RuleResult, Score, Violation
from potakt.model import Constraint, Coverage, DecisionModel, Preference, ProblemInstance, Solution


def check(dm: DecisionModel, instance: ProblemInstance, solution: Solution) -> CheckResult:
    """Score ``solution`` against ``dm`` without trusting whoever produced it.

    The inputs must be valid: ``validate_data`` and ``validate_solution`` report no issues.
    Raises ``ValueError`` if the solution assigns ids that are not in the instance.
    """
    facts = Facts(dm, instance, solution)
    results = tuple(_score_rule(rule, facts.view(rule.scope)) for rule in dm.constraints)
    hard = sum(r.contribution for r in results if r.hardness == "hard")
    soft = sum(r.contribution for r in results if r.hardness == "soft")
    return CheckResult(Score(hard=hard, soft=soft), results)


def _score_rule(rule: Constraint, context: RuleContext) -> RuleResult:
    violations = sorted(EVALUATORS[rule.template](rule, context), key=Violation.sort_key)
    amount = sum(violation.amount for violation in violations)
    weight = None if isinstance(rule, Coverage) else rule.weight
    if weight is None:
        contribution = amount
    else:
        reward = isinstance(rule, Preference) and rule.params.kind == "prefer"
        contribution = (-1 if reward else 1) * weight * amount
    return RuleResult(
        rule_id=rule.id,
        template=rule.template,
        hardness=rule.hardness,
        weight=weight,
        amount=amount,
        contribution=contribution,
        violations=tuple(violations),
    )
