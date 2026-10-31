"""A readable text report of a check result."""

from potakt.checker.result import CheckResult, RuleResult
from potakt.model import DecisionModel


def render_check_result(result: CheckResult, dm: DecisionModel) -> str:
    """Score line, one block per rule with violations, then the clean rules on one line."""
    score = result.score
    feasibility = "feasible" if score.feasible else "infeasible"
    blocks = [
        f"{dm.title} ({dm.id} v{dm.version})\n"
        f"Score: hard {score.hard}, soft {score.soft} ({feasibility})\n"
    ]
    blocks += [_rule_block(rule) for rule in result.rules if rule.amount]
    clean = [rule.rule_id for rule in result.rules if not rule.amount]
    if clean:
        blocks.append(f"Rules without violations: {', '.join(clean)}\n")
    return "\n".join(blocks)


def _rule_block(rule: RuleResult) -> str:
    weight = "" if rule.weight is None else f", weight {rule.weight}"
    header = (
        f"{rule.rule_id} — {rule.template}, {rule.hardness}{weight}: "
        f"amount {rule.amount}, {rule.hardness} {rule.contribution:+d}\n"
    )
    return header + "".join(f"  - {violation.message}\n" for violation in rule.violations)
