"""The structured result of a check: score, per-rule totals and violations (ADR-0005 §5).

Frozen standard-library dataclasses rather than Pydantic models: the checker may depend only on
``potakt.model`` and the standard library, and a result is trusted output that needs no
validation. ``to_dict`` / ``to_json`` give the stable JSON form used by later tools and evals.
"""

import json
from dataclasses import dataclass
from typing import Any

from potakt.model.common import Hardness


@dataclass(frozen=True, slots=True)
class Violation:
    """One violation of one rule; ``amount`` is a positive integer."""

    resource_id: str | None
    task_ids: tuple[str, ...]
    period: str | None
    amount: int
    message: str

    def sort_key(self) -> tuple[str, str, tuple[str, ...]]:
        """Resource id (none first), period, task ids."""
        return (self.resource_id or "", self.period or "", self.task_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "task_ids": list(self.task_ids),
            "period": self.period,
            "amount": self.amount,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class RuleResult:
    """The total of one rule: ``contribution`` goes to ``hard`` or ``soft`` by ``hardness``."""

    rule_id: str
    template: str
    hardness: Hardness
    weight: int | None
    amount: int
    contribution: int
    violations: tuple[Violation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "template": self.template,
            "hardness": self.hardness,
            "weight": self.weight,
            "amount": self.amount,
            "contribution": self.contribution,
            "violations": [violation.to_dict() for violation in self.violations],
        }


@dataclass(frozen=True, slots=True, order=True)
class Score:
    """``(hard, soft)``, compared lexicographically: lower hard wins, then lower soft."""

    hard: int
    soft: int

    @property
    def feasible(self) -> bool:
        return self.hard == 0

    def to_dict(self) -> dict[str, Any]:
        return {"hard": self.hard, "soft": self.soft, "feasible": self.feasible}


@dataclass(frozen=True, slots=True)
class CheckResult:
    """The score of a solution and the result of every rule, in DM order."""

    score: Score
    rules: tuple[RuleResult, ...]

    def rule(self, rule_id: str) -> RuleResult:
        for result in self.rules:
            if result.rule_id == rule_id:
                return result
        raise KeyError(rule_id)

    def to_dict(self) -> dict[str, Any]:
        return {"score": self.score.to_dict(), "rules": [r.to_dict() for r in self.rules]}

    def to_json(self) -> str:
        """Indented UTF-8 JSON with a trailing newline; identical inputs give identical text."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"
