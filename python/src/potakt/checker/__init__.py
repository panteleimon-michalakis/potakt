"""Independent checker (DESIGN §8.3): scores any solution against a Decision Model.

It trusts no solver and is the reference semantics of the catalog (ADR-0005). Depends only on
``potakt.model`` and the standard library.
"""

from potakt.checker.evaluators import EVALUATORS
from potakt.checker.report import render_check_result
from potakt.checker.result import CheckResult, RuleResult, Score, Violation
from potakt.checker.scoring import check

__all__ = [
    "EVALUATORS",
    "CheckResult",
    "RuleResult",
    "Score",
    "Violation",
    "check",
    "render_check_result",
]
