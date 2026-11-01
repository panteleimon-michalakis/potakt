"""Phase 0 evaluation (ADR-0007): the scenario corpus and the Test A comparison metric.

Library code only: no LLM, no network. Depends on ``potakt.model``, ``potakt.checker``, Pydantic
and the standard library; neither of those core packages imports this one.
"""

from potakt.evaluation.scenario import (
    Disruption,
    LoadedScenario,
    ReferenceScore,
    Scenario,
    ScenarioError,
    ScenarioIssue,
    UnsupportedRequest,
    load_scenario,
)

__all__ = [
    "Disruption",
    "LoadedScenario",
    "ReferenceScore",
    "Scenario",
    "ScenarioError",
    "ScenarioIssue",
    "UnsupportedRequest",
    "load_scenario",
]
