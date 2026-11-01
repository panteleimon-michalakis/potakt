"""Phase 0 evaluation (ADR-0007): the scenario corpus and the Test A comparison metric.

Library code only: no LLM, no network. Depends on ``potakt.model``, ``potakt.checker``, Pydantic
and the standard library; neither of those core packages imports this one.
"""

from potakt.evaluation.compare import (
    CATEGORIES,
    Category,
    Comparison,
    Denotation,
    InvalidModel,
    LoadedModel,
    MatchedPair,
    UnmatchedCandidate,
    UnmatchedTruth,
    compare_models,
    denote,
    load_model,
)
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
    "CATEGORIES",
    "Category",
    "Comparison",
    "Denotation",
    "Disruption",
    "InvalidModel",
    "LoadedModel",
    "LoadedScenario",
    "MatchedPair",
    "ReferenceScore",
    "Scenario",
    "ScenarioError",
    "ScenarioIssue",
    "UnmatchedCandidate",
    "UnmatchedTruth",
    "UnsupportedRequest",
    "compare_models",
    "denote",
    "load_model",
    "load_scenario",
]
