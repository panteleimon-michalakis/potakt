"""Phase 0 evaluation (ADR-0007): the scenario corpus and the Test A comparison metric.

Library code only: no LLM, no network. Depends on ``potakt.model``, ``potakt.checker``, Pydantic
and the standard library; neither of those core packages imports this one.
"""

from potakt.evaluation.bundle import Bundle, BundleFile, build_bundle, render_bundle
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
    SCENARIO_TAGS,
    Disruption,
    LoadedScenario,
    ReferenceScore,
    Scenario,
    ScenarioError,
    ScenarioIssue,
    ScenarioTag,
    UnsupportedRequest,
    load_scenario,
)
from potakt.evaluation.summary import (
    CORPUS_PATH,
    PRICES_USD_PER_MILLION,
    Summary,
    model_directory_name,
    summarize,
)
from potakt.evaluation.test_a import (
    INSTRUCTION_PATH,
    STRONG_MODEL,
    Prompt,
    UnknownModelError,
    evaluate_candidate,
    extract_candidate,
    render_payload,
    render_test_a,
    request_payload,
)
from potakt.evaluation.vacuity import is_vacuous

__all__ = [
    "CATEGORIES",
    "CORPUS_PATH",
    "INSTRUCTION_PATH",
    "PRICES_USD_PER_MILLION",
    "SCENARIO_TAGS",
    "STRONG_MODEL",
    "Bundle",
    "BundleFile",
    "Category",
    "Comparison",
    "Denotation",
    "Disruption",
    "InvalidModel",
    "LoadedModel",
    "LoadedScenario",
    "MatchedPair",
    "Prompt",
    "ReferenceScore",
    "Scenario",
    "ScenarioError",
    "ScenarioIssue",
    "ScenarioTag",
    "Summary",
    "UnknownModelError",
    "UnmatchedCandidate",
    "UnmatchedTruth",
    "UnsupportedRequest",
    "build_bundle",
    "compare_models",
    "denote",
    "evaluate_candidate",
    "extract_candidate",
    "is_vacuous",
    "load_model",
    "load_scenario",
    "model_directory_name",
    "render_bundle",
    "render_payload",
    "render_test_a",
    "request_payload",
    "summarize",
]
