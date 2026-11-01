"""Test A: the prompt, the candidate taken from an answer and its evaluation (ADR-0007 §7).

Library code only: no LLM, no network, no credentials. The HTTP calls and the run command live
elsewhere (TASK-005c2); this module fixes what they send and how an answer is scored.
"""

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from potakt.evaluation.bundle import build_bundle, render_bundle
from potakt.evaluation.compare import Comparison, LoadedModel, compare_models, load_model
from potakt.evaluation.scenario import LoadedScenario
from potakt.model import render_json_schema

# The instruction text of ADR-0007 §7, resolved from the repository root (src layout:
# python/src/potakt/evaluation/test_a.py). Its SHA256 is pinned in the ADR and by a test.
INSTRUCTION_PATH = (
    Path(__file__).resolve().parents[4] / "docs" / "validation" / "prompts" / "test-a.md"
)
SCHEMA_HEADING = "## Decision Model JSON Schema"
_FENCE = "```"
_OPENERS = ("```json\n", "```\n")
_CLOSER = "\n```"


@dataclass(frozen=True, slots=True)
class Prompt:
    """The system instruction and the user message of one Test A request."""

    system: str
    user: str


def render_test_a(scenario: LoadedScenario) -> Prompt:
    """The Test A prompt of one scenario (ADR-0007 §7 «Prompt»).

    The system text is the prompt file, decoded as UTF-8 and otherwise unchanged. The user
    message is the rendered bundle, one blank line, the schema heading, one blank line and the
    exact text ``potakt schema`` prints.
    """
    system = INSTRUCTION_PATH.read_bytes().decode("utf-8")
    bundle = render_bundle(build_bundle(scenario))
    user = f"{bundle}\n{SCHEMA_HEADING}\n\n{render_json_schema()}"
    return Prompt(system, user)


def extract_candidate(text: str) -> str:
    """The candidate DM text of an answer (ADR-0007 §7 «Answer»); never raises.

    If the text, leading and trailing whitespace ignored, is exactly one Markdown code fence
    (an opening line ```` ```json ```` or ```` ``` ````, a closing line ```` ``` ```` and no
    other fence line inside), the inside is returned. Otherwise the text is returned unchanged,
    so any text around or between fences makes the candidate invalid.
    """
    stripped = text.strip()
    for opener in _OPENERS:
        if stripped.startswith(opener) and stripped.endswith(_CLOSER):
            inside = stripped[len(opener) : -len(_CLOSER)]  # "" when the two lines touch
            if any(line.startswith(_FENCE) for line in inside.split("\n")):
                return text  # more than one fence
            return inside
    return text


def evaluate_candidate(scenario: LoadedScenario, text: str) -> Comparison:
    """Validate a candidate against a copy of the scenario's CSV files only and compare it with
    the truth (ADR-0007 §7 «Validation»).

    No ``truth.json``, ``solution.json``, ``scenario.json`` or ``request.md`` is present where the
    candidate's ``source`` paths resolve, so a candidate cannot read them as data. The temporary
    directory is always removed.
    """
    truth = LoadedModel(scenario.truth, scenario.instance)
    with tempfile.TemporaryDirectory(prefix="potakt-candidate-") as folder:
        directory = Path(folder)
        for name in scenario.scenario.csv_files:
            shutil.copyfile(scenario.directory / name, directory / name)
        candidate = load_model(text, directory)
    return compare_models(truth, candidate)
