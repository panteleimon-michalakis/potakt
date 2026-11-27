"""Test A: the prompt, the candidate taken from an answer and its evaluation (ADR-0007 §7).

Library code only: no LLM, no network, no credentials. The HTTP calls and the run command live
elsewhere (TASK-005c2); this module fixes what they send and how an answer is scored.
"""

import json
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from potakt.evaluation.bundle import build_bundle, render_bundle
from potakt.evaluation.compare import (
    Comparison,
    InvalidModel,
    LoadedModel,
    compare_models,
    load_model,
)
from potakt.evaluation.scenario import LoadedScenario
from potakt.model import (
    DecisionModel,
    DecisionModelError,
    Unavailable,
    ValidationIssue,
    load_instance,
    parse_decision_model,
    render_json_schema,
    validate_data,
)
from potakt.model.lookalike import lookalike
from potakt.model.validation import issue_at

# The instruction text of ADR-0007 §7, resolved from the repository root (src layout:
# python/src/potakt/evaluation/test_a.py). Its SHA256 is pinned in the ADR and by a test.
INSTRUCTION_PATH = (
    Path(__file__).resolve().parents[4] / "docs" / "validation" / "prompts" / "test-a.md"
)
SCHEMA_HEADING = "## Decision Model JSON Schema"
_FENCE = "```"
_OPENERS = ("```json\n", "```\n")
_CLOSER = "\n```"

# The models and settings of ADR-0007 §7 «Models», «Settings». ``REGISTERED_MODELS`` is in the
# run order of the amendment «Order» (re-exported by ``records``, which builds on this module).
REGISTERED_MODELS = ("claude-haiku-5-5", "gemini-3.8-flash", "claude-opus-5-5")
STRONG_MODEL = "claude-opus-5-5"
EFFORT = "high"  # documented for both Claude models (sent to Haiku too, ADR §7)
ANTHROPIC_MAX_TOKENS: dict[str, int] = {"claude-opus-5-5": 64000, "claude-haiku-5-5": 64000}
GEMINI_MAX_OUTPUT_TOKENS = 64000
ANTHROPIC_PREFIX = "claude-"
# ADR-0008 D9: the prefixes of the data-level messages of ``potakt.model.validation`` that name
# an unknown id, and the kind of data id each one names.
UNKNOWN_ID_MESSAGES = (("unknown resource id ", "resource"), ("unknown task id ", "task"))


class UnknownModelError(ValueError):
    """A model id that is not in ``REGISTERED_MODELS``: ADR-0007 §7 fixes no settings for it."""


def check_registered(model: str) -> None:
    """Raise ``UnknownModelError`` unless ``model`` is one of ``REGISTERED_MODELS``."""
    if model not in REGISTERED_MODELS:
        *first, last = REGISTERED_MODELS
        known = f"{', '.join(first)} or {last}"
        raise UnknownModelError(f"no pre-registered Test A settings for {model!r}; use {known}")


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


def request_payload(prompt: Prompt, model: str) -> dict[str, Any]:
    """The JSON body of the one Test A request for ``model`` (ADR-0007 §7), without credentials.

    Only ``REGISTERED_MODELS`` are accepted; any other id raises ``UnknownModelError``. The
    Claude models get the Anthropic Messages API body, ``gemini-3.8-flash`` the Gemini
    ``generateContent`` body, whose model is part of the URL. No tools, no sampling parameters,
    thinking left to the API default.
    """
    check_registered(model)
    if model.startswith(ANTHROPIC_PREFIX):
        return {
            "model": model,
            "max_tokens": ANTHROPIC_MAX_TOKENS[model],
            "system": prompt.system,
            "messages": [{"role": "user", "content": prompt.user}],
            "output_config": {"effort": EFFORT},
        }
    return {
        "systemInstruction": {"parts": [{"text": prompt.system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt.user}]}],
        "generationConfig": {"maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS},
    }


def render_payload(payload: dict[str, Any]) -> str:
    """The payload as indented UTF-8 JSON with sorted keys and a trailing ``\\n``: the exact text
    ``potakt eval prompt`` prints and a run records as ``request.json``."""
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


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
    the truth (ADR-0007 §7 «Validation» and its amendment «Data files»).

    Every ``source`` of a candidate that parses must name one of the scenario's CSV files
    exactly (``source_issues``); otherwise the candidate is invalid and the loader does not run,
    so validity never depends on how the OS resolves a file name. No ``truth.json``,
    ``solution.json``, ``scenario.json`` or ``request.md`` is present where the candidate's
    ``source`` paths resolve, so a candidate cannot read them as data. The temporary directory
    is always removed.
    """
    truth = LoadedModel(scenario.truth, scenario.instance)
    try:
        parsed = parse_decision_model(text)
    except DecisionModelError:
        parsed = None  # the loader reports it below
    if parsed is not None:
        rejected = source_issues(parsed, scenario.scenario.csv_files)
        if rejected:
            return compare_models(truth, InvalidModel(rejected, "data_file"))
    with tempfile.TemporaryDirectory(prefix="potakt-candidate-") as folder:
        directory = Path(folder)
        for name in scenario.scenario.csv_files:
            shutil.copyfile(scenario.directory / name, directory / name)
        candidate = load_model(text, directory)
    return compare_models(truth, candidate)


def replace_lookalike_ids(scenario: LoadedScenario, text: str) -> str | None:
    """The candidate ``text`` with its look-alike ids replaced (ADR-0008 D9, second point), or
    ``None`` if nothing was replaced. Raises ``OSError`` only when the scenario's files cannot
    be copied (for example a full disk; TASK-005e5 decision 2); nothing else raises.

    Only a candidate that reaches data-level validation can change: it parses, every ``source``
    is one of the scenario's files (``source_issues``) and a copy of those files loads, as in
    ``evaluate_candidate``. For each issue of ``validate_data`` whose message starts with a
    prefix of ``UNKNOWN_ID_MESSAGES``, the string its JSON pointer leads to in the parsed
    document is replaced by the one data id of that kind (of the candidate's own bindings) that
    is its look-alike (``lookalike``). An id with no look-alike, or more than one, stays as
    written; so does an id read from a data file: its issue points into the ``source`` binding
    (``.../params/source/resource_id``), at the column name the candidate wrote, which is never
    an id and is never replaced (``_IN_DATA_FILE``). The result is
    ``json.dumps(document, ensure_ascii=False)``.
    """
    try:
        dm = parse_decision_model(text)
    except DecisionModelError:
        return None
    if source_issues(dm, scenario.scenario.csv_files):
        return None
    with tempfile.TemporaryDirectory(prefix="potakt-candidate-") as folder:
        directory = Path(folder)
        for name in scenario.scenario.csv_files:
            shutil.copyfile(scenario.directory / name, directory / name)
        try:
            instance = load_instance(dm, directory)
        except DecisionModelError:
            return None
    known = {
        "resource": frozenset(resource.id for resource in instance.resources),
        "task": frozenset(task.id for task in instance.tasks),
    }
    document: object = json.loads(text)  # parse_decision_model read it as JSON
    replaced = False
    for issue in validate_data(dm, instance):
        if _IN_DATA_FILE in issue.path:  # an id read from a data file (ADR-0008 D9 amendment)
            continue
        for prefix, kind in UNKNOWN_ID_MESSAGES:
            if issue.message.startswith(prefix) and _replace_at(document, issue.path, known[kind]):
                replaced = True
    return json.dumps(document, ensure_ascii=False) if replaced else None


_MISSING = object()  # what a JSON pointer leads to when it leads nowhere
# Where ``validate_data`` reports an unknown id read from a data file: an ``unavailable`` rule's
# ``source`` binding, whose ``resource_id`` holds a column name, not an id.
_IN_DATA_FILE = "/params/source/"


def _child(node: object, token: str) -> object:
    """The member ``token`` of an object, or the element ``token`` of an array (RFC 6901: ASCII
    decimal digits without a leading zero, below the length); ``_MISSING`` otherwise."""
    if isinstance(node, dict):
        return cast(dict[str, object], node).get(token, _MISSING)
    if isinstance(node, list):
        items = cast(list[object], node)
        if _is_index(token) and int(token) < len(items):
            return items[int(token)]
    return _MISSING


def _is_index(token: str) -> bool:
    return token.isascii() and token.isdigit() and (token == "0" or not token.startswith("0"))


def _replace_at(document: object, pointer: str, known: frozenset[str]) -> bool:
    """Replace the string that the JSON pointer ``pointer`` leads to in ``document`` by its one
    look-alike in ``known``; ``True`` if it was replaced. A pointer that leads nowhere, to the
    root or to a value that is not a string replaces nothing.

    RFC 6901: the pointer is split on ``/``, then each token is decoded, ``~1`` to ``/`` first
    and then ``~0`` to ``~``."""
    if not pointer.startswith("/"):
        return False
    tokens = [token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/")]
    parent = document
    for token in tokens[:-1]:
        parent = _child(parent, token)
    last = tokens[-1]
    value = _child(parent, last)
    if not isinstance(value, str):
        return False
    match = lookalike(value, known)
    if match is None or match == value:
        return False
    if isinstance(parent, dict):
        cast(dict[str, object], parent)[last] = match
    else:
        cast(list[object], parent)[int(last)] = match
    return True


def source_issues(dm: DecisionModel, csv_files: Sequence[str]) -> tuple[ValidationIssue, ...]:
    """One issue per ``source`` of ``dm`` that is not exactly one of ``csv_files``.

    A source is accepted iff its POSIX path parts are exactly one file name, so ``x.csv``,
    ``./x.csv`` and ``x.csv/`` are accepted, while another letter case, a backslash, a trailing
    dot or space and a subdirectory are not. Checked in order: the resources binding, the tasks
    binding, then each ``unavailable`` rule's window source in constraint order.
    """
    names = ", ".join(sorted(csv_files, key=lambda name: name.encode("utf-8")))
    sources: list[tuple[tuple[str | int, ...], str]] = [
        (("data", "resources", "source"), dm.data.resources.source),
        (("data", "tasks", "source"), dm.data.tasks.source),
    ]
    sources += [
        (("constraints", index, "params", "source", "source"), rule.params.source.source)
        for index, rule in enumerate(dm.constraints)
        if isinstance(rule, Unavailable) and rule.params.source is not None
    ]
    allowed = {(name,) for name in csv_files}
    return tuple(
        issue_at(location, f'"{source}" is not one of the scenario\'s data files: {names}')
        for location, source in sources
        if PurePosixPath(source).parts not in allowed
    )
