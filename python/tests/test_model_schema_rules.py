"""The committed JSON Schema against the validator (TASK-005e1, ADR-0008 D2).

The schema is what a model sees (ADR-0007 §7 «Prompt»), the validator is the reference. The
differential test checks, with an independent draft 2020-12 implementation (``jsonschema``), that
the schema accepts every Decision Model the validator accepts in the repository and every valid
mutation of the corpus truths, so that showing a rule in the schema never changes the language.
"""

import json
import re
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Literal, cast, get_args

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from jsonschema import Draft202012Validator
from message_templates import HOLE, collect_templates, template_pattern

from potakt.evaluation import CORPUS_PATH
from potakt.evaluation.test_a import source_issues
from potakt.model import (
    DecisionModelError,
    load_instance,
    parse_decision_model,
    render_json_schema,
    validate_data,
)
from potakt.model.catalog import _HardOrSoft
from potakt.model.common import RESERVED_ATTRIBUTE_NAMES

PYTHON = Path(__file__).resolve().parents[1]
REPO = PYTHON.parent
SCHEMA_FILE = REPO / "spec" / "decision-model.schema.json"
SCHEMA: dict[str, Any] = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
VALIDATOR = Draft202012Validator(SCHEMA)
FIXTURES = PYTHON / "tests" / "fixtures"

# Templates that take either hardness, read from the catalog (``_HardOrSoft`` subclasses).
HARD_OR_SOFT = frozenset(
    get_args(cls.model_fields["template"].annotation)[0] for cls in _HardOrSoft.__subclasses__()
)

Document = dict[str, Any]


def _parses(text: str) -> bool:
    try:
        parse_decision_model(text)
    except DecisionModelError:
        return False
    return True


def _truth_paths() -> list[Path]:
    return sorted(CORPUS_PATH.glob("*/truth.json"))


def _fixture_models() -> list[Path]:
    """Every file under ``tests/fixtures`` whose text is a Decision Model the validator accepts."""
    candidates = sorted(
        path for path in FIXTURES.rglob("*") if path.is_file() and path.suffix in (".json", ".txt")
    )
    return [path for path in candidates if _parses(path.read_text(encoding="utf-8-sig"))]


def repository_models() -> list[Path]:
    """The ten corpus truths, both examples and every DM fixture that parses."""
    return [*_truth_paths(), *sorted(REPO.glob("examples/*/model.json")), *_fixture_models()]


def schema_errors(document: Any) -> list[str]:
    """The schema's error messages for ``document``, each with its JSON path; empty if valid."""
    return [
        f"{error.json_path}: {error.message}"
        for error in sorted(VALIDATOR.iter_errors(document), key=lambda e: e.json_path)
    ]


def test_the_committed_schema_is_a_valid_draft_2020_12_schema() -> None:
    assert SCHEMA["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    Draft202012Validator.check_schema(SCHEMA)


def test_the_repository_has_the_models_the_differential_test_needs() -> None:
    names = [path.relative_to(REPO).as_posix() for path in repository_models()]

    assert len(_truth_paths()) == 10
    assert "examples/clinic-el/model.json" in names
    assert "examples/retail-en/model.json" in names
    assert len(_fixture_models()) >= 1, "no DM fixture parses: the fixture part would be vacuous"


def test_the_hard_or_soft_templates_are_the_catalog_ones() -> None:
    expected = {
        "requires_attribute",
        "no_overlap",
        "min_rest",
        "max_per_period",
        "min_per_period",
        "max_consecutive_days",
        "unavailable",
        "apart",
    }

    assert sorted(HARD_OR_SOFT) == sorted(expected)


def test_the_schema_accepts_every_model_the_validator_accepts_in_the_repository() -> None:
    rejected = {}
    for path in repository_models():
        text = path.read_text(encoding="utf-8-sig")
        parse_decision_model(text)  # each one is valid for the validator
        errors = schema_errors(json.loads(text))
        if errors:
            rejected[path.relative_to(REPO).as_posix()] = errors

    assert rejected == {}


TRUTHS: list[Document] = [json.loads(path.read_text(encoding="utf-8")) for path in _truth_paths()]


@st.composite
def valid_mutations(draw: st.DrawFn) -> Document:
    """A corpus truth with some non-coverage rules dropped, the rules shuffled and some
    hard-or-soft rules flipped (hard: weight removed; soft: weight 1-10000)."""
    truth = cast("Document", json.loads(json.dumps(draw(st.sampled_from(TRUTHS)))))
    rules: list[Document] = truth["constraints"]
    kept = [
        rule
        for rule in rules
        if rule["template"] == "coverage" or not draw(st.booleans(), label="drop")
    ]
    kept = draw(st.permutations(kept), label="order")
    for rule in kept:
        if rule["template"] in HARD_OR_SOFT and draw(st.booleans(), label="flip"):
            if rule["hardness"] == "hard":
                rule["hardness"] = "soft"
                rule["weight"] = draw(st.integers(1, 10_000), label="weight")
            else:
                rule["hardness"] = "hard"
                rule.pop("weight", None)
    truth["constraints"] = kept
    return truth


@settings(deadline=None, max_examples=200)
@given(document=valid_mutations())
def test_the_schema_accepts_every_valid_mutation_of_the_truths(document: Document) -> None:
    # ``assume``: Hypothesis fails the test if too few mutations parse, so it is never vacuous.
    assume(_parses(json.dumps(document, ensure_ascii=False)))

    assert schema_errors(document) == []


# --- the inventory of every validation rule (decision 2) --------------------------------------

SOURCES = [
    *sorted((PYTHON / "src" / "potakt" / "model").glob("*.py")),
    PYTHON / "src" / "potakt" / "evaluation" / "test_a.py",
]
INSTRUCTION = REPO / "docs" / "validation" / "prompts" / "test-a.md"
# The pointer of a rule stated by the Test A instruction rather than by the schema.
INSTRUCTION_POINTER = "docs/validation/prompts/test-a.md"


@dataclass(frozen=True)
class Slip:
    """A real model with one rule broken, and a substring of the validator's message.

    ``files`` are extra CSV files written next to the scenario's; with ``test_a`` the sources are
    checked against the scenario's data files (``source_issues``) instead of loading the data."""

    scenario: str
    mutate: Callable[[Document], object]
    message: str
    files: Mapping[str, str] = field(default_factory=dict[str, str])
    test_a: bool = False


@dataclass(frozen=True)
class Rule:
    """One validation rule, shown by a schema keyword (S) or a description (D), or not shown (X,
    with its ``reason``). ``pointer`` is where the committed schema shows it; a D rule's
    description there contains ``phrase``."""

    id: str
    kind: Literal["S", "D", "X"]
    pointer: str | None = None
    phrase: str | None = None
    slips: tuple[Slip, ...] = ()
    reason: str | None = None


def constraint(document: Document, rule_id: str) -> Document:
    return next(rule for rule in document["constraints"] if rule["id"] == rule_id)


def params(document: Document, rule_id: str) -> Document:
    return constraint(document, rule_id)["params"]


def resources(document: Document) -> Document:
    return document["data"]["resources"]


def tasks(document: Document) -> Document:
    return document["data"]["tasks"]


def put(target: Document, key: str, value: object) -> None:
    target[key] = value


def drop_coverage(document: Document) -> None:
    document["constraints"] = [r for r in document["constraints"] if r["template"] != "coverage"]


def add_coverage(document: Document) -> None:
    document["constraints"].append({**constraint(document, "coverage"), "id": "coverage-again"})


def add_window(rule_id: str, start: str, end: str) -> Callable[[Document], None]:
    def mutate(document: Document) -> None:
        windows = params(document, rule_id)["windows"]
        windows.append({**windows[0], "start": start, "end": end})

    return mutate


def scope_task(rule_id: str, condition: Document) -> Callable[[Document], None]:
    def mutate(document: Document) -> None:
        scope = constraint(document, rule_id).setdefault("scope", {})
        scope.setdefault("tasks", []).append(condition)

    return mutate


def rebind_resource_id(column: str) -> Callable[[Document], None]:
    """Another id column; the rules that name resource ids are dropped, so only the id rule
    breaks."""

    def mutate(document: Document) -> None:
        document["constraints"] = [
            r for r in document["constraints"] if r["template"] not in ("together", "apart")
        ]
        resources(document)["id"] = column

    return mutate


def duplicate_first_baseline_entry(document: Document) -> None:
    baseline = params(document, "keep-the-circulated-draft")["baseline"]
    baseline.append(dict(baseline[0]))


C01, C02, C03, C04, C06 = (
    "01-pharmacy",
    "02-hotel-reception",
    "03-call-centre",
    "04-restaurant",
    "06-ambulance-crews",
)
DATA_BINDINGS = "/$defs/DataBindings"

# The texts of decision 3, verbatim.
CONSTRAINTS_TEXT = (
    "The rules of the model. Exactly one constraint has `template` `coverage`. Every constraint "
    "`id` is unique within the model."
)
ASSUMPTIONS_TEXT = "Every assumption `id` is unique within the model."
PARENT_VERSION_TEXT = "When given, lower than `version`."
TIMEZONE_TEXT = (
    "An IANA time zone key exactly as the tz database writes it (case-sensitive), e.g. "
    "`Europe/Athens`. Every local date-time of the model and of its data files is in this zone; "
    "a local time skipped by a daylight-saving change is invalid."
)
WEIGHT_TEXT = (
    "Required (an integer from 1 to 10000) when `hardness` is `soft`; must be omitted or null "
    "when `hardness` is `hard`."
)
QUOTE_TEXT = "Required when `kind` is `utterance`: the words of the request the rule comes from."
SOURCE_TEXT = (
    "Required when `kind` is `data`: where in the data the rule comes from (free text, e.g. a "
    "file and a column)."
)
ATTRIBUTE_TEXT = (
    "`id`, or the name of an attribute bound under `data.resources.attributes` (in resource "
    "conditions) or `data.tasks.attributes` (in task conditions). Task conditions may also use "
    "the derived attributes of each task's local start: `weekday` (`mon`, `tue`, `wed`, `thu`, "
    "`fri`, `sat`, `sun`), `start_hour` (an integer 0\N{EN DASH}23), `duration_hours` (a number) "
    "and `date` "
    "(`YYYY-MM-DD`). A condition on an attribute that a resource or task has no value for never "
    "holds, also with `ne` and `not_in`."
)
VALUE_TEXT = (
    "A list for `in` and `not_in`; a single value (not a list) for `contains`, which holds when a "
    "`list` attribute contains it; a number or a string for `gte` and `lte`. Ids are written "
    "exactly as in the data: a condition on an id that does not exist selects nothing with `eq` "
    "and everything with `ne`."
)
ATTRIBUTES_TEXT = (
    "Keys are attribute names: lower-case snake_case (`^[a-z][a-z0-9_]*$`), none of `id`, "
    "`start`, `end`, `demand`, `weekday`, `start_hour`, `duration_hours`, `date` (reserved). The "
    "CSV column itself may have any name (`column`)."
)
SEPARATOR_TEXT = "Only for `type` `list`: the separator between the items of one cell. Default `;`."
TYPE_TEXT = (
    "How every non-empty cell of the column is read: `str` as is; `int` a whole number (digits "
    "with an optional sign); `float` a number written with the file's `decimal` separator (with "
    "decimal `,` a `.` is rejected, never read as a thousands separator); `bool` one of "
    "`true`/`false`, `yes`/`no`, `1`/`0` in any letter case; `list` items split by `separator`. "
    "An empty cell means no value. A cell that cannot be read this way makes the model invalid."
)
DECIMAL_TEXT = "`,` needs a `delimiter` other than `,`."
SOURCE_PATH_TEXT = (
    "A path relative to the model's folder, e.g. `staff.csv`; it may not leave that folder."
)
TASK_COLUMNS_TEXT = (
    "`start` and `end` cells are ISO-8601 local date-times without a UTC offset, in whole "
    "minutes (e.g. `2026-11-02T07:00` or `2026-11-02 07:00`); `end` is after `start`, at most "
    "24 h later, and every task starts inside the horizon. `demand` cells are whole numbers from "
    "1 to 10000. `id` values are unique; the file has at least one task."
)
RESOURCE_COLUMNS_TEXT = "`id` values are unique; the file has at least one resource."
WINDOW_COLUMNS_TEXT = (
    "`start` and `end` cells as in the tasks file; `resource_id` cells are resource ids from the "
    "data, written exactly as there."
)
WINDOW_TEXT = (
    "`end` is after `start`; `resource_id` is a resource id from the data, written exactly as "
    "there."
)
UNAVAILABLE_TEXT = "At least one of `windows` (non-empty) or `source`."
PAIR_TEXT = "Two different resource ids from the data, written exactly as there."
BASELINE_TEXT = (
    "Pairs of a task id and a resource id from the data, written exactly as there; each pair at "
    "most once."
)
REQUIRES_TEXT = (
    "`task_attribute` and `resource_attribute` name attributes bound in `data.tasks.attributes` "
    "and `data.resources.attributes`."
)
LIMIT_TEXT = (
    'A number, or `{"attribute": …}` naming a resource attribute bound as `int` or `float` that '
    "every in-scope resource has. Limits are at least 0; with `measure` `count` whole numbers up "
    "to 100000, with `measure` `hours` up to 8784 in whole minutes."
)
HORIZON_TEXT = "`end` is not before `start`."
# Texts of the extra rows found by the AST test (for the reviewer's ruling).
HOURS_TEXT = "Hours, more than 0 and in whole minutes, e.g. `11` or `7.5` (not `7.501`)."
CSV_FILE_TEXT = (
    "Every `source` (here and in `unavailable` rules) names an existing CSV file in the model's "
    "folder"
)
CSV_ENCODING_TEXT = (
    "written in the declared `encoding` (a UTF-8 byte order mark is allowed only with `utf-8`)"
)
CSV_HEADER_TEXT = (
    "its first row is the header. Each column that a binding names must appear exactly once in "
    "the header (names and cells are trimmed); a row may not have more fields than the header."
)
REQUIRED_CELLS_TEXT = (
    "Cells of the `id`, `start`, `end`, `demand` and `resource_id` columns must not be empty."
)
TEST_A_SOURCE_TEXT = "by their file names"
DATETIME_BOUNDS_TEXT = "from 1970-01-01T00:00 to 3000-01-01T00:00"

HARD_OR_SOFT_DEFS = (
    "Apart",
    "MaxConsecutiveDays",
    "MaxPerPeriod",
    "MinPerPeriod",
    "MinRest",
    "NoOverlap",
    "RequiresAttribute",
    "Unavailable",
)

# Every text of decision 3 at every place it is attached.
TEXTS: tuple[tuple[str, str], ...] = (
    ("/properties/constraints", CONSTRAINTS_TEXT),
    ("/properties/assumptions", ASSUMPTIONS_TEXT),
    ("/properties/parent_version", PARENT_VERSION_TEXT),
    ("/properties/timezone", TIMEZONE_TEXT),
    *((f"/$defs/{name}/properties/weight", WEIGHT_TEXT) for name in HARD_OR_SOFT_DEFS),
    ("/$defs/Provenance/properties/quote", QUOTE_TEXT),
    ("/$defs/Provenance/properties/source", SOURCE_TEXT),
    ("/$defs/Condition/properties/attribute", ATTRIBUTE_TEXT),
    ("/$defs/Condition/properties/value", VALUE_TEXT),
    ("/$defs/ResourceBinding/properties/attributes", ATTRIBUTES_TEXT),
    ("/$defs/TaskBinding/properties/attributes", ATTRIBUTES_TEXT),
    ("/$defs/AttributeBinding/properties/separator", SEPARATOR_TEXT),
    ("/$defs/AttributeBinding/properties/type", TYPE_TEXT),
    ("/$defs/CsvDialect/properties/decimal", DECIMAL_TEXT),
    ("/$defs/ResourceBinding/properties/source", SOURCE_PATH_TEXT),
    ("/$defs/TaskBinding/properties/source", SOURCE_PATH_TEXT),
    ("/$defs/WindowsBinding/properties/source", SOURCE_PATH_TEXT),
    ("/$defs/TaskBinding", TASK_COLUMNS_TEXT),
    ("/$defs/ResourceBinding", RESOURCE_COLUMNS_TEXT),
    ("/$defs/WindowsBinding", WINDOW_COLUMNS_TEXT),
    ("/$defs/TimeWindow", WINDOW_TEXT),
    ("/$defs/UnavailableParams", UNAVAILABLE_TEXT),
    ("/$defs/ResourcePairParams/properties/resource_ids", PAIR_TEXT),
    ("/$defs/StabilityParams/properties/baseline", BASELINE_TEXT),
    ("/$defs/RequiresAttributeParams", REQUIRES_TEXT),
    ("/$defs/PeriodLimitParams/properties/limit", LIMIT_TEXT),
    ("/$defs/Horizon", HORIZON_TEXT),
)

RULES: tuple[Rule, ...] = (
    Rule(
        "one-coverage",
        "S",
        "/properties/constraints/contains",
        slips=(
            Slip(C01, drop_coverage, "exactly one 'coverage' constraint is required"),
            Slip(C01, add_coverage, "exactly one 'coverage' constraint is allowed"),
        ),
    ),
    Rule(
        "unique-constraint-ids",
        "D",
        "/properties/constraints",
        "Every constraint `id` is unique within the model.",
        (
            Slip(
                C03,
                lambda d: put(d["constraints"][2], "id", d["constraints"][1]["id"]),
                "duplicate constraint id 'no-parallel-shifts' (first used at /constraints/1)",
            ),
        ),
    ),
    Rule(
        "unique-assumption-ids",
        "D",
        "/properties/assumptions",
        ASSUMPTIONS_TEXT,
        (
            Slip(
                C03,
                lambda d: put(d["assumptions"][1], "id", d["assumptions"][0]["id"]),
                "duplicate assumption id",
            ),
        ),
    ),
    Rule(
        "parent-version",
        "D",
        "/properties/parent_version",
        PARENT_VERSION_TEXT,
        (Slip(C03, lambda d: put(d, "parent_version", 1), "must be lower than version (1)"),),
    ),
    Rule(
        "timezone",
        "D",
        "/properties/timezone",
        TIMEZONE_TEXT,
        (
            Slip(C03, lambda d: put(d, "timezone", "America/new_york"), "unknown IANA timezone"),
            Slip(
                C01,
                add_window("ioanna-leave", "2026-03-29T03:30", "2026-03-29T05:00"),
                "local time 2026-03-29T03:30 does not exist in Europe/Athens (DST change)",
            ),
        ),
    ),
    Rule(
        "weight-iff-soft",
        "S",
        "/$defs/MinRest/if",
        slips=(
            Slip(
                C03,
                lambda d: put(constraint(d, "no-parallel-shifts"), "weight", 5),
                "weight is only allowed on soft rules",
            ),
            Slip(
                C03,
                lambda d: constraint(d, "rest-12h").pop("weight"),
                "a soft rule needs a weight",
            ),
        ),
    ),
    Rule(
        "provenance-quote",
        "S",
        "/$defs/Provenance/allOf/0",
        slips=(
            Slip(
                C03,
                lambda d: put(constraint(d, "rest-12h"), "provenance", {"kind": "utterance"}),
                "quote is required when kind is 'utterance'",
            ),
        ),
    ),
    Rule(
        "provenance-source",
        "S",
        "/$defs/Provenance/allOf/1",
        slips=(
            Slip(
                C03,
                lambda d: put(constraint(d, "rest-12h"), "provenance", {"kind": "data"}),
                "source is required when kind is 'data'",
            ),
        ),
    ),
    Rule(
        "condition-attribute",
        "D",
        "/$defs/Condition/properties/attribute",
        ATTRIBUTE_TEXT,
        (
            Slip(
                C04,
                scope_task("fair-weekends", {"attribute": "start", "op": "eq", "value": "x"}),
                "unknown task attribute 'start'",
            ),
        ),
    ),
    Rule(
        "condition-value-shape",
        "S",
        "/$defs/Condition/allOf",
        slips=(
            Slip(
                C03,
                scope_task(
                    "jamal-early", {"attribute": "shift_type", "op": "contains", "value": ["early"]}
                ),
                "value must be a single value for op 'contains'",
            ),
            Slip(
                C03,
                scope_task(
                    "jamal-early", {"attribute": "shift_type", "op": "in", "value": "early"}
                ),
                "value must be a list for op 'in'",
            ),
            Slip(
                C03,
                scope_task("jamal-early", {"attribute": "start_hour", "op": "gte", "value": True}),
                "value must be a number or a string for op 'gte'",
            ),
        ),
    ),
    Rule(
        "attribute-names",
        "S",
        "/$defs/ResourceBinding/properties/attributes/propertyNames",
        slips=(
            Slip(
                C03,
                lambda d: put(
                    resources(d)["attributes"], "date", {"column": "name", "type": "str"}
                ),
                "attribute name 'date' is reserved",
            ),
            Slip(
                C03,
                lambda d: put(tasks(d)["attributes"], "Shift", {"column": "end", "type": "str"}),
                "attribute name 'Shift' must be snake_case",
            ),
        ),
    ),
    Rule(
        "separator-only-list",
        "S",
        "/$defs/AttributeBinding/if",
        slips=(
            Slip(
                C03,
                lambda d: put(resources(d)["attributes"]["name"], "separator", ";"),
                "separator is only allowed for 'list' columns",
            ),
        ),
    ),
    # A default, not a check: no message (02 without `separator` stays valid, see below).
    Rule(
        "list-separator-default",
        "D",
        "/$defs/AttributeBinding/properties/separator",
        "Default `;`.",
    ),
    Rule(
        "cell-types",
        "D",
        "/$defs/AttributeBinding/properties/type",
        TYPE_TEXT,
        tuple(
            Slip(
                C03,
                lambda d, kind=kind: put(resources(d)["attributes"]["name"], "type", kind),
                f"'Maya Johnson' is not {what}",
            )
            for kind, what in (("int", "an integer"), ("bool", "a boolean"), ("float", "a number"))
        ),
    ),
    Rule(
        "decimal-delimiter",
        "S",
        "/$defs/CsvDialect/if",
        slips=(
            Slip(
                C03,
                lambda d: put(resources(d), "csv", {"decimal": ","}),
                "decimal ',' needs a delimiter other than ','",
            ),
        ),
    ),
    Rule(
        "source-path",
        "D",
        "/$defs/ResourceBinding/properties/source",
        SOURCE_PATH_TEXT,
        (
            Slip(
                C03,
                lambda d: put(resources(d), "source", "../agents.csv"),
                "must be a relative path inside the model directory",
            ),
        ),
    ),
    Rule(
        "task-columns",
        "D",
        "/$defs/TaskBinding",
        TASK_COLUMNS_TEXT,
        (
            Slip(
                C03,
                lambda d: put(d["horizon"], "start", "2026-12-08"),
                "starts on 2026-12-07, outside the horizon 2026-12-08 to 2026-12-13",
            ),
            Slip(
                C03,
                lambda d: put(tasks(d), "start", "shift_type"),
                "'early' is not an ISO-8601 local datetime",
            ),
            Slip(
                C03,
                lambda d: put(tasks(d), "end", "start"),
                "end 2026-12-07T07:00 must be after start 2026-12-07T07:00",
            ),
        ),
    ),
    Rule(
        "resource-columns",
        "D",
        "/$defs/ResourceBinding",
        RESOURCE_COLUMNS_TEXT,
        (Slip(C03, rebind_resource_id("team"), "duplicate resource id 'billing' in agents.csv"),),
    ),
    Rule(
        "window-columns",
        "D",
        "/$defs/WindowsBinding",
        WINDOW_COLUMNS_TEXT,
        (
            Slip(
                C04,
                lambda d: put(params(d, "approved-time-off")["source"], "resource_id", "reason"),
                "unknown resource id 'holiday' in time_off.csv",
            ),
        ),
    ),
    Rule(
        "window",
        "D",
        "/$defs/TimeWindow",
        WINDOW_TEXT,
        (
            Slip(
                C01,
                add_window("ioanna-leave", "2026-11-19T00:00", "2026-11-19T00:00"),
                "end must be after start",
            ),
        ),
    ),
    Rule(
        "unavailable-not-empty",
        "S",
        "/$defs/UnavailableParams/anyOf",
        slips=tuple(
            Slip(
                C01,
                lambda d, value=value: put(constraint(d, "ioanna-leave"), "params", value),
                "at least one of 'windows' or 'source' is required",
            )
            for value in ({}, {"windows": []}, {"windows": [], "source": None})
        ),
    ),
    Rule(
        "pair-ids",
        "D",
        "/$defs/ResourcePairParams/properties/resource_ids",
        PAIR_TEXT,
        (
            Slip(
                C03,
                lambda d: put(params(d, "leo-shadows-priya"), "resource_ids", ["A03", "A03"]),
                "the two resource ids must be different",
            ),
            Slip(
                C03,
                lambda d: put(params(d, "leo-shadows-priya"), "resource_ids", ["A03", "Z99"]),
                "unknown resource id 'Z99'",
            ),
        ),
    ),
    Rule(
        "baseline",
        "D",
        "/$defs/StabilityParams/properties/baseline",
        BASELINE_TEXT,
        (
            Slip(
                C06,
                lambda d: put(
                    params(d, "keep-the-circulated-draft")["baseline"][0], "task_id", "no-such"
                ),
                "unknown task id 'no-such'",
            ),
            Slip(C06, duplicate_first_baseline_entry, "duplicate baseline assignment"),
        ),
    ),
    Rule(
        "requires-attribute-names",
        "D",
        "/$defs/RequiresAttributeParams",
        REQUIRES_TEXT,
        (
            Slip(
                C03,
                lambda d: put(params(d, "late-needs-billing"), "task_attribute", "skill"),
                "unknown task attribute 'skill'",
            ),
        ),
    ),
    Rule(
        "period-limit",
        "D",
        "/$defs/PeriodLimitParams/properties/limit",
        LIMIT_TEXT,
        (
            Slip(
                C01,
                lambda d: put(params(d, "one-shift-per-day"), "limit", 1.5),
                "limit must be a whole number when measure is 'count'",
            ),
            Slip(
                C03,
                lambda d: put(params(d, "weekly-max-hours"), "limit", {"attribute": "contract"}),
                "limit attribute 'contract' is bound as 'str'",
            ),
            Slip(
                C01,
                lambda d: put(params(d, "assistants-max-25h"), "limit", 7.501),
                "7.501 hours is not a whole number of minutes",
            ),
        ),
    ),
    Rule(
        "period-limit-maximum",
        "S",
        "/$defs/PeriodLimitParams/allOf",
        slips=(
            Slip(
                C01,
                lambda d: put(params(d, "one-shift-per-day"), "limit", 100001),
                "limit 100001 is above the maximum of 100000",
            ),
            Slip(
                C01,
                lambda d: put(params(d, "assistants-max-25h"), "limit", 8785),
                "limit 8785 hours is above the maximum of 8784 (366 days)",
            ),
        ),
    ),
    Rule(
        "limit-shape",
        "S",
        "/$defs/PeriodLimitParams/properties/limit/oneOf",
        slips=(
            Slip(
                C01,
                lambda d: put(params(d, "one-shift-per-day"), "limit", "1"),
                'limit must be a number (at least 0) or {"attribute": "<resource attribute>"}',
            ),
        ),
    ),
    Rule(
        "hardness-per-template",
        "S",
        "/$defs/Preference/properties/hardness",
        slips=(
            Slip(
                C01,
                lambda d: put(constraint(d, "stelios-avoids-afternoons"), "hardness", "hard"),
                "template 'preference' allows only soft rules",
            ),
        ),
    ),
    Rule(
        "hours-whole-minutes",
        "D",
        "/$defs/MinRestParams/properties/hours",
        HOURS_TEXT,
        (
            Slip(
                C03,
                lambda d: put(params(d, "rest-12h"), "hours", 7.501),
                "7.501 hours is not a whole number of minutes",
            ),
        ),
    ),
    Rule(
        "local-datetime",
        "S",
        "/$defs/TimeWindow/properties/start/pattern",
        slips=(
            Slip(
                C01,
                add_window("ioanna-leave", "2026-11-19T00:00:30", "2026-11-20T00:00"),
                "must be a whole minute (no seconds)",
            ),
            Slip(
                C01,
                add_window("ioanna-leave", "2026-11-19T00:00+02:00", "2026-11-20T00:00"),
                "must be a local time without a UTC offset (the DM timezone applies)",
            ),
        ),
    ),
    Rule(
        "datetime-bounds",
        "D",
        "/$defs/TimeWindow/properties/start",
        DATETIME_BOUNDS_TEXT,
        (
            Slip(
                C01,
                add_window("ioanna-leave", "1969-12-31T00:00", "2026-11-20T00:00"),
                "1969-12-31T00:00 is before the minimum of 1970-01-01T00:00",
            ),
            Slip(
                C01,
                add_window("ioanna-leave", "2026-11-20T00:00", "3000-01-01T00:01"),
                "3000-01-01T00:01 is after the maximum of 3000-01-01T00:00",
            ),
        ),
    ),
    Rule(
        "horizon",
        "D",
        "/$defs/Horizon",
        HORIZON_TEXT,
        (
            Slip(
                C03,
                lambda d: put(d["horizon"], "end", "2026-12-06"),
                "horizon end 2026-12-06 is before its start 2026-12-07",
            ),
        ),
    ),
    Rule(
        "csv-file",
        "D",
        DATA_BINDINGS,
        CSV_FILE_TEXT,
        (
            Slip(
                C03,
                lambda d: put(resources(d), "source", "missing.csv"),
                "cannot read missing.csv: file not found",
            ),
            Slip(
                C03,
                lambda d: put(resources(d), "source", "nul.csv"),
                "cannot read nul.csv: reserved device name",
            ),
        ),
    ),
    Rule(
        "csv-encoding",
        "D",
        DATA_BINDINGS,
        CSV_ENCODING_TEXT,
        (
            Slip(
                C02,
                lambda d: put(resources(d), "csv", {**resources(d)["csv"], "encoding": "utf-8"}),
                "not valid utf-8",
            ),
        ),
    ),
    Rule(
        "csv-header",
        "D",
        DATA_BINDINGS,
        CSV_HEADER_TEXT,
        (
            Slip(
                C03,
                lambda d: put(resources(d), "id", "agent"),
                "column 'agent' not found in agents.csv",
            ),
            Slip(
                C03,
                lambda d: put(resources(d), "source", "empty.csv"),
                "empty.csv has no header row",
                files={"empty.csv": ""},
            ),
        ),
    ),
    Rule(
        "required-cells",
        "D",
        DATA_BINDINGS,
        REQUIRED_CELLS_TEXT,
        (
            Slip(
                C03,
                lambda d: put(tasks(d), "id", "skill_required"),
                "column 'skill_required': a value is required",
            ),
        ),
    ),
    Rule(
        "test-a-source-file",
        "D",
        INSTRUCTION_POINTER,
        TEST_A_SOURCE_TEXT,
        (
            Slip(
                C03,
                lambda d: put(resources(d), "source", "data/agents.csv"),
                '"data/agents.csv" is not one of the scenario\'s data files: '
                "agents.csv, shifts.csv",
                test_a=True,
            ),
        ),
    ),
    Rule(
        "json-syntax",
        "X",
        reason="not a schema rule: a text that is not JSON has no schema to check (the "
        "instruction asks for a single JSON object)",
    ),
    Rule(
        "json-depth",
        "X",
        reason="internal bound: JSON nested more than 64 levels (a Decision Model has fewer "
        "than 10)",
    ),
    Rule(
        "json-number",
        "X",
        reason="internal bound: an integer longer than Python's int() accepts (thousands of "
        "digits)",
    ),
    Rule(
        "file-read",
        "X",
        reason="the operating system could not read a file (permissions, I/O): not a property of "
        "the model's text",
    ),
    Rule(
        "hours-overflow",
        "X",
        reason="internal guard: hours whose minutes overflow a float; unreachable after the "
        "maximum checks (8784 hours)",
    ),
    Rule("solution", "X", reason="solution input, not part of a Decision Model"),
    Rule(
        "lookalike-hint",
        "X",
        reason="a hint appended to an unknown-id message (ADR-0008 D9), not a rule of its own",
    ),
    Rule("wrapper", "X", reason="wrapper: passes another rule's message on"),
)
RULE_BY_ID = {rule.id: rule for rule in RULES}

MESSAGE_TO_RULE: dict[str, str] = {
    '"{}" is not one of the scenario\'s data files: {}': "test-a-source-file",
    "JSON nested more than {} levels deep": "json-depth",
    "JSON number too large": "json-number",
    "a soft rule needs a weight (integer ≥ 1)": "weight-iff-soft",
    "a value is required": "required-cells",
    "at least one of 'windows' or 'source' is required": "unavailable-not-empty",
    "attribute name {} is reserved": "attribute-names",
    "attribute name {} must be snake_case": "attribute-names",
    "cannot read {}: file not found": "csv-file",
    "cannot read {}: not a regular file": "csv-file",
    "cannot read {}: not valid {}": "csv-encoding",
    "cannot read {}: reserved device name": "csv-file",
    "cannot read {}: {}": "file-read",
    "column {} appears more than once in {}": "csv-header",
    "column {} not found in {}{}": "csv-header",
    "decimal ',' needs a delimiter other than ','": "decimal-delimiter",
    "demand must be at least 1, got {}": "task-columns",
    "demand {} is above the maximum of {}": "task-columns",
    "duplicate assignment of resource {} to task {} (first at /assignments/{})": "solution",
    (
        "duplicate baseline assignment of resource {} to task {} "
        "(first at /constraints/{}/params/baseline/{})"
    ): "baseline",
    "duplicate {} id {} (first used at /{}/{})": "unique-constraint-ids",
    "duplicate {} id {} in {}": "resource-columns",
    "end must be after start": "window",
    "exactly one 'coverage' constraint is allowed; another one is at /constraints/{}": (
        "one-coverage"
    ),
    "exactly one 'coverage' constraint is required, found none": "one-coverage",
    "horizon end {} is before its start {}": "horizon",
    "horizon is {} days long; at most {} days are allowed": "horizon",
    "invalid JSON at line {}, column {}: {}": "json-syntax",
    "limit attribute {} is bound as {}; it must be bound as 'int' or 'float'": "period-limit",
    'limit must be a number (at least 0) or {"attribute": "<resource attribute>"}': "limit-shape",
    "limit must be a whole number when measure is 'count'": "period-limit",
    "limit {}": "period-limit-maximum",
    "must be a local time without a UTC offset (the DM timezone applies)": "local-datetime",
    "must be a whole minute (no seconds)": "local-datetime",
    "parent_version ({}) must be lower than version ({})": "parent-version",
    "quote is required when kind is 'utterance'": "provenance-quote",
    "resource {} has no {}; every resource in the rule's scope needs a limit value": "period-limit",
    "resource {}: {} {}": "period-limit",
    "resource {}: {} {} is negative; a limit must be at least 0": "period-limit",
    "resource {}: {} {} is not a number": "period-limit",
    "resource {}: {} {} is not a whole number (measure 'count')": "period-limit",
    "separator is only allowed for 'list' columns": "separator-only-list",
    "solution is for model version {}, not {}": "solution",
    "solution is for model {}, not {}": "solution",
    "source is required when kind is 'data'": "provenance-source",
    "source {} must be a relative path inside the model directory": "source-path",
    "template {} allows only {} rules": "hardness-per-template",
    "the two resource ids must be different": "pair-ids",
    "unknown IANA timezone {}": "timezone",
    "unknown resource id {}{}": "solution",
    "unknown resource id {}{}{}": "pair-ids",
    "unknown task id {}{}": "baseline",
    "; did you mean {}? ({})": "lookalike-hint",
    "unknown {} attribute {}; known: {}": "condition-attribute",
    "value must be {} for op {}": "condition-value-shape",
    "weight is only allowed on soft rules": "weight-iff-soft",
    "{message}": "wrapper",
    "{}": "wrapper",
    "{} contains no {}s; at least one is required": "resource-columns",
    "{} has no header row": "csv-header",
    "{} hours is above the maximum of {} (366 days)": "period-limit-maximum",
    "{} hours is not a whole number of minutes": "hours-whole-minutes",
    "{} hours is too large": "hours-overflow",
    "{} is above the maximum of {}": "period-limit-maximum",
    "{} is after the maximum of {}": "datetime-bounds",
    "{} is before the minimum of {}": "datetime-bounds",
    "{} is not a boolean (use true/false, yes/no or 1/0)": "cell-types",
    "{} is not a finite number": "cell-types",
    "{} is not a number{}": "cell-types",
    "{} is not an ISO-8601 local datetime": "task-columns",
    "{} is not an integer": "cell-types",
    "{} line {}, column {}: {}": "wrapper",
    "{} line {}: more fields than the header": "csv-header",
    "{} line {}: {}": "wrapper",
    "{} resolves outside the model directory": "source-path",
    '{} starts with a UTF-8 byte order mark; declare "encoding": "utf-8"': "csv-encoding",
    "{} {}": "wrapper",
    "{}: end {} must be after start {}": "task-columns",
    "{}: lasts {} h, more than the maximum of 24 h": "task-columns",
    "{}: local time {} does not exist in {} (DST change)": "timezone",
    "{}: starts on {}, outside the horizon {} to {}": "task-columns",
}

# The run-time loop: the templates a slip's message may match (rules S and D), and the prefixes
# of the wrappers that pass a message on (``<file> line <n>, column <c>: <message>``).
LIVE_PATTERNS = tuple(
    template_pattern(template)
    for template, rule in MESSAGE_TO_RULE.items()
    if RULE_BY_ID[rule].kind != "X"
)
WRAPPER_PREFIXES = tuple(
    template_pattern(template.removesuffix(HOLE))
    for template, rule in MESSAGE_TO_RULE.items()
    if rule == "wrapper" and template.endswith(HOLE) and template != HOLE
)


@cache
def matches_a_template(message: str) -> bool:
    """``message`` is one of a rule's templates, possibly passed on by a wrapper, or several
    messages joined by ``"; "`` (Pydantic merges the messages of one path)."""
    if any(pattern.fullmatch(message) for pattern in LIVE_PATTERNS):
        return True
    for prefix in WRAPPER_PREFIXES:
        for end in range(1, len(message)):
            if prefix.fullmatch(message[:end]) and matches_a_template(message[end:]):
                return True
    parts = message.split("; ")
    return len(parts) > 1 and all(matches_a_template(part) for part in parts)


def resolve(pointer: str) -> Any:
    """The node of the committed schema at a JSON pointer (RFC 6901)."""
    node: Any = SCHEMA
    for token in pointer.split("/")[1:]:
        key = token.replace("~1", "/").replace("~0", "~")
        node = node[int(key)] if isinstance(node, list) else node[key]
    return node


def validator_messages(document: Document, slip: Slip, folder: Path) -> list[str]:
    """What the validator says about ``document``: structure (``parse_decision_model``), then
    the Test A data files (``source_issues``) or the data (``load_instance``, ``validate_data``)
    of the slip's scenario, copied into ``folder``."""
    scenario = CORPUS_PATH / slip.scenario
    try:
        dm = parse_decision_model(json.dumps(document, ensure_ascii=False))
    except DecisionModelError as error:
        return [issue.message for issue in error.issues]
    csv_files = sorted(path.name for path in scenario.glob("*.csv"))
    if slip.test_a:
        return [issue.message for issue in source_issues(dm, csv_files)]
    for name in csv_files:
        shutil.copyfile(scenario / name, folder / name)
    for name, text in slip.files.items():
        (folder / name).write_text(text, encoding="utf-8")
    try:
        instance = load_instance(dm, folder)
    except DecisionModelError as error:
        return [issue.message for issue in error.issues]
    return [issue.message for issue in validate_data(dm, instance)]


def truth(scenario: str) -> Document:
    return json.loads((CORPUS_PATH / scenario / "truth.json").read_text(encoding="utf-8"))


SLIPS = [(rule, slip) for rule in RULES for slip in rule.slips]
SLIP_IDS = [f"{rule.id}-{k}" for rule in RULES for k, _ in enumerate(rule.slips)]
TEMPLATES = collect_templates(SOURCES)


def test_every_message_template_has_a_rule_and_no_row_is_stale() -> None:
    unmapped = sorted(TEMPLATES - MESSAGE_TO_RULE.keys())
    stale = sorted(MESSAGE_TO_RULE.keys() - TEMPLATES)

    assert unmapped == [], "a validation message without a RULES row"
    assert stale == [], "a MESSAGE_TO_RULE template that no longer exists in the code"
    assert set(MESSAGE_TO_RULE.values()) <= RULE_BY_ID.keys()


def test_the_collector_finds_messages_of_every_kind_of_sink() -> None:
    # One witness per collection rule of decision 2: (a) argument, (b) local name, (c) module
    # constant, (d) returned by a function; and the nested return of ``above_limit_maximum``.
    for template in (
        "end must be after start",
        "limit must be a whole number when measure is 'count'",
        "weight is only allowed on soft rules",
        "unknown {} attribute {}; known: {}",
        "{} is above the maximum of {}",
        '"{}" is not one of the scenario\'s data files: {}',
    ):
        assert template in TEMPLATES, template


def test_every_rule_row_is_well_formed() -> None:
    assert len(RULE_BY_ID) == len(RULES), "duplicate rule id"
    targets = set(MESSAGE_TO_RULE.values())
    for rule in RULES:
        assert re.fullmatch("[a-z0-9]+(-[a-z0-9]+)*", rule.id), rule.id
        if rule.kind == "X":
            assert rule.reason and rule.pointer is None and not rule.slips, rule.id
            assert rule.id in targets, f"{rule.id}: an X row without a message"
            continue
        assert rule.pointer is not None and rule.reason is None, rule.id
        assert (rule.kind == "D") == (rule.phrase is not None), rule.id
        if rule.id != "list-separator-default":  # a default: no message to break
            assert rule.slips, f"{rule.id}: no slip"


@pytest.mark.parametrize(("rule", "slip"), SLIPS, ids=SLIP_IDS)
def test_each_slip_breaks_its_rule(rule: Rule, slip: Slip, tmp_path: Path) -> None:
    document = truth(slip.scenario)
    assert validator_messages(document, slip, tmp_path) == [], "the unchanged truth is valid"
    slip.mutate(document)

    messages = validator_messages(document, slip, tmp_path)

    assert any(slip.message in message for message in messages), messages
    assert [message for message in messages if not matches_a_template(message)] == []
    if rule.kind == "S":
        assert schema_errors(document) != [], "the schema accepts a slip of an S rule"


@pytest.mark.parametrize("rule", [r for r in RULES if r.kind == "S"], ids=lambda r: r.id)
def test_each_s_rule_points_at_its_keyword(rule: Rule) -> None:
    assert rule.pointer is not None
    assert resolve(rule.pointer) is not None


@pytest.mark.parametrize("rule", [r for r in RULES if r.kind == "D"], ids=lambda r: r.id)
def test_each_d_rule_is_stated_where_it_points(rule: Rule) -> None:
    assert rule.pointer is not None and rule.phrase is not None
    if rule.pointer == INSTRUCTION_POINTER:
        stated = INSTRUCTION.read_text(encoding="utf-8")
    else:
        stated = resolve(rule.pointer)["description"]

    assert rule.phrase in stated


@pytest.mark.parametrize(("pointer", "text"), TEXTS, ids=[p for p, _ in TEXTS])
def test_the_texts_of_decision_3_are_in_the_schema_verbatim(pointer: str, text: str) -> None:
    printed = json.loads(render_json_schema())  # what `potakt schema` prints
    node: Any = printed
    for token in pointer.split("/")[1:]:
        node = node[token]

    assert text in node["description"]


WEIGHT_IFF_SOFT = {
    "if": {"properties": {"hardness": {"const": "soft"}}},
    "then": {"required": ["weight"], "properties": {"weight": {"type": "integer"}}},
    "else": {"properties": {"weight": {"type": "null"}}},
}


def test_every_hard_or_soft_template_carries_the_weight_rule() -> None:
    defs = SCHEMA["$defs"]
    names = sorted(
        name
        for name, definition in defs.items()
        if definition.get("properties", {}).get("template", {}).get("const") in HARD_OR_SOFT
    )

    assert names == sorted(HARD_OR_SOFT_DEFS)
    for name in names:
        assert {key: defs[name][key] for key in WEIGHT_IFF_SOFT} == WEIGHT_IFF_SOFT, name


def test_the_schema_reserves_exactly_the_reserved_attribute_names() -> None:
    names = resolve("/$defs/ResourceBinding/properties/attributes/propertyNames/not/enum")

    assert sorted(names) == sorted(RESERVED_ATTRIBUTE_NAMES)
    assert len(names) == len(set(names))


def test_02_without_a_separator_stays_valid_with_the_default(tmp_path: Path) -> None:
    # The reviewer's probe: the default `;` is described, not changed; 02's list is `,`-separated.
    document = truth(C02)
    languages = next(
        binding
        for binding in resources(document)["attributes"].values()
        if binding["type"] == "list"
    )
    languages.pop("separator")

    assert validator_messages(document, Slip(C02, lambda _: None, ""), tmp_path) == []
    assert schema_errors(document) == []
