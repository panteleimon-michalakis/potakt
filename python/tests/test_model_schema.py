"""The committed JSON Schema must match the one generated from the Pydantic models."""

import json
import re
from pathlib import Path

from potakt.model import decision_model_json_schema, render_json_schema

SCHEMA_FILE = Path(__file__).resolve().parents[2] / "spec" / "decision-model.schema.json"


def test_committed_schema_matches_generated_schema() -> None:
    committed = SCHEMA_FILE.read_text(encoding="utf-8")

    assert committed == render_json_schema(), (
        "spec/decision-model.schema.json is out of date; regenerate it with "
        "`uv run potakt schema --out ../spec/decision-model.schema.json` (from python/)"
    )


def test_schema_declares_dialect_and_lists_every_template() -> None:
    schema = decision_model_json_schema()

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    rendered = json.dumps(schema)
    for template in (
        "coverage",
        "requires_attribute",
        "no_overlap",
        "min_rest",
        "max_per_period",
        "min_per_period",
        "max_consecutive_days",
        "unavailable",
        "preference",
        "balance",
        "together",
        "apart",
        "stability",
    ):
        assert f'"const": "{template}"' in rendered


def test_rendered_schema_is_stable_json_with_trailing_newline() -> None:
    text = render_json_schema()

    assert text.endswith("}\n")
    assert json.loads(text) == decision_model_json_schema()


def test_local_datetimes_are_described_without_utc_offset() -> None:
    window = decision_model_json_schema()["$defs"]["TimeWindow"]["properties"]["start"]

    assert "format" not in window
    pattern = re.compile(window["pattern"])
    for accepted in ("2026-11-02T07:00", "2026-11-02T07:00:00", "2026-11-02 07:00"):
        assert pattern.fullmatch(accepted), accepted
    for rejected in ("2026-11-02T07:00+02:00", "2026-11-02T07:00Z", "2026-11-02T07:00:30"):
        assert not pattern.fullmatch(rejected), rejected
