"""Test A prompt, answer extraction and CSV-only evaluation (ADR-0007 §7; TASK-005c1 D3)."""

import hashlib
import json
import re
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import potakt.evaluation.test_a as test_a_module
from potakt.evaluation import (
    INSTRUCTION_PATH,
    InvalidModel,
    LoadedModel,
    Prompt,
    build_bundle,
    evaluate_candidate,
    extract_candidate,
    load_scenario,
    render_bundle,
    render_test_a,
)
from potakt.model import render_json_schema

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "docs" / "validation" / "scenarios"
SCENARIOS = sorted(path.name for path in CORPUS.iterdir() if path.is_dir())
GOLDEN = Path(__file__).resolve().parent / "golden"
ADR = REPO / "docs" / "adr" / "0007-phase-0-validation-protocol.md"
SCHEMA_HEADING = "## Decision Model JSON Schema\n\n"


def adr_section_7() -> str:
    """ADR-0007 §7 with its whitespace normalised (lines are wrapped anywhere)."""
    text = ADR.read_text(encoding="utf-8")
    section = text[text.index("### 7. Test A harness") : text.index("\n## Consequences")]
    return " ".join(section.split())


def pinned_sha256(pattern: str) -> str:
    found = re.search(pattern + r" \(SHA256 `([0-9a-f]{64})`\)", adr_section_7())
    assert found, f"no SHA256 after {pattern!r} in ADR-0007 §7"
    return found.group(1)


# --- the prompt ------------------------------------------------------------------------------


def test_instruction_path_is_the_committed_prompt_file() -> None:
    assert INSTRUCTION_PATH == REPO / "docs" / "validation" / "prompts" / "test-a.md"


def test_full_user_message_for_02_matches_the_golden_byte_for_byte() -> None:
    prompt = render_test_a(load_scenario(CORPUS / "02-hotel-reception"))

    golden = (GOLDEN / "test-a-02-hotel-reception.user.txt").read_bytes()
    assert prompt.user.encode("utf-8") == golden


def test_system_text_is_the_prompt_file() -> None:
    prompt = render_test_a(load_scenario(CORPUS / "02-hotel-reception"))

    assert prompt.system.encode("utf-8") == INSTRUCTION_PATH.read_bytes()


@pytest.mark.parametrize("name", SCENARIOS)
def test_user_message_is_the_bundle_then_the_schema(name: str) -> None:
    scenario = load_scenario(CORPUS / name)

    prompt = render_test_a(scenario)

    bundle, schema = prompt.user.split(SCHEMA_HEADING)
    assert bundle == render_bundle(build_bundle(scenario)) + "\n"  # one blank line between
    assert schema == render_json_schema()


def test_schema_section_is_the_committed_schema_and_its_sha256_is_pinned_in_the_adr() -> None:
    schema = render_test_a(load_scenario(CORPUS / "01-pharmacy")).user.split(SCHEMA_HEADING)[1]

    data = schema.encode("utf-8")
    assert data == (REPO / "spec" / "decision-model.schema.json").read_bytes()
    assert hashlib.sha256(data).hexdigest() == pinned_sha256(r"output of `potakt schema`")


def test_prompt_file_sha256_is_pinned_in_the_adr() -> None:
    digest = hashlib.sha256(INSTRUCTION_PATH.read_bytes()).hexdigest()

    assert digest == pinned_sha256(r"`docs/validation/prompts/test-a\.md`")


def test_render_is_deterministic() -> None:
    scenario = load_scenario(CORPUS / "06-ambulance-crews")

    assert render_test_a(scenario) == render_test_a(scenario)
    assert isinstance(render_test_a(scenario), Prompt)


# --- answer extraction -----------------------------------------------------------------------

BARE = '{"schema_version": "0.2"}'
TWO_FENCES = f"```json\n{BARE}\n```\n\n```json\n{BARE}\n```"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (BARE, BARE),  # bare JSON
        (f"  \n{BARE}\n", f"  \n{BARE}\n"),  # bare JSON is never changed, whitespace included
        (f"```json\n{BARE}\n```", BARE),
        (f"```\n{BARE}\n```", BARE),
        (f"\n  ```json\n{BARE}\n```  \n", BARE),  # whitespace around the fence is ignored
        ('```json\n{\n  "a": 1\n}\n```', '{\n  "a": 1\n}'),
        (f"Here it is:\n```json\n{BARE}\n```", f"Here it is:\n```json\n{BARE}\n```"),
        (f"```json\n{BARE}\n```\nDone.", f"```json\n{BARE}\n```\nDone."),
        (TWO_FENCES, TWO_FENCES),
        (f"```JSON\n{BARE}\n```", f"```JSON\n{BARE}\n```"),  # only ```json or ``` open a fence
        (f"```json {BARE}```", f"```json {BARE}```"),  # fence lines need their newlines
        ("```\n```", ""),  # an empty fence: empty inside
        ("", ""),
    ],
    ids=[
        "bare",
        "bare-with-whitespace",
        "json-fence",
        "plain-fence",
        "fence-with-surrounding-whitespace",
        "multi-line",
        "text-before",
        "text-after",
        "two-fences",
        "upper-case-language",
        "one-line",
        "empty-fence",
        "empty",
    ],
)
def test_extract_candidate(text: str, expected: str) -> None:
    assert extract_candidate(text) == expected


JSON_VALUES = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(),
    lambda inner: st.lists(inner) | st.dictionaries(st.text(), inner),
    max_leaves=20,
)


@settings(max_examples=200, deadline=None)
@given(st.text())
def test_extract_candidate_never_raises(text: str) -> None:
    assert isinstance(extract_candidate(text), str)


@settings(max_examples=200, deadline=None)
@given(JSON_VALUES, st.sampled_from([None, 2]), st.booleans())
def test_a_json_fence_around_json_text_gives_the_text_back(
    value: object, indent: int | None, ascii_only: bool
) -> None:
    text = json.dumps(value, indent=indent, ensure_ascii=ascii_only)
    if "`" in text:
        return  # the property is stated for JSON text without backticks

    assert extract_candidate("```json\n" + text + "\n```") == text


# --- CSV-only evaluation ---------------------------------------------------------------------

SCENARIO_03 = CORPUS / "03-call-centre"


@pytest.fixture
def private_tempdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Temporary directories go to an empty folder, so the test can see what is left behind."""
    base = tmp_path / "temp"
    base.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(base))
    return base


def truth_text(directory: Path) -> str:
    return (directory / "truth.json").read_text(encoding="utf-8")


def with_resource_source(text: str, source: str) -> str:
    document = json.loads(text)
    document["data"]["resources"]["source"] = source
    return json.dumps(document, ensure_ascii=False)


def test_the_truth_itself_recalls_everything(private_tempdir: Path) -> None:
    result = evaluate_candidate(load_scenario(SCENARIO_03), truth_text(SCENARIO_03))

    assert result.valid
    assert (result.recall, result.precision) == (1.0, 1.0)


@pytest.mark.parametrize("source", ["truth.json", "../truth.json", "request.md", "scenario.json"])
def test_a_candidate_cannot_read_the_scenario_files_other_than_csv(
    source: str, private_tempdir: Path
) -> None:
    # Only the CSV files are copied, so naming any other scenario file as data finds nothing.
    candidate = with_resource_source(truth_text(SCENARIO_03), source)

    result = evaluate_candidate(load_scenario(SCENARIO_03), candidate)

    assert not result.valid
    assert result.recall == 0.0
    assert [issue.path for issue in result.issues] == ["/data/resources/source"]


def test_the_temporary_directory_is_removed(private_tempdir: Path) -> None:
    evaluate_candidate(load_scenario(SCENARIO_03), truth_text(SCENARIO_03))

    assert list(private_tempdir.iterdir()) == []


def test_the_temporary_directory_is_removed_after_an_invalid_candidate(
    private_tempdir: Path,
) -> None:
    result = evaluate_candidate(load_scenario(SCENARIO_03), "not json")

    assert not result.valid
    assert list(private_tempdir.iterdir()) == []


def test_the_candidate_is_validated_in_a_directory_with_the_csv_files_only(
    private_tempdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[set[str]] = []
    real = test_a_module.load_model

    def spy(text: str, directory: Path) -> LoadedModel | InvalidModel:
        seen.append({path.name for path in directory.iterdir()})
        assert directory.parent == private_tempdir
        return real(text, directory)

    monkeypatch.setattr(test_a_module, "load_model", spy)

    evaluate_candidate(load_scenario(SCENARIO_03), truth_text(SCENARIO_03))

    assert seen == [{"agents.csv", "shifts.csv"}]
