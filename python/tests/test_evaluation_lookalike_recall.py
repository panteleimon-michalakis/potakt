"""Replacing look-alike ids in an invalid candidate (ADR-0008 D9, second point; TASK-005e4).

The corpus truths write ids in Greek capitals that look exactly like Latin letters, and some in
Latin letters only. A candidate that writes one with the other alphabet is invalid at the data
stage; ``replace_lookalike_ids`` puts back the one data id of the same kind that is its
look-alike, so the summary can report the recall after that replacement without changing the gate.
"""

import copy
import functools
import json
from typing import Any, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from potakt.evaluation import (
    CORPUS_PATH,
    UNKNOWN_ID_MESSAGES,
    LoadedScenario,
    evaluate_candidate,
    load_scenario,
    replace_lookalike_ids,
)
from potakt.evaluation.test_a import _replace_at  # pyright: ignore[reportPrivateUsage]
from potakt.model.lookalike import skeleton

# (scenario, pointer in truth.json, the id as a candidate writes it). The written ids quote Greek
# capitals verbatim, which ruff reads as Latin look-alikes (RUF001); each line names the letter.
ROWS = (
    ("02-hotel-reception", "/constraints/7/params/windows/0/resource_id", "Y5"),  # Latin Y
    ("03-call-centre", "/constraints/8/params/resource_ids/1", "Α04"),  # noqa: RUF001 Greek Alpha
    ("04-restaurant", "/constraints/6/params/resource_ids/0", "Κ1"),  # noqa: RUF001 Greek Kappa
    ("05-supermarket-checkouts", "/constraints/6/params/resource_ids/1", "T2"),  # Latin T
    ("06-ambulance-crews", "/constraints/8/params/baseline/0/task_id", "09-Π-O"),  # Latin O
    ("06-ambulance-crews", "/constraints/8/params/baseline/0/resource_id", "E03"),  # Latin E
)
SUPERMARKET = "05-supermarket-checkouts"
APART_IDS = "/constraints/6/params/resource_ids"  # 05's apart rule
TAU1 = "\N{GREEK CAPITAL LETTER TAU}1"
# Every Greek capital with a Latin look-alike and that Latin letter, both ways.
TWINS = {
    greek: skeleton(greek) for greek in map(chr, range(0x391, 0x3AA)) if skeleton(greek) != greek
}
TWINS |= {latin: greek for greek, latin in TWINS.items()}


@functools.cache
def scenario(name: str) -> LoadedScenario:
    return load_scenario(CORPUS_PATH / name)


def truth(name: str) -> dict[str, Any]:
    return json.loads((CORPUS_PATH / name / "truth.json").read_text(encoding="utf-8"))


def tokens(pointer: str) -> list[str | int]:
    """The parts of a plain pointer of this file (no escapes), array indices as ints."""
    return [int(part) if part.isdigit() else part for part in pointer[1:].split("/")]


def get_at(document: Any, pointer: str) -> Any:
    for part in tokens(pointer):
        document = document[part]
    return document


def set_at(document: Any, pointer: str, value: object) -> None:
    *parents, last = tokens(pointer)
    for part in parents:
        document = document[part]
    document[last] = value


def written(name: str, *changes: tuple[str, object]) -> str:
    """The truth of ``name`` with each (pointer, value) set, as candidate text."""
    document = truth(name)
    for pointer, value in changes:
        set_at(document, pointer, value)
    return json.dumps(document, ensure_ascii=False)


def test_the_prefixes_are_those_of_the_data_level_messages() -> None:
    assert UNKNOWN_ID_MESSAGES == (
        ("unknown resource id ", "resource"),
        ("unknown task id ", "task"),
    )


@pytest.mark.parametrize(("name", "pointer", "as_written"), ROWS)
def test_a_lookalike_id_is_replaced_by_the_truths_id(
    name: str, pointer: str, as_written: str
) -> None:
    text = written(name, (pointer, as_written))
    before = evaluate_candidate(scenario(name), text)
    assert not before.valid
    assert before.invalid_stage == "data"
    assert get_at(truth(name), pointer) != as_written

    replaced = replace_lookalike_ids(scenario(name), text)

    assert replaced is not None
    assert json.loads(replaced) == truth(name)
    after = evaluate_candidate(scenario(name), replaced)
    assert after.valid
    assert after.matched_count == after.truth_count


@pytest.mark.parametrize(
    "case",
    [
        (),  # the truth: valid, nothing to replace
        "not json",
        "[]",
        '{"id": "x"}',
        ((f"{APART_IDS}/1", "Z9"),),  # no look-alike
        (("/data/resources/source", "other.csv"),),  # not a scenario file
        # The staff file as the tasks: loading fails (stage data_file).
        (("/data/tasks/source", "tameia_prosopiko.csv"),),
    ],
)
def test_a_candidate_with_nothing_to_replace_gives_none(
    case: str | tuple[tuple[str, object], ...],
) -> None:
    # Raw text, or the changes made to 05's truth (read here, never at import).
    text = case if isinstance(case, str) else written(SUPERMARKET, *case)

    assert replace_lookalike_ids(scenario(SUPERMARKET), text) is None


def test_the_data_file_case_fails_at_loading() -> None:
    text = written(SUPERMARKET, ("/data/tasks/source", "tameia_prosopiko.csv"))
    assert evaluate_candidate(scenario(SUPERMARKET), text).invalid_stage == "data_file"


def test_an_id_without_a_lookalike_stays_as_written() -> None:
    text = written(SUPERMARKET, (APART_IDS, ["T1", "Z9"]))

    replaced = replace_lookalike_ids(scenario(SUPERMARKET), text)

    assert replaced is not None
    assert get_at(json.loads(replaced), APART_IDS) == [TAU1, "Z9"]
    assert not evaluate_candidate(scenario(SUPERMARKET), replaced).valid


def test_every_replaceable_id_of_a_candidate_is_replaced_in_one_call() -> None:
    name = "06-ambulance-crews"
    text = written(name, *((pointer, value) for row, pointer, value in ROWS if row == name))

    replaced = replace_lookalike_ids(scenario(name), text)

    assert replaced is not None
    assert json.loads(replaced) == truth(name)


@st.composite
def flipped(draw: st.DrawFn) -> tuple[str, str, str]:
    """A row of ``ROWS`` with one look-alike letter of the truth's id flipped to its twin."""
    name, pointer, _ = draw(st.sampled_from(ROWS))
    original = cast(str, get_at(truth(name), pointer))
    positions = [index for index, letter in enumerate(original) if letter in TWINS]
    index = draw(st.sampled_from(positions))
    changed = original[:index] + TWINS[original[index]] + original[index + 1 :]
    return name, pointer, changed


@settings(deadline=None, max_examples=40)
@given(flipped())
def test_any_flipped_lookalike_letter_is_put_back(case: tuple[str, str, str]) -> None:
    name, pointer, changed = case

    replaced = replace_lookalike_ids(scenario(name), written(name, (pointer, changed)))

    assert replaced is not None
    assert json.loads(replaced) == truth(name)


# --- the JSON pointer helper (RFC 6901) ---------------------------------------------------------

DOCUMENT = {
    "a": [TAU1],
    "b/c": TAU1,
    "d~e": TAU1,
    "f": {"source": "x.csv"},
    "g~1h": TAU1,
    "g/h": {"not": "a string"},
}


@pytest.mark.parametrize(
    ("pointer", "path"),
    [("/a/0", ("a", 0)), ("/b~1c", ("b/c",)), ("/d~0e", ("d~e",)), ("/g~01h", ("g~1h",))],
)
def test_a_pointer_to_a_lookalike_string_replaces_it(
    pointer: str, path: tuple[str | int, ...]
) -> None:
    document = copy.deepcopy(DOCUMENT)

    assert _replace_at(document, pointer, frozenset({"T1"}))

    node: Any = document
    for part in path:
        node = node[part]
    assert node == "T1"
    assert document != DOCUMENT


@pytest.mark.parametrize(
    "pointer",
    [
        "/a/00",
        "/a/5",
        "/a/-1",
        "/a/\N{ARABIC-INDIC DIGIT ZERO}",  # a decimal digit, but not ASCII
        "/f",
        "/a/0/x",
        "/a/0/x/y",
        "/missing",
        "",
        "/",
    ],
)
def test_a_pointer_that_leads_to_no_string_replaces_nothing(pointer: str) -> None:
    document = copy.deepcopy(DOCUMENT)

    assert not _replace_at(document, pointer, frozenset({"T1"}))

    assert document == DOCUMENT
