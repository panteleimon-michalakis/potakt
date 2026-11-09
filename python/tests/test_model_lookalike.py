"""Look-alike ids in unknown-id messages (TASK-005e1 decision 5, ADR-0008 D9).

Greek letters are written as ``\\N{...}`` escapes, so no line quotes ambiguous characters.
"""

import json
import shutil
from pathlib import Path
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from potakt.evaluation import CORPUS_PATH
from potakt.model import (
    ProblemInstance,
    load_instance,
    parse_decision_model,
    parse_solution,
    validate_data,
    validate_solution,
)
from potakt.model.decision_model import DecisionModel
from potakt.model.lookalike import lookalike, lookalike_hint, skeleton

ALPHA = "\N{GREEK CAPITAL LETTER ALPHA}"
BETA = "\N{GREEK CAPITAL LETTER BETA}"
EPSILON = "\N{GREEK CAPITAL LETTER EPSILON}"
NU = "\N{GREEK CAPITAL LETTER NU}"
OMICRON = "\N{GREEK CAPITAL LETTER OMICRON}"
PI = "\N{GREEK CAPITAL LETTER PI}"
TAU = "\N{GREEK CAPITAL LETTER TAU}"
GREEK = (
    "\N{GREEK CAPITAL LETTER ALPHA}\N{GREEK CAPITAL LETTER BETA}"
    "\N{GREEK CAPITAL LETTER EPSILON}\N{GREEK CAPITAL LETTER ZETA}"
    "\N{GREEK CAPITAL LETTER ETA}\N{GREEK CAPITAL LETTER IOTA}"
    "\N{GREEK CAPITAL LETTER KAPPA}\N{GREEK CAPITAL LETTER MU}"
    "\N{GREEK CAPITAL LETTER NU}\N{GREEK CAPITAL LETTER OMICRON}"
    "\N{GREEK CAPITAL LETTER RHO}\N{GREEK CAPITAL LETTER TAU}"
    "\N{GREEK CAPITAL LETTER UPSILON}\N{GREEK CAPITAL LETTER CHI}"
)
LATIN = "ABEZHIKMNOPTYX"
E_FOR_EPSILON = "LATIN CAPITAL LETTER E where the data has GREEK CAPITAL LETTER EPSILON"

Document = dict[str, Any]


def test_the_skeleton_maps_exactly_the_fourteen_greek_capitals() -> None:
    code_points = [f"U+{ord(letter):04X}" for letter in GREEK]

    assert code_points == [
        "U+0391",
        "U+0392",
        "U+0395",
        "U+0396",
        "U+0397",
        "U+0399",
        "U+039A",
        "U+039C",
        "U+039D",
        "U+039F",
        "U+03A1",
        "U+03A4",
        "U+03A5",
        "U+03A7",
    ]
    assert skeleton(GREEK) == LATIN
    assert skeleton(LATIN) == LATIN
    # Greek letters without a Latin look-alike, and small letters, are unchanged.
    other = (
        "\N{GREEK CAPITAL LETTER GAMMA}\N{GREEK CAPITAL LETTER DELTA}\N{GREEK SMALL LETTER ALPHA}"
    )
    assert skeleton(other) == other


def test_a_latin_letter_where_the_data_has_a_greek_one() -> None:
    known = {f"{EPSILON}03", f"{EPSILON}04"}

    assert lookalike("E03", known) == f"{EPSILON}03"
    assert lookalike_hint("E03", known) == f"; did you mean '{EPSILON}03'? ({E_FOR_EPSILON})"


def test_a_greek_letter_where_the_data_has_a_latin_one() -> None:
    assert lookalike(f"{EPSILON}03", {"E03"}) == "E03"
    assert lookalike_hint(f"{EPSILON}03", {"E03"}) == (
        "; did you mean 'E03'? "
        "(GREEK CAPITAL LETTER EPSILON where the data has LATIN CAPITAL LETTER E)"
    )


def test_two_letters_in_one_id_are_both_named() -> None:
    known = {f"09-{PI}-{OMICRON}", f"{ALPHA}{NU}-1"}

    assert lookalike_hint("AN-1", known) == (
        f"; did you mean '{ALPHA}{NU}-1'? (LATIN CAPITAL LETTER A where the data has GREEK "
        "CAPITAL LETTER ALPHA, LATIN CAPITAL LETTER N where the data has GREEK CAPITAL LETTER NU)"
    )


def test_an_ambiguous_skeleton_gives_no_hint() -> None:
    known = {f"{ALPHA}B", f"A{BETA}"}  # both have the skeleton "AB"

    assert lookalike("AB", known) is None
    assert lookalike_hint("AB", known) == ""


def test_an_id_unknown_in_every_alphabet_gives_no_hint() -> None:
    known = {f"{EPSILON}03", "A01"}

    assert lookalike("Z99", known) is None
    assert lookalike_hint("Z99", known) == ""


IDS = st.text(alphabet=GREEK + LATIN + "abc019-", max_size=6)


@settings(deadline=None)
@given(text=st.text())
def test_the_skeleton_is_idempotent(text: str) -> None:
    assert skeleton(skeleton(text)) == skeleton(text)


@settings(deadline=None)
@given(known=st.sets(IDS, min_size=1, max_size=8), data=st.data())
def test_a_known_id_is_its_own_lookalike_and_gets_no_hint(
    known: set[str], data: st.DataObject
) -> None:
    written = data.draw(st.sampled_from(sorted(known)))

    assert lookalike(written, known) == written
    assert lookalike_hint(written, known) == ""


@settings(deadline=None)
@given(known=st.sets(IDS, max_size=8), written=IDS)
def test_a_lookalike_is_a_known_id_with_the_same_skeleton(known: set[str], written: str) -> None:
    match = lookalike(written, known)

    if match is not None:
        assert match in known
        assert skeleton(match) == skeleton(written)


# --- the messages of the validator (validity unchanged: the issue is still reported) ----------


def _truth(name: str) -> Document:
    return json.loads((CORPUS_PATH / name / "truth.json").read_text(encoding="utf-8"))


def _load(document: Document, name: str, folder: Path) -> tuple[DecisionModel, ProblemInstance]:
    for csv in (CORPUS_PATH / name).glob("*.csv"):
        shutil.copyfile(csv, folder / csv.name)
    dm = parse_decision_model(json.dumps(document, ensure_ascii=False))
    return dm, load_instance(dm, folder)


def _data_messages(document: Document, name: str, folder: Path) -> list[str]:
    return [issue.message for issue in validate_data(*_load(document, name, folder))]


def _stability(document: Document) -> Document:
    return next(r for r in document["constraints"] if r["template"] == "stability")


def test_a_latin_e_in_06s_baseline_gets_the_hint(tmp_path: Path) -> None:
    # The reviewer's probe (TASK-005r1 B2-2): one Latin E in the 56-entry baseline of 06.
    document = _truth("06-ambulance-crews")
    baseline = _stability(document)["params"]["baseline"]
    entry = next(e for e in baseline if e["resource_id"] == f"{EPSILON}03")
    entry["resource_id"] = "E03"

    messages = _data_messages(document, "06-ambulance-crews", tmp_path)

    assert messages == [f"unknown resource id 'E03'; did you mean '{EPSILON}03'? ({E_FOR_EPSILON})"]


def test_a_latin_o_in_a_baseline_task_id_gets_the_hint(tmp_path: Path) -> None:
    document = _truth("06-ambulance-crews")
    entry = _stability(document)["params"]["baseline"][0]
    meant = entry["task_id"]
    assert OMICRON in meant
    written = meant.replace(OMICRON, "O")
    entry["task_id"] = written

    messages = _data_messages(document, "06-ambulance-crews", tmp_path)

    assert messages == [
        f"unknown task id {written!r}; did you mean {meant!r}? "
        "(LATIN CAPITAL LETTER O where the data has GREEK CAPITAL LETTER OMICRON)"
    ]


def test_window_and_pair_ids_get_the_hint(tmp_path: Path) -> None:
    document = _truth("05-supermarket-checkouts")
    for rule in document["constraints"]:
        if rule["template"] == "unavailable":
            rule["params"]["windows"][0]["resource_id"] = "T3"
        if rule["template"] == "apart":
            rule["params"]["resource_ids"] = [f"{TAU}1", "T2"]

    messages = _data_messages(document, "05-supermarket-checkouts", tmp_path)

    hint = "LATIN CAPITAL LETTER T where the data has GREEK CAPITAL LETTER TAU"
    assert sorted(messages) == [
        f"unknown resource id 'T2'; did you mean '{TAU}2'? ({hint})",
        f"unknown resource id 'T3'; did you mean '{TAU}3'? ({hint})",
    ]


def test_an_id_with_no_lookalike_keeps_the_plain_message(tmp_path: Path) -> None:
    document = _truth("06-ambulance-crews")
    _stability(document)["params"]["baseline"][0]["resource_id"] = "Z99"

    messages = _data_messages(document, "06-ambulance-crews", tmp_path)

    assert messages == ["unknown resource id 'Z99'"]


def test_solution_assignments_get_the_hint(tmp_path: Path) -> None:
    name = "06-ambulance-crews"
    dm, instance = _load(_truth(name), name, tmp_path)
    solution = json.loads((CORPUS_PATH / name / "solution.json").read_text(encoding="utf-8"))
    assignment = solution["assignments"][0]
    meant = assignment["resource_id"]
    assert meant.startswith(EPSILON)
    assignment["resource_id"] = skeleton(meant)

    issues = validate_solution(dm, instance, parse_solution(json.dumps(solution)))

    assert [issue.message for issue in issues] == [
        f"unknown resource id {skeleton(meant)!r}; did you mean {meant!r}? ({E_FOR_EPSILON})"
    ]
