"""The Test A / Test B input bundle (ADR-0007 §3, §7 «Prompt»; TASK-005c1 D2)."""

import codecs
import json
from pathlib import Path

import pytest
from evaluation_helpers import scenario_metadata, write_scenario_dir
from helpers import constraint, minimal_dm

from potakt.evaluation import Bundle, BundleFile, build_bundle, load_scenario, render_bundle

CORPUS = Path(__file__).resolve().parents[2] / "docs" / "validation" / "scenarios"
SCENARIOS = sorted(path.name for path in CORPUS.iterdir() if path.is_dir())
GOLDEN = Path(__file__).resolve().parent / "golden"
# Python codecs for the declared encodings, written out here so the test does not reuse the code.
CODECS = {"utf-8": "utf-8", "windows-1253": "cp1253", "iso-8859-7": "iso8859_7"}


def utf8_order(names: list[str]) -> list[str]:
    return sorted(names, key=lambda name: name.encode("utf-8"))


# --- goldens ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["02-hotel-reception", "06-ambulance-crews"])
def test_rendered_bundle_matches_the_golden_byte_for_byte(name: str) -> None:
    # 02: windows-1253 with ';' and ','; 06: proschedio.csv is not bound by the truth (utf-8).
    bundle = build_bundle(load_scenario(CORPUS / name))

    assert render_bundle(bundle).encode("utf-8") == (GOLDEN / f"bundle-{name}.txt").read_bytes()


def test_scenario_02_files_are_windows_1253_and_06_draft_uses_the_defaults() -> None:
    hotel = build_bundle(load_scenario(CORPUS / "02-hotel-reception"))
    ambulance = build_bundle(load_scenario(CORPUS / "06-ambulance-crews"))

    assert [(f.name, f.encoding) for f in hotel.files] == [
        ("ypodoxi_prosopiko.csv", "windows-1253"),
        ("ypodoxi_vardies.csv", "windows-1253"),
    ]
    assert [(f.name, f.encoding) for f in ambulance.files] == [
        ("pliromata.csv", "utf-8"),
        ("proschedio.csv", "utf-8"),
        ("vardies_pliromaton.csv", "utf-8"),
    ]


# --- every scenario --------------------------------------------------------------------------


@pytest.mark.parametrize("name", SCENARIOS)
def test_bundle_files_are_the_csv_files_in_utf8_byte_order(name: str) -> None:
    scenario = load_scenario(CORPUS / name)

    bundle = build_bundle(scenario)

    assert [f.name for f in bundle.files] == utf8_order(scenario.scenario.csv_files)


@pytest.mark.parametrize("name", SCENARIOS)
def test_bundle_text_re_encodes_to_the_file_bytes(name: str) -> None:
    # Apart from a removed byte order mark and CRLF line endings, nothing is changed.
    scenario = load_scenario(CORPUS / name)

    for file in build_bundle(scenario).files:
        raw = (scenario.directory / file.name).read_bytes()
        raw = raw.removeprefix(codecs.BOM_UTF8).replace(b"\r\n", b"\n")
        assert file.text.encode(CODECS[file.encoding]) == raw, file.name


@pytest.mark.parametrize("name", SCENARIOS)
def test_bundle_request_is_the_request_file(name: str) -> None:
    scenario = load_scenario(CORPUS / name)

    raw = (scenario.directory / "request.md").read_bytes()

    assert build_bundle(scenario).request == raw.decode("utf-8").replace("\r\n", "\n")


# --- decoding --------------------------------------------------------------------------------


def test_a_byte_order_mark_and_crlf_are_removed(tmp_path: Path) -> None:
    # Bytes, not text mode: the test decides the line endings, not the OS.
    directory = write_scenario_dir(tmp_path)
    (directory / "request.md").write_bytes(b"Line one\r\nLine two\r\n")
    resources = (directory / "resources.csv").read_text(encoding="utf-8")
    (directory / "resources.csv").write_bytes(
        codecs.BOM_UTF8 + resources.replace("\n", "\r\n").encode("utf-8")
    )

    bundle = build_bundle(load_scenario(directory))

    assert bundle.request == "Line one\nLine two\n"
    assert bundle.files[0] == BundleFile("resources.csv", "utf-8", resources)


def test_a_file_bound_twice_with_one_dialect_is_shown_once(tmp_path: Path) -> None:
    truth = _absences_truth({}, {})
    directory = _absences_scenario(tmp_path, truth)

    bundle = build_bundle(load_scenario(directory))

    assert [(f.name, f.encoding) for f in bundle.files] == [
        ("absences.csv", "utf-8"),
        ("resources.csv", "utf-8"),
        ("tasks.csv", "utf-8"),
    ]


def test_a_file_bound_twice_with_different_dialects_is_an_error(tmp_path: Path) -> None:
    # An ASCII file decodes under both encodings, so the scenario itself loads.
    truth = _absences_truth({}, {"encoding": "iso-8859-7"})
    directory = _absences_scenario(tmp_path, truth)
    scenario = load_scenario(directory)

    with pytest.raises(ValueError, match=r"absences\.csv is bound with two different CSV dialects"):
        build_bundle(scenario)


def test_an_unbound_file_that_is_not_utf8_is_an_error(tmp_path: Path) -> None:
    # Unbound files use the defaults (UTF-8); the scenario loader never reads them.
    metadata = scenario_metadata()
    metadata["csv_files"] = ["resources.csv", "tasks.csv", "draft.csv"]
    directory = write_scenario_dir(tmp_path, metadata=metadata)
    (directory / "draft.csv").write_bytes("Κωδικός\n".encode("cp1253"))
    scenario = load_scenario(directory)

    with pytest.raises(ValueError, match=r"draft\.csv is not valid utf-8"):
        build_bundle(scenario)


def _absences_truth(first: dict[str, str], second: dict[str, str]) -> dict[str, object]:
    truth = minimal_dm()
    for index, dialect in enumerate((first, second)):
        source: dict[str, object] = {
            "source": "absences.csv",
            "resource_id": "person",
            "start": "from",
            "end": "to",
        }
        if dialect:
            source["csv"] = dialect
        truth["constraints"].append(
            constraint(
                "unavailable", id=f"away-{index}", hardness="hard", params={"source": source}
            )
        )
    return truth


def _absences_scenario(tmp_path: Path, truth: dict[str, object]) -> Path:
    metadata = scenario_metadata()
    metadata["csv_files"] = ["resources.csv", "tasks.csv", "absences.csv"]
    directory = write_scenario_dir(tmp_path, metadata=metadata, truth=truth)
    (directory / "absences.csv").write_text(
        "person,from,to\nr1,2026-11-02T00:00,2026-11-03T00:00\n", encoding="utf-8"
    )
    return directory


# --- rendering -------------------------------------------------------------------------------


def test_render_uses_the_headings_fences_and_blank_lines_of_adr_7() -> None:
    bundle = Bundle(
        request="Please plan.\n",
        files=(
            BundleFile("a.csv", "windows-1253", "id\nr1\n"),
            BundleFile("b.csv", "utf-8", "id\nt1"),  # no final newline: one is added before ```
        ),
    )

    assert render_bundle(bundle) == (
        "## Request\n"
        "\n"
        "Please plan.\n"
        "\n"
        "## Data files\n"
        "\n"
        "### a.csv (encoding: windows-1253)\n"
        "\n"
        "```\n"
        "id\n"
        "r1\n"
        "```\n"
        "\n"
        "### b.csv (encoding: utf-8)\n"
        "\n"
        "```\n"
        "id\n"
        "t1\n"
        "```\n"
    )


def test_render_is_deterministic() -> None:
    scenario = load_scenario(CORPUS / "02-hotel-reception")

    assert render_bundle(build_bundle(scenario)) == render_bundle(build_bundle(scenario))


def test_the_bundle_shows_no_scenario_metadata_and_names_no_truth() -> None:
    rendered = render_bundle(build_bundle(load_scenario(CORPUS / "06-ambulance-crews")))
    metadata = json.loads((CORPUS / "06-ambulance-crews" / "scenario.json").read_text("utf-8"))

    assert metadata["notes"] not in rendered
    assert "truth.json" not in rendered
