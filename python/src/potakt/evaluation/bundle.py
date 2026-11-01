"""The input bundle of Test A and Test B (ADR-0007 §3, §7 «Prompt»).

The bundle is the text of ``request.md`` and every CSV file of the scenario, each decoded with the
dialect of the truth's binding of that file and shown with its file name and encoding. It holds no
truth, no solution and no metadata.
"""

from dataclasses import dataclass
from pathlib import Path

from potakt.evaluation.scenario import LoadedScenario
from potakt.model import DecisionModel, ResourceBinding, TaskBinding, Unavailable, WindowsBinding
from potakt.model.data import CsvDialect, CsvEncoding

# Python codecs for the DM's encoding names, chosen here rather than looked up from the OS. Plain
# "utf-8" keeps a byte order mark as U+FEFF, which ``_decode`` removes.
_CODECS: dict[CsvEncoding, str] = {
    "utf-8": "utf-8",
    "windows-1253": "cp1253",
    "iso-8859-7": "iso8859_7",
}
_BYTE_ORDER_MARK = "\N{ZERO WIDTH NO-BREAK SPACE}"
_FENCE = "```"


@dataclass(frozen=True, slots=True)
class BundleFile:
    """One CSV file: its name, its declared encoding and its decoded text (LF line endings)."""

    name: str
    encoding: str
    text: str


@dataclass(frozen=True, slots=True)
class Bundle:
    """The request and the CSV files, in UTF-8 byte order of file name."""

    request: str
    files: tuple[BundleFile, ...]


def build_bundle(scenario: LoadedScenario) -> Bundle:
    """The bundle of one loaded scenario.

    Each file uses the CSV dialect of the truth's binding of that file (resources, tasks or an
    ``unavailable`` window source); a file the truth does not bind uses the defaults (UTF-8).
    Raises ``ValueError`` if the truth binds one file with two different dialects, or if a file
    cannot be decoded.
    """
    directory = scenario.directory
    dialects = _declared_dialects(scenario.truth, directory)
    names = sorted(scenario.scenario.csv_files, key=lambda name: name.encode("utf-8"))
    files: list[BundleFile] = []
    for name in names:
        path = (directory / name).resolve()
        dialect = dialects.get(path, CsvDialect())
        raw = path.read_bytes()
        try:
            text = _decode(raw, _CODECS[dialect.encoding])
        except UnicodeDecodeError as error:
            raise ValueError(f"{name} is not valid {dialect.encoding}") from error
        files.append(BundleFile(name, dialect.encoding, text))
    return Bundle(_normalized(scenario.request), tuple(files))


def render_bundle(bundle: Bundle) -> str:
    """The ``## Request`` and ``## Data files`` sections of ADR-0007 §7, separated by one blank
    line; each file's text is shown unchanged in a fenced block."""
    sections = ["## Request", bundle.request.rstrip("\n"), "## Data files"]
    for file in bundle.files:
        sections.append(f"### {file.name} (encoding: {file.encoding})")
        sections.append(f"{_FENCE}\n{_with_final_newline(file.text)}{_FENCE}")
    return "\n\n".join(sections) + "\n"


def _declared_dialects(truth: DecisionModel, directory: Path) -> dict[Path, CsvDialect]:
    """The dialect of every file the truth binds, by resolved path (``x.csv`` and ``./x.csv`` are
    one file)."""
    bindings: list[ResourceBinding | TaskBinding | WindowsBinding] = [
        truth.data.resources,
        truth.data.tasks,
    ]
    bindings += [
        rule.params.source
        for rule in truth.constraints
        if isinstance(rule, Unavailable) and rule.params.source is not None
    ]
    dialects: dict[Path, CsvDialect] = {}
    for binding in bindings:
        known = dialects.setdefault((directory / binding.source).resolve(), binding.csv)
        if known != binding.csv:
            raise ValueError(f"{binding.source} is bound with two different CSV dialects")
    return dialects


def _decode(raw: bytes, codec: str) -> str:
    return _normalized(raw.decode(codec).removeprefix(_BYTE_ORDER_MARK))


def _normalized(text: str) -> str:
    """Line endings as LF: CRLF and lone CR both become LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _with_final_newline(text: str) -> str:
    return text if not text or text.endswith("\n") else text + "\n"
