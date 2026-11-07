"""Answers, run records and offline rescoring (ADR-0007 §7 and its amendment; TASK-005c2 E3).

Response shapes follow the official references, looked up on 2026-11-02: Anthropic Messages
(``content`` blocks of type ``text`` / ``thinking``, ``stop_reason`` values from «Handling stop
reasons») and Gemini ``generateContent`` (``candidates[].content.parts[]`` with ``text`` and
``thought``, ``finishReason``, ``promptFeedback.blockReason``).
"""

import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from runner_helpers import RefusedReplace

import potakt.evaluation.records as records_module
from potakt.evaluation import (
    REGISTERED_MODELS,
    Answer,
    Comparison,
    LoadedScenario,
    UnknownModelError,
    answer_of,
    evaluate_candidate,
    load_scenario,
    render_payload,
    render_test_a,
    request_payload,
    rescore_run,
    score_answer,
    summarize,
    write_answer,
    write_request,
)
from potakt.evaluation.records import write_atomic

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "docs" / "validation" / "scenarios"
PHARMACY = CORPUS / "01-pharmacy"
TRUTH_01 = (PHARMACY / "truth.json").read_text(encoding="utf-8")
FENCED_01 = "```json\n" + TRUTH_01 + "\n```"
OPUS, HAIKU, GEMINI = "claude-opus-5-5", "claude-haiku-5-5", "gemini-3.8-flash"


@pytest.fixture(scope="module")
def pharmacy() -> LoadedScenario:
    return load_scenario(PHARMACY)


@pytest.fixture(autouse=True)
def no_temporary_files_are_left(tmp_path: Path) -> Iterator[None]:
    yield
    assert sorted(tmp_path.rglob("*.tmp")) == []


def anthropic(*blocks: dict[str, object], stop: str = "end_turn") -> dict[str, object]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": OPUS,
        "content": list(blocks),
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def text_block(text: str) -> dict[str, object]:
    return {"type": "text", "text": text}


def gemini(*parts: dict[str, object], finish: str = "STOP") -> dict[str, object]:
    return {
        "candidates": [
            {"content": {"role": "model", "parts": list(parts)}, "finishReason": finish}
        ],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
    }


# --- registered models -----------------------------------------------------------------------


def test_registered_models_in_run_order() -> None:
    assert REGISTERED_MODELS == ("claude-haiku-5-5", "gemini-3.8-flash", "claude-opus-5-5")


# --- Anthropic answers -----------------------------------------------------------------------


def test_anthropic_end_turn_with_one_fenced_text_block() -> None:
    answer = answer_of(OPUS, anthropic(text_block(FENCED_01)))

    assert answer == Answer(FENCED_01, "end_turn", True)


def test_anthropic_text_blocks_are_joined_without_a_separator_and_thinking_is_ignored() -> None:
    response = anthropic(
        {"type": "thinking", "thinking": "Let me think.", "signature": "abc"},
        {"type": "redacted_thinking", "data": "xyz"},
        text_block('{"a":'),
        text_block(" 1}"),
    )

    assert answer_of(HAIKU, response) == Answer('{"a": 1}', "end_turn", True)


@pytest.mark.parametrize("stop", ["refusal", "max_tokens", "model_context_window_exceeded"])
def test_anthropic_other_stops_are_not_a_normal_end(stop: str) -> None:
    answer = answer_of(OPUS, anthropic(text_block("partial"), stop=stop))

    assert answer == Answer("partial", stop, False)


@pytest.mark.parametrize(
    "response",
    [
        {"content": "text", "stop_reason": "end_turn"},
        {"content": [text_block("x")]},
        {"content": [text_block("x")], "stop_reason": None},
        {"content": [{"type": "text", "text": 5}], "stop_reason": "end_turn"},
        {"content": [{"text": "x"}], "stop_reason": "end_turn"},
        {"content": ["x"], "stop_reason": "end_turn"},
        [],
        None,
    ],
    ids=[
        "content-not-a-list",
        "stop-reason-missing",
        "stop-reason-null",
        "text-not-a-string",
        "block-without-type",
        "block-not-an-object",
        "not-an-object",
        "null",
    ],
)
def test_a_malformed_anthropic_response(response: object) -> None:
    assert answer_of(OPUS, response) == Answer("", "malformed", False)


# --- Gemini answers --------------------------------------------------------------------------


def test_gemini_stop_with_a_thought_part_then_text_parts() -> None:
    response = gemini(
        {"text": "I should answer in JSON.", "thought": True},
        {"text": '{"a":', "thought": False},
        {"text": " 1}"},
        {"functionCall": {"name": "ignored"}},  # a part without text is skipped
    )

    assert answer_of(GEMINI, response) == Answer('{"a": 1}', "STOP", True)


@pytest.mark.parametrize("finish", ["MAX_TOKENS", "SAFETY", "RECITATION"])
def test_gemini_other_finish_reasons_are_not_a_normal_end(finish: str) -> None:
    assert answer_of(GEMINI, gemini({"text": "par"}, finish=finish)) == Answer("par", finish, False)


def test_gemini_a_candidate_without_content_has_no_text() -> None:
    response = {"candidates": [{"finishReason": "SAFETY", "index": 0}]}

    assert answer_of(GEMINI, response) == Answer("", "SAFETY", False)


def test_gemini_a_blocked_prompt() -> None:
    response = {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}, "usageMetadata": {}}

    assert answer_of(GEMINI, response) == Answer("", "blocked:PROHIBITED_CONTENT", False)


@pytest.mark.parametrize(
    "response",
    [{}, {"candidates": []}, {"candidates": [], "promptFeedback": {"safetyRatings": []}}],
    ids=["absent", "empty", "feedback-without-block-reason"],
)
def test_gemini_no_candidate(response: object) -> None:
    assert answer_of(GEMINI, response) == Answer("", "no-candidate", False)


@pytest.mark.parametrize(
    "response",
    [
        {"candidates": {"content": {}}},
        {"candidates": ["x"]},
        {"candidates": [{"content": {"parts": [{"text": "x"}]}}]},
        {"candidates": [{"content": "x", "finishReason": "STOP"}]},
        {"candidates": [{"content": {"parts": "x"}, "finishReason": "STOP"}]},
        {"candidates": [{"content": {"parts": ["x"]}, "finishReason": "STOP"}]},
        {"candidates": [{"content": {"parts": [{"text": 5}]}, "finishReason": "STOP"}]},
        gemini({"text": "x", "thought": "yes"}),
        {"promptFeedback": "blocked"},
        {"promptFeedback": {"blockReason": 3}},
        "text",
    ],
    ids=[
        "candidates-not-a-list",
        "candidate-not-an-object",
        "finish-reason-missing",
        "content-not-an-object",
        "parts-not-a-list",
        "part-not-an-object",
        "text-not-a-string",
        "thought-not-a-boolean",
        "feedback-not-an-object",
        "block-reason-not-a-string",
        "not-an-object",
    ],
)
def test_a_malformed_gemini_response(response: object) -> None:
    assert answer_of(GEMINI, response) == Answer("", "malformed", False)


@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "gemini-2.5-flash", ""])
def test_answer_of_an_unregistered_model_is_refused(model: str) -> None:
    with pytest.raises(UnknownModelError, match=r"claude-haiku-5-5, gemini-3\.8-flash or "):
        answer_of(model, anthropic(text_block("x")))


# --- scoring ---------------------------------------------------------------------------------


def test_a_normal_answer_is_scored_by_evaluate_candidate(pharmacy: LoadedScenario) -> None:
    candidate, comparison = score_answer(pharmacy, Answer(FENCED_01, "end_turn", True))

    assert candidate == TRUTH_01
    assert comparison == evaluate_candidate(pharmacy, TRUTH_01)
    assert (comparison.recall, comparison.precision) == (1.0, 1.0)


def test_an_answer_that_did_not_end_normally_is_invalid(pharmacy: LoadedScenario) -> None:
    # Even a complete DM counts as nothing when the answer was cut off (ADR-0007 §7 «Answer»).
    candidate, comparison = score_answer(pharmacy, Answer(FENCED_01, "max_tokens", False))

    assert candidate == TRUTH_01
    assert not comparison.valid
    assert comparison.recall == 0.0
    assert [(i.path, i.message) for i in comparison.issues] == [
        ("/", "the answer did not end normally (stop: max_tokens)")
    ]


# --- records ---------------------------------------------------------------------------------


def opus_payload(scenario: LoadedScenario) -> dict[str, object]:
    return request_payload(render_test_a(scenario), OPUS)


def test_write_request_writes_the_rendered_payload(
    tmp_path: Path, pharmacy: LoadedScenario
) -> None:
    run = tmp_path / OPUS / "01-pharmacy"
    payload = opus_payload(pharmacy)

    write_request(run, payload)
    write_request(run, payload)  # the same bytes again: accepted, nothing changes

    assert (run / "request.json").read_bytes() == render_payload(payload).encode("utf-8")


def test_write_request_refuses_to_replace_a_different_request(
    tmp_path: Path, pharmacy: LoadedScenario
) -> None:
    run = tmp_path / "01-pharmacy"
    write_request(run, opus_payload(pharmacy))
    before = (run / "request.json").read_bytes()

    with pytest.raises(FileExistsError):
        write_request(run, request_payload(render_test_a(pharmacy), HAIKU))

    assert (run / "request.json").read_bytes() == before


def test_write_answer_writes_the_four_files_as_specified(
    tmp_path: Path, pharmacy: LoadedScenario
) -> None:
    run = tmp_path / OPUS / "01-pharmacy"
    response = anthropic(text_block(FENCED_01), text_block(""))
    write_request(run, opus_payload(pharmacy))

    comparison = write_answer(run, pharmacy, OPUS, response)

    assert sorted(path.name for path in run.iterdir()) == [
        "candidate.txt",
        "comparison.json",
        "request.json",
        "response.json",
    ]
    assert (run / "response.json").read_bytes() == (
        json.dumps(response, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    assert (run / "candidate.txt").read_bytes() == TRUTH_01.encode("utf-8")
    assert (run / "comparison.json").read_bytes() == comparison.to_json().encode("utf-8")
    assert comparison == evaluate_candidate(pharmacy, TRUTH_01)
    assert b"\r\n" not in b"".join(path.read_bytes() for path in run.iterdir())


def test_write_answer_refuses_a_second_call(tmp_path: Path, pharmacy: LoadedScenario) -> None:
    run = tmp_path / "01-pharmacy"
    write_answer(run, pharmacy, OPUS, anthropic(text_block(FENCED_01)))
    before = {path.name: path.read_bytes() for path in run.iterdir()}

    with pytest.raises(FileExistsError):
        write_answer(run, pharmacy, OPUS, anthropic(text_block("other"), stop="refusal"))

    assert {path.name: path.read_bytes() for path in run.iterdir()} == before


def test_response_json_survives_a_failure_of_the_derived_writes_and_rescore_completes_it(
    tmp_path: Path, pharmacy: LoadedScenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "01-pharmacy"
    response = anthropic(text_block(FENCED_01))

    def broken(scenario: LoadedScenario, text: str) -> Comparison:
        raise RuntimeError("a harness bug")

    with monkeypatch.context() as patch:
        patch.setattr(records_module, "evaluate_candidate", broken)
        with pytest.raises(RuntimeError, match="a harness bug"):
            write_answer(run, pharmacy, OPUS, response)

    assert sorted(path.name for path in run.iterdir()) == ["response.json"]
    assert json.loads((run / "response.json").read_bytes()) == response

    assert rescore_run(run, pharmacy, OPUS) == ("candidate.txt", "comparison.json")
    assert (run / "candidate.txt").read_bytes() == TRUTH_01.encode("utf-8")
    expected = evaluate_candidate(pharmacy, TRUTH_01).to_json().encode("utf-8")
    assert (run / "comparison.json").read_bytes() == expected


def recorded_run(tmp_path: Path, pharmacy: LoadedScenario) -> Path:
    run = tmp_path / "01-pharmacy"
    write_request(run, opus_payload(pharmacy))
    write_answer(run, pharmacy, OPUS, anthropic(text_block(FENCED_01)))
    return run


def test_rescore_reports_nothing_for_an_unchanged_run(
    tmp_path: Path, pharmacy: LoadedScenario
) -> None:
    run = recorded_run(tmp_path, pharmacy)
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in run.iterdir()}

    assert rescore_run(run, pharmacy, OPUS) == ()
    assert {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in run.iterdir()
    } == before


def test_rescore_repairs_a_deleted_comparison(tmp_path: Path, pharmacy: LoadedScenario) -> None:
    run = recorded_run(tmp_path, pharmacy)
    expected = (run / "comparison.json").read_bytes()
    (run / "comparison.json").unlink()

    assert rescore_run(run, pharmacy, OPUS) == ("comparison.json",)
    assert (run / "comparison.json").read_bytes() == expected


def test_rescore_restores_an_edited_candidate_and_never_touches_the_raw_files(
    tmp_path: Path, pharmacy: LoadedScenario
) -> None:
    run = recorded_run(tmp_path, pharmacy)
    raw = {name: (run / name).read_bytes() for name in ("request.json", "response.json")}
    (run / "candidate.txt").write_bytes(b"{}")

    assert rescore_run(run, pharmacy, OPUS) == ("candidate.txt",)
    assert (run / "candidate.txt").read_bytes() == TRUTH_01.encode("utf-8")
    assert {name: (run / name).read_bytes() for name in raw} == raw


def test_rescore_follows_a_changed_rule_without_a_new_response(
    tmp_path: Path, pharmacy: LoadedScenario
) -> None:
    # A recorded answer that did not end normally is rescored as invalid, from response.json alone.
    run = tmp_path / "01-pharmacy"
    write_answer(run, pharmacy, OPUS, anthropic(text_block(FENCED_01), stop="max_tokens"))
    (run / "comparison.json").write_bytes(evaluate_candidate(pharmacy, TRUTH_01).to_json().encode())

    assert rescore_run(run, pharmacy, OPUS) == ("comparison.json",)
    assert json.loads((run / "comparison.json").read_bytes())["valid"] is False


def test_a_response_json_that_is_not_json_is_rescored_as_malformed(
    tmp_path: Path, pharmacy: LoadedScenario
) -> None:
    run = tmp_path / "01-pharmacy"
    run.mkdir()
    (run / "response.json").write_bytes(b"{not json")

    assert rescore_run(run, pharmacy, OPUS) == ("candidate.txt", "comparison.json")
    document = json.loads((run / "comparison.json").read_bytes())
    assert document["issues"] == [
        {"path": "/", "message": "the answer did not end normally (stop: malformed)"}
    ]
    assert (run / "candidate.txt").read_bytes() == b""


def test_rescore_without_a_response_raises(tmp_path: Path, pharmacy: LoadedScenario) -> None:
    with pytest.raises(FileNotFoundError):
        rescore_run(tmp_path, pharmacy, OPUS)


def test_a_leftover_temporary_file_is_overwritten(tmp_path: Path, pharmacy: LoadedScenario) -> None:
    run = tmp_path / "01-pharmacy"
    run.mkdir()
    (run / "candidate.txt.tmp").write_bytes(b"left over by a crash")

    write_answer(run, pharmacy, OPUS, anthropic(text_block(FENCED_01)))

    assert (run / "candidate.txt").read_bytes() == TRUTH_01.encode("utf-8")
    assert not (run / "candidate.txt.tmp").exists()


# --- TASK-005c3d H8: answers that are not valid Unicode ---------------------------------------

SURROGATE = "\ud83d"  # an unpaired surrogate: json.loads accepts "\ud83d", UTF-8 cannot encode it
MALFORMED = Answer("", "malformed", False)


@pytest.mark.parametrize(
    ("model", "response"),
    [
        (OPUS, anthropic(text_block(FENCED_01), text_block(SURROGATE))),
        (GEMINI, gemini({"text": "fine " + SURROGATE + " text"})),
    ],
    ids=["anthropic-text", "gemini-text"],
)
def test_a_lone_surrogate_in_the_text_is_stored_exactly_and_is_malformed(
    tmp_path: Path, pharmacy: LoadedScenario, model: str, response: dict[str, object]
) -> None:
    # U2: .encode("utf-8") raised in _response_bytes, so the batch stayed open on every run.
    run = tmp_path / model / "01-pharmacy"

    comparison = write_answer(run, pharmacy, model, response)

    raw = (run / "response.json").read_bytes()
    assert raw.isascii()
    assert b"\\ud83d" in raw
    assert json.loads(raw) == response
    assert (
        raw == (json.dumps(response, indent=2, ensure_ascii=True, sort_keys=True) + "\n").encode()
    )
    assert answer_of(model, response) == MALFORMED
    assert (run / "candidate.txt").read_bytes() == b""
    assert comparison == score_answer(pharmacy, MALFORMED)[1]
    assert not comparison.valid
    assert (run / "comparison.json").read_bytes() == comparison.to_json().encode("utf-8")
    assert rescore_run(run, pharmacy, model) == ()
    summary = summarize(tmp_path / model, CORPUS)
    assert summary.stops == {"malformed": 1}
    assert "Stops: malformed 1\n" in summary.to_text()


@pytest.mark.parametrize(
    ("model", "response"),
    [
        (OPUS, anthropic(text_block(FENCED_01), stop="end" + SURROGATE)),
        (GEMINI, gemini({"text": "x"}, finish="STOP" + SURROGATE)),
        (GEMINI, {"promptFeedback": {"blockReason": SURROGATE + "SAFETY"}}),
    ],
    ids=["stop_reason", "finishReason", "blockReason"],
)
def test_a_lone_surrogate_in_the_stop_is_malformed(
    tmp_path: Path, pharmacy: LoadedScenario, model: str, response: dict[str, object]
) -> None:
    assert answer_of(model, response) == MALFORMED

    comparison = write_answer(tmp_path / "run", pharmacy, model, response)

    assert json.loads((tmp_path / "run" / "response.json").read_bytes()) == response
    assert comparison == score_answer(pharmacy, MALFORMED)[1]


def test_a_paired_surrogate_escape_is_valid_text(pharmacy: LoadedScenario) -> None:
    # "😀" in JSON is one character, U+1F600: valid Unicode, kept as text.
    response = json.loads(json.dumps(anthropic(text_block(FENCED_01 + "\U0001f600"))))

    assert answer_of(OPUS, response) == Answer(FENCED_01 + "\U0001f600", "end_turn", True)


def test_valid_text_is_still_stored_as_utf_8(tmp_path: Path, pharmacy: LoadedScenario) -> None:
    response = anthropic(text_block(FENCED_01 + " café"))

    write_answer(tmp_path, pharmacy, OPUS, response)

    raw = (tmp_path / "response.json").read_bytes()
    assert "café".encode() in raw
    assert (
        raw == (json.dumps(response, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()
    )


# --- TASK-005c3d H11: durable writes ----------------------------------------------------------

DIRECTORY_FD = 987654  # the descriptor the fake os.open gives a directory: never a real one


class Disk:
    """Fakes of ``os.open``, ``os.fsync``, ``os.close`` and ``os.replace`` that record the order
    of the steps; real file descriptors are synced for real. Our code decides, not the OS: on
    every OS the fake ``os.open`` opens a directory (or refuses, if ``refuse_directory``)."""

    def __init__(self, target: Path, data: bytes) -> None:
        self.target = target
        self.data = data
        self.events: list[str] = []
        self.refuse_directory = False
        self.fail_directory_fsync = False
        self.fail_file_fsync = False
        self.real_fsync = os.fsync
        self.real_replace = os.replace

    def open(self, path: object, flags: int, *args: object) -> int:
        assert Path(str(path)) == self.target.parent
        assert flags == os.O_RDONLY
        self.events.append("open directory")
        if self.refuse_directory:
            raise PermissionError(13, "Permission denied")
        return DIRECTORY_FD

    def fsync(self, fd: int) -> None:
        if fd == DIRECTORY_FD:
            self.events.append("fsync directory")
            if self.fail_directory_fsync:
                raise OSError(22, "Invalid argument")
            return
        temporary = self.target.with_name(self.target.name + ".tmp")
        # The bytes are flushed to the temporary file, and the target is not replaced yet.
        assert temporary.read_bytes() == self.data
        self.events.append(f"fsync file (target exists: {self.target.exists()})")
        if self.fail_file_fsync:
            raise OSError(5, "Input/output error")
        self.real_fsync(fd)

    def close(self, fd: int) -> None:
        assert fd == DIRECTORY_FD
        self.events.append("close directory")

    def replace(self, source: Path, destination: Path) -> None:
        self.events.append(f"replace {source.name} -> {destination.name}")
        self.real_replace(source, destination)


@pytest.fixture
def disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Disk:
    fake = Disk(tmp_path / "out.json", b'{"a": 1}\n')
    monkeypatch.setattr(os, "open", fake.open)
    monkeypatch.setattr(os, "fsync", fake.fsync)
    monkeypatch.setattr(os, "close", fake.close)
    monkeypatch.setattr(os, "replace", fake.replace)
    return fake


def test_write_atomic_syncs_the_file_before_the_replace_then_the_directory(disk: Disk) -> None:
    # U6: no fsync, so a power loss right after the create POST could lose the creating entry.
    write_atomic(disk.target, disk.data)

    assert disk.events == [
        "fsync file (target exists: False)",
        "replace out.json.tmp -> out.json",
        "open directory",
        "fsync directory",
        "close directory",
    ]
    assert disk.target.read_bytes() == disk.data


def test_a_directory_that_cannot_be_opened_is_skipped(disk: Disk) -> None:
    # Windows cannot open a directory with os.open; its os.replace is one file-system operation.
    disk.refuse_directory = True

    write_atomic(disk.target, disk.data)

    assert disk.events[-1] == "open directory"
    assert disk.target.read_bytes() == disk.data


def test_a_failed_directory_fsync_is_ignored_and_the_directory_closed(disk: Disk) -> None:
    disk.fail_directory_fsync = True

    write_atomic(disk.target, disk.data)

    assert disk.events[-2:] == ["fsync directory", "close directory"]
    assert disk.target.read_bytes() == disk.data


def test_a_failed_file_fsync_is_raised_and_leaves_the_target_unchanged(disk: Disk) -> None:
    # A write that may not be durable is not reported as done; the temporary file is removed.
    disk.target.write_bytes(b"old\n")
    disk.fail_file_fsync = True

    with pytest.raises(OSError, match="Input/output error"):
        write_atomic(disk.target, disk.data)

    assert disk.events == ["fsync file (target exists: True)"]
    assert disk.target.read_bytes() == b"old\n"
    assert not disk.target.with_name("out.json.tmp").exists()


# --- TASK-005c3g G3: a bounded retry when Windows refuses a replace ---------------------------

# min(0.05 * 2**attempt, 0.5) after the attempts 0..8: 9 pauses between 10 attempts, 3.25 s.
REFUSED_PAUSES = [0.05, 0.1, 0.2, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5]


@pytest.fixture
def pauses(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """``time.sleep`` recorded instead of slept."""
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    return slept


def test_the_pauses_add_up_to_about_three_seconds() -> None:
    assert sum(REFUSED_PAUSES) == pytest.approx(3.25)


@pytest.mark.parametrize("times", [1, 9])
def test_a_refused_replace_is_retried_until_it_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pauses: list[float], times: int
) -> None:
    refused = RefusedReplace(times)
    monkeypatch.setattr(os, "replace", refused)
    target = tmp_path / "out.json"

    write_atomic(target, b"new\n")

    assert target.read_bytes() == b"new\n"
    assert refused.refused == times
    assert pauses == REFUSED_PAUSES[:times]


def test_ten_refused_replaces_raise_the_last_error_and_remove_the_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pauses: list[float]
) -> None:
    refused = RefusedReplace(10)
    monkeypatch.setattr(os, "replace", refused)
    target = tmp_path / "out.json"
    target.write_bytes(b"old\n")

    with pytest.raises(PermissionError):
        write_atomic(target, b"new\n")

    assert refused.refused == 10
    assert pauses == REFUSED_PAUSES
    assert target.read_bytes() == b"old\n"
    assert not (tmp_path / "out.json.tmp").exists()


def test_eleven_refusals_are_not_retried_beyond_ten_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pauses: list[float]
) -> None:
    refused = RefusedReplace(11)
    monkeypatch.setattr(os, "replace", refused)

    with pytest.raises(PermissionError):
        write_atomic(tmp_path / "out.json", b"new\n")

    assert refused.refused == 10


def test_another_os_error_of_the_replace_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pauses: list[float]
) -> None:
    calls: list[str] = []

    def failing(source: object, destination: object) -> None:
        calls.append(str(destination))
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(os, "replace", failing)

    with pytest.raises(OSError, match="Input/output error"):
        write_atomic(tmp_path / "out.json", b"new\n")

    assert len(calls) == 1
    assert pauses == []
    assert not (tmp_path / "out.json.tmp").exists()


def refused_open(monkeypatch: pytest.MonkeyPatch, name: str, times: int) -> list[int]:
    """``Path.open`` of the file ``name`` refused with ``PermissionError`` ``times`` times."""
    real_open = Path.open
    refused: list[int] = []

    def open_(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.name == name and len(refused) < times:
            refused.append(1)
            raise PermissionError(13, "Permission denied")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_)
    return refused


@pytest.mark.parametrize("times", [1, 9])
def test_a_refused_open_of_the_temporary_file_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pauses: list[float], times: int
) -> None:
    refused = refused_open(monkeypatch, "out.json.tmp", times)

    write_atomic(tmp_path / "out.json", b"new\n")

    assert (tmp_path / "out.json").read_bytes() == b"new\n"
    assert len(refused) == times
    assert pauses == REFUSED_PAUSES[:times]


def test_ten_refused_opens_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pauses: list[float]
) -> None:
    refused = refused_open(monkeypatch, "out.json.tmp", 10)

    with pytest.raises(PermissionError):
        write_atomic(tmp_path / "out.json", b"new\n")

    assert len(refused) == 10
    assert pauses == REFUSED_PAUSES
    assert not (tmp_path / "out.json").exists()


def test_an_injected_sleep_is_used_instead_of_time_sleep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pauses: list[float]
) -> None:
    monkeypatch.setattr(os, "replace", RefusedReplace(2))
    injected: list[float] = []

    write_atomic(tmp_path / "out.json", b"new\n", sleep=injected.append)

    assert injected == [0.05, 0.1]
    assert pauses == []


def test_an_answer_whose_replace_is_refused_ten_times_is_kept_in_its_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pauses: list[float], pharmacy: LoadedScenario
) -> None:
    # G5: never lose a first answer; the next run stops on response.json.tmp instead.
    monkeypatch.setattr(os, "replace", RefusedReplace(10, "response.json"))
    response = anthropic({"type": "text", "text": TRUTH_01})

    with pytest.raises(PermissionError):
        write_answer(tmp_path, pharmacy, HAIKU, response)

    kept = tmp_path / "response.json.tmp"
    assert json.loads(kept.read_bytes()) == response
    assert not (tmp_path / "response.json").exists()
    kept.unlink()  # for the fixture that checks no temporary file is left
