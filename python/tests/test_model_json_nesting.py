"""One JSON nesting limit, decided by our code on the raw text (ADR-0006 §4).

The JSON decoder's recursion guard depends on the thread's real C stack in Python 3.14, so deep
input decodes on a machine with a large stack and fails on a small one. These tests reproduce the
large-stack machine (the ubuntu 3.14 CI runner) with a thread of their own instead of relying on
this machine's defaults.
"""

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from evaluation_helpers import write_scenario_dir
from hypothesis import given, settings
from hypothesis import strategies as st

from potakt.evaluation import ScenarioError, load_scenario, scenario
from potakt.model import (
    DecisionModelError,
    json_nesting_exceeds,
    loader,
    parse_decision_model,
    parse_solution,
)
from potakt.model.loader import _nesting_depth  # pyright: ignore[reportPrivateUsage]

TOO_DEEP = "JSON nested more than 64 levels deep"
# 128 MiB: valid everywhere (Windows rejects 256 MiB and more) and already large enough for the 3.14
# decoder to decode 100 000 levels, as on the CI runner.
LARGE_STACK = 128 * 1024 * 1024
DEEP_INPUTS = {
    "unbalanced": "[" * 100_000,
    "balanced": "[" * 100_000 + "]" * 100_000,
}


def in_large_stack_thread[T](call: Callable[[], T]) -> T:
    """Run ``call`` in a large-stack thread, like a machine whose decoder does not overflow."""
    outcome: list[T] = []
    failure: list[BaseException] = []

    def run() -> None:
        try:
            outcome.append(call())
        except BaseException as error:  # re-raised in the calling thread
            failure.append(error)

    previous = threading.stack_size(LARGE_STACK)
    try:
        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
    finally:
        threading.stack_size(previous)
    if failure:
        raise failure[0]
    return outcome[0]


def in_this_thread[T](call: Callable[[], T]) -> T:
    return call()


RUNNERS = {"this-thread": in_this_thread, "large-stack-thread": in_large_stack_thread}


def model_issues(parse: Callable[[str], object], text: str) -> list[str]:
    try:
        parse(text)
    except DecisionModelError as error:
        return [str(issue) for issue in error.issues]
    return []


def scenario_issues(directory: Path) -> list[str]:
    try:
        load_scenario(directory)
    except ScenarioError as error:
        return [str(issue) for issue in error.issues]
    return []


@pytest.mark.parametrize("runner", RUNNERS.values(), ids=RUNNERS)
@pytest.mark.parametrize("text", DEEP_INPUTS.values(), ids=DEEP_INPUTS)
@pytest.mark.parametrize(
    "parse", [parse_decision_model, parse_solution], ids=["decision-model", "solution"]
)
def test_deep_model_json_gives_the_nesting_issue_on_any_stack(
    runner: Callable[[Callable[[], list[str]]], list[str]],
    text: str,
    parse: Callable[[str], object],
) -> None:
    assert runner(lambda: model_issues(parse, text)) == [f"/: {TOO_DEEP}"]


@pytest.mark.parametrize("runner", RUNNERS.values(), ids=RUNNERS)
@pytest.mark.parametrize("text", DEEP_INPUTS.values(), ids=DEEP_INPUTS)
def test_deep_scenario_json_gives_the_nesting_issue_on_any_stack(
    tmp_path: Path,
    runner: Callable[[Callable[[], list[str]]], list[str]],
    text: str,
) -> None:
    directory = write_scenario_dir(tmp_path)
    (directory / "scenario.json").write_text(text, encoding="utf-8")

    assert runner(lambda: scenario_issues(directory)) == [f"scenario.json#/: {TOO_DEEP}"]


# --- the decoder is never reached for deep input --------------------------------------------------


def nested(depth: int) -> str:
    """A JSON array ``depth`` levels deep (``[]`` is one level)."""
    return "[" * depth + "]" * depth


def _decoder_must_not_run(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("json.loads was called for input deeper than the limit")


def test_the_decoder_is_never_called_for_input_deeper_than_the_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = write_scenario_dir(tmp_path)
    for name in ("scenario.json", "truth.json", "solution.json"):
        (directory / name).write_text(nested(65), encoding="utf-8")
    monkeypatch.setattr(json, "loads", _decoder_must_not_run)

    assert model_issues(parse_decision_model, nested(65)) == [f"/: {TOO_DEEP}"]
    assert model_issues(parse_solution, nested(65)) == [f"/: {TOO_DEEP}"]
    assert sorted(scenario_issues(directory)) == [
        f"scenario.json#/: {TOO_DEEP}",
        f"solution.json#/: {TOO_DEEP}",
        f"truth.json#/: {TOO_DEEP}",
    ]


def _scanner_misses(*_args: object) -> bool:
    return False


def _decoder_overflows(*_args: object, **_kwargs: object) -> object:
    raise RecursionError("maximum recursion depth exceeded")


@pytest.mark.parametrize(
    "parse", [parse_decision_model, parse_solution], ids=["decision-model", "solution"]
)
def test_defence_in_depth_without_the_raw_text_check(
    monkeypatch: pytest.MonkeyPatch, parse: Callable[[str], object]
) -> None:
    # Should the raw-text check ever miss, the decoded depth and the RecursionError handler
    # still give the same issue.
    monkeypatch.setattr(loader, "json_nesting_exceeds", _scanner_misses)

    assert model_issues(parse, nested(65)) == [f"/: {TOO_DEEP}"]
    monkeypatch.setattr(json, "loads", _decoder_overflows)
    assert model_issues(parse, nested(65)) == [f"/: {TOO_DEEP}"]


def test_defence_in_depth_for_scenario_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = write_scenario_dir(tmp_path)
    (directory / "scenario.json").write_text(nested(65), encoding="utf-8")
    monkeypatch.setattr(scenario, "json_nesting_exceeds", _scanner_misses)
    monkeypatch.setattr(scenario.json, "loads", _decoder_overflows)

    assert f"scenario.json#/: {TOO_DEEP}" in scenario_issues(directory)


# --- the raw-text scanner ------------------------------------------------------------------------


def mixed(depth: int) -> str:
    """Alternating objects and arrays, ``depth`` levels deep."""
    text = "0"
    for level in range(depth):
        text = f'{{"k": {text}}}' if level % 2 else f"[{text}]"
    return text


@pytest.mark.parametrize(
    "text",
    [
        '"[[[["',
        '"{"',
        '["[[[[", "{{{{"]',
        '"a\\"[["',
        '["\\\\", "x"]',
        "",
        "not json at all",
    ],
    ids=[
        "brackets",
        "brace",
        "strings-in-array",
        "escaped-quote",
        "escaped-backslash",
        "empty",
        "text",
    ],
)
def test_brackets_inside_strings_do_not_count(text: str) -> None:
    assert not json_nesting_exceeds(text, 1)


def test_a_bracket_after_an_escaped_backslash_counts() -> None:
    # "\\" closes the string, so the two brackets after it are real: depth 3 with the outer array.
    text = '["\\\\", [[]]]'

    assert json_nesting_exceeds(text, 2)
    assert not json_nesting_exceeds(text, 3)


@pytest.mark.parametrize(
    "build",
    [nested, lambda depth: "{" + '"k": {' * (depth - 1) + "}" * depth, mixed],
    ids=["arrays", "objects", "mixed"],
)
def test_64_levels_are_accepted_and_65_rejected(build: Callable[[int], str]) -> None:
    assert json.loads(build(64)) is not None  # the builders make real JSON
    assert not json_nesting_exceeds(build(64))
    assert json_nesting_exceeds(build(65))


@pytest.mark.parametrize(
    "text",
    ["[" * 10, '["unterminated', '{"a": "\\', "]]]]", "[]]]][[", '"\\', "\\[]"],
    ids=[
        "open-brackets",
        "open-string",
        "trailing-escape",
        "closing-only",
        "unbalanced",
        "escape",
        "backslash-outside-a-string",
    ],
)
def test_unterminated_or_unbalanced_input_returns_without_raising(text: str) -> None:
    # Left to the decoder, which reports it with a position.
    assert json_nesting_exceeds(text, 10) is False


def test_a_flat_array_of_a_million_items_is_accepted_quickly() -> None:
    text = "[" + ",".join(["1"] * 1_000_000) + "]"

    started = time.perf_counter()
    exceeds = json_nesting_exceeds(text)
    elapsed = time.perf_counter() - started

    assert not exceeds
    # Linear: under a second locally. The bound is loose on purpose (slow CI runners); a quadratic
    # scan of 2 million characters would take hours.
    assert elapsed < 30.0


json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(alphabet='[]{}"\\ab', max_size=5),
    lambda children: (
        st.lists(children, max_size=2) | st.dictionaries(st.text(max_size=3), children, max_size=2)
    ),
    max_leaves=80,
)


def _wrapped(value: Any, depth: int) -> Any:
    for level in range(depth):
        value = [value] if level % 2 else {"k": value}
    return value


@settings(deadline=None)
@given(json_values, st.integers(min_value=0, max_value=70))
def test_the_scanner_agrees_with_the_depth_of_the_decoded_value(value: Any, extra: int) -> None:
    document = _wrapped(value, extra)
    text = json.dumps(document)

    assert json_nesting_exceeds(text, 64) == (_nesting_depth(document) > 64)
