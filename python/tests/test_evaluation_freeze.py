"""TASK-005e6 decision 4 (ADR-0008 D1 and its amendment): no corpus scenario is sent before the
freeze.

This module has no fixture that sets ``CORPUS_FROZEN``: it checks the repository's value and what
``eval run`` does with it. A test that needs the frozen state sets it explicitly. No test reads a
real key or opens a connection: ``read_key`` is recorded and every transport refuses to send.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

import potakt.cli as cli
import potakt.evaluation.runner as runner_module
from potakt.cli import app
from potakt.evaluation.transport import HttpResponse, RunStopped

HAIKU = "claude-haiku-5-5"
GEMINI = "gemini-3.8-flash"
NOT_FROZEN = (
    "no corpus scenario is sent before the commit tagged phase0-freeze (ADR-0008 D1); run the "
    "development set with --dev; nothing was sent"
)
NOT_FROZEN_PLAN = (
    "Not sent before the freeze: --yes is refused until the commit tagged phase0-freeze "
    "(ADR-0008 D1); the development set runs with --dev."
)
RUN_AGAIN = "Run again with --yes to send."

runner = CliRunner()


class NoTransport:
    """A transport that fails the test if anything is sent."""

    def send(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
    ) -> HttpResponse:
        raise AssertionError(f"nothing may be sent: {method} {url}")


@pytest.fixture(autouse=True)
def keys_read(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """``read_key`` records the variable it is asked for and then reads nothing (the key
    variables are unset, conftest); nothing can be sent."""
    read: list[str] = []
    real = cli.read_key

    def record(variable: str) -> str:
        read.append(variable)
        return real(variable)

    monkeypatch.setattr(cli, "read_key", record)
    monkeypatch.setattr(cli, "make_transport", NoTransport)
    return read


def eval_run(*args: str) -> Result:
    return runner.invoke(app, ["eval", "run", *args])


def test_the_corpus_is_not_frozen_in_the_repository() -> None:
    assert runner_module.CORPUS_FROZEN is False


@pytest.mark.parametrize("model", [HAIKU, GEMINI])
def test_a_corpus_yes_run_before_the_freeze_stops_and_writes_nothing(
    tmp_path: Path, keys_read: list[str], model: str
) -> None:
    runs = tmp_path / "runs"

    result = eval_run("--model", model, "--runs", str(runs), "--yes")

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == NOT_FROZEN + "\n"  # not "... is not set in this shell"
    assert keys_read == []
    assert not runs.exists()  # no lock, no record


def test_the_freeze_stop_comes_before_the_release_check(
    tmp_path: Path, keys_read: list[str]
) -> None:
    result = eval_run(
        "--model", GEMINI, "--runs", str(tmp_path / "runs"), "--yes", "--release", "x"
    )

    assert result.exit_code == 1
    assert result.stderr == NOT_FROZEN + "\n"  # not "x is not a held batch of gemini-3.8-flash"
    assert keys_read == []


def test_the_corpus_dry_run_before_the_freeze_ends_with_the_plan_line(tmp_path: Path) -> None:
    result = eval_run("--model", HAIKU, "--runs", str(tmp_path / "runs"))

    assert result.exit_code == 0, result.stderr
    assert result.stdout.startswith("Test A run plan: claude-haiku-5-5\n")
    assert result.stdout.endswith("\n" + NOT_FROZEN_PLAN + "\n")
    assert RUN_AGAIN not in result.stdout


@pytest.mark.parametrize("frozen", [False, True])
def test_the_dev_dry_run_ends_with_run_again_whatever_the_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen: bool
) -> None:
    monkeypatch.setattr(runner_module, "CORPUS_FROZEN", frozen)

    result = eval_run("--model", HAIKU, "--dev", "--runs", str(tmp_path / "runs"))

    assert result.exit_code == 0, result.stderr
    assert result.stdout.endswith("\n" + RUN_AGAIN + "\n")
    assert NOT_FROZEN_PLAN not in result.stdout


def test_a_dev_yes_run_before_the_freeze_goes_on_to_read_the_key(
    tmp_path: Path, keys_read: list[str]
) -> None:
    result = eval_run("--model", GEMINI, "--dev", "--runs", str(tmp_path / "runs"), "--yes")

    assert result.exit_code == 1
    assert result.stderr == "POTAKT_GEMINI_API_KEY is not set in this shell\n"
    assert keys_read == ["POTAKT_GEMINI_API_KEY"]


def test_the_flag_is_read_when_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keys_read: list[str]
) -> None:
    monkeypatch.setattr(runner_module, "CORPUS_FROZEN", True)

    yes = eval_run("--model", HAIKU, "--runs", str(tmp_path / "runs"), "--yes")
    dry = eval_run("--model", HAIKU, "--runs", str(tmp_path / "runs"))

    assert yes.exit_code == 1
    assert yes.stderr == "POTAKT_ANTHROPIC_API_KEY is not set in this shell\n"
    assert keys_read == ["POTAKT_ANTHROPIC_API_KEY"]
    assert dry.stdout.endswith("\n" + RUN_AGAIN + "\n")


def test_check_frozen_stops_only_a_run_without_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    runner_module.check_frozen(True)
    with pytest.raises(RunStopped) as raised:
        runner_module.check_frozen(False)
    monkeypatch.setattr(runner_module, "CORPUS_FROZEN", True)
    runner_module.check_frozen(False)

    assert str(raised.value) == NOT_FROZEN
