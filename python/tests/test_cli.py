from importlib.metadata import version

from typer.testing import CliRunner

from potakt.cli import app

runner = CliRunner()


def test_version_prints_package_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout == f"potakt {version('potakt')}\n"
