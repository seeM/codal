from pathlib import Path

import black
import isort.main
import pyright
import pytest
from click.testing import CliRunner

code_root = Path(__file__).parent.parent


def test_black() -> None:
    runner = CliRunner()
    result = runner.invoke(black.main, [str(code_root), "--check"])
    assert result.exit_code == 0, result.output


def test_isort(capsys: pytest.CaptureFixture[str]) -> None:
    try:
        isort.main.main(["--check-only", str(code_root)])
    except SystemExit as exception:
        captured = capsys.readouterr()
        assert exception.code == 0, captured.err


def test_pyright() -> None:
    process = pyright.run(
        str(code_root),
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stdout
