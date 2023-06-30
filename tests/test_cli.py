import os
from pathlib import Path
from typing import Iterable
from click.testing import CliRunner
from unittest import mock

import pytest


@pytest.fixture
def runner(tmp_path: Path) -> Iterable[CliRunner]:
    runner = CliRunner(mix_stderr=False)
    cache_home = tmp_path / "cache"
    with mock.patch.dict(
        os.environ, {"XDG_CACHE_HOME": str(cache_home)}
    ), runner.isolated_filesystem(temp_dir=tmp_path):
        yield runner


def test_embed_invalid_repo(runner: CliRunner) -> None:
    from codal.cli import cli

    result = runner.invoke(
        cli, ["embed", "not-a-repo-identifier"], catch_exceptions=False
    )
    assert result.stderr.splitlines()[-1].startswith("Error: REPO")
    assert result.exit_code == 2
