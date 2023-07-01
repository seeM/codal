from typing import Iterable, Set

import pytest
from click.testing import CliRunner
from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from codal.cli import cli
from codal.database import SessionLocal, engine, metadata, migrate
from codal.models import Repo
from codal.settings import settings


@pytest.fixture(scope="session")
def runner() -> Iterable[CliRunner]:
    yield CliRunner(mix_stderr=False)


@pytest.fixture(scope="session")
def db() -> Iterable[Session]:
    yield SessionLocal()


def _inspect_table_names() -> Set[str]:
    insp = inspect(engine)
    table_names = insp.get_table_names()
    return set(table_names)


# NOTE: This test is currently order-dependent; it must run first since invoking `embed` calls
#       migrate, and we reuse the same database for all tests.
def test_migrate() -> None:
    """
    Running `migrate` creates all tables in the database.
    """
    settings.DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    table_names = _inspect_table_names()
    assert table_names == set()

    migrate()

    table_names = _inspect_table_names()
    expected = metadata.tables.keys() | {"alembic_version"}
    assert table_names == expected


def test_embed_invalid_repo(runner: CliRunner) -> None:
    """
    Running `embed` with an invalid repo identifier prints an error message.
    """
    result = runner.invoke(
        cli, ["embed", "not-a-repo-identifier"], catch_exceptions=False
    )
    assert result.stderr.splitlines()[-1].startswith("Error: REPO")
    assert result.exit_code == 2


# def test_embed_first_run(runner: CliRunner, db: Session) -> None:
#     result = runner.invoke(
#         cli,
#         ["embed", "seem/test-codal-repo"],
#         catch_exceptions=False,
#     )

#     repos = db.execute(select(Repo)).scalars().all()
#     assert 0
