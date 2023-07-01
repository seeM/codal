from datetime import datetime
from pathlib import Path
from typing import Iterable, Set

import pytest
from click.testing import CliRunner
from git.repo import Repo as GitRepo
from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from codal.ai import load_index
from codal.cli import cli
from codal.database import SessionLocal, engine, metadata, migrate
from codal.models import Chunk, Commit, Document, DocumentVersion, Org, Repo
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


def test_embed_first_run(runner: CliRunner, db: Session) -> None:
    repo_arg = "seem/test-codal-repo"
    result = runner.invoke(
        cli,
        ["embed", repo_arg],
        catch_exceptions=False,
    )

    assert result.exit_code == 0

    # Clones the repo
    git_dir = settings.REPO_DIR / repo_arg
    assert git_dir.exists()

    git_repo = GitRepo(git_dir)
    assert not git_repo.bare

    # Checks out the latest commit
    head_commit_sha = "f51a972d1c611636847bffddd868c3329f9588d1"
    assert git_repo.head.commit.hexsha == head_commit_sha

    # Creates an Org in the database
    org = db.execute(select(Org)).scalar_one()
    org_name, repo_name = repo_arg.split("/")
    assert org.name == org_name

    # Creates a Repo in the database
    repo = db.execute(select(Repo)).scalar_one()
    assert repo.org_id == org.id
    assert repo.name == repo_name
    assert repo.default_branch == "main"
    assert repo.head_commit.sha == head_commit_sha

    # Creates a Commit in the database
    commit = db.execute(select(Commit)).scalar_one()
    committed_datetime = datetime(2023, 6, 30, 14, 1, 38)
    assert commit.repo_id == repo.id
    assert commit.sha == head_commit_sha
    assert commit.message == "rename"
    assert commit.author_name == "Wasim Lorgat"
    assert commit.author_email and commit.author_email.endswith(".com")
    assert commit.authored_datetime == committed_datetime
    assert commit.committer_name == "GitHub"
    assert commit.committer_email == "noreply@github.com"
    assert commit.committed_datetime == committed_datetime

    # Creates a Document in the database
    document = db.execute(select(Document)).scalar_one()
    assert document.repo_id == repo.id
    assert document.path == Path("README.md")

    # Creates a DocumentVersion in the database
    document_version = db.execute(select(DocumentVersion)).scalar_one()
    assert document_version.document_id == document.id
    assert document_version.commit_id == commit.id
    assert document_version.text == "# test-codal-repo\n"
    assert document_version.num_tokens == 7
    assert document_version.processed == True

    # Creates Chunks in the database
    chunks = db.execute(select(Chunk)).scalars().all()
    assert len(chunks) == 1

    chunks = sorted(chunks, key=lambda x: x.start)

    # Chunks are linked to documents and document versions
    for chunk in chunks:
        assert chunk.document_id == document.id
        assert chunk.document_versions == [document_version]

    # Chunks cover the entire document
    assert chunks[0].start == 0
    assert chunks[-1].end == len(document_version.text.strip())

    # Each chunk's start should be the previous chunk's end
    for i in range(len(chunks) - 1):
        assert chunks[i].end == chunks[i + 1].start

    # Joining the chunk texts should equal the document text
    # TODO: This might not actually work when we have multiple chunks because we may currently
    #       be losing newlines...
    chunk_text = "".join(chunk.text for chunk in chunks)
    assert chunk_text == document_version.text.strip()

    # TODO: Creates the default branch file? Or test the end result?
    #       If we run with an existing repo dir but no DB entry, we should still set a default branch
    #       There's also an assert in embed so maybe that's good enough

    # Creates an index file that fully covers chunks.
    index_path = settings.INDEX_DIR / f"{org_name}-{repo_name}.bin"
    # TODO: Don't hardcode the dimension
    index = load_index(index_path, dim=1536)
    assert index is not None
    assert index.get_ids_list() == [chunk.id for chunk in chunks]
