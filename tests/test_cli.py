from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence, Set

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


@pytest.fixture
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
        cli, ["embed", "not-a-repo-identifier"], catch_exceptions=False  # type: ignore
    )
    assert result.stderr.splitlines()[-1].startswith("Error: REPO")
    assert result.exit_code == 2


def test_embed_first_run(runner: CliRunner, db: Session) -> None:
    repo_arg = "seem/test-codal-repo"
    org_name, repo_name = repo_arg.split("/")
    head_commit_sha = "f51a972d1c611636847bffddd868c3329f9588d1"

    result = runner.invoke(
        cli,  # type: ignore
        ["embed", repo_arg, "--head", head_commit_sha],
        catch_exceptions=False,
    )

    assert result.exit_code == 0

    # Clones the repo
    git_dir = settings.REPO_DIR / repo_arg
    assert git_dir.exists()

    git_repo = GitRepo(git_dir)
    assert not git_repo.bare

    # Checks out the latest commit
    assert git_repo.head.commit.hexsha == head_commit_sha

    # Creates an Org in the database
    org = db.execute(select(Org)).scalar_one()
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

    # Check the chunks themselves
    _test_chunks(document_version)

    # Index file is updated to reflect chunks
    _test_index(chunks, org_name, repo_name)


def test_embed_updated_file(runner: CliRunner, db: Session) -> None:
    repo_arg = "seem/test-codal-repo"
    org_name, repo_name = repo_arg.split("/")
    head_commit_sha = "e1ed69c96180e7282d0b10f1bb36088a2789ccea"

    result = runner.invoke(
        cli,  # type: ignore
        ["embed", repo_arg, "--head", head_commit_sha],
        catch_exceptions=False,
    )

    assert result.exit_code == 0

    # Cloned repo points to the new head
    git_dir = settings.REPO_DIR / repo_arg
    git_repo = GitRepo(git_dir)
    assert git_repo.head.commit.hexsha == head_commit_sha

    # Database Repo points to the new head
    repo = db.execute(select(Repo)).scalar_one()
    assert repo.head_commit.sha == head_commit_sha

    # New Commit in the database
    commits = db.execute(select(Commit)).scalars().all()
    assert len(commits) == 2
    commit = next(commit for commit in commits if commit.sha == head_commit_sha)
    committed_datetime = datetime(2023, 7, 1, 11, 32, 49)
    assert commit.repo_id == repo.id
    assert commit.sha == head_commit_sha
    assert commit.message == "update readme\n"
    assert commit.author_name == "seem"
    assert commit.author_email and commit.author_email.endswith(".com")
    assert commit.authored_datetime == committed_datetime
    assert commit.committer_name == "seem"
    assert commit.committer_email and commit.author_email.endswith(".com")
    assert commit.committed_datetime == committed_datetime

    # Still the same Document in the database
    document = db.execute(select(Document)).scalar_one()
    assert document.repo_id == repo.id
    assert document.path == Path("README.md")

    # New DocumentVersion in the database
    document_versions = db.execute(select(DocumentVersion)).scalars().all()
    assert len(document_versions) == 2
    document_version = next(
        document_version
        for document_version in document_versions
        if document_version.commit_id == commit.id
    )
    assert document_version.document_id == document.id
    assert document_version.commit_id == commit.id
    assert (
        document_version.text
        == "# test-codal-repo\n\nThis repo is used solely to test [Codal](https://github.com/seeM/codal).\n"
    )
    assert document_version.num_tokens == 28
    assert document_version.processed == True

    # New Chunks in the database
    chunks = db.execute(select(Chunk)).scalars().all()
    assert len(chunks) == 2

    # Check the chunks themselves
    _test_chunks(document_version)

    # Index file is updated to reflect chunks
    head_chunks = [
        chunk for chunk in chunks if chunk.document_versions == [document_version]
    ]
    _test_index(head_chunks, org_name, repo_name)


def test_embed_new_file(runner: CliRunner, db: Session) -> None:
    repo_arg = "seem/test-codal-repo"
    org_name, repo_name = repo_arg.split("/")
    head_commit_sha = "58c9dd47390b6003c4006a29b2b9f1f01026b4c7"
    # TODO: The db should allow us to get the parent commit sha instead of hardcoding
    prev_head_commit_sha = "e1ed69c96180e7282d0b10f1bb36088a2789ccea"

    result = runner.invoke(
        cli,  # type: ignore
        ["embed", repo_arg, "--head", head_commit_sha],
        catch_exceptions=False,
    )

    assert result.exit_code == 0

    # New Document in the database
    documents = db.execute(select(Document)).scalars().all()
    assert len(documents) == 2

    unchanged_document = next(
        document for document in documents if document.path == Path("README.md")
    )

    new_document = next(
        document for document in documents if document.path == Path(".gitignore")
    )

    # New DocumentVersion in the database
    document_versions = db.execute(select(DocumentVersion)).scalars().all()
    assert len(document_versions) == 4

    # The unchanged document should have a new version that reuses the same chunks
    unchanged_document_version = next(
        document_version
        for document_version in document_versions
        if document_version.commit.sha == head_commit_sha
        and document_version.document_id == unchanged_document.id
    )
    unchanged_previous_document_version = next(
        document_version
        for document_version in document_versions
        if document_version.commit.sha == prev_head_commit_sha
        and document_version.document_id == unchanged_document.id
    )
    assert unchanged_document_version.text == unchanged_previous_document_version.text
    assert (
        unchanged_document_version.num_tokens
        == unchanged_previous_document_version.num_tokens
    )
    assert (
        unchanged_document_version.chunks == unchanged_previous_document_version.chunks
    )

    # There should also be a new version corresponding to the new file
    new_document_version = next(
        document_version
        for document_version in document_versions
        if document_version.commit.sha == head_commit_sha
        and document_version.document_id == new_document.id
    )

    assert new_document_version.text == "*.sqlite\n"
    assert new_document_version.num_tokens == 3
    assert new_document_version.processed == True

    # New Chunks in the database
    all_chunks = db.execute(select(Chunk)).scalars().all()
    assert len(all_chunks) == 3

    # Check the chunks themselves
    _test_chunks(new_document_version)

    # Index file is updated to reflect chunks
    head_chunks = [
        chunk
        for chunk in all_chunks
        if unchanged_document_version in chunk.document_versions
        or new_document_version in chunk.document_versions
    ]
    _test_index(head_chunks, org_name, repo_name)


def _test_chunks(document_version: DocumentVersion) -> None:
    chunks = sorted(document_version.chunks, key=lambda x: x.start)

    # Chunks are linked to documents and document versions
    for chunk in chunks:
        assert chunk.document_id == document_version.document.id
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


def _test_index(chunks: Sequence[Chunk], org_name: str, repo_name: str) -> None:
    # Creates an index file that fully covers chunks.
    index_path = settings.INDEX_DIR / f"{org_name}-{repo_name}.bin"
    # TODO: Don't hardcode the dimension
    index = load_index(index_path, dim=1536)
    assert index is not None
    assert index.get_ids_list() == [chunk.id for chunk in chunks]
