from typing import Iterable

import pytest
import sqlite_utils
from click.testing import CliRunner
from git.repo import Repo as GitRepo

from codal.ai import load_index
from codal.cli import cli
from codal.settings import settings


@pytest.fixture(scope="session")
def runner() -> Iterable[CliRunner]:
    yield CliRunner(mix_stderr=False)


@pytest.mark.serial
def test_embed_invalid_repo(runner: CliRunner) -> None:
    """
    Running `embed` with an invalid repo identifier prints an error message.
    """
    result = runner.invoke(
        cli, ["embed", "not-a-repo-identifier"], catch_exceptions=False  # type: ignore
    )
    assert result.stderr.splitlines()[-1].startswith("Error: REPO")
    assert result.exit_code == 1


@pytest.mark.serial
def test_embed_first_run(runner: CliRunner) -> None:
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

    # TODO: Replace with an in-memory database
    db = sqlite_utils.Database(settings.DB_PATH)

    # Creates an Org in the database
    [org] = db["orgs"].rows
    assert org["name"] == org_name

    # Creates a Repo in the database
    [repo] = db["repos"].rows
    assert repo["org_id"] == org["id"]
    assert repo["name"] == repo_name
    assert repo["default_branch"] == "main"
    assert repo["head_commit_id"] == 1

    # Creates a Commit in the database
    [commit] = db["commits"].rows
    committed_datetime = "2023-06-30T14:01:38+02:00"
    assert commit["repo_id"] == repo["id"]
    assert commit["sha"] == head_commit_sha
    assert commit["message"] == "rename"
    assert commit["author_name"] == "Wasim Lorgat"
    assert commit["author_email"] and commit["author_email"].endswith(".com")
    assert commit["authored_datetime"] == committed_datetime
    assert commit["committer_name"] == "GitHub"
    assert commit["committer_email"] == "noreply@github.com"
    assert commit["committed_datetime"] == committed_datetime

    # Creates a Document in the database
    [document] = db["documents"].rows
    assert document["repo_id"] == repo["id"]
    assert document["path"] == "README.md"

    # Creates a DocumentVersion in the database
    [document_version] = db["document_versions"].rows
    assert document_version["document_id"] == document["id"]
    assert document_version["commit_id"] == commit["id"]
    assert document_version["text"] == "# test-codal-repo\n"
    assert document_version["num_tokens"] == 7
    assert document_version["processed"] == True

    # Creates Chunks in the database
    chunks = list(db["chunks"].rows)
    assert len(chunks) == 1

    # Check the chunks themselves
    _test_chunks(document_version, db)

    # Index file is updated to reflect chunks
    _test_index(chunks, org_name, repo_name)


@pytest.mark.serial
def test_embed_updated_file(runner: CliRunner) -> None:
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

    # TODO: Replace with an in-memory database
    db = sqlite_utils.Database(settings.DB_PATH)

    # Database Repo points to the new head
    [repo] = db["repos"].rows
    assert db["commits"].get(repo["head_commit_id"])["sha"] == head_commit_sha

    # New Commit in the database
    commits = list(db["commits"].rows)
    assert len(commits) == 2
    commit = next(commit for commit in commits if commit["sha"] == head_commit_sha)
    committed_datetime = "2023-07-01T11:32:49+02:00"
    assert commit["repo_id"] == repo["id"]
    assert commit["sha"] == head_commit_sha
    assert commit["message"] == "update readme\n"
    assert commit["author_name"] == "seem"
    assert commit["author_email"] and commit["author_email"].endswith(".com")
    assert commit["authored_datetime"] == committed_datetime
    assert commit["committer_name"] == "seem"
    assert commit["committer_email"] and commit["author_email"].endswith(".com")
    assert commit["committed_datetime"] == committed_datetime

    # Still the same Document in the database
    [document] = db["documents"].rows
    assert document["repo_id"] == repo["id"]
    assert document["path"] == "README.md"

    # New DocumentVersion in the database
    document_versions = list(db["document_versions"].rows)
    assert len(document_versions) == 2
    document_version = next(
        document_version
        for document_version in document_versions
        if document_version["commit_id"] == commit["id"]
    )
    assert document_version["document_id"] == document["id"]
    assert document_version["commit_id"] == commit["id"]
    assert (
        document_version["text"]
        == "# test-codal-repo\n\nThis repo is used solely to test [Codal](https://github.com/seeM/codal).\n"
    )
    assert document_version["num_tokens"] == 28
    assert document_version["processed"] == True

    # New Chunks in the database
    chunks = list(db["chunks"].rows)
    assert len(chunks) == 2

    # Check the chunks themselves
    _test_chunks(document_version, db)

    # Index file is updated to reflect chunks
    head_chunks = [
        chunk
        for chunk in chunks
        if list(db["document_version_chunks"].rows_where("chunk_id = ?", [chunk["id"]]))
        == [{"document_version_id": document_version["id"], "chunk_id": chunk["id"]}]
    ]
    _test_index(head_chunks, org_name, repo_name)


@pytest.mark.serial
def test_embed_new_file(runner: CliRunner) -> None:
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

    db = sqlite_utils.Database(settings.DB_PATH)

    # New Document in the database
    documents = list(db["documents"].rows)
    assert len(documents) == 2

    unchanged_document = next(
        document for document in documents if document["path"] == "README.md"
    )

    new_document = next(
        document for document in documents if document["path"] == ".gitignore"
    )

    # New DocumentVersion in the database
    document_versions = list(db["document_versions"].rows)
    assert len(document_versions) == 4

    # The unchanged document should have a new version that reuses the same chunks
    unchanged_document_version = next(
        document_version
        for document_version in document_versions
        if db["commits"].get(document_version["commit_id"])["sha"] == head_commit_sha
        and document_version["document_id"] == unchanged_document["id"]
    )
    unchanged_previous_document_version = next(
        document_version
        for document_version in document_versions
        if db["commits"].get(document_version["commit_id"])["sha"]
        == prev_head_commit_sha
        and document_version["document_id"] == unchanged_document["id"]
    )
    assert (
        unchanged_document_version["text"]
        == unchanged_previous_document_version["text"]
    )
    assert (
        unchanged_document_version["num_tokens"]
        == unchanged_previous_document_version["num_tokens"]
    )
    assert set(
        row["chunk_id"]
        for row in db["document_version_chunks"].rows_where(
            "document_version_id = ?", [unchanged_document_version["id"]]
        )
    ) == set(
        row["chunk_id"]
        for row in db["document_version_chunks"].rows_where(
            "document_version_id = ?", [unchanged_previous_document_version["id"]]
        )
    )

    # There should also be a new version corresponding to the new file
    new_document_version = next(
        document_version
        for document_version in document_versions
        if db["commits"].get(document_version["commit_id"])["sha"] == head_commit_sha
        and document_version["document_id"] == new_document["id"]
    )

    assert new_document_version["text"] == "*.sqlite\n"
    assert new_document_version["num_tokens"] == 3
    assert new_document_version["processed"] == True

    # New Chunks in the database
    all_chunks = list(db["chunks"].rows)
    assert len(all_chunks) == 3

    # Check the chunks themselves
    _test_chunks(new_document_version, db)

    # Index file is updated to reflect chunks
    head_chunks = [
        chunk
        for chunk in all_chunks
        if {unchanged_document_version["id"], new_document_version["id"]}
        & set(
            row["document_version_id"]
            for row in db["document_version_chunks"].rows_where(
                "chunk_id = ?", [chunk["id"]]
            )
        )
    ]
    _test_index(head_chunks, org_name, repo_name)


def _test_chunks(document_version, db) -> None:
    chunk_ids = [
        row["chunk_id"]
        for row in db["document_version_chunks"].rows_where(
            "document_version_id = ?", [document_version["id"]]
        )
    ]
    chunks = list(db["chunks"].rows_where("id in (?)", chunk_ids, order_by="start"))

    # Chunks are linked to documents and document versions
    for chunk in chunks:
        assert chunk["document_id"] == document_version["document_id"]
        assert list(
            db["document_version_chunks"].rows_where("chunk_id = ?", [chunk["id"]])
        ) == [
            {
                "chunk_id": chunk["id"],
                "document_version_id": document_version["id"],
            }
        ]

    # Chunks cover the entire document
    assert chunks[0]["start"] == 0
    assert chunks[-1]["end"] == len(document_version["text"].strip())

    # Each chunk's start should be the previous chunk's end
    for i in range(len(chunks) - 1):
        assert chunks[i]["end"] == chunks[i + 1]["start"]

    # Joining the chunk texts should equal the document text
    # TODO: This might not actually work when we have multiple chunks because we may currently
    #       be losing newlines...
    chunk_text = "".join(chunk["text"] for chunk in chunks)
    assert chunk_text == document_version["text"].strip()

    # TODO: Creates the default branch file? Or test the end result?
    #       If we run with an existing repo dir but no DB entry, we should still set a default branch
    #       There's also an assert in embed so maybe that's good enough


def _test_index(chunks, org_name: str, repo_name: str) -> None:
    # Creates an index file that fully covers chunks.
    index_path = settings.INDEX_DIR / f"{org_name}-{repo_name}.bin"
    # TODO: Don't hardcode the dimension
    index = load_index(index_path, dim=1536)
    assert index is not None
    assert index.get_ids_list() == [chunk["id"] for chunk in chunks]
