from functools import wraps
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple, TypeVar

import click
import hnswlib
import numpy as np
import tiktoken
import uvicorn
from git.repo import Repo as GitRepo
from git.exc import GitCommandError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session

from . import crud
from .database import SessionLocal, migrate
from .models import Chunk, Repo
from .ai import get_chat_completion, get_embedding
from .schemas import (
    CommitCreate,
    DocumentCreate,
    DocumentVersionCreate,
    DocumentVersionUpdate,
    OrgCreate,
    RepoCreate,
    RepoUpdate,
)
from .settings import INDEX_DIR, EMBEDDING_MODEL_NAME, REPO_DIR
from .version import __version__


def pretty_print(obj: Any):
    if isinstance(obj, Path):
        return str(obj).replace(str(Path.home()), "~")
    return str(obj)


def echo_progress(prefix: str, current: int, total: int) -> None:
    if total == 0:
        return
    percentage = f"{100 * current / total:.0f}%"
    template = prefix + ": {percentage} ({current}/{total})"
    message = template.format(percentage=percentage, current=current, total=total)
    if current < total:
        message += "\r"
        nl = False
    else:
        message += ", done"
        nl = True
    click.echo(message, err=True, nl=nl)


T = TypeVar("T")


def progress(it: Sequence[T], prefix: str) -> Iterable[T]:
    total = len(it)
    for current, obj in enumerate(it):
        echo_progress(prefix, current + 1, total)
        yield obj


def _provide_db(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        migrate()
        with SessionLocal() as db:
            kwargs["db"] = db
            return func(*args, **kwargs)

    return wrapper


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """
    Codal is an open source tool for understanding code repos using large language models.
    """
    pass


_REPO_DESCRIPTION = (
    'REPO should uniquely identify a valid GitHub repo, e.g. "seem/codal".'
)


@cli.command()
@click.argument("repo")
@click.option("--head", help="Commit hash to embed.")
@_provide_db
def embed(repo, db: Session, head: Optional[str]) -> None:
    f"""
    Embed a GitHub REPO.

    {_REPO_DESCRIPTION}

    The first run clones the repo. Subsequent runs pull the latest changes.

    Example:

        codal embed seem/codal
    """
    repo_arg = repo
    try:
        org_name, repo_name = repo_arg.split("/")
    except ValueError:
        raise click.UsageError(
            _REPO_DESCRIPTION,
        )

    # Clone or pull the repo
    git_url = f"https://github.com/{org_name}/{repo_name}.git"
    git_dir = REPO_DIR / org_name / repo_name
    default_branch_path = git_dir / ".codal" / "DEFAULT_BRANCH"
    if (git_dir / ".git").exists():
        git_repo = GitRepo(git_dir)
        origin = git_repo.remote(name="origin")
        click.echo(f"Fetching latest changes: {git_dir}", err=True)
        origin.fetch()
    else:
        click.echo(f"Cloning repo: {git_url} -> {pretty_print(git_dir)}", err=True)
        try:
            git_repo = GitRepo.clone_from(git_url, git_dir)
        except GitCommandError as exception:
            if "remote: Repository not found." in exception.stderr:
                raise click.exceptions.ClickException(f"Repo not found: {git_url}")
            raise

        # Store the default branch as a file so that we can recover it if the db is lost
        default_branch_path.parent.mkdir(parents=True, exist_ok=True)
        default_branch_path.write_text(git_repo.active_branch.name + "\n")

    # Get or create the organization and repo
    org_name, repo_name = repo_arg.split("/")
    org = crud.org.get_or_create(db, OrgCreate(name=org_name))
    repo = crud.repo.get_or_create(db, RepoCreate(name=repo_name, org_id=org.id))

    # Update the default branch
    if repo.default_branch is None:
        default_branch = default_branch_path.read_text().strip()
        repo = crud.repo.update(
            db, db_obj=repo, obj_in=RepoUpdate(default_branch=default_branch)
        )

    assert repo.default_branch is not None

    if head is None:
        head = f"origin/{repo.default_branch}"

    if git_repo.head.commit != git_repo.commit(head):
        click.echo(f"Checking out: {head}")
        git_repo.git.checkout(head)

    git_commit = git_repo.head.commit
    commit = crud.commit.get_or_create(
        db,
        CommitCreate(
            repo_id=repo.id,
            sha=git_commit.hexsha,
            message=str(git_commit.message),
            author_name=git_commit.author.name,
            author_email=git_commit.author.email,
            authored_datetime=git_commit.authored_datetime,
            committer_name=git_commit.committer.name,
            committer_email=git_commit.committer.email,
            committed_datetime=git_commit.committed_datetime,
        ),
    )

    prev_head = repo.head_commit

    # Read documents from the repo
    encoder = tiktoken.encoding_for_model(EMBEDDING_MODEL_NAME)
    document_versions = []
    for path in progress(
        list(git_dir.rglob("*")),
        "Checking files for changes",
    ):
        if path.is_dir():
            continue
        if ".git" in path.parts:
            continue

        with path.open() as fp:
            try:
                text = fp.read()
            except UnicodeDecodeError:
                continue

            if not text:
                continue

            path = path.relative_to(git_dir)

            # Get or create the document and version
            document = crud.document.get_or_create(
                db, DocumentCreate(repo_id=repo.id, path=path)
            )
            document_version = crud.document_version.get_or_create(
                db,
                DocumentVersionCreate(
                    document_id=document.id,
                    commit_id=commit.id,
                    text=text,
                    num_tokens=len(encoder.encode(text)),
                ),
            )

            # Use the chunks from the previous head, if we can.
            if not document_version.processed and prev_head is not None:
                previous_document_version = crud.document_version.get(
                    db,
                    document_id=document.id,
                    commit_id=prev_head.id,
                )
                if (
                    previous_document_version is not None
                    and previous_document_version.processed
                    and document_version.text == previous_document_version.text
                ):
                    crud.document_version.update(
                        db,
                        document_version,
                        DocumentVersionUpdate(
                            chunks=previous_document_version.chunks,
                            processed=True,
                        ),
                    )

            document_versions.append(document_version)

    unprocessed_document_versions = [
        document_version
        for document_version in document_versions
        if not document_version.processed
    ]

    # Log stats about documents, tokens, and cost to process
    dollars_per_1k_tokens = 0.0004
    num_unprocessed_documents = len(unprocessed_document_versions)
    num_unprocessed_tokens = sum(
        document_version.num_tokens
        for document_version in unprocessed_document_versions
    )
    estimated_cost = dollars_per_1k_tokens * num_unprocessed_tokens / 1000
    click.echo(
        f" {num_unprocessed_documents} file(s) changed, {num_unprocessed_tokens} tokens to embed, ${estimated_cost:.3f} estimated cost"
    )
    # TODO: Estimated time

    # Process documents
    splitter = RecursiveCharacterTextSplitter(chunk_overlap=0, chunk_size=1000)
    num_processed_tokens = 0
    embed_message = "Embedding tokens"
    echo_progress(
        embed_message,
        num_processed_tokens,
        num_unprocessed_tokens,
    )
    for document_version in unprocessed_document_versions:
        chunks = []
        start = 0
        end = start
        for chunk_text in splitter.split_text(document_version.text):
            end += len(chunk_text)

            embedding = get_embedding(chunk_text)
            chunk = Chunk(
                document=document_version.document,
                start=start,
                end=end,
                text=chunk_text,
                embedding=embedding,
            )

            chunks.append(chunk)

            start = end

            num_processed_tokens += len(encoder.encode(chunk_text))
            echo_progress(
                "Embedding changed files, tokens processed",
                num_processed_tokens,
                num_unprocessed_tokens,
            )

        crud.document_version.update(
            db, document_version, DocumentVersionUpdate(chunks=chunks, processed=True)
        )

    # NOTE: I'm not sure why chunk tokens don't add up to document version tokens.
    #       Maybe because of the way we split the text e.g. stripping chunks?
    if num_processed_tokens != num_unprocessed_tokens:
        echo_progress(
            embed_message,
            num_unprocessed_tokens,
            num_unprocessed_tokens,
        )

    # Finally, bump the head commit
    if repo.head_commit != commit:
        crud.repo.update(db, repo, RepoUpdate(head_commit_id=commit.id))
        click.echo(f"Repo head updated: {commit.sha[:7]}", err=True)

    reindex(repo, db)


def _get_repo_or_raise(db: Session, org_and_repo: str) -> Repo:
    org_name, repo_name = org_and_repo.split("/")
    repo = crud.repo.get(db, name=repo_name, org_name=org_name)
    if repo is None:
        click.echo(
            f"Repo does not exist. Have you run `codal embed {org_and_repo}`?", err=True
        )
        raise click.exceptions.Exit(1)
    return repo


def reindex(repo: Repo, db: Session) -> hnswlib.Index:
    """
    Rebuild the vector search index for a repo.
    """
    # Make the vector search index
    chunks = crud.chunk.get_multi_by_repo(db, repo=repo)

    # TODO: Need to investigate, but I found in tests that LICENSE always appears at the top of
    #       my search results. I suspect because it's in English and is therefore the most similar
    #       to the query. For now, just remove it from the index.
    chunks = [chunk for chunk in chunks if chunk.document.path.name != "LICENSE"]

    click.echo(
        f"Found {len(chunks)} chunks for this repo at head: {repo.head_commit.sha}"
    )

    embeddings = np.array([chunk.embedding for chunk in chunks])
    chunk_ids = np.array([chunk.id for chunk in chunks])

    click.echo(f"Embeddings shape: {embeddings.shape}")

    max_elements = embeddings.shape[0]
    dim = embeddings.shape[1]
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=max_elements)
    index.add_items(embeddings, chunk_ids)

    click.echo(f"Created index: {index}")

    # TODO: Safely overwrite the previous index?
    # TODO: Make Repo property?
    index_path = INDEX_DIR / f"{repo.org.name}-{repo.name}.bin"
    index_path.parent.mkdir(exist_ok=True, parents=True)
    index.save_index(str(index_path))
    return index


# TODO: How to reuse parameter defaults like num_neighbors?
def _ask(repo, question: str, db: Session, num_neighbors=10, debug=False) -> str:
    repo_arg = repo

    repo = _get_repo_or_raise(db, repo)

    # TODO: Make Repo property
    index_path = INDEX_DIR / f"{repo.org.name}-{repo.name}.bin"

    # TODO: Do we really have to store the dimension somewhere?
    #       SQL table?
    index = hnswlib.Index(space="cosine", dim=1536)
    if not index_path.is_file():
        click.echo(
            f"Index does not exist. Have you run `codal embed {repo_arg}`?", err=True
        )
        raise click.exceptions.Exit(1)
    index.load_index(str(index_path))

    # Find the top nearest neighbours

    query = get_embedding(question)
    nn_ids, _ = index.knn_query(query, k=num_neighbors)

    # Elements have to be Python ints (not numpy ints) else the sqlalchemy query below returns empty
    nn_ids = [int(id) for id in nn_ids[0]]

    nn_chunks = crud.chunk.get_multi_by_id(db, ids=nn_ids)

    context = "\n\n".join(
        [f"### {chunk.document.path}\n\n" + chunk.text for chunk in nn_chunks]
    )

    # Make the LLM prompt given the vector search context
    prompt = (
        "Given the following context and code, answer the following question.\n"
        "- Do not use outside context, and do not assume the user can see the provided context.\n"
        "- Try to be as detailed as possible and reference the components that you are looking at.\n"
        "- Keep in mind that these are only code snippets, and more snippets may be added during the conversation.\n"
        "- Do not generate code, only reference the exact code snippets that you have been provided with.\n"
        "- If you are going to write code, make sure to specify the language of the code.\n"
        "    - For example, if you were writing Python, you would write the following:\n"
        "      ```python\n"
        "      <python code goes here>\n"
        "      ```\n\n"
        "## Context\n\n"
        f"{context}\n\n"
        "## Question\n\n"
        f"{question}\n\n"
        "## Answer\n\n"
    )

    if debug:
        nn_paths = [str(chunk.document.path) for chunk in nn_chunks]
        click.echo(f"Nearest neighbours: {nn_paths}", err=True)
        click.echo(prompt, err=True)

    num_tokens, cost = _estimate_cost(prompt)
    if num_tokens > 5000:
        click.echo(f"Unexpectedly high number of tokens: {num_tokens}", err=True)
        raise click.exceptions.Exit(1)

    # click.echo(f"Number of tokens: {num_tokens}", err=True)
    # click.echo(f"Estimated cost: ${cost}", err=True)

    # Get confirmation from the user
    # if not click.confirm("Continue?", err=True):
    #     raise click.Abort()

    result = _complete(prompt, "gpt-4")
    return result


@cli.command()
@click.argument("repo")
@click.argument("question")
@click.option("--num-neighbors", default=10)
@click.option("--debug", is_flag=True)
@_provide_db
def ask(repo, question, db: Session, num_neighbors: int, debug: bool) -> None:
    """
    Ask a question about a GitHub repo.

    The repo must already have been embedded (see `codal embed`).

    For example:

        codal ask seem/codal 'What dependencies does this project have?'
    """
    _ask(repo, question, db, num_neighbors, debug)


def _estimate_cost(prompt: str) -> Tuple[int, float]:
    encoder = tiktoken.encoding_for_model("gpt-4")
    num_tokens = len(encoder.encode(prompt))
    # TODO: More accurate estimate
    cost_per_token = 0.004 / 1000  # Conservative estimate
    cost = num_tokens * cost_per_token
    return num_tokens, cost


def _complete(prompt: str, model: str) -> str:
    result = []
    content = None
    for content in get_chat_completion(prompt, model=model):
        if content is not None:
            result.append(content)
            click.echo(content, nl=False)

    click.echo()

    return "".join(result)


@cli.command()
def serve():
    """
    Serve the Codal API.
    """
    uvicorn.run("codal.api:app")


@cli.command()
@_provide_db
def check(db: Session):
    """
    Check that the database is in a consistent state.
    """
    from sqlalchemy import select

    from .models import DocumentVersion

    document_versions = (
        db.execute(
            select(DocumentVersion).where(
                DocumentVersion.chunks == None, DocumentVersion.processed == True
            )
        )
        .scalars()
        .all()
    )

    if document_versions:
        click.echo(
            f"Found {len(document_versions)} processed document versions with no chunks, resetting processed to false",
            err=True,
        )
        for document_version in document_versions:
            crud.document_version.update(
                db, document_version, DocumentVersionUpdate(processed=False)
            )
