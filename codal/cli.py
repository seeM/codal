import time
from functools import wraps
from typing import List, Optional, Tuple

import click
import numpy as np
import openai
import tiktoken
from git.repo import Repo as GitRepo
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy import select
from sqlalchemy.orm import Session
from tqdm import tqdm

from .database import SessionLocal
from .models import Chunk, Document, DocumentVersion, Org, Repo, Commit
from .settings import MODEL_NAME, REPO_DIR
from .version import __version__


def _provide_db(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with SessionLocal() as db:
            kwargs["db"] = db
            return func(*args, **kwargs)

    return wrapper


def _get_embedding(text, model=MODEL_NAME, max_attempts=5, retry_delay=1) -> List[int]:
    text = text.replace("\n", " ")
    for attempt in range(max_attempts):
        try:
            return openai.Embedding.create(input=[text], model=model)["data"][0][  # type: ignore
                "embedding"
            ]
        # TODO: Only retry on 500 error
        except openai.OpenAIError as exception:
            click.echo(f"OpenAI error: {exception}")
            if attempt < max_attempts - 1:  # No delay on last attempt, TODO: why not?
                time.sleep(retry_delay)
            else:
                raise


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """
    Codal is an open source tool for understanding code repos using large language models.
    """
    pass


@cli.command()
@click.argument("repo")
@click.option("--head", help="Commit hash to embed.")
@_provide_db
def embed(repo, db: Session, head: Optional[str]) -> None:
    """
    Embed a GitHub repo.

    This includes downloading the repo, splitting each document into chunks, and embedding those
    chunks.

    For example, to embed the Codal repo:

        codal embed seem/codal
    """
    # Create the organization and repo
    org_name, repo_name = repo.split("/")

    org = db.execute(select(Org).where(Org.name == org_name)).scalar_one_or_none()
    if org is None:
        org = Org(name=org_name)
        db.add(org)

    db.commit()

    repo = db.execute(
        select(Repo).where(Repo.name == repo_name, Repo.org == org)
    ).scalar_one_or_none()
    if repo is None:
        repo = Repo(name=repo_name, org=org)

    # Clone or pull the repo
    git_url = f"https://github.com/{repo.org.name}/{repo.name}.git"
    git_dir = REPO_DIR / repo.org.name / repo.name
    git_dir.mkdir(parents=True, exist_ok=True)
    default_branch_path = git_dir / ".codal" / "DEFAULT_BRANCH"
    if (git_dir / ".git").exists():
        click.echo(f"Fetching latest changes: {git_dir}")
        git_repo = GitRepo(git_dir)
        origin = git_repo.remote(name="origin")
        origin.fetch()
    else:
        click.echo(f"Cloning repo: {git_url} -> {git_dir}")
        git_repo = GitRepo.clone_from(git_url, git_dir)
        default_branch_path.parent.mkdir(parents=True, exist_ok=True)
        default_branch_path.write_text(git_repo.active_branch.name + "\n")

    if repo.default_branch is None:
        repo.default_branch = default_branch_path.read_text().strip()

    if head is not None:
        click.echo(f"Checking out: {head}")
        git_repo.git.checkout(head)
    else:
        click.echo(f"Checking out: origin/{repo.default_branch}")
        git_repo.git.checkout(f"origin/{repo.default_branch}")

    git_commit = git_repo.head.commit

    db.add(repo)
    db.commit()

    commit = db.execute(
        select(Commit).where(Commit.repo == repo, Commit.sha == git_commit.hexsha)
    ).scalar_one_or_none()
    if commit is None:
        commit = Commit(
            repo=repo,
            sha=git_commit.hexsha,
            message=str(git_commit.message),
            author_name=git_commit.author.name,
            author_email=git_commit.author.email,
            authored_datetime=git_commit.authored_datetime,
            committer_name=git_commit.committer.name,
            committer_email=git_commit.committer.email,
            committed_datetime=git_commit.committed_datetime,
        )

    prev_head = repo.head_commit
    db.add(commit)
    db.commit()

    # Read documents from the repo
    encoder = tiktoken.encoding_for_model(MODEL_NAME)
    document_versions = []
    for path in git_dir.rglob("*"):
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

            # Get or create the document
            document = db.execute(
                select(Document).where(
                    Document.repo == repo,
                    Document.path == path,
                )
            ).scalar_one_or_none()
            if document is None:
                document = Document(repo=repo, path=path)
                db.add(document)
                # TODO: need to commit?
                db.commit()

            # If a version does not exist for this commit, create one.
            document_version = db.execute(
                select(DocumentVersion).where(
                    DocumentVersion.document == document,
                    DocumentVersion.commit == commit,
                )
            ).scalar_one_or_none()
            if document_version is None:
                document_version = DocumentVersion(
                    document=document,
                    commit=commit,
                    text=text,
                    num_tokens=len(encoder.encode(text)),
                )
            else:
                assert document_version.text == text

            # Use the chunks from the previous head, if we can.
            if not document_version.processed:
                previous_document_version = db.execute(
                    select(DocumentVersion).where(
                        DocumentVersion.document == document,
                        DocumentVersion.commit == prev_head,
                    )
                ).scalar_one_or_none()
                if (
                    previous_document_version is not None
                    and previous_document_version.processed
                    and document_version.text == previous_document_version.text
                ):
                    document_version.chunks = previous_document_version.chunks
                    document_version.processed = True

            db.add_all([document, document_version])
            db.commit()

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
    click.echo(f"Documents to process: {num_unprocessed_documents}")
    click.echo(f"Tokens to process: {num_unprocessed_tokens}")
    click.echo(
        f"Estimated price: ${dollars_per_1k_tokens * num_unprocessed_tokens / 1000}"
    )
    # TODO: Estimated time

    # Process documents
    splitter = RecursiveCharacterTextSplitter(chunk_overlap=0, chunk_size=1000)
    for document_version in tqdm(unprocessed_document_versions):
        chunks = []
        start = 0
        end = start
        for chunk_text in splitter.split_text(document_version.text):
            end += len(chunk_text)

            embedding_raw = _get_embedding(chunk_text)
            embedding = np.array(embedding_raw)
            chunk = Chunk(
                document=document_version.document,
                start=start,
                end=end,
                text=chunk_text,
                embedding=embedding,
            )

            chunks.append(chunk)

            start = end

        document_version.chunks = chunks
        document_version.processed = True

        db.add(document_version)
        db.commit()

    # Finally, bump the head commit
    repo.head_commit = commit
    db.add(repo)
    db.commit()

    click.echo(f"Repo updated to head: {commit.sha}")


@cli.command()
@click.argument("repo")
@click.argument("question")
@click.option("--num-neighbors", default=10)
@_provide_db
def ask(repo, question, db: Session, num_neighbors: int) -> None:
    """
    Ask a question about a GitHub repo.

    The repo must already have been embedded (see `codal embed`).

    For example:

        codal ask seem/codal 'What dependencies does this project have?'
    """
    repo_arg = repo
    org_name, repo_name = repo_arg.split("/")
    repo = db.execute(
        select(Repo).join(Repo.org).where(Repo.name == repo_name, Org.name == org_name)
    ).scalar_one_or_none()
    if repo is None:
        click.echo(
            f"Repo does not exist. Have you run `codal embed {repo_arg}`?", err=True
        )
        raise click.exceptions.Exit(1)

    # TODO: This should probably live elsewhere, maybe in embed or a separate command
    # Make the vector search index
    chunks = (
        db.execute(
            (
                select(Chunk)
                .join(Chunk.document_versions)
                .join(DocumentVersion.document)
                .where(
                    Document.repo == repo, DocumentVersion.commit == repo.head_commit
                )
            )
        )
        .scalars()
        .all()
    )

    # TODO: Need to investigate
    chunks = [chunk for chunk in chunks if chunk.document.path.name != "LICENSE"]

    # click.echo(
    #     f"Found {len(chunks)} chunks for this repo at head: {repo.head_commit.sha}"
    # )
    # click.echo(f"First chunk: {chunks[0]}")

    embeddings = np.array([chunk.embedding for chunk in chunks])
    chunk_ids = np.array([chunk.id for chunk in chunks])

    # click.echo(f"Embeddings shape: {embeddings.shape}")

    import hnswlib

    max_elements = embeddings.shape[0]
    dim = embeddings.shape[1]
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=max_elements)
    index.add_items(embeddings, chunk_ids)

    # click.echo(f"Created index: {index}")

    # TODO:
    # index.save_index(path)
    # index = hnswlib.Index(space="cosine", dim=dim)
    # index.load_index(path, max_elements=new_max_elements)

    # Find the top nearest neighbours

    # TODO: Customize k
    query = np.array(_get_embedding(question))
    nn_ids, distances = index.knn_query(query, k=num_neighbors)

    # Elements have to be Python ints (not numpy ints) else the sqlalchemy query below returns empty
    nn_ids = [int(id) for id in nn_ids[0]]

    # click.echo(f"Found knn: {nn_ids}, distances: {distances[0]}")

    nn_chunks = db.execute(select(Chunk).where(Chunk.id.in_(nn_ids))).scalars().all()

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

    # click.echo(prompt, err=True)

    num_tokens, cost = estimate_cost(prompt)
    if num_tokens > 5000:
        click.echo(f"Unexpectedly high number of tokens: {num_tokens}", err=True)
        raise click.exceptions.Exit(1)

    # click.echo(f"Number of tokens: {num_tokens}", err=True)
    # click.echo(f"Estimated cost: ${cost}", err=True)

    # Get confirmation from the user
    # if not click.confirm("Continue?", err=True):
    #     raise click.Abort()

    result = complete(prompt, "gpt-4")


def estimate_cost(prompt: str) -> Tuple[int, float]:
    encoder = tiktoken.encoding_for_model("gpt-4")
    num_tokens = len(encoder.encode(prompt))
    cost_per_token = 0.004 / 1000  # Conservative estimate
    cost = num_tokens * cost_per_token
    return num_tokens, cost


def complete(prompt: str, model: str) -> str:
    result = []
    content = None
    for chunk in openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=True,
    ):
        content = chunk.choices[0].get("delta", {}).get("content")  # type: ignore
        if content is not None:
            result.append(content)
            click.echo(content, nl=False)

    click.echo()

    return "".join(result)
