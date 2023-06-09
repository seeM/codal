import time
from functools import wraps

import click
import numpy as np
import openai
import tiktoken
from git import Repo as GitRepo
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy import select
from sqlalchemy.orm import Session
from tqdm import tqdm

from . import __version__
from .database import SessionLocal
from .models import Chunk, Document, Org, Repo
from .settings import MODEL_NAME, REPO_DIR


def _provide_db(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with SessionLocal() as db:
            kwargs["db"] = db
            return func(*args, **kwargs)

    return wrapper


def _get_embedding(text, model=MODEL_NAME, max_attempts=5, retry_delay=1):
    text = text.replace("\n", " ")
    for attempt in range(max_attempts):
        try:
            return openai.Embedding.create(input=[text], model=model)["data"][0][
                "embedding"
            ]
        # TODO: Only retry on 500 error
        except openai.error.OpenAIError as exception:
            click.echo(f"OpenAI error: {exception}")
            if attempt < max_attempts - 1:  # No delay on last attempt, TODO: why not?
                time.sleep(retry_delay)
            else:
                raise


@click.group()
@click.version_option(version=__version__)
def cli():
    """
    Codal is an open source tool for understanding code repos using large language models.
    """
    pass


@cli.command()
@click.argument("repo")
@_provide_db
def embed(repo, db: Session):
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

    repo = db.execute(
        select(Repo).where(Repo.name == repo_name, Repo.org_id == org.id)
    ).scalar_one_or_none()
    if repo is None:
        repo = Repo(name=repo_name, org=org)
        db.add(repo)

    db.commit()

    # Clone or pull the repo
    git_url = f"https://github.com/{repo.org.name}/{repo.name}.git"
    git_dir = REPO_DIR / repo.org.name / repo.name
    git_dir.mkdir(parents=True, exist_ok=True)
    if (git_dir / ".git").exists():
        click.echo(f"Updating repo: {git_dir}")
        git_repo = GitRepo(git_dir)
        origin = git_repo.remote(name="origin")
        origin.pull()
    else:
        click.echo(f"Cloning repo: {git_url} -> {git_dir}")
        git_repo = GitRepo.clone_from(git_url, git_dir)

    head = git_repo.head.commit.hexsha

    # Read documents from the repo
    encoder = tiktoken.encoding_for_model(MODEL_NAME)
    documents = []
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

            document = db.execute(
                select(Document).where(
                    Document.repo_id == repo.id,
                    Document.path == path,
                    Document.head == head,
                )
            ).scalar_one_or_none()

            if document is None:
                document = Document(
                    path=path,
                    repo=repo,
                    head=head,
                    text=text,
                    num_tokens=len(encoder.encode(text)),
                )
            else:
                assert document.text == text

            documents.append(document)

    db.add_all(documents)
    db.commit()

    unprocessed_documents = [
        document for document in documents if not document.processed
    ]

    # Log stats about documents, tokens, and cost to process
    dollars_per_1k_tokens = 0.0004
    num_documents = len(documents)
    num_tokens = sum(document.num_tokens for document in documents)
    num_unprocessed_documents = len(unprocessed_documents)
    num_unprocessed_tokens = sum(
        document.num_tokens for document in unprocessed_documents
    )
    click.echo(f"Documents: {num_documents}")
    click.echo(f"Documents to process: {num_unprocessed_documents}")
    click.echo(f"Tokens: {num_tokens}")
    click.echo(f"Tokens to process: {num_unprocessed_tokens}")
    click.echo(
        f"Estimated price: ${dollars_per_1k_tokens * num_unprocessed_tokens / 1000}"
    )
    # TODO: Estimated time

    # Process documents
    splitter = RecursiveCharacterTextSplitter(chunk_overlap=0, chunk_size=1000)
    for document in tqdm(unprocessed_documents):
        chunks = []
        start = 0
        end = start
        for chunk_num, chunk_text in enumerate(splitter.split_text(document.text)):
            end += len(chunk_text)

            chunk = db.execute(
                select(Chunk).where(
                    Chunk.document_id == document.id, Chunk.start == start
                )
            ).scalar_one_or_none()
            if chunk is None:
                embedding_raw = _get_embedding(chunk_text)
                embedding = np.array(embedding_raw)
                chunk = Chunk(
                    document_id=document.id,
                    start=start,
                    end=end,
                    text=chunk_text,
                    embedding=embedding,
                )
            else:
                assert chunk.text == chunk_text
                assert chunk.end == end

            chunks.append(chunk)

            start = end

        document.processed = True

        db.add_all(chunks)
        db.add(document)
        db.commit()

    repo.head = head
    db.add(repo)
    db.commit()
