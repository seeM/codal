import hashlib
import json
import time
from functools import wraps
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import click
import numpy as np
import openai
import tiktoken
from django.core.management.base import BaseCommand, CommandError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from tqdm import tqdm

from . import __version__
from .database import SessionLocal
from .models import Chunk, Document, Org, Repo
from .settings import ARCHIVE_DIR, CACHE_DIR, EMBEDDING_DIR, MODEL_NAME


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
        except openai.error.OpenAIError as exception:
            print(f"OpenAI error: {exception.reason}")
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

    # Download the repo zip file
    zip_url = (
        f"https://github.com/{repo.org.name}/{repo.name}/archive/refs/heads/main.zip"
    )
    if repo.zip_path is not None and repo.zip_path.exists():
        click.echo(f"Using the cached archive: {repo.zip_path}")
    else:
        repo.zip_path = ARCHIVE_DIR / repo.org.name / f"{repo.name}.zip"
        repo.zip_path.parent.mkdir(parents=True, exist_ok=True)
        click.echo(f"Downloading the repo: {zip_url} -> {repo.zip_path}")
        with urlopen(zip_url) as response, repo.zip_path.open("wb") as out_fp:
            data = response.read()
            out_fp.write(data)
        db.add(repo)

    db.commit()

    # Read the documents from the archiv
    encoder = tiktoken.encoding_for_model(MODEL_NAME)
    documents = []
    with repo.zip_path.open("rb") as archive_fp:
        with ZipFile(archive_fp) as archive:
            for filename in archive.namelist():
                path = Path(filename)
                if path.is_dir():
                    continue
                with archive.open(filename) as fp:
                    try:
                        text = fp.read().decode()
                    except UnicodeDecodeError:
                        continue

                    path = path.relative_to(path.parts[0])

                    document = db.execute(
                        select(Document).where(
                            Document.path == path, Document.repo_id == repo.id
                        )
                    ).scalar_one_or_none()
                    if document is None:
                        document = Document(path=path, repo=repo)
                    else:
                        # A document already exists. If the text has changed, clear all of its
                        # chunks
                        if document.text != text:
                            num_deleted = db.execute(
                                delete(Chunk).where(Chunk.document_id == document.id)
                            ).rowcount
                            click.echo(
                                f"Document: {document.path} has changed. Deleting {num_deleted} "
                                "existing chunks"
                            )
                    document.text = text
                    document.num_tokens = len(encoder.encode(text))
                    documents.append(document)

    db.add_all(documents)
    # TODO: Should we commit here? Since if we error during chunk/embedding creation, we can end up
    #       in an unusable state.
    db.commit()

    dollars_per_1k_tokens = 0.0004
    num_tokens = sum(document.num_tokens for document in documents)
    click.echo(f"Documents: {len(documents)}")
    click.echo(f"Tokens: {num_tokens}")
    click.echo(f"Estimated price: ${dollars_per_1k_tokens * num_tokens / 1000}")
    # TODO: Estimated time

    embedding_dir = EMBEDDING_DIR / repo.org.name / repo.name
    embedding_dir.mkdir(parents=True, exist_ok=True)

    # Split each document into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_overlap=0, chunk_size=1000)
    chunks = []
    for document in tqdm(documents):
        if not document.text:
            continue

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
                    text=text,
                    embedding=embedding,
                )
                db.add(chunk)
                db.commit()
            else:
                assert chunk.text == chunk_text
                assert chunk.end == end

            chunks.append(chunk)

            start = end
