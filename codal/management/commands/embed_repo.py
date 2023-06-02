import hashlib
import json
import time
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import openai
import tiktoken
from django.core.management.base import BaseCommand, CommandError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from codal.constants import ARCHIVE_DIR, CACHE_DIR, EMBEDDING_DIR, MODEL_NAME
from codal.models import Chunk, Document, Embedding, Organization, Repo


class Command(BaseCommand):
    help = (
        "Download a GitHub repo, split each document into chunks, and get embeddings."
    )

    def add_arguments(self, parser):
        parser.add_argument("repo", help="Repo identifier e.g. 'seeM/codal'")

    def handle(self, repo, **options):
        # Create the organization and repo
        org_name, repo_name = repo.split("/")
        org, _ = Organization.objects.update_or_create(name=org_name)
        repo, _ = Repo.objects.update_or_create(name=repo_name, organization=org)

        # Download the repo zip file
        zip_url = (
            f"https://github.com/{org.name}/{repo.name}/archive/refs/heads/main.zip"
        )
        if repo.zip_path:
            zip_path = Path(repo.zip_path)
            if zip_path.exists():
                self.stdout.write(f"Using the cached archive: {zip_path}")
        else:
            zip_path = ARCHIVE_DIR / f"{repo}.zip"
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            self.stdout.write(f"Downloading the repo: {zip_url} -> {zip_path}")
            with urlopen(zip_url) as response, zip_path.open("wb") as out_fp:
                data = response.read()
                out_fp.write(data)
            repo.zip_path = str(zip_path)
            repo.save()

        # Read the documents from the archive
        documents = []
        with zip_path.open("rb") as archive_fp:
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

                        if not len(text):
                            continue

                        path = path.relative_to(path.parts[0])
                        document, _ = Document.objects.update_or_create(
                            path=path, repo=repo, defaults={"text": text}
                        )
                        documents.append(document)

        exit()

        embedding_dir = EMBEDDING_DIR / args.repo
        embedding_dir.mkdir(parents=True, exist_ok=True)

        # Load the embeddings metadata file
        embedding_metadata_path = embedding_dir / "metadata.json"
        if embedding_metadata_path.exists():
            with embedding_metadata_path.open() as fp:
                embedding_metadata = json.load(fp)
        else:
            embedding_metadata = {"documents": []}
        for doc in embedding_metadata["documents"]:
            doc["path"] = Path(doc["path"])
        embedding_metadata["documents"] = {
            doc["path"]: doc for doc in embedding_metadata["documents"]
        }

        # Split each document into chunks
        encoder = tiktoken.encoding_for_model(MODEL_NAME)
        splitter = RecursiveCharacterTextSplitter(chunk_overlap=0, chunk_size=1000)
        chunks = []
        for document in documents:
            for chunk_num, chunk_text in enumerate(
                splitter.split_text(document["text"])
            ):
                # Determine if the embedding needs to be updated
                embedding_path = (
                    embedding_dir / f"{document['path']}.{chunk_num:04}.npy"
                )
                needs_update = True
                if embedding_path.exists():
                    embedding = np.load(embedding_path)

                    # Check if the hash changed
                    # hasher =

                # TODO: Validate the hash of the embedding

                chunk = {
                    "path": document["path"],
                    "chunk_num": chunk_num,
                    "text": chunk_text,
                    "embedding_path": embedding_path,
                    "needs_update": document["needs_update"],
                    "num_tokens": len(encoder.encode(chunk_text)),
                }
                chunks.append(chunk)

        print(f"Documents: {len(documents)}")
        print(
            f"Documents to update: {sum(document['needs_update'] for document in documents)}"
        )
        print(f"Chunks: {len(chunks)}")
        print(f"Chunks to update: {sum(chunk['needs_update'] for chunk in chunks)}")
        print(f"Tokens: {sum(chunk['num_tokens'] for chunk in chunks)}")
        tokens_to_update = sum(
            chunk["num_tokens"] for chunk in chunks if chunk["needs_update"]
        )
        print(f"Tokens to update: {tokens_to_update}")
        DOLLARS_PER_1K_TOKENS = 0.0004
        print(f"Estimated price: ${DOLLARS_PER_1K_TOKENS * tokens_to_update / 1000}")
        # print(f"Estimated time: {sum(document['needs_update'] for document in documents)}")

        # Get the embeddings for documents that need to be updated
        for chunk in tqdm(chunks):
            if not chunk["needs_update"]:
                continue
            chunk_num = chunk["chunk_num"]
            chunk_text = chunk["text"]
            # Get the embedding and save it to file -- if it doesn't already exist
            embedding_path = embedding_dir / f"{document['path']}.{chunk_num:04}.npy"
            embedding_path.parent.mkdir(exist_ok=True, parents=True)
            if embedding_path.exists():
                # Make sure it's not corrupted
                embedding = np.load(embedding_path)

                # TODO: Validate the hash of the embedding
            else:
                embedding_raw = _get_embedding(chunk_text)
                embedding = np.array(embedding_raw)

                # Get the embedding filename by hashing the document path and chunk id
                # hasher = hashlib.sha256()
                # hasher.update(document['path'].encode())
                # hasher.update(str(chunk_id).encode())
                # embedding_fn = hasher.digest()

                np.save(embedding_path, embedding)

        # Update the embeddings metadata file
        for document in documents:
            document["path"] = str(document["path"])
            document.pop("text")
            document.pop("needs_update")
        for chunk in chunks:
            chunk["path"] = str(chunk["path"])
            chunk["embedding_path"] = str(chunk["embedding_path"])
            chunk.pop("text")
        embedding_metadata["documents"] = documents
        embedding_metadata["chunks"] = chunks
        with embedding_metadata_path.open("w") as f:
            json.dump(embedding_metadata, f)
