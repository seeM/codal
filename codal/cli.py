import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import openai
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

CACHE_DIR = Path.home() / ".cache/codal"
ARCHIVE_DIR = CACHE_DIR / "archives"
EMBEDDING_DIR = CACHE_DIR / "embeddings"

MODEL_NAME = "text-embedding-ada-002"


def _get_embedding(text, model=MODEL_NAME, max_attempts=5, retry_delay=1):
    text = text.replace("\n", " ")
    for attempt in range(max_attempts):
        try:
            return openai.Embedding.create(input=[text], model=model)["data"][0][
                "embedding"
            ]
        except openai.error.OpenAIError as exception:
            print(f"OpenAI error: {exception.reason}")
            if attempt < max_attempts - 1:  # No delay on last attempt
                time.sleep(retry_delay)
            else:
                raise


def cli(args=None):
    parser = argparse.ArgumentParser(
        description="Download a GitHub repo, split the code, and get code search embeddings."
    )
    parser.add_argument("repo")
    args = parser.parse_args(args=args)

    # Download the zip file
    zip_url = f"https://github.com/{args.repo}/archive/refs/heads/main.zip"
    zip_path = ARCHIVE_DIR / f"{args.repo}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        with urlopen(zip_url) as response, zip_path.open("wb") as out_fp:
            data = response.read()
            out_fp.write(data)

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
                    document = {"path": path, "text": text}
                    documents.append(document)

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

    # Determine which documents need to be updated
    encoder = tiktoken.encoding_for_model(MODEL_NAME)
    for document in documents:
        path = document["path"]
        text = document["text"]

        # Hash the document's contents
        hasher = hashlib.sha256()
        hasher.update(text.encode())
        cur_hash = hasher.hexdigest()

        # Check if the hash changed since the last run
        prev_hash = embedding_metadata["documents"].get(path, {}).get("hash")
        document["needs_update"] = cur_hash != prev_hash
        document["hash"] = cur_hash
        document["num_tokens"] = len(encoder.encode(text))

    # Get document chunks. We get these for all documents (even those that don't need updates)
    # to calculate stats about the repo
    splitter = RecursiveCharacterTextSplitter(chunk_overlap=0, chunk_size=1000)
    chunks = []
    for document in documents:
        if not document["needs_update"]:
            continue

        for chunk_num, chunk_text in enumerate(splitter.split_text(document["text"])):
            embedding_path = embedding_dir / f"{document['path']}.{chunk_num:04}.npy"
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
