from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import numpy as np
from sqlalchemy import (
    ForeignKey,
    ForeignKeyConstraint,
    LargeBinary,
    String,
    TypeDecorator,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing_extensions import Annotated

from .database import Base


class FilePath(TypeDecorator):
    impl = String
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return str(value)

    def process_result_value(self, value, dialect):
        if value is not None:
            return Path(value)


class NumpyArray(TypeDecorator):
    impl = LargeBinary

    def process_bind_param(self, value, dialect):
        if value is not None:
            out = BytesIO()
            np.save(out, value)
            out.seek(0)
            return out.read()

    def process_result_value(self, value, dialect):
        if value is not None:
            out = BytesIO(value)
            out.seek(0)
            return np.load(out)


array = Annotated[np.ndarray, mapped_column(NumpyArray)]
path = Annotated[Path, mapped_column(FilePath)]


class Org(Base):
    __tablename__ = "orgs"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True, index=True)

    repos: Mapped[List[Repo]] = relationship(back_populates="org")


class Repo(Base):
    __tablename__ = "repos"

    id: Mapped[int] = mapped_column(primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("orgs.id"))
    name: Mapped[str]

    head_commit_id: Mapped[Optional[int]] = mapped_column(ForeignKey("commits.id"))
    default_branch: Mapped[str] = mapped_column()

    org: Mapped[Org] = relationship(back_populates="repos")
    head_commit: Mapped[Commit] = relationship(
        foreign_keys=head_commit_id, post_update=True
    )
    commits: Mapped[List[Commit]] = relationship(
        back_populates="repo", foreign_keys="Commit.repo_id"
    )
    documents: Mapped[List[Document]] = relationship(back_populates="repo")

    __table_args__ = (
        ForeignKeyConstraint(
            ["id", "head_commit_id"], ["commits.repo_id", "commits.id"]
        ),
        Index("org_id", "name", unique=True),
    )


class Commit(Base):
    __tablename__ = "commits"

    id: Mapped[int] = mapped_column(primary_key=True)
    repo_id: Mapped[int] = mapped_column(ForeignKey("repos.id"))

    sha: Mapped[str]
    message: Mapped[str]
    author_name: Mapped[Optional[str]]
    author_email: Mapped[Optional[str]]
    authored_datetime: Mapped[datetime]
    committer_name: Mapped[Optional[str]]
    committer_email: Mapped[Optional[str]]
    committed_datetime: Mapped[datetime]

    repo: Mapped[Repo] = relationship(back_populates="commits", foreign_keys=repo_id)
    document_versions: Mapped[List[DocumentVersion]] = relationship(
        back_populates="commit"
    )

    __table_args__ = (UniqueConstraint("repo_id", "sha"),)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    repo_id: Mapped[int] = mapped_column(ForeignKey("repos.id"))
    path: Mapped[path]

    repo: Mapped[Repo] = relationship(back_populates="documents")
    chunks: Mapped[List[Chunk]] = relationship(back_populates="document")
    versions: Mapped[List[DocumentVersion]] = relationship(back_populates="document")

    __table_args__ = (Index("repo_id", "path", unique=True),)


class DocumentVersionChunk(Base):
    __tablename__ = "document_version_chunks"

    document_version_id: Mapped[int] = mapped_column(
        ForeignKey("document_versions.id"), primary_key=True
    )
    chunk_id: Mapped[int] = mapped_column(ForeignKey("chunks.id"), primary_key=True)

    document_version: Mapped[DocumentVersion] = relationship(back_populates="chunks")
    chunk: Mapped[Chunk] = relationship(back_populates="document_versions")


class DocumentVersion(Base):
    __tablename__ = "document_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    commit_id: Mapped[int] = mapped_column(ForeignKey("commits.id"))
    text: Mapped[str]
    num_tokens: Mapped[int]
    processed: Mapped[bool] = mapped_column()

    document: Mapped[Document] = relationship(back_populates="versions")
    commit: Mapped[Commit] = relationship(back_populates="document_versions")
    chunks: Mapped[List[DocumentVersionChunk]] = relationship(
        back_populates="document_version",
    )

    __table_args__ = (Index("document_id", "commit_id", unique=True),)


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    start: Mapped[Optional[int]]
    end: Mapped[int]
    text: Mapped[str]
    embedding: Mapped[array]

    document: Mapped[Document] = relationship(back_populates="chunks")
    document_versions: Mapped[List[DocumentVersionChunk]] = relationship(
        back_populates="chunk",
    )
