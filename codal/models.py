from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import List, Optional

import numpy as np
from sqlalchemy import (
    Column,
    ForeignKey,
    LargeBinary,
    String,
    Table,
    TypeDecorator,
    UniqueConstraint,
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

    # TODO: Can we remove index on unique??
    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    name: Mapped[str] = mapped_column(unique=True, index=True)

    repos: Mapped[List[Repo]] = relationship(default_factory=list, back_populates="org")


class Repo(Base):
    __tablename__ = "repos"

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("orgs.id"), init=False)
    name: Mapped[str]

    org: Mapped[Org] = relationship(back_populates="repos")
    head_commit: Mapped[Commit] = relationship(foreign_keys="Repo.head_commit_id")
    commits: Mapped[List[Commit]] = relationship(
        default_factory=list, back_populates="repo", foreign_keys="Commit.repo_id"
    )
    documents: Mapped[List[Document]] = relationship(
        default_factory=list, back_populates="repo"
    )

    default_branch: Mapped[str] = mapped_column(default=None)
    head_commit_id: Mapped[int] = mapped_column(ForeignKey("commits.id"), default=None)

    __table_args__ = (UniqueConstraint("org_id", "name"),)


class Commit(Base):
    __tablename__ = "commits"

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    repo_id: Mapped[int] = mapped_column(ForeignKey("repos.id"), init=False)
    sha: Mapped[str]
    message: Mapped[str]
    author_name: Mapped[Optional[str]]
    author_email: Mapped[Optional[str]]
    committer_name: Mapped[Optional[str]]
    committer_email: Mapped[Optional[str]]

    repo: Mapped[Repo] = relationship(back_populates="commits")
    document_versions: Mapped[List[DocumentVersion]] = relationship(
        default_factory=list, back_populates="commit"
    )

    __table_args__ = (UniqueConstraint("repo_id", "sha"),)


# NOTE: For a Core table, we use the sqlalchemy.Column construct, not sqlalchemy.orm.mapped_column
document_version_chunks = Table(
    "document_version_chunks",
    Base.metadata,
    Column("document_version_id", ForeignKey("document_versions.id"), primary_key=True),
    Column("chunk_id", ForeignKey("chunks.id"), primary_key=True),
)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    repo_id: Mapped[int] = mapped_column(ForeignKey("repos.id"), init=False)
    path: Mapped[path]

    repo: Mapped[Repo] = relationship(back_populates="documents")
    chunks: Mapped[List[Chunk]] = relationship(
        default_factory=list, back_populates="document"
    )
    versions: Mapped[List[DocumentVersion]] = relationship(
        default_factory=list, back_populates="document"
    )

    __table_args__ = (UniqueConstraint("repo_id", "path"),)


class DocumentVersion(Base):
    __tablename__ = "document_versions"

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), init=False)
    commit_id: Mapped[int] = mapped_column(ForeignKey("commits.id"), init=False)
    text: Mapped[str]
    num_tokens: Mapped[int]
    processed: Mapped[bool] = mapped_column(default=False)

    document: Mapped[Document] = relationship(default=None, back_populates="versions")
    commit: Mapped[Commit] = relationship(
        default=None, back_populates="document_versions"
    )
    chunks: Mapped[List[Chunk]] = relationship(
        default_factory=list,
        secondary=document_version_chunks,
        back_populates="document_versions",
    )

    __table_args__ = (UniqueConstraint("document_id", "commit_id"),)


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), init=False)
    start: Mapped[Optional[int]]
    end: Mapped[int]
    text: Mapped[str]
    embedding: Mapped[array]

    document: Mapped[Document] = relationship(
        default=None, back_populates="chunks", repr=False
    )
    document_versions: Mapped[List[DocumentVersion]] = relationship(
        default_factory=list,
        secondary=document_version_chunks,
        back_populates="chunks",
        repr=False,
    )
