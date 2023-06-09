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

    def __repr__(self):
        return f"<Org: {self.name}>"


class Repo(Base):
    __tablename__ = "repos"

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("orgs.id"), init=False)
    name: Mapped[str]

    org: Mapped[Org] = relationship(back_populates="repos")
    documents: Mapped[List[Document]] = relationship(
        default_factory=list, back_populates="repo"
    )

    default_branch: Mapped[str] = mapped_column(default=None)
    head: Mapped[str] = mapped_column(default=None)

    __table_args__ = (UniqueConstraint("org_id", "name"),)

    def __repr__(self):
        return f"<Repo: {self.org.name}/{self.name}>"


# note for a Core table, we use the sqlalchemy.Column construct,
# not sqlalchemy.orm.mapped_column
documents_chunks = Table(
    "documents_chunks",
    Base.metadata,
    Column("document_id", ForeignKey("documents.id"), primary_key=True),
    Column("chunk_id", ForeignKey("chunks.id"), primary_key=True),
)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    repo_id: Mapped[int] = mapped_column(ForeignKey("repos.id"), init=False)
    head: Mapped[str]
    path: Mapped[path]
    text: Mapped[str]
    num_tokens: Mapped[int]

    repo: Mapped[Repo] = relationship(back_populates="documents")
    chunks: Mapped[List[Chunk]] = relationship(
        default_factory=list, secondary=documents_chunks
    )

    processed: Mapped[bool] = mapped_column(default=False)

    __table_args__ = (UniqueConstraint("repo_id", "path", "head"),)


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    start: Mapped[Optional[int]]
    end: Mapped[int]
    text: Mapped[str]
    embedding: Mapped[array]
