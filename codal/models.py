from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import List

import numpy as np
from sqlalchemy import ForeignKey, LargeBinary, String, TypeDecorator, UniqueConstraint
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


int_primary_key = Annotated[int, mapped_column(primary_key=True)]
array = Annotated[np.ndarray, mapped_column(NumpyArray)]
path = Annotated[Path, mapped_column(FilePath)]


class Org(Base):
    __tablename__ = "orgs"

    # TODO: Can we remove index on unique??
    id: Mapped[int_primary_key]
    name: Mapped[str] = mapped_column(unique=True, index=True)

    repos: Mapped[List[Repo]] = relationship(back_populates="org")

    def __repr__(self):
        return f"<Org: {self.name}>"


class Repo(Base):
    __tablename__ = "repos"

    id: Mapped[int_primary_key]
    org_id: Mapped[int] = mapped_column(ForeignKey("orgs.id"))
    name: Mapped[str]
    head: Mapped[str]

    org: Mapped[Org] = relationship(back_populates="repos")
    documents: Mapped[List[Document]] = relationship(back_populates="repo")

    __table_args__ = (
        UniqueConstraint("org_id", "name", name="unique__repo__org_id__name"),
    )

    def __repr__(self):
        return f"<Repo: {self.org.name}/{self.name}>"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int_primary_key]
    repo_id: Mapped[int] = mapped_column(ForeignKey("repos.id"))
    head: Mapped[str]
    path: Mapped[path]
    text: Mapped[str]
    num_tokens: Mapped[int]
    processed: Mapped[bool]

    repo: Mapped[Repo] = relationship(back_populates="documents")
    chunks: Mapped[List[Chunk]] = relationship(back_populates="document")

    __table_args__ = (
        UniqueConstraint(
            "repo_id", "path", "head", name="unique__document__repo__path__head"
        ),
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int_primary_key]
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    start: Mapped[int]
    end: Mapped[int]
    text: Mapped[str]
    embedding: Mapped[array]

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint(
            "document_id", "start", name="unique__chunk__document_id__start"
        ),
        UniqueConstraint("document_id", "end", name="unique__chunk__document_id__end"),
    )
