from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import List

import numpy as np
from sqlalchemy import (
    Boolean,
    Column,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

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


class Org(Base):
    __tablename__ = "orgs"

    id: int = Column(Integer, primary_key=True, index=True)
    name: str = Column(String, unique=True, index=True)

    repos: List[Repo] = relationship("Repo", back_populates="org")

    def __repr__(self):
        return f"<Org: {self.name}>"


class Repo(Base):
    __tablename__ = "repos"

    id: int = Column(Integer, primary_key=True, index=True)
    org_id: int = Column(ForeignKey("orgs.id"))
    name: str = Column(String)
    head: str = Column(String)

    org: Org = relationship("Org", back_populates="repos")
    documents: List[Document] = relationship("Document", back_populates="repo")

    __table_args__ = (UniqueConstraint("org_id", "name"),)

    def __repr__(self):
        return f"<Repo: {self.org.name}/{self.name}>"


class Document(Base):
    __tablename__ = "documents"

    id: int = Column(Integer, primary_key=True, index=True)
    repo_id: int = Column(ForeignKey("repos.id"))
    head: str = Column(String)
    path: Path = Column(FilePath)
    text: str = Column(String)
    num_tokens: int = Column(Integer)
    processed: bool = Column(Boolean)

    repo: Repo = relationship("Repo", back_populates="documents")
    chunks: List[Chunk] = relationship("Chunk", back_populates="document")

    __table_args__ = (
        UniqueConstraint(
            "repo_id", "path", "head", name="unique__document__repo_path_head"
        ),
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: int = Column(Integer, primary_key=True, index=True)
    document_id: int = Column(ForeignKey("documents.id"))
    start: int = Column(Integer)
    end: int = Column(Integer)
    text: str = Column(String)
    embedding: np.ndarray = Column(NumpyArray)

    document: Document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "start"),
        UniqueConstraint("document_id", "end"),
    )
