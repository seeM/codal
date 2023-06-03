from io import BytesIO
from pathlib import Path

import numpy as np
from sqlalchemy import (
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

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)

    repos = relationship("Repo", back_populates="org")

    def __repr__(self):
        return f"<Org: {self.name}>"


class Repo(Base):
    __tablename__ = "repos"

    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(ForeignKey("orgs.id"))
    name = Column(String)
    zip_path = Column(FilePath, unique=True, nullable=True)

    org = relationship("Org", back_populates="repos")
    documents = relationship("Document", back_populates="repo")

    __table_args__ = (UniqueConstraint("org_id", "name"),)

    def __repr__(self):
        return f"<Repo: {self.org.name}/{self.name}>"


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    repo_id = Column(ForeignKey("repos.id"))
    path = Column(FilePath)
    text = Column(String)
    num_tokens = Column(Integer)

    repo = relationship("Repo", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document")

    __table_args__ = (UniqueConstraint("repo_id", "path"),)


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(ForeignKey("documents.id"))
    start = Column(Integer)
    end = Column(Integer)
    text = Column(String)
    embedding = Column(NumpyArray)

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "start"),
        UniqueConstraint("document_id", "end"),
    )
