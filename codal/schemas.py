# This feels a little overengineered at the moment, but it does have a few benefits:
# 1. We'd need it to expose a CRUD web API anyway.
# 2. It doubles up for type checking and auto-completing the arguments to create/update methods.

# TODO: Why do we use ids instead of objects?
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from pydantic import BaseModel

from .models import Chunk


class CommitBase(BaseModel):
    repo_id: Optional[int] = None
    sha: Optional[str] = None
    message: Optional[str] = None
    author_name: Optional[str]
    author_email: Optional[str]
    authored_datetime: Optional[datetime]
    committer_name: Optional[str]
    committer_email: Optional[str]
    committed_datetime: Optional[datetime]


class CommitCreate(CommitBase):
    repo_id: int
    sha: str
    message: str
    authored_datetime: datetime
    committed_datetime: datetime


class CommitUpdate(CommitBase):
    pass


class DocumentBase(BaseModel):
    repo_id: Optional[int] = None
    path: Optional[Path] = None


class DocumentCreate(DocumentBase):
    repo_id: int
    path: Path


class DocumentUpdate(DocumentBase):
    pass


class DocumentVersionBase(BaseModel):
    document_id: Optional[int] = None
    commit_id: Optional[int] = None
    text: Optional[str] = None
    num_tokens: Optional[int] = None
    processed: Optional[bool] = None

    # TODO: Not yet sure how to do collections via an API
    chunks: Optional[List[Chunk]] = []

    class Config:
        arbitrary_types_allowed = True


class DocumentVersionCreate(DocumentVersionBase):
    document_id: int
    commit_id: int
    text: str
    num_tokens: int


class DocumentVersionUpdate(DocumentVersionBase):
    pass


class ChunkBase(BaseModel):
    document_id: Optional[int] = None
    start: Optional[int] = None
    end: Optional[int] = None
    text: Optional[str] = None
    embedding: Optional[np.ndarray] = None

    class Config:
        arbitrary_types_allowed = True


class ChunkCreate(ChunkBase):
    document_id: int
    start: int
    end: int
    text: str
    embedding: np.ndarray


class ChunkUpdate(ChunkBase):
    pass
