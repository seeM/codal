# This feels a little overengineered at the moment, but it does have a few benefits:
# 1. We'd need it to expose a CRUD web API anyway.
# 2. It doubles up for type checking and auto-completing the arguments to create/update methods.

# TODO: Why do we use ids instead of objects?
from typing import List, Optional

import numpy as np
from pydantic import BaseModel

from .models import Chunk


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
