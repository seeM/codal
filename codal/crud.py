from pathlib import Path
from typing import Any, Dict, Generic, List, Sequence, Type, TypeVar, Union

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import Base
from .models import Chunk, Document, DocumentVersion
from .schemas import ChunkCreate, ChunkUpdate

ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    def __init__(self, model: Type[ModelType]):
        self.model = model

    def create(self, db: Session, obj_in: CreateSchemaType) -> ModelType:
        obj_in_data = obj_in.dict()
        db_obj = self.model(**obj_in_data)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
        self,
        db: Session,
        db_obj: ModelType,
        obj_in: Union[UpdateSchemaType, Dict[str, Any]],
    ) -> ModelType:
        obj_data = jsonable_encoder(db_obj)
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)
        for field in obj_data:
            if field in update_data:
                setattr(db_obj, field, update_data[field])
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj


class CRUDChunk(CRUDBase[Chunk, ChunkCreate, ChunkUpdate]):
    def get_multi_by_repo(self, db: Session, *, repo) -> Sequence[Chunk]:
        chunks = (
            db.execute(
                (
                    select(Chunk)
                    .join(Chunk.document_versions)
                    .join(DocumentVersion.document)
                    .where(
                        Document.repo_id == repo["id"],
                        DocumentVersion.commit_id == repo["head_commit_id"],
                    )
                )
            )
            .scalars()
            .all()
        )
        return chunks

    def get_multi_by_id(self, db: Session, *, ids: List[int]) -> Sequence[Chunk]:
        chunks = db.execute(select(Chunk).where(Chunk.id.in_(ids))).scalars().all()
        return chunks


chunk = CRUDChunk(Chunk)
