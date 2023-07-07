from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Sequence, Type, TypeVar, Union

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import Base
from .models import Chunk, Commit, Document, DocumentVersion
from .schemas import (
    ChunkCreate,
    ChunkUpdate,
    CommitCreate,
    CommitUpdate,
    DocumentCreate,
    DocumentUpdate,
    DocumentVersionCreate,
    DocumentVersionUpdate,
)

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


class CRUDCommit(CRUDBase[Commit, CommitCreate, CommitUpdate]):
    def get(self, db: Session, *, repo_id: int, sha: str) -> Optional[Commit]:
        commit = db.execute(
            select(Commit).where(Commit.repo_id == repo_id, Commit.sha == sha)
        ).scalar_one_or_none()
        return commit

    def get_or_create(self, db: Session, obj_in: CommitCreate) -> Commit:
        commit = self.get(db, repo_id=obj_in.repo_id, sha=obj_in.sha)
        if commit:
            return commit
        return self.create(db, obj_in=obj_in)


class CRUDDocument(CRUDBase[Document, DocumentCreate, DocumentUpdate]):
    def get(self, db: Session, *, repo_id: int, path: Path) -> Optional[Document]:
        document = db.execute(
            select(Document).where(Document.repo_id == repo_id, Document.path == path)
        ).scalar_one_or_none()
        return document

    def get_or_create(self, db: Session, obj_in: DocumentCreate) -> Document:
        document = self.get(db, repo_id=obj_in.repo_id, path=obj_in.path)
        if document:
            return document
        return self.create(db, obj_in=obj_in)


class CRUDDocumentVersion(
    CRUDBase[DocumentVersion, DocumentVersionCreate, DocumentVersionUpdate]
):
    def get(
        self, db: Session, *, document_id: int, commit_id: int
    ) -> Optional[DocumentVersion]:
        document_version = db.execute(
            select(DocumentVersion).where(
                DocumentVersion.document_id == document_id,
                DocumentVersion.commit_id == commit_id,
            )
        ).scalar_one_or_none()
        return document_version

    def get_or_create(
        self, db: Session, obj_in: DocumentVersionCreate
    ) -> DocumentVersion:
        document_version = self.get(
            db, document_id=obj_in.document_id, commit_id=obj_in.commit_id
        )
        if document_version:
            return document_version
        return self.create(db, obj_in=obj_in)

    def set_chunks(
        self, db: Session, document_version: DocumentVersion, chunks: List[Chunk]
    ) -> DocumentVersion:
        # NOTE: We have to access document_version.chunks, else `update` doesn't override it,
        #       possibly due to SQLAlchemy's lazy relationship loading.
        # TODO: Is there a better way to handle this in `update` or via schemas?
        #       Maybe this is why we should use ids in the schemas?
        assert document_version.chunks == []
        return self.update(
            db, document_version, DocumentVersionUpdate(chunks=chunks, processed=True)
        )


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


commit = CRUDCommit(Commit)
document = CRUDDocument(Document)
document_version = CRUDDocumentVersion(DocumentVersion)
chunk = CRUDChunk(Chunk)
