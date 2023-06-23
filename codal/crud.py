from typing import Any, Dict, Generic, Optional, Type, TypeVar, TypedDict, Union
from typing_extensions import Unpack

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import select

from codal.models import Repo
from codal.schemas import RepoCreate

from .database import Base
from .models import Org, Repo, Commit
from .schemas import (
    OrgCreate,
    OrgUpdate,
    RepoCreate,
    RepoUpdate,
    CommitCreate,
    CommitUpdate,
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


class CRUDOrg(CRUDBase[Org, OrgCreate, OrgUpdate]):
    def get(self, db: Session, *, name: str) -> Optional[Org]:
        org = db.execute(select(Org).where(Org.name == name)).scalar_one_or_none()
        return org

    def get_or_create(self, db: Session, obj_in: OrgCreate) -> Org:
        org = self.get(db, name=obj_in.name)
        if org:
            return org
        return self.create(db, obj_in=obj_in)


class CRUDRepo(CRUDBase[Repo, RepoCreate, RepoUpdate]):
    def get(self, db: Session, *, name: str, org_name: str) -> Optional[Repo]:
        repo = db.execute(
            select(Repo).join(Org).where(Repo.name == name, Org.name == org_name)
        ).scalar_one_or_none()
        return repo

    def get_or_create(self, db: Session, obj_in: RepoCreate) -> Repo:
        repo = db.execute(
            select(Repo).where(Repo.org_id == obj_in.org_id, Repo.name == obj_in.name)
        ).scalar_one_or_none()
        if repo:
            return repo
        return self.create(db, obj_in=obj_in)


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


org = CRUDOrg(Org)
repo = CRUDRepo(Repo)
commit = CRUDCommit(Commit)
