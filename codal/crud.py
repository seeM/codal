from typing import Any, Dict, Generic, Optional, Type, TypeVar, Union

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import select

from codal.models import Repo
from codal.schemas import RepoCreate

from .database import Base
from .models import Org, Repo
from .schemas import OrgCreate, OrgUpdate, RepoCreate, RepoUpdate


ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    def __init__(self, model: Type[ModelType]):
        self.model = model

    def create(self, db: Session, *, obj_in: CreateSchemaType) -> ModelType:
        obj_in_data = jsonable_encoder(obj_in)
        db_obj = self.model(**obj_in_data)  # type: ignore
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
        self,
        db: Session,
        *,
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

    def get_or_create(self, db: Session, *, name: str) -> Org:
        org = self.get(db, name=name)
        if org:
            return org
        return self.create(db, obj_in=OrgCreate(name=name))


class CRUDRepo(CRUDBase[Repo, RepoCreate, RepoUpdate]):
    def get(self, db: Session, *, org_id: int, name: str) -> Optional[Repo]:
        repo = db.execute(
            select(Repo).where(Repo.org_id == org_id, Repo.name == name)
        ).scalar_one_or_none()
        return repo

    def get_by_name(self, db: Session, *, org_name: str, name: str) -> Optional[Repo]:
        repo = db.execute(
            select(Repo).join(Org).where(Org.name == org_name, Repo.name == name)
        ).scalar_one_or_none()
        return repo

    def get_or_create(self, db: Session, *, org_id: int, name: str) -> Repo:
        repo = self.get(db, org_id=org_id, name=name)
        if repo:
            return repo
        return self.create(db, obj_in=RepoCreate(org_id=org_id, name=name))

    def create(self, db: Session, *, obj_in: RepoCreate) -> Repo:
        repo = super().create(db, obj_in=obj_in)

        return repo


org = CRUDOrg(Org)
repo = CRUDRepo(Repo)
