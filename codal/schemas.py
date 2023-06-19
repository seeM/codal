from typing import Optional

from pydantic import BaseModel


class OrgCreate(BaseModel):
    name: str


class OrgUpdate(BaseModel):
    pass


class RepoBase(BaseModel):
    org_id: Optional[int] = None
    name: Optional[str] = None
    head_commit_id: Optional[int] = None
    default_branch: Optional[str] = None


class RepoCreate(RepoBase):
    org_id: int
    name: str


class RepoUpdate(RepoBase):
    pass
