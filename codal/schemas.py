from datetime import datetime
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
