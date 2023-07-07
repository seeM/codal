from datetime import datetime
from typing import Callable, List

MIGRATIONS: List[Callable] = []
migration = MIGRATIONS.append


def migrate(db):
    ensure_migrations_table(db)
    already_applied = {r["name"] for r in db["_codal_migrations"].rows}
    for fn in MIGRATIONS:
        name = fn.__name__
        if name not in already_applied:
            fn(db)
            db["_codal_migrations"].insert(
                {"name": name, "applied_at": str(datetime.utcnow())}
            )
            already_applied.add(name)


def ensure_migrations_table(db):
    if not db["_codal_migrations"].exists():
        db["_codal_migrations"].create(
            {
                "name": str,
                "applied_at": str,
            },
            pk="name",
        )


@migration
def m001_initial(db):
    # Ensure the original table design exists, so other migrations can run
    if not db["orgs"].exists():
        db["orgs"].create(
            {
                "id": int,
                "name": str,
            },
            pk="id",
            not_null={"id", "name"},
        )
        db["orgs"].create_index(["name"], unique=True)

    if not db["commits"].exists():
        db["commits"].create(
            {
                "id": int,
                "repo_id": int,
                "sha": str,
                "message": str,
                "author_name": str,
                "author_email": str,
                "authored_datetime": str,
                "committer_name": str,
                "committer_email": str,
                "committed_datetime": str,
            },
            pk="id",
            not_null={
                "id",
                "repo_id",
                "sha",
                "message",
                "authored_datetime",
                "committed_datetime",
            },
        )
        db["commits"].create_index(["repo_id", "sha"], unique=True)
        # TODO: Including this because it's in the SQLAlchemy example, but not yet sure why
        db["commits"].create_index(["id", "repo_id"], unique=True)

    if not db["repos"].exists():
        db["repos"].create(
            {
                "id": int,
                "org_id": int,
                "name": str,
                "head_commit_id": int,
                "default_branch": str,
            },
            pk="id",
            not_null={"id", "org_id", "name"},
            foreign_keys=[
                ("org_id", "orgs", "id"),
                ("head_commit_id", "commits", "id"),
            ],
        )
        db["repos"].create_index(["org_id", "name"], unique=True)

    if not db["documents"].exists():
        db["documents"].create(
            {
                "id": int,
                "repo_id": int,
                "path": str,
            },
            pk="id",
            not_null={"id", "repo_id", "path"},
            foreign_keys=[("repo_id", "repos", "id")],
        )
        db["documents"].create_index(["repo_id", "path"], unique=True)

    if not db["document_versions"].exists():
        db["document_versions"].create(
            {
                "id": int,
                "document_id": int,
                "commit_id": int,
                "text": str,
                "num_tokens": int,
                "processed": bool,
            },
            pk="id",
            not_null={
                "id",
                "document_id",
                "commit_id",
                "text",
                "num_tokens",
                "processed",
            },
            defaults={"processed": False},
            foreign_keys=[
                ("document_id", "documents", "id"),
                ("commit_id", "commits", "id"),
            ],
        )
        db["document_versions"].create_index(["document_id", "commit_id"], unique=True)

    if not db["chunks"].exists():
        db["chunks"].create(
            {
                "id": int,
                "document_id": int,
                "start": int,
                "end": int,
                "text": str,
                "embedding": bytes,
            },
            pk="id",
            not_null={"id", "document_id", "start", "end", "text", "embedding"},
            foreign_keys=[("document_id", "documents", "id")],
        )

    if not db["document_version_chunks"].exists():
        db["document_version_chunks"].create(
            {
                "document_version_id": int,
                "chunk_id": int,
            },
            pk=("document_version_id", "chunk_id"),
            not_null={"document_version_id", "chunk_id"},
            foreign_keys=[
                ("document_version_id", "document_versions", "id"),
                ("chunk_id", "chunks", "id"),
            ],
        )
