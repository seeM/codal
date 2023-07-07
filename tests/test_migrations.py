import sqlite_utils

from codal.migrations import migrate


def test_migrate_blank():
    # We probably won't need this test in future. It was mainly useful when migrating from
    # SQLAlchemy to sqlite-utils.

    db = sqlite_utils.Database(memory=True)

    migrate(db)

    assert set(db.table_names()) == {
        "_codal_migrations",
        "orgs",
        "commits",
        "repos",
        "documents",
        "document_versions",
        "chunks",
        "document_version_chunks",
    }

    class Ignored:
        def __eq__(self, _):
            return True

        def __repr__(self):
            return "<IGNORED>"

    ignored = Ignored()

    def column(name, type, notnull=1, default_value=None, is_pk=0):
        return sqlite_utils.db.Column(
            ignored, name, type, notnull, default_value, is_pk
        )

    def index(name, columns, unique=1, origin="c", partial=0):
        return sqlite_utils.db.Index(ignored, name, unique, origin, partial, columns)

    def foreign_key(table, column, other_table, other_column):
        return sqlite_utils.db.ForeignKey(table, column, other_table, other_column)

    assert db["orgs"].columns == [
        column("id", "INTEGER", is_pk=1),
        column("name", "TEXT"),
    ]
    assert db["orgs"].pks == ["id"]
    assert db["orgs"].indexes == [index("idx_orgs_name", ["name"])]
    assert db["orgs"].foreign_keys == []

    assert db["commits"].columns == [
        column("id", "INTEGER", is_pk=1),
        column("repo_id", "INTEGER"),
        column("sha", "TEXT"),
        column("message", "TEXT"),
        column("author_name", "TEXT", notnull=0),
        column("author_email", "TEXT", notnull=0),
        column("authored_datetime", "TEXT"),
        column("committer_name", "TEXT", notnull=0),
        column("committer_email", "TEXT", notnull=0),
        column("committed_datetime", "TEXT"),
    ]
    assert db["commits"].indexes == [
        index("idx_commits_id_repo_id", ["id", "repo_id"]),
        index("idx_commits_repo_id_sha", ["repo_id", "sha"]),
    ]
    assert db["commits"].pks == ["id"]

    assert db["repos"].columns_dict == {
        "id": int,
        "org_id": int,
        "name": str,
        "head_commit_id": int,
        "default_branch": str,
    }
    assert db["repos"].columns == [
        column("id", "INTEGER", is_pk=1),
        column("org_id", "INTEGER"),
        column("name", "TEXT"),
        column("head_commit_id", "INTEGER", notnull=0),
        column("default_branch", "TEXT", notnull=0),
    ]
    assert db["repos"].pks == ["id"]
    assert db["repos"].indexes == [index("idx_repos_org_id_name", ["org_id", "name"])]
    assert db["repos"].foreign_keys == [
        foreign_key("repos", "head_commit_id", "commits", "id"),
        foreign_key("repos", "org_id", "orgs", "id"),
    ]

    assert db["documents"].columns == [
        column("id", "INTEGER", is_pk=1),
        column("repo_id", "INTEGER"),
        column("path", "TEXT"),
    ]
    assert db["documents"].pks == ["id"]
    assert db["documents"].indexes == [
        index("idx_documents_repo_id_path", ["repo_id", "path"])
    ]
    assert db["documents"].foreign_keys == [
        foreign_key("documents", "repo_id", "repos", "id")
    ]

    assert db["document_versions"].columns_dict == {
        "id": int,
        "document_id": int,
        "commit_id": int,
        "text": str,
        "num_tokens": int,
        "processed": int,
    }
    assert db["document_versions"].columns == [
        column("id", "INTEGER", is_pk=1),
        column("document_id", "INTEGER"),
        column("commit_id", "INTEGER"),
        column("text", "TEXT"),
        column("num_tokens", "INTEGER"),
        column("processed", "INTEGER", default_value="0"),
    ]
    assert db["document_versions"].pks == ["id"]
    assert db["document_versions"].indexes == [
        index(
            "idx_document_versions_document_id_commit_id", ["document_id", "commit_id"]
        )
    ]
    assert db["document_versions"].foreign_keys == [
        foreign_key("document_versions", "commit_id", "commits", "id"),
        foreign_key("document_versions", "document_id", "documents", "id"),
    ]

    assert db["chunks"].columns_dict == {
        "id": int,
        "document_id": int,
        "start": int,
        "end": int,
        "text": str,
        "embedding": bytes,
    }
    assert db["chunks"].columns == [
        column("id", "INTEGER", is_pk=1),
        column("document_id", "INTEGER"),
        column("start", "INTEGER"),
        column("end", "INTEGER"),
        column("text", "TEXT"),
        column("embedding", "BLOB"),
    ]
    assert db["chunks"].pks == ["id"]
    assert db["chunks"].indexes == []
    assert db["chunks"].foreign_keys == [
        foreign_key("chunks", "document_id", "documents", "id")
    ]

    assert db["document_version_chunks"].columns == [
        column("document_version_id", "INTEGER", is_pk=1),
        column("chunk_id", "INTEGER", is_pk=2),
    ]
    assert db["document_version_chunks"].pks == ["document_version_id", "chunk_id"]
    assert db["document_version_chunks"].indexes == [
        index(
            "sqlite_autoindex_document_version_chunks_1",
            ["document_version_id", "chunk_id"],
            origin="pk",
        )
    ]
    assert db["document_version_chunks"].foreign_keys == [
        foreign_key("document_version_chunks", "chunk_id", "chunks", "id"),
        foreign_key(
            "document_version_chunks", "document_version_id", "document_versions", "id"
        ),
    ]
