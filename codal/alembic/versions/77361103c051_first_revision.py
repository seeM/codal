"""first revision

Revision ID: 77361103c051
Revises: 
Create Date: 2023-06-09 10:05:24.936681

"""
import sqlalchemy as sa

import codal
from alembic import op

# revision identifiers, used by Alembic.
revision = "77361103c051"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("start", sa.Integer(), nullable=True),
        sa.Column("end", sa.Integer(), nullable=False),
        sa.Column("text", sa.String(), nullable=False),
        sa.Column("embedding", codal.models.NumpyArray(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chunks")),
    )
    op.create_table(
        "orgs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_orgs")),
    )
    with op.batch_alter_table("orgs", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_orgs_name"), ["name"], unique=True)

    op.create_table(
        "repos",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("org_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("default_branch", sa.String(), nullable=False),
        sa.Column("head", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(
            ["org_id"], ["orgs.id"], name=op.f("fk_repos_org_id_orgs")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_repos")),
        sa.UniqueConstraint("org_id", "name", name=op.f("uq_repos_org_id")),
    )
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("repo_id", sa.Integer(), nullable=False),
        sa.Column("head", sa.String(), nullable=False),
        sa.Column("path", codal.models.FilePath(), nullable=False),
        sa.Column("text", sa.String(), nullable=False),
        sa.Column("num_tokens", sa.Integer(), nullable=False),
        sa.Column("processed", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ["repo_id"], ["repos.id"], name=op.f("fk_documents_repo_id_repos")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
        sa.UniqueConstraint(
            "repo_id", "path", "head", name=op.f("uq_documents_repo_id")
        ),
    )
    op.create_table(
        "documents_chunks",
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("chunk_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["chunks.id"],
            name=op.f("fk_documents_chunks_chunk_id_chunks"),
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_documents_chunks_document_id_documents"),
        ),
        sa.PrimaryKeyConstraint(
            "document_id", "chunk_id", name=op.f("pk_documents_chunks")
        ),
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table("documents_chunks")
    op.drop_table("documents")
    op.drop_table("repos")
    with op.batch_alter_table("orgs", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_orgs_name"))

    op.drop_table("orgs")
    op.drop_table("chunks")
    # ### end Alembic commands ###