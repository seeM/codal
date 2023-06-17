"""fix indices

Revision ID: ba31d3d85648
Revises: 802a3b210f33
Create Date: 2023-06-17 17:05:00.571430

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "ba31d3d85648"
down_revision = "802a3b210f33"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("document_versions", schema=None) as batch_op:
        batch_op.drop_index("document_id")
        batch_op.create_index(
            batch_op.f("ix_document_versions_document_id"),
            ["document_id", "commit_id"],
            unique=True,
        )

    with op.batch_alter_table("documents", schema=None) as batch_op:
        batch_op.drop_index("repo_id")
        batch_op.create_index(
            batch_op.f("ix_documents_repo_id"), ["repo_id", "path"], unique=True
        )

    with op.batch_alter_table("repos", schema=None) as batch_op:
        batch_op.drop_index("org_id")
        batch_op.create_index(
            batch_op.f("ix_repos_org_id"), ["org_id", "name"], unique=True
        )

    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("repos", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_repos_org_id"))
        batch_op.create_index("org_id", ["name"], unique=False)

    with op.batch_alter_table("documents", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_documents_repo_id"))
        batch_op.create_index("repo_id", ["path"], unique=False)

    with op.batch_alter_table("document_versions", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_document_versions_document_id"))
        batch_op.create_index("document_id", ["commit_id"], unique=False)

    # ### end Alembic commands ###
