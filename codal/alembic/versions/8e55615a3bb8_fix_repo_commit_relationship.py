"""fix repo commit relationship

Revision ID: 8e55615a3bb8
Revises: ad467cc16a19
Create Date: 2023-07-01 12:09:19.859335

"""
import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "8e55615a3bb8"
down_revision = "ad467cc16a19"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("commits", schema=None) as batch_op:
        batch_op.create_unique_constraint(
            batch_op.f("uq_commits_id"), ["id", "repo_id"]
        )

    with op.batch_alter_table("repos", schema=None) as batch_op:
        batch_op.drop_constraint("fk_repos_head_commit_id_commits", type_="foreignkey")

    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("repos", schema=None) as batch_op:
        batch_op.create_foreign_key(
            "fk_repos_head_commit_id_commits", "commits", ["head_commit_id"], ["id"]
        )

    with op.batch_alter_table("commits", schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f("uq_commits_id"), type_="unique")

    # ### end Alembic commands ###
