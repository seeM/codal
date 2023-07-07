"""make repos.head_commit_head nullable

Revision ID: 8894f8ab014f
Revises: 3812ca2ef6c8
Create Date: 2023-06-17 16:27:52.885848

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "8894f8ab014f"
down_revision = "3812ca2ef6c8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("repos", schema=None) as batch_op:
        batch_op.alter_column(
            "head_commit_id", existing_type=sa.INTEGER(), nullable=True
        )

    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("repos", schema=None) as batch_op:
        batch_op.alter_column(
            "head_commit_id", existing_type=sa.INTEGER(), nullable=False
        )

    # ### end Alembic commands ###
