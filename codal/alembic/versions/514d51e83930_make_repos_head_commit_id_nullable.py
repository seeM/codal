"""make repos.head_commit_id nullable

Revision ID: 514d51e83930
Revises: a904ab0ac9e5
Create Date: 2023-06-17 16:01:25.666331

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '514d51e83930'
down_revision = 'a904ab0ac9e5'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('repos', schema=None) as batch_op:
        batch_op.alter_column('head_commit_id',
               existing_type=sa.INTEGER(),
               nullable=True)

    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('repos', schema=None) as batch_op:
        batch_op.alter_column('head_commit_id',
               existing_type=sa.INTEGER(),
               nullable=False)

    # ### end Alembic commands ###