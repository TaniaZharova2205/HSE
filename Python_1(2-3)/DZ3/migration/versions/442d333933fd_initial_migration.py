"""Initial migration

Revision ID: 442d333933fd
Revises: 3558c2fc6c92
Create Date: 2025-03-24 08:42:08.265046

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '442d333933fd'
down_revision: Union[str, None] = '3558c2fc6c92'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index('ix_user_id', table_name='user')
    op.create_index(op.f('ix_user_email'), 'user', ['email'], unique=True)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_user_email'), table_name='user')
    op.create_index('ix_user_id', 'user', ['id'], unique=False)
    # ### end Alembic commands ###
