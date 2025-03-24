"""Initial migration

Revision ID: 4c93a6d5e7af
Revises: 99fdbc16d61a
Create Date: 2025-03-24 15:02:09.413366

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4c93a6d5e7af'
down_revision: Union[str, None] = '99fdbc16d61a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column('shortlink', sa.Column('short_code', sa.String(), nullable=True))
    op.execute(
        """
        UPDATE shortlink
        SET short_code = regexp_replace(short_link, '^https://slay/', '')
        """
    )
    op.alter_column('shortlink', 'short_code', nullable=False)

def downgrade():
    op.drop_column('shortlink', 'short_code')
