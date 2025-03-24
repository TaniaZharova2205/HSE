"""Initial migration

Revision ID: 5e63078d647e
Revises: b83754956520
Create Date: 2025-03-24 09:36:13.572020

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import uuid

# revision identifiers, used by Alembic.
revision: str = '5e63078d647e'
down_revision: Union[str, None] = 'b83754956520'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None
new_column_type = sa.dialects.postgresql.UUID(as_uuid=True)

def upgrade():
    # Добавляем временный столбец с типом UUID
    op.add_column('shortlink', sa.Column('user_id_uuid', new_column_type, nullable=True))

    # Переносим данные, преобразуя int в UUID (используем фиктивные UUID)
    op.execute(
        "UPDATE shortlink SET user_id_uuid = gen_random_uuid()"
    )

    # Удаляем старый столбец
    op.drop_column('shortlink', 'user_id')

    # Переименовываем новый столбец в user_id
    op.alter_column('shortlink', 'user_id_uuid', new_column_name='user_id', existing_type=new_column_type, nullable=False)
    # ### end Alembic commands ###


def downgrade():
    old_column_type = sa.Integer()

    # Добавляем обратно user_id как INTEGER
    op.add_column('shortlink', sa.Column('user_id_int', old_column_type, nullable=True))

    # Откатываем данные (если есть логика для обратного преобразования)
    op.execute(
        "UPDATE shortlink SET user_id_int = NULL"
    )

    # Удаляем новый user_id
    op.drop_column('shortlink', 'user_id')

    # Переименовываем обратно
    op.alter_column('shortlink', 'user_id_int', new_column_name='user_id', existing_type=old_column_type, nullable=True)
