from sqlalchemy import Table, Column, Integer, MetaData, String, DateTime
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP
from datetime import datetime
metadata = MetaData()

shortlink = Table(
    "shortlink",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", UUID(as_uuid=True)),
    Column("long_link", String, nullable=False),
    Column("short_code", String, nullable=False, unique=True),

    Column("created_at", TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False),
    Column("expires_at", TIMESTAMP(timezone=True), nullable=True),

    Column("clicks_count", Integer, default=0, nullable=False),
    Column("last_clicked_at", DateTime, nullable=True)
)

