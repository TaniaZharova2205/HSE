from sqlalchemy import Table, Column, Integer, MetaData, String
metadata = MetaData()

shortlink = Table(
    "shortlink",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, nullable=False),
    Column("long_link", String, nullable=False),
    Column("short_link", String, nullable=False),
)

