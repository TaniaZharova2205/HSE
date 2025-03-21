from sqlalchemy import Table, Column, Integer, DateTime, MetaData, String
metadata = MetaData()

booking = Table(
    "booking",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer),
    Column("room_id", Integer),
    Column("category", String),
    Column("start_time", DateTime, nullable=False),
    Column("end_time", DateTime, nullable=False),
)

