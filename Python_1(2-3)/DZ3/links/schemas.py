from datetime import datetime

from pydantic import BaseModel


class BookingCreate(BaseModel):
    user_id: int
    room_id: int
    category: str
    start_time: datetime
    end_time: datetime
