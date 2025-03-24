from datetime import datetime

from pydantic import BaseModel, HttpUrl
from uuid import UUID

class LinkCreate(BaseModel):
    long_link: HttpUrl
    expires_at: datetime | None = None

class LinkUpdate(BaseModel):
    long_link: HttpUrl | None = None

class LinkStats(BaseModel):
    id: int
    user_id: UUID 
    long_link: str
    short_code: str
    created_at: datetime
    expires_at: datetime | None
    clicks_count: int
    last_clicked_at: datetime | None

class ShortCode(BaseModel):
    short_code: str