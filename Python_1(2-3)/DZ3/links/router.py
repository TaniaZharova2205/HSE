from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, update, insert, delete
from sqlalchemy.ext.asyncio import AsyncSession
from .models import shortlink
from .schemas import LinkCreate, LinkStats, ShortCode, LinkUpdate
import string
import random
from datetime import datetime
from auth.db import User, get_async_session
from auth.users import current_active_user
from typing import Optional
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from redis.asyncio import Redis

# Инициализация кэша
redis = Redis(host='localhost', port=6379, decode_responses=True)

def generate_short_code(length: int = 6) -> str:
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=length))

router = APIRouter(prefix="/links", tags=["Links"])
security = HTTPBearer(auto_error=False)

async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    session: AsyncSession = Depends(get_async_session)
) -> Optional[int]:
    if not credentials:
        return None
    try:
        user = await current_active_user(credentials.credentials, session)
        return user.id if user else None
    except HTTPException as e:
        return None

@router.post("/shorten", response_model=ShortCode)
async def create_short_code(
    link: LinkCreate,
    session: AsyncSession = Depends(get_async_session),
    user_id: Optional[int] = Depends(get_optional_user)
):
    if link.custom_alias:
        stmt = select(shortlink).where(shortlink.c.short_code == link.custom_alias)
        result = await session.execute(stmt)
        if result.scalar():
            raise HTTPException(status_code=400, detail="Alias already taken")
        short_code = link.custom_alias
    else:
        while True:
            short_code = generate_short_code()
            stmt = select(shortlink).where(shortlink.c.short_code == short_code)
            result = await session.execute(stmt)
            if not result.scalar():  
                break
    
    now = datetime.utcnow()
    stmt = insert(shortlink).values(
        user_id=user_id,
        long_link=str(link.long_link),
        short_code=short_code,
        created_at=now,
        expires_at=link.expires_at,
        clicks_count=0,
        last_clicked_at=None
    )
    await session.execute(stmt)
    await session.commit()
    return ShortCode(short_code=short_code)

@router.get("/{short_code}")
async def redirect_link(short_code: str, session: AsyncSession = Depends(get_async_session)):
    cached_url = await redis.get(f"short:{short_code}")
    if cached_url:
        return {"redirect_to": cached_url}
    
    stmt = select(shortlink).where(shortlink.c.short_code == short_code)
    result = await session.execute(stmt)
    link = result.fetchone()
    if not link:
        raise HTTPException(status_code=404, detail="Code not found")
    link_data = link._mapping
    
    if link_data['expires_at'] and link_data['expires_at'] < datetime.utcnow():
        raise HTTPException(status_code=410, detail="Link expired")
    
    upd_stmt = update(shortlink).where(shortlink.c.id == link_data['id']).values(
        clicks_count=link_data['clicks_count'] + 1,
        last_clicked_at=datetime.utcnow()
    )
    await session.execute(upd_stmt)
    await session.commit()
    
    await redis.setex(f"short:{short_code}", 3600, link_data['long_link'])
    return {"redirect_to": link_data['long_link']}

@router.delete("/{short_code}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_link(
    short_code: str, 
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user)
):
    stmt = select(shortlink).where(shortlink.c.short_code == short_code)
    result = await session.execute(stmt)
    link = result.fetchone()
    if not link:
        raise HTTPException(status_code=404, detail="Code not found")
    if link.user_id != user.id and link.user_id is not None:
        raise HTTPException(status_code=403, detail="Forbidden")
    del_stmt = delete(shortlink).where(shortlink.c.id == link.id)
    await session.execute(del_stmt)
    await session.commit()
    await redis.delete(f"short:{short_code}")

@router.put("/{short_code}", response_model=LinkStats)
async def update_link(
    short_code: str, 
    data: LinkUpdate, 
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user)
):
    stmt = select(shortlink).where(shortlink.c.short_code == short_code)
    result = await session.execute(stmt)
    link = result.fetchone()
    if not link:
        raise HTTPException(status_code=404, detail="Link not found")
    if link.user_id != user.id and link.user_id is not None:
        raise HTTPException(status_code=403, detail="Forbidden")
    upd_data = {}
    if data.long_link:
        upd_data["long_link"] = str(data.long_link)
    upd_stmt = update(shortlink).where(shortlink.c.id == link.id).values(**upd_data).returning(shortlink)
    result = await session.execute(upd_stmt)
    await session.commit()
    updated_link = result.fetchone()
    await redis.delete(f"short:{short_code}")
    return LinkStats(**updated_link._mapping)

@router.get("/search")
async def search_link(original_url: str, session: AsyncSession = Depends(get_async_session)):
    stmt = select(shortlink).where(shortlink.c.long_link == original_url)
    result = await session.execute(stmt)
    links = result.fetchall()
    if not links:
        raise HTTPException(status_code=404, detail="No links found")
    return [{"short_code": row.short_code, "created_at": row.created_at} for row in links]
