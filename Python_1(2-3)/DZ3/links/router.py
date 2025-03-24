from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, update, insert, delete
from sqlalchemy.ext.asyncio import AsyncSession
from .models import shortlink
from .schemas import LinkCreate, LinkStats, ShortCode, LinkUpdate
import string
import random
from datetime import datetime
from auth.db import User, get_async_session
from auth.users import current_active_user
from fastapi import status
from typing import Optional
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

def generate_short_code(length: int = 6) -> str:
    """Генерирует случайный короткий код без домена"""
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=length))


router = APIRouter(
    prefix="/links",
    tags=["Links"]
)

security = HTTPBearer(auto_error=False)

async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    session: AsyncSession = Depends(get_async_session)
) -> Optional[User]:
    """Получаем пользователя, если есть токен, иначе возвращаем None"""
    if not credentials:
        return None
    try:
        return await current_active_user(credentials.credentials, session)
    except HTTPException:
        return None

@router.post("/shorten", response_model=ShortCode)
async def create_short_code(
    link: LinkCreate,
    session: AsyncSession = Depends(get_async_session),
    user: Optional[User] = Depends(get_optional_user)  # Теперь пользователь по умолчанию None
):
    """Создает короткую ссылку, автоматически генерируя short_code"""
    
    while True:
        short_code = generate_short_code()
        stmt = select(shortlink).where(shortlink.c.short_code == short_code)
        result = await session.execute(stmt)
        if not result.scalar():  
            break

    now = datetime.utcnow()
    stmt = insert(shortlink).values(
        user_id=user.id if user else None,  # Если user нет, ставим NULL
        long_link=str(link.long_link),
        short_code=short_code,
        created_at=now,
        expires_at=link.expires_at,
        clicks_count=0,
        last_clicked_at=None
    ).returning(shortlink.c.short_code)

    result = await session.execute(stmt)
    await session.commit()
    
    return ShortCode(short_code=result.scalar_one())

@router.get("/{short_code}")
async def redirect_link(short_code: str, session: AsyncSession = Depends(get_async_session)):
    """Перенаправляет по короткой ссылке"""

    stmt = select(shortlink).where(shortlink.c.short_code == short_code)
    result = await session.execute(stmt)
    link = result.fetchone()

    if not link:
        raise HTTPException(status_code=404, detail="Code not found")

    # result.fetchone() возвращает Row, поэтому доступ по индексу или через маппинг
    link_data = link._mapping  # Превращаем в словарь-подобный объект

    # Обновляем количество кликов
    upd_stmt = update(shortlink).where(shortlink.c.id == link_data['id']).values(
        clicks_count=link_data['clicks_count'] + 1,
        last_clicked_at=datetime.utcnow()
    )
    await session.execute(upd_stmt)
    await session.commit()

    return {"redirect_to": link_data['long_link']}

@router.delete("/{short_code}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_link(
    short_code: str, 
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user)  # Обязательная аутентификация
):
    stmt = select(shortlink).where(shortlink.c.short_code == short_code)
    result = await session.execute(stmt)
    link = result.fetchone()

    if not link:
        raise HTTPException(status_code=404, detail="Code not found")

    # Проверяем, что пользователь владелец ссылки
    if link.user_id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")

    del_stmt = delete(shortlink).where(shortlink.c.id == link.id)
    await session.execute(del_stmt)
    await session.commit()


@router.put("/{short_code}", response_model=LinkStats)
async def update_link(
    short_code: str, 
    data: LinkUpdate, 
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user)  # Обязательная аутентификация
):
    stmt = select(shortlink).where(shortlink.c.short_code == short_code)
    result = await session.execute(stmt)
    link = result.fetchone()

    if not link:
        raise HTTPException(status_code=404, detail="Link not found")

    if link.user_id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")

    upd_data = {}
    if data.long_link:
        upd_data["long_link"] = str(data.long_link)

    upd_stmt = update(shortlink).where(shortlink.c.id == link.id).values(**upd_data).returning(shortlink)
    result = await session.execute(upd_stmt)
    await session.commit()

    updated_link = result.fetchone()

    return LinkStats(**updated_link._mapping)
