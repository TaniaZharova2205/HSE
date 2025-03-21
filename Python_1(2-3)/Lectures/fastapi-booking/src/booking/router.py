from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, insert
from sqlalchemy.ext.asyncio import AsyncSession
import time
from database import get_async_session
from .models import booking
from .schemas import BookingCreate
from fastapi_cache.decorator import cache


router = APIRouter(
    prefix="/booking",
    tags=["Booking"]
)


@router.get("/long")
@cache(expire=60)
async def get_long():
    time.sleep(5)
    return "hello"


@router.get("/")
async def get_category_bookings(category: str, session: AsyncSession = Depends(get_async_session)):
    try:
        query = select(booking).where(booking.c.category == category)
        result = await session.execute(query)
        # https://stackoverflow.com/questions/76322342/fastapi-sqlalchemy-cannot-convert-dictionary-update-sequence-element-0-to-a-seq
        return {
            "status": "success",
            "data": result.scalars().all()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "status": "error",
            "data": None,
        })


@router.post("/")
async def add_booking(new_booking: BookingCreate, session: AsyncSession = Depends(get_async_session)):
    statement = insert(booking).values(**new_booking.dict())
    await session.execute(statement)
    await session.commit()
    return {"status": "success"}
