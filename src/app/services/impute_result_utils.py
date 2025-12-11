from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from src.app.schemas import (
    ImputeResultResonse
)
from src.app.database_tables import ImputeResultTable
from src.app.exceptions import NotFoundException, InvalidUUIDException

async def find_impute_results(
        model_id: str,
        road_id: int,
        start_time: int,
        end_time: int,
        db: AsyncSession):

    try:
        uuid = UUID(model_id)
    except ValueError:
        raise InvalidUUIDException(f"Invalid UUID format: {model_id}")
    
    result = await db.execute(
        select(
            ImputeResultTable.tms,
            ImputeResultTable.value,
            ImputeResultTable.imputed
        )
        .where(
            ImputeResultTable.model_id == uuid,
            ImputeResultTable.road_id == road_id,
            ImputeResultTable.tms.between(start_time, end_time)
        ))

    res = result.all()
    response = [ImputeResultResonse(
        tms=r[0],
        value=r[1],
        imputed=r[2]
        ) for r in res]
    return response

async def find_road_ids(model_id: str, db: AsyncSession):
    try:
        uuid = UUID(model_id)
    except ValueError:
        raise InvalidUUIDException(f"Invalid UUID format: {model_id}")
    
    result = await db.execute(
        select(ImputeResultTable.road_id)
        .where(ImputeResultTable.model_id == uuid)
        )
    roads = result.scalars().all()
    return roads

async def find_timespan(model_id: str,  road_id: int, db: AsyncSession):
    try:
        uuid = UUID(model_id)
    except ValueError:
        raise InvalidUUIDException(f"Invalid UUID format: {model_id}")
    
    result = await db.execute(
        select(
            func.min(ImputeResultTable.tms),
            func.max(ImputeResultTable.tms)
        )
        .where(ImputeResultTable.model_id == uuid,
               ImputeResultTable.road_id == road_id)
    )
    min_time, max_time = result.one()
    return min_time, max_time
