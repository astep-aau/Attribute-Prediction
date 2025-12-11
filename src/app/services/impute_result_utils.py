from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime
from src.app.schemas import (
    ImputeResultResonse, RoadResponse, TimeIntervalResponse
)
from src.app.database_tables import ImputeResultTable

async def find_impute_results(
        model_id: UUID,
        road_id: int,
        start_time: datetime,
        end_time: datetime,
        db: AsyncSession):
    result = await db.execute(
        select(
            ImputeResultTable.tms,
            ImputeResultTable.value,
            ImputeResultTable.imputed
        )
        .where(
            ImputeResultTable.model_id == model_id,
            ImputeResultTable.road_id == road_id,
            ImputeResultTable.tms.between(start_time, end_time)
        ))

    impute_results = result.scalars().all()
    return impute_results

async def find_road_ids(model_id: UUID, db: AsyncSession):
    result = await db.execute(
        select(ImputeResultTable.road_id)
        .where(ImputeResultTable.model_id == model_id)
        )
    roads = result.scalars().all()
    return roads

async def find_timespan(model_id: UUID,  road_id: int, db: AsyncSession):
    result = await db.execute(
        select(
            func.min(ImputeResultTable.tms),
            func.max(ImputeResultTable.tms)
        )
        .where(ImputeResultTable.model_id == model_id,
               ImputeResultTable.road_id == road_id)
    )
    min_time, max_time = result.one()
    return min_time, max_time
