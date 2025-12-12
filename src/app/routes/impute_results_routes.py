from fastapi import APIRouter, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from src.app.database import get_db
from src.app.services import impute_result_utils as utils
from src.app.schemas import (
    RoadIdResponse,
    TimeIntervalResponse,
    ImputeResultResponse,
    )
from src.app.exceptions import NotFoundException

router = APIRouter(prefix="/impute-result", tags=["impute result"])

@router.get(
        "/{model_id}/{road_id}/{start_time}/{end_time}",
        response_model=List[ImputeResultResponse],
        status_code=status.HTTP_200_OK
        )
async def get_impute_result(
        model_id: str,
        road_id: str,
        start_time: int,
        end_time: int,
        db: AsyncSession = Depends(get_db)
    ):
    response = await utils.find_impute_results(
        model_id=model_id,
        road_id=road_id,
        start_time=start_time,
        end_time=end_time,
        db=db)

    return response

@router.get(
        "/roads/{model_id}",
        response_model=List[RoadIdResponse],
        status_code=status.HTTP_200_OK
        )
async def get_road_ids(model_id: str, db: AsyncSession = Depends(get_db)):
    response = await utils.find_road_ids(model_id=model_id, db=db)

    return response

@router.get(
        "/time-interval/{model_id}/{road_id}",
        response_model=TimeIntervalResponse,
        status_code=status.HTTP_200_OK
        )
async def get_time_interval(
    model_id: str,
    road_id: str,
    db: AsyncSession = Depends(get_db)
    ):

    response = await utils.find_timespan(
        model_id=model_id,
        road_id=road_id,
        db=db)

    return response
