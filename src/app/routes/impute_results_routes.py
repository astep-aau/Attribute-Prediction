from fastapi import APIRouter, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from src.app.database import get_db
from src.app.services import impute_result_utils as utils
from src.app.schemas import (
    RoadIdResponse,
    TimeIntervalResponse,
    ImputeResultResponse,
    ImputeResultCreate,
    )
from src.app.exceptions import NotFoundException

router = APIRouter(prefix="/impute-result", tags=["impute result"])

@router.post(
    "/",
    response_model=ImputeResultResponse,
    status_code=status.HTTP_201_CREATED
)
async def create_impute_result(
    result_data: ImputeResultCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new imputation result entry

    Args:
        result_data: Imputation result data including model_id, road_id, tms, value, and imputed
        db: Database session

    Returns:
        Created imputation result

    Raises:
        ForeignKeyViolationException: If model_id doesn't exist
        IntegrityError: If combination of model_id, road_id, and tms already exists
    """
    result = await utils.create_impute_result(result_data, db)
    return result

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
    """
    Get imputation results for a specific model and road within a time range

    Args:
        model_id: UUID string of the model
        road_id: ID of the road
        start_time: Start time as Unix timestamp
        end_time: End time as Unix timestamp
        db: Database session

    Returns:
        List of imputation results with timestamps, values, and imputed flags

    Raises:
        InvalidUUIDException: If model_id is not a valid UUID
        NotFoundException: If no results found for the given parameters
    """
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
    """
    Get all distinct road IDs that have imputation results for a specific model

    Args:
        model_id: UUID string of the model
        db: Database session

    Returns:
        List of road IDs with available imputation data

    Raises:
        InvalidUUIDException: If model_id is not a valid UUID
        NotFoundException: If no roads found for the given model
    """
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
    """
    Get the time range (earliest and latest timestamps) of available imputation data

    Args:
        model_id: UUID string of the model
        road_id: ID of the road
        db: Database session

    Returns:
        Time interval with start_time and end_time as Unix timestamps

    Raises:
        InvalidUUIDException: If model_id is not a valid UUID
        NotFoundException: If no data found for the given model and road
    """

    response = await utils.find_timespan(
        model_id=model_id,
        road_id=road_id,
        db=db)

    return response
