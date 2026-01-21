from fastapi import APIRouter, status
from typing import List
from src.app.services import impute_result_utils as utils
from src.app.schemas import (
    RoadIdResponse,
    TimeIntervalResponse,
    ImputeResultResponse,
    ImputeResultCreate,
)

router = APIRouter(prefix="/impute-result", tags=["impute result"])

@router.post(
    "/",
    response_model=ImputeResultResponse,
    status_code=status.HTTP_201_CREATED
)
async def create_impute_result(
    result_data: ImputeResultCreate
):
    """
    Create a new imputation result entry

    Args:
        result_data: Imputation result data including model_id, road_id, tms, value, and imputed

    Returns:
        Created imputation result

    Raises:
        InvalidUUIDException: If model_id is not a valid UUID
    """
    result = await utils.create_impute_result(result_data)
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
        end_time: int
    ):
    """
    Get imputation results for a specific model and road within a time range

    Args:
        model_id: UUID string of the model
        road_id: ID of the road
        start_time: Start time as Unix timestamp
        end_time: End time as Unix timestamp

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
        end_time=end_time)

    return response

@router.get(
        "/roads/{model_id}",
        response_model=List[RoadIdResponse],
        status_code=status.HTTP_200_OK
        )
async def get_road_ids(model_id: str):
    """
    Get all distinct road IDs that have imputation results for a specific model

    Args:
        model_id: UUID string of the model

    Returns:
        List of road IDs with available imputation data

    Raises:
        InvalidUUIDException: If model_id is not a valid UUID
        NotFoundException: If no roads found for the given model
    """
    response = await utils.find_road_ids(model_id=model_id)

    return response

@router.get(
        "/time-interval/{model_id}/{road_id}",
        response_model=TimeIntervalResponse,
        status_code=status.HTTP_200_OK
        )
async def get_time_interval(
    model_id: str,
    road_id: str
    ):
    """
    Get the time range (earliest and latest timestamps) of available imputation data

    Args:
        model_id: UUID string of the model
        road_id: ID of the road

    Returns:
        Time interval with start_time and end_time as Unix timestamps

    Raises:
        InvalidUUIDException: If model_id is not a valid UUID
        NotFoundException: If no data found for the given model and road
    """

    response = await utils.find_timespan(
        model_id=model_id,
        road_id=road_id)

    return response
