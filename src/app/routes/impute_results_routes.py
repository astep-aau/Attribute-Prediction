from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.app.database import get_db
from src.app.services import impute_result_utils as utils
from src.app.schemas import (
    RoadResponse,
    TimeIntervalResponse,
    ImputeResultResonse,
    )
from datetime import datetime

router = APIRouter(prefix="/impute-result", tags=["impute result"])

@router.get(
        "/{model_id}/{road_id}/{start_time}/{end_time}",
        response_model=ImputeResultResonse,
        status_code=status.HTTP_200_OK
        )
async def get_impute_result(
        model_id: str,
        road_id: int,
        start_time: int,
        end_time: int,
        db: AsyncSession = Depends(get_db)
    ):
    try:
        response = await utils.find_impute_results(
            model_id=model_id,
            road_id=road_id,
            start_time=start_time,
            end_time=end_time,
            db=db)

        if not response:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Not Found")

        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e))

@router.get(
        "/roads/{model_id}",
        response_model=RoadResponse,
        status_code=status.HTTP_200_OK
        )
async def get_road_ids(model_id: str, db: AsyncSession = Depends(get_db)):
    try:
        response = await utils.find_road_ids(model_id=model_id, db=db)

        if not response:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No roads for {model_id} or invalid id"
                )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
        "/time-interval/{model_id}/{road_id}",
        response_model=TimeIntervalResponse,
        status_code=status.HTTP_200_OK
        )
async def get_time_interval(
    model_id: str,
    road_id: int,
    db: AsyncSession = Depends(get_db)
    ):
    try:
        response = await utils.find_timespan(
            model_id= model_id,
            road_id= road_id,
            db= db)

        if not response:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
