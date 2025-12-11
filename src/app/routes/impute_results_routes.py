from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.app.database import get_db
from src.app.schemas import (
    RoadResponse,
    TimeIntervalResponse,
    ImputeResultResonse,
    PlaceHolder
    )

router = APIRouter(prefix="/impute-result", tags=["impute result"])

@router.get("/{model_id}", response_model=PlaceHolder)
def get_impute_result(model_id: str):
    return PlaceHolder(id="uuid goes here")

@router.get(
        "/roads/{model_id}",
        response_model=RoadResponse,
        status_code=status.HTTP_200_OK)
def get_road_ids(model_id: str, db: AsyncSession = Depends(get_db)):
    pass

@router.get(
        "/time-interval/{model_id}/{road_id}",
        response_model=TimeIntervalResponse,
        status_code=status.HTTP_200_OK)
def get_time_interval(model_id: str, road_id: int):
    pass
