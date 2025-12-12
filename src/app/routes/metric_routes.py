from fastapi import APIRouter, HTTPException, status, Depends
from typing import List
from src.app.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from src.app.schemas import ModelMetricsResponse
from src.app.services.metric_utils import find_metric
from src.app.exceptions import NotFoundException

router = APIRouter(prefix="/model-metrics", tags=["metrics"])

@router.get("/{model_type}", response_model=List[ModelMetricsResponse], status_code=status.HTTP_200_OK)
async def get_metrics(model_type: str, db: AsyncSession = Depends(get_db)):
    """
    Gets the metrics of all models , based on model type

    Returns:
        The models metrics
    """
    result = await find_metric(model_type, db)

    if not result:
        raise NotFoundException(f"No models for type: {model_type}")

    return result
