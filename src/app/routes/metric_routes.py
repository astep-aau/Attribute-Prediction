from fastapi import APIRouter, HTTPException, status, Depends
from typing import List
from src.app.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from src.app.schemas import ModelMetrics, Hyperparam, ModelLoss
from src.app.services.metric_finder import find_metric

router = APIRouter(prefix="/model-metrics", tags=["metrics"])

@router.get("/{model_type}", response_model=List[ModelMetrics], status_code=status.HTTP_200_OK)
async def get_metrics(model_type: str, db: AsyncSession = Depends(get_db)):
    """
    Gets the metrics of all models , based on model type

    Returns:
        The models metrics
    """
    try:
        result = await find_metric(model_type, db)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
