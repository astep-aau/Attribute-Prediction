from fastapi import APIRouter, status, Depends
from typing import List
from src.app.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from src.app.schemas import (
    ModelMetricsResponse,
    ModelMetricsCreate,
    ModelMetricsCreateResponse,
    Hyperparam,
    ModelLoss
)
from src.app.services.metric_utils import (
    find_metric,
    create_metric,
    create_hyperparam as create_hyperparam_service,
    create_loss as create_loss_service
)
from src.app.exceptions import NotFoundException

router = APIRouter(prefix="/model-metrics", tags=["metrics"])

@router.post(
    "/create",
    response_model=ModelMetricsCreateResponse,
    status_code=status.HTTP_201_CREATED
)
async def create_model_metric(
    metric_data: ModelMetricsCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new model metric entry

    Args:
        metric_data: Model metrics data including model_type, train_time_min, bias, gap, and path_to_save
        db: Database session

    Returns:
        Created model metric record with generated ID and timestamp

    Raises:
        ForeignKeyViolationException: If model_type does not exist
    """
    result = await create_metric(metric_data, db)
    return result

@router.post(
        "/hyperparam/create",
        response_model=Hyperparam,
        status_code=status.HTTP_201_CREATED
)
async def create_hyperparam(
    hyperparam_data: Hyperparam,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new hyperparameter entry for a model

    Args:
        hyperparam_data: Hyperparameter data including model_id, param_name, and param_value
        db: Database session

    Returns:
        Created hyperparameter record

    Raises:
        ForeignKeyViolationException: If model_id does not exist
    """
    result = await create_hyperparam_service(hyperparam_data, db)
    return result

@router.post(
        "/loss/create",
        response_model=ModelLoss,
        status_code=status.HTTP_201_CREATED
)
async def create_loss(
    loss_data: ModelLoss,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new loss entry for a model

    Args:
        loss_data: Loss data including model_id, type, loss_value, and loss_unit
        db: Database session

    Returns:
        Created loss record

    Raises:
        ForeignKeyViolationException: If model_id does not exist
    """
    result = await create_loss_service(loss_data, db)
    return result


@router.get("/{model_type}", response_model=List[ModelMetricsResponse], status_code=status.HTTP_200_OK)
async def get_metrics(model_type: str, db: AsyncSession = Depends(get_db)):
    """
    Get metrics for all models of a specific type

    Args:
        model_type: UUID string of the model type
        db: Database session

    Returns:
        List of model metrics including hyperparameters and loss data

    Raises:
        InvalidUUIDException: If model_type is not a valid UUID
        NotFoundException: If no models found for the given type
    """
    result = await find_metric(model_type, db)

    return result
