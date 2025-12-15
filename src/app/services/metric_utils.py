from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.app.database_tables import ModelMetricsTable, HyperparamTable, LossTable
from src.app.schemas import (
    ModelMetricsResponse,
    ModelMetricsCreate,
)
from src.app.exceptions import InvalidUUIDException, NotFoundException

async def find_metric(model_type: str, db: AsyncSession):
    try:
        uuid = UUID(model_type)
    except ValueError:
        raise InvalidUUIDException(f"Invalid UUID format: {model_type}")
    result = await db.execute(
        select(ModelMetricsTable)
        .where(ModelMetricsTable.model_type == uuid)
        )

    metric_seq = result.scalars().all()
    if not metric_seq:
        raise NotFoundException(f"No metrics for type: {model_type}")

    model_metrics_list = []
    for metric in metric_seq:
        m = ModelMetricsResponse(
            id= metric.id,
            model_type= metric.model_type,
            train_time_min= metric.train_time_min,
            bias= metric.bias,
            gap= metric.gap,
            hyperparameters= await find_hyperparams(metric.id, db),
            loss= await find_loss(metric.id, db)
        )
        model_metrics_list.append(m)

    if not model_metrics_list:
        raise NotFoundException(f"No models for type: {model_type}")

    return model_metrics_list

async def find_hyperparams(model_id: UUID, db: AsyncSession):
    result = await db.execute(
        select(HyperparamTable)
        .where(HyperparamTable.model_id == model_id)
        )
    return result.scalars().all()

async def find_loss(model_id: UUID, db: AsyncSession):
    result = await db.execute(
        select(LossTable)
        .where(LossTable.model_id == model_id)
        )
    return result.scalars().all()

async def create_metric(metric_data: ModelMetricsCreate, db: AsyncSession):
    new_metric = ModelMetricsTable(
        model_type=metric_data.model_type,
        train_time_min=metric_data.train_time_min,
        bias=metric_data.bias,
        gap=metric_data.gap,
        path_to_save=metric_data.path_to_save
    )

    db.add(new_metric)
    await db.commit()
    await db.refresh(new_metric)

    return new_metric
