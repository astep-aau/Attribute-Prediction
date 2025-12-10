from fastapi import APIRouter, Depends
from uuid import UUID
from src.app.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.app.database_tables import ModelMetricsTable, HyperparamTable, LossTable
from src.app.schemas import ModelMetrics, Hyperparam, ModelLoss

async def find_metric(model_type: str, db: AsyncSession):
    uuid_type = UUID(model_type)
    result = await db.execute(
        select(ModelMetricsTable)
        .where(ModelMetricsTable.model_type == uuid_type)
        )

    metric_seq = result.scalars().all()

    model_metrics_list = []
    for metric in metric_seq:
        m = ModelMetrics(
            id= metric.id,
            model_type= metric.model_type,
            train_time_min= metric.train_time_min,
            bias= metric.bias,
            gap= metric.gap,
            hyperparameters= await find_hyperparams(metric.id, db),
            loss= await find_loss(metric.id, db)
        )
        model_metrics_list.append(m)

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
