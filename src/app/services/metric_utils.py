from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from src.app.database_tables import (
    ModelMetricsTable,
    HyperparamTable,
    LossTable
)
from src.app.schemas import (
    ModelMetricsResponse,
    ModelMetricsCreate,
    Hyperparam,
    ModelLoss
)
from src.app.exceptions import InvalidUUIDException, NotFoundException, ForeignKeyViolationException

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
    try:
        await db.commit()
        await db.refresh(new_metric)
    except IntegrityError as e:
        await db.rollback()
        if "foreign key" in str(e).lower():
            raise ForeignKeyViolationException(f"Invalid model_type ID: {metric_data.model_type}. Model type does not exist.")
        raise

    return new_metric

async def create_hyperparam(hyperparam_data: Hyperparam, db: AsyncSession):
    new_hyperparam = HyperparamTable(
        model_id = hyperparam_data.model_id,
        param_name = hyperparam_data.param_name,
        param_value = hyperparam_data.param_value
    )

    db.add(new_hyperparam)
    try:
        await db.commit()
        await db.refresh(new_hyperparam)
    except IntegrityError as e:
        await db.rollback()
        if "foreign key" in str(e).lower():
            raise ForeignKeyViolationException(f"Invalid model_id: {hyperparam_data.model_id}. Model does not exist.")
        raise

    return new_hyperparam

async def create_loss(loss_data: ModelLoss, db: AsyncSession):
    new_loss = LossTable(
        model_id = loss_data.model_id,
        type = loss_data.type,
        loss_value = loss_data.loss_value,
        loss_unit = loss_data.loss_unit
    )

    db.add(new_loss)
    try:
        await db.commit()
        await db.refresh(new_loss)
    except IntegrityError as e:
        await db.rollback()
        if "foreign key" in str(e).lower():
            raise ForeignKeyViolationException(f"Invalid model_id: {loss_data.model_id}. Model does not exist.")
        raise

    return new_loss
