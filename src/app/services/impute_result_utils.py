from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError
from src.app.schemas import (
    ImputeResultResponse,
    RoadIdResponse,
    TimeIntervalResponse,
    ImputeResultCreate,
)
from src.app.database_tables import ImputeResultTable
from src.app.exceptions import NotFoundException, InvalidUUIDException, ForeignKeyViolationException

async def find_impute_results(
        model_id: str,
        road_id: int,
        start_time: int,
        end_time: int,
        db: AsyncSession):
    """
    Find imputation results for a specific model and road within a time range

    Args:
        model_id: UUID string of the model
        road_id: ID of the road
        start_time: Start time as Unix timestamp
        end_time: End time as Unix timestamp
        db: Database session

    Returns:
        List of ImputeResultResponse objects with timestamps, values, and imputed flags

    Raises:
        InvalidUUIDException: If model_id is not a valid UUID
        NotFoundException: If no results found for the given parameters
    """

    try:
        uuid = UUID(model_id)
    except ValueError:
        raise InvalidUUIDException(f"Invalid UUID format: {model_id}")

    result = await db.execute(
        select(
            ImputeResultTable.tms,
            ImputeResultTable.value,
            ImputeResultTable.imputed
        )
        .where(
            ImputeResultTable.model_id == uuid,
            ImputeResultTable.road_id == road_id,
            ImputeResultTable.tms.between(start_time, end_time)
        ))

    res = result.all()

    if not res:
        raise NotFoundException(f"No impute results found for model {model_id}, road {road_id}")

    response = [ImputeResultResponse(
        tms=r[0],
        value=r[1],
        imputed=r[2]
        ) for r in res]

    return response

async def find_road_ids(model_id: str, db: AsyncSession):
    """
    Find all distinct road IDs that have imputation data for a model

    Args:
        model_id: UUID string of the model
        db: Database session

    Returns:
        List of RoadIdResponse objects containing unique road IDs

    Raises:
        InvalidUUIDException: If model_id is not a valid UUID
        NotFoundException: If no roads found for the model
    """
    try:
        uuid = UUID(model_id)
    except ValueError:
        raise InvalidUUIDException(f"Invalid UUID format: {model_id}")

    result = await db.execute(
        select(ImputeResultTable.road_id)
        .where(ImputeResultTable.model_id == uuid)
        .distinct()
        )
    roads = result.scalars().all()

    if not roads:
        raise NotFoundException(f"No roads found for model {model_id}")

    return [RoadIdResponse(road_id=road_id) for road_id in roads]

async def find_timespan(model_id: str, road_id: str, db: AsyncSession):
    """
    Find the minimum and maximum timestamps for imputation data

    Args:
        model_id: UUID string of the model
        road_id: ID of the road
        db: Database session

    Returns:
        TimeIntervalResponse with start_time and end_time as Unix timestamps

    Raises:
        InvalidUUIDException: If model_id is not a valid UUID
        NotFoundException: If no timestamps found for the given model and road
    """
    try:
        uuid = UUID(model_id)
    except ValueError:
        raise InvalidUUIDException(f"Invalid UUID format: {model_id}")

    result = await db.execute(
        select(
            func.min(ImputeResultTable.tms),
            func.max(ImputeResultTable.tms)
        )
        .where(ImputeResultTable.model_id == uuid,
               ImputeResultTable.road_id == road_id)
    )
    min_time, max_time = result.one()

    if not min_time:
        raise NotFoundException(f"No min time found for model {model_id}, road {road_id}")

    if not max_time:
        raise NotFoundException(f"No max time found for model {model_id}, road {road_id}")

    return TimeIntervalResponse(start_time= min_time, end_time= max_time)

async def create_impute_result(result_data: ImputeResultCreate, db: AsyncSession):
    """
    Create a new imputation result entry

    Args:
        result_data: Imputation result data including model_id, road_id, tms, value, and imputed
        db: Database session

    Returns:
        Created ImputeResultTable object

    Raises:
        ForeignKeyViolationException: If model_id doesn't exist
        IntegrityError: For other database constraint violations (e.g., duplicate primary key)
    """
    new_result = ImputeResultTable(
        model_id=result_data.model_id,
        road_id=result_data.road_id,
        tms=result_data.tms,
        value=result_data.value,
        imputed=result_data.imputed
    )

    db.add(new_result)
    try:
        await db.commit()
        await db.refresh(new_result)
    except IntegrityError as e:
        await db.rollback()
        if "foreign key" in str(e).lower():
            raise ForeignKeyViolationException(f"Invalid model_id: {result_data.model_id}. Model does not exist.")
        raise

    return new_result
