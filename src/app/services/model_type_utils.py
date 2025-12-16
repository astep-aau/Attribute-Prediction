from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from src.app.database_tables import ModelTypeTable
from src.app.schemas import ModelTypeCreate
from src.app.exceptions import NotFoundException, ForeignKeyViolationException
import logging

logger = logging.getLogger(__name__)


async def create_model_type(
    model_type_data: ModelTypeCreate,
    db: AsyncSession
):
    """
    Create a new model type in the database

    Args:
        model_type_data: Model type data with name
        db: Database session

    Returns:
        Created model type with generated ID

    Raises:
        ForeignKeyViolationException: If database constraint is violated
    """
    try:
        logger.info(f"Creating new model type: {model_type_data.name}")
        new_model_type = ModelTypeTable(
            name=model_type_data.name
        )

        db.add(new_model_type)
        await db.commit()
        await db.refresh(new_model_type)
        logger.info(f"Model type created successfully with ID: {new_model_type.id}")

        return new_model_type
    except IntegrityError as e:
        await db.rollback()
        logger.error(f"Integrity error creating model type '{model_type_data.name}': {str(e)}")
        raise ForeignKeyViolationException(f"Database constraint violated: {str(e)}")


async def get_all_model_types(db: AsyncSession):
    """
    Get all available model types from the database

    Args:
        db: Database session

    Returns:
        List of all model types with their IDs and names (empty list if none found)
    """
    logger.debug("Fetching all model types")
    result = await db.execute(select(ModelTypeTable))
    models = result.scalars().all()
    logger.info(f"Retrieved {len(models)} model types")
    return models


async def get_model_type_by_name(name: str, db: AsyncSession) -> ModelTypeTable | None:
    """
    Find a model type by name

    Args:
        name: Name of the model type to find
        db: Database session

    Returns:
        ModelTypeTable if found, None otherwise
    """
    logger.debug(f"Looking up model type by name: {name}")
    result = await db.execute(
        select(ModelTypeTable).where(ModelTypeTable.name == name)
    )
    model_type = result.scalars().first()
    if model_type:
        logger.debug(f"Model type '{name}' found with ID: {model_type.id}")
    else:
        logger.debug(f"Model type '{name}' not found")
    return model_type
