from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from src.app.database_tables import ModelTypeTable
from src.app.schemas import ModelTypeCreate
from src.app.exceptions import NotFoundException, ForeignKeyViolationException


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
        new_model_type = ModelTypeTable(
            name=model_type_data.name
        )

        db.add(new_model_type)
        await db.commit()
        await db.refresh(new_model_type)

        return new_model_type
    except IntegrityError as e:
        await db.rollback()
        raise ForeignKeyViolationException(f"Database constraint violated: {str(e)}")


async def get_all_model_types(db: AsyncSession):
    """
    Get all available model types from the database

    Args:
        db: Database session

    Returns:
        List of all model types with their IDs and names

    Raises:
        NotFoundException: If no model types found in the database
    """
    result = await db.execute(select(ModelTypeTable))
    models = result.scalars().all()

    if not models:
        raise NotFoundException("No model types found")

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
    result = await db.execute(
        select(ModelTypeTable).where(ModelTypeTable.name == name)
    )
    return result.scalar_one_or_none()
