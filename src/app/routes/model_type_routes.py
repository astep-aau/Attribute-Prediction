from fastapi import APIRouter, status, Depends
from typing import List
from src.app.schemas import ModelTypeResponse, ModelTypeCreate
from src.app.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from src.app.services.model_type_utils import (
    create_model_type as create_model_type_service,
    get_all_model_types
)

router = APIRouter(prefix="/model-types", tags=["models types"])

@router.post("/", response_model=ModelTypeResponse, status_code=status.HTTP_201_CREATED)
async def create_model_type(
    model_type_data: ModelTypeCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new model type

    Args:
        model_type_data: Model type data with name
        db: Database session

    Returns:
        Created model type with generated ID

    Raises:
        ForeignKeyViolationException: If database constraint violated
    """
    return await create_model_type_service(model_type_data, db)

@router.get("/", response_model=List[ModelTypeResponse], status_code=status.HTTP_200_OK)
async def get_models(db: AsyncSession = Depends(get_db)):
    """
    Get all available model types

    Args:
        db: Database session

    Returns:
        List of all model types with their IDs and names

    Raises:
        NotFoundException: If no model types found in the database
    """
    return await get_all_model_types(db)
