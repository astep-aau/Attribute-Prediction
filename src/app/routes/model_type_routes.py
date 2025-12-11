from fastapi import APIRouter, HTTPException, status, Depends
from typing import List
from src.app.schemas import ModelType
from src.app.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.app.database_tables import ModelTypeTable
from src.app.exceptions import NotFoundException

router = APIRouter(prefix="/model-types", tags=["models types"])

@router.get("/", response_model=List[ModelType], status_code=status.HTTP_200_OK)
async def get_models(db: AsyncSession = Depends(get_db)):
    """
    Retrieve all model types from the database.

    Returns:
        List of model type objects
    """
    result = await db.execute(select(ModelTypeTable))
    models = result.scalars().all()

    if not models:
        raise NotFoundException(f"no model types found")

    return models
