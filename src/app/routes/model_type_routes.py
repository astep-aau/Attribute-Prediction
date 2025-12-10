from fastapi import APIRouter, HTTPException, status, Depends
from typing import List
from src.app.schemas import ModelType
from src.app.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.app.database_tables import ModelTypeTable

router = APIRouter(prefix="/model-types", tags=["models types"])

@router.get("/", response_model=List[ModelType], status_code=status.HTTP_200_OK)
async def get_models(db: AsyncSession = Depends(get_db)):
    """
    Retrieve all model types from the database.

    Returns:
        List of model type objects
    """
    try:
        result = await db.execute(select(ModelTypeTable))
        models = result.scalars().all()
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
