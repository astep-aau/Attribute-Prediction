from fastapi import APIRouter, Depends
from typing import List
from src.app.schemas import ModelType
from src.app.database import get_db, engine, Base
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.app.database_tables import ModelTypeTable

router = APIRouter(prefix="/model-types", tags=["models types"])

@router.get("/", response_model=List[ModelType])
async def get_models(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ModelTypeTable))
    models = result.scalars().all()
    return models
