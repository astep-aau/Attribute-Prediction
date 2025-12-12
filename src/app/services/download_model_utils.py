from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.app.database_tables import ImputeResultTable
from src.app.exceptions import NotFoundException, InvalidUUIDException

async def get_model_path():
    pass
