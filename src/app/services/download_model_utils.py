from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi.responses import FileResponse
from src.app.database_tables import (
    ModelMetricsTable
)
from src.app.exceptions import (
    NotFoundException,
    InvalidUUIDException,
)

async def build_file_response(model_id :str, db: AsyncSession):
    file_path = await get_model_path(model_id, db)
    file_name = get_file_name(file_path)

    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=file_name)

async def get_model_path(model_id: str, db: AsyncSession):
    try:
        uuid = UUID(model_id)
    except ValueError:
        raise InvalidUUIDException(f"Invalid UUID format: {model_id}")

    result = await db.execute(
        select(ModelMetricsTable.path_to_save)
        .where(ModelMetricsTable.id == uuid)
    )
    path = result.scalar_one_or_none()

    if not path:
        raise NotFoundException(f"No path for model: {model_id}")

    return path

def get_file_name(filepath: str):
    return filepath.split("/")[-1]
