from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi.responses import FileResponse
import os
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

    response = FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=file_name)

    return response

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

    if not os.path.isfile(path):
        raise NotFoundException(f"Model wfile not found at: {path}")
    return path

def get_file_name(filepath: str):
    base_name = filepath.split("/")[-1]
    file_name = base_name.rsplit(".", 1)[0] + ".pth"
    return file_name
