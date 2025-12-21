from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi.responses import FileResponse
import os
import logging
from src.app.database_tables import (
    ModelMetricsTable
)
from src.app.exceptions import (
    NotFoundException,
    InvalidUUIDException,
)

logger = logging.getLogger(__name__)

async def build_file_response(model_id :str, db: AsyncSession):
    """
    Build a FileResponse for downloading a model file

    Args:
        model_id: UUID string of the model
        db: Database session

    Returns:
        FileResponse configured for model file download with .pth extension

    Raises:
        InvalidUUIDException: If model_id is not a valid UUID
        NotFoundException: If model path not found or file doesn't exist
    """
    logger.info(f"Building file response for model: {model_id}")
    file_path = await get_model_path(model_id, db)
    file_name = get_file_name(file_path)
    logger.debug(f"File path resolved to: {file_path}, filename: {file_name}")

    response = FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=file_name)

    return response

async def get_model_path(model_id: str, db: AsyncSession):
    """
    Retrieve and validate the file path for a model

    Args:
        model_id: UUID string of the model
        db: Database session

    Returns:
        Validated file path to the model file

    Raises:
        InvalidUUIDException: If model_id is not a valid UUID
        NotFoundException: If model path not found in database or file doesn't exist on disk
    """
    try:
        uuid = UUID(model_id)
    except ValueError:
        logger.warning(f"Invalid UUID format provided: {model_id}")
        raise InvalidUUIDException(f"Invalid UUID format: {model_id}")

    logger.debug(f"Querying database for model path: {model_id}")
    result = await db.execute(
        select(ModelMetricsTable.path_to_save)
        .where(ModelMetricsTable.id == uuid)
    )
    path = result.scalar_one_or_none()

    if not path:
        logger.error(f"No path found in database for model: {model_id}")
        raise NotFoundException(f"No path for model: {model_id}")

    if not os.path.isfile(path):
        logger.error(f"Model file not found on disk at path: {path}")
        raise NotFoundException(f"Model file not found at: {path}")
    
    logger.debug(f"Model path validated: {path}")
    return path

def get_file_name(filepath: str):
    """
    Extract filename from path and replace extension with .pth

    Args:
        filepath: Full path to the file

    Returns:
        Filename with .pth extension
    """
    base_name = filepath.split("/")[-1]
    file_name = base_name.rsplit(".", 1)[0] + ".pth"
    return file_name
