from fastapi import APIRouter, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import FileResponse
from src.app.services.download_model_utils import get_model_path

router = APIRouter(prefix="/download_model", tags=["download model"])

@router.get(
        "/{model_id}",
        response_class=FileResponse,
        status_code=status.HTTP_200_OK)
async def download_model(model_id: str, db: AsyncSession):

    file_path = get_model_path(model_id, db)
    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=f"{model_id}.txt"
)
