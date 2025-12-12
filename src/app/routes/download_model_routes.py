from fastapi import APIRouter, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import FileResponse
from src.app.database import get_db
from src.app.services.download_model_utils import build_file_response

router = APIRouter(prefix="/download_model", tags=["download model"])

@router.get(
        "/{model_id}",
        response_class=FileResponse,
        status_code=status.HTTP_200_OK)
async def download_model(model_id: str, db: AsyncSession = Depends(get_db)):
    file_response = build_file_response(model_id, db)
    return file_response
