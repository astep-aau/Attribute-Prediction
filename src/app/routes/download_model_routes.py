from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os

router = APIRouter(prefix="/download_model", tags=["download model"])

@router.get("/{model_id}", response_class=FileResponse)
def download_model(model_id: str):

    file_path = "testfile.txt"
    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=f"{model_id}.txt"
)
