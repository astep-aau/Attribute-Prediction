from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(prefix="/download_model", tags=["download", "model", "download model"])

@router.get("/{model_id}", response_class=FileResponse)
def download_model(model_id: str):

    return FileResponse(
        path='testfile.txt',
        media_type="application/octet-stream",
        filename=f"{model_id}.txt"
    )
