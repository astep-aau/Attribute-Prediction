from fastapi import APIRouter
from src.app.data_models.model_type import ModelType

router = APIRouter(prefix="/models", tags=["models"])

@router.get("/", response_model=ModelType)
def get_models():
    return ModelType(uuid= "Test", name= "Test")
