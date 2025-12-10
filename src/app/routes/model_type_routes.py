from fastapi import APIRouter
from typing import List
from src.app.data_models.model_type import ModelType

router = APIRouter(prefix="/model-types", tags=["models", "types"])

@router.get("/", response_model=List[ModelType])
def get_models():
    return [ModelType(uuid= "Test", name= "Test")]
