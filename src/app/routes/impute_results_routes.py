from fastapi import APIRouter
from src.app.data_models.place_holder import PlaceHolder

router = APIRouter(prefix="/impute-result", tags=["impute", "impute result"])

@router.get("/{model_id}", response_model=PlaceHolder)
def get_impute_result(model_id: str):
    return PlaceHolder(id="uuid goes here")
