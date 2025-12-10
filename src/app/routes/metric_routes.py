from fastapi import APIRouter
from src.app.data_models.model_metrics import ModelMetrics
from src.app.data_models import hyperparameters, model_loss

router = APIRouter(prefix="/model-metrics", tags=["metrics"])

@router.get("/{model_id}", response_model=ModelMetrics)
def get_metrics(model_id: str):
    return ModelMetrics(id=model_id,
                        model_type="asda",
                        train_time_min=21,
                        bias= 2.2,
                        gap= .2,
                        hyperparameters=[hyperparameters.Hyperparameters(model_id=model_id, param_name="Your mama", param_value="Joe")],
                        loss=[model_loss.ModelLoss(model_id=model_id, type="Training", loss_unit="MEA", loss_value=183.2 )])
