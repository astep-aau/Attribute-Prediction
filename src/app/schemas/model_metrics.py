from pydantic import BaseModel
from src.app.schemas.hyperparameters import Hyperparameters
from src.app.schemas.model_loss import ModelLoss
from typing import List

class ModelMetrics(BaseModel):
    id: str
    model_type: str
    train_time_min: int
    bias: float
    gap: float
    hyperparameters: List[Hyperparameters]
    loss: List[ModelLoss]
