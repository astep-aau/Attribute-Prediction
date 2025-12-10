from pydantic import BaseModel, ConfigDict
from src.app.schemas.hyperparam import Hyperparam
from src.app.schemas.model_loss import ModelLoss
from typing import List, Optional
from uuid import UUID

class ModelMetrics(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    model_type: UUID
    train_time_min: int
    bias: Optional[float] = None
    gap: Optional[float] = None
    hyperparameters: List[Hyperparam]
    loss: List[ModelLoss]
