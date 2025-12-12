from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from uuid import UUID

class Hyperparam(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_id: UUID
    param_name: str
    param_value: str

class ModelLoss(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_id: UUID
    type: str
    loss_value: float
    loss_unit: str

class ModelMetricsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    model_type: UUID
    train_time_min: int
    bias: Optional[float] = None
    gap: Optional[float] = None
    hyperparameters: List[Hyperparam]
    loss: List[ModelLoss]
