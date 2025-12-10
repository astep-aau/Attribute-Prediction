from pydantic import BaseModel

class ModelLoss(BaseModel):
    model_id: str
    type: str
    loss_value: float
    loss_unit: float
