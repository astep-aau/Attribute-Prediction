from pydantic import BaseModel

class Hyperparameters(BaseModel):
    model_id: str
    param_name: str
    param_value: str
