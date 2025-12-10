from pydantic import BaseModel, ConfigDict
from uuid import UUID

class Hyperparam(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_id: UUID
    param_name: str
    param_value: str
