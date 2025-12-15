from pydantic import BaseModel, ConfigDict
from uuid import UUID
from typing import Optional

class ModelTypeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str

class ModelTypeCreate(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
