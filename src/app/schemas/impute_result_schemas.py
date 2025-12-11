from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from uuid import UUID

class Test(BaseModel):
    pass

class RoadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    road_id: int
