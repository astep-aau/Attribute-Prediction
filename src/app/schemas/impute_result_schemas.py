from pydantic import BaseModel, ConfigDict
from typing import List

class RoadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    road_id: List[int]

class TimeIntervalResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    start_time: int
    end_time: int

class ImputeResultResonse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    tms: int
    value: float
    imputed: bool
