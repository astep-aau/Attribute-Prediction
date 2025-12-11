from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import List

class RoadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    road_id: List[int]

class TimeIntervalResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    start_time: datetime
    end_time: datetime

class ImputeResultResonse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    tms: datetime
    value: float
    imputed: bool
