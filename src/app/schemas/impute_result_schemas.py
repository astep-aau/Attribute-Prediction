from pydantic import BaseModel, ConfigDict
from uuid import UUID

class RoadIdResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    road_id: str

class TimeIntervalResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    start_time: int
    end_time: int

class ImputeResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    tms: int
    value: float
    imputed: bool

class ImputeResultCreate(BaseModel):
    model_id: UUID
    road_id: str
    tms: int
    value: float
    imputed: bool
