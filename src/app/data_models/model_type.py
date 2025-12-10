from pydantic import BaseModel

class ModelType(BaseModel):
    uuid: str
    name: str
