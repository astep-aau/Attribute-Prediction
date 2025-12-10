from pydantic import BaseModel

class PlaceHolder(BaseModel):
    id: int
    name: str
    email: str
