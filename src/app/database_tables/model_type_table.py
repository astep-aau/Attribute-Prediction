from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import UUID
import uuid
from src.app.database import Base

class ModelTypeTable(Base):
    __tablename__ = "model_type"
    __table_args__ = {"schema": "models"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
