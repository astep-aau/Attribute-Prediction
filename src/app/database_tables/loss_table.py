from sqlalchemy import Column, String, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
import uuid
from src.app.database import Base

class LossTable(Base):
    __tablename__ = "loss"
    __table_args__ = {"schema": "models"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey("model_metrics.id"), nullable=False)
    type = Column(String(100), nullable=False)
    loss_value = Column(Float, nullable=False)
    loss_unit = Column(String(50), nullable=False)
