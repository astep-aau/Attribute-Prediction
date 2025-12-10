from sqlalchemy import Column, String, Integer, Float, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
from app.database import Base

class ModelMetricsTable(Base):
    __tablename__ = "model_metrics"
    __table_args__ = {"schema": "models"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_type = Column(UUID(as_uuid=True), ForeignKey("model_type.id"), nullable=False)
    train_time_min = Column(Integer, nullable=False)
    bias = Column(Float)
    gap = Column(Float)
    path_to_save = Column(String(500), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
