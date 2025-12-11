from sqlalchemy import Column, String, Float, ForeignKey, TIMESTAMP, Boolean
from sqlalchemy.dialects.postgresql import UUID
import uuid
from src.app.database import Base

class ImputeResult(Base):
    __tablename__ = "impute_result"

    road_id = Column(String, primary_key=True)
    model_id = Column(UUID(as_uuid=True), ForeignKey("model_metrics.id"), primary_key=True)
    tms = Column(TIMESTAMP, primary_key=True)

    value = Column(Float, nullable=False)
    imputed = Column(Boolean, nullable=False)
