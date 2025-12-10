from sqlalchemy import Column, String, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
import uuid
from app.database import Base

class HyperparamTable(Base):
    __tablename__ = "hyperparam"
    __table_args__ = {"schema": "models"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey("model_metrics.id"), nullable=False)
    param_name = Column(String(255), nullable=False)
    param_value = Column(String(500), nullable=False)
