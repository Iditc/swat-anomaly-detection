"""SQLAlchemy models for storing predictions and incident details."""

from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime, ForeignKey, Text,
)
from sqlalchemy.orm import relationship

from src.api.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ae_score = Column(Float, nullable=False)
    svm_score = Column(Float, nullable=False)
    is_anomaly = Column(Boolean, nullable=False)
    detected_by = Column(String(20), nullable=True)
    attack_type = Column(String(50), nullable=True)
    description = Column(Text, nullable=True)
    evidence = Column(Text, nullable=True)

    sensors = relationship(
        "IncidentSensor", back_populates="prediction", cascade="all, delete-orphan",
    )


class IncidentSensor(Base):
    __tablename__ = "incident_sensors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    sensor = Column(String(20), nullable=False)
    stage = Column(Integer, nullable=False)
    contribution_pct = Column(Float, nullable=False)

    prediction = relationship("Prediction", back_populates="sensors")
