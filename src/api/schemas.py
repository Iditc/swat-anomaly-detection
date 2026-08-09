"""Pydantic models for API request/response validation."""

from pydantic import BaseModel


class SensorReading(BaseModel):
    """A single row of sensor readings from the SWaT plant.

    All 51 sensors must be provided. The API computes all derived
    features (rolling stats, contradiction, frozen detection, etc.)
    internally — callers only send raw readings.
    """
    FIT101: float
    LIT101: float
    MV101: int
    P101: int
    P102: int
    AIT201: float
    AIT202: float
    AIT203: float
    FIT201: float
    MV201: int
    P201: int
    P202: int
    P203: int
    P204: int
    P205: int
    P206: int
    DPIT301: float
    FIT301: float
    LIT301: float
    MV301: int
    MV302: int
    MV303: int
    MV304: int
    P301: int
    P302: int
    AIT401: float
    AIT402: float
    FIT401: float
    LIT401: float
    P401: int
    P402: int
    P403: int
    P404: int
    UV401: int
    AIT501: float
    AIT502: float
    AIT503: float
    AIT504: float
    FIT501: float
    FIT502: float
    FIT503: float
    FIT504: float
    P501: int
    P502: int
    PIT501: float
    PIT502: float
    PIT503: float
    FIT601: float
    P601: int
    P602: int
    P603: int


class InvolvedSensor(BaseModel):
    sensor: str
    stage: int
    contribution_pct: float


class DetectionDetail(BaseModel):
    ae_score: float
    ae_anomaly: bool
    svm_score: float
    svm_anomaly: bool


class PredictionResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    detected_by: str | None
    attack_type: str | None
    description: str | None
    involved_sensors: list[InvolvedSensor]
    evidence: str | None
    detection: DetectionDetail


class BatchRequest(BaseModel):
    readings: list[SensorReading]


class BatchResponse(BaseModel):
    total: int
    anomalies: int
    results: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_features: int
    threshold: float
    buffer_size: int
