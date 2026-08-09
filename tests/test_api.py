"""API tests using FastAPI TestClient."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


NORMAL_READING = {
    "FIT101": 2.5, "LIT101": 500.0, "MV101": 1, "P101": 1, "P102": 0,
    "AIT201": 200.0, "AIT202": 7.5, "AIT203": 300.0, "FIT201": 2.0, "MV201": 1,
    "P201": 1, "P202": 0, "P203": 0, "P204": 0, "P205": 0, "P206": 0,
    "DPIT301": 50.0, "FIT301": 2.0, "LIT301": 800.0,
    "MV301": 1, "MV302": 0, "MV303": 0, "MV304": 0, "P301": 1, "P302": 0,
    "AIT401": 100.0, "AIT402": 200.0, "FIT401": 2.0, "LIT401": 600.0,
    "P401": 1, "P402": 0, "P403": 0, "P404": 0, "UV401": 1,
    "AIT501": 50.0, "AIT502": 300.0, "AIT503": 400.0, "AIT504": 0.5,
    "FIT501": 2.0, "FIT502": 1.5, "FIT503": 1.0, "FIT504": 1.0,
    "P501": 1, "P502": 0, "PIT501": 50.0, "PIT502": 40.0, "PIT503": 30.0,
    "FIT601": 2.0, "P601": 1, "P602": 0, "P603": 0,
}


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_model_loaded(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True
        assert data["n_features"] == 349
        assert data["status"] == "healthy"


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        response = client.post("/predict", json=NORMAL_READING)
        assert response.status_code == 200

    def test_predict_response_fields(self, client):
        data = client.post("/predict", json=NORMAL_READING).json()
        assert "anomaly_score" in data
        assert "is_anomaly" in data
        assert "detection" in data
        assert "ae_score" in data["detection"]
        assert "svm_score" in data["detection"]
        assert "ae_anomaly" in data["detection"]
        assert "svm_anomaly" in data["detection"]

    def test_predict_missing_sensor_returns_422(self, client):
        incomplete = {"FIT101": 2.5, "LIT101": 500.0}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422


class TestBatchEndpoint:
    def test_batch_returns_200(self, client):
        batch = {"readings": [NORMAL_READING, NORMAL_READING]}
        response = client.post("/predict/batch", json=batch)
        assert response.status_code == 200

    def test_batch_response_fields(self, client):
        batch = {"readings": [NORMAL_READING]}
        data = client.post("/predict/batch", json=batch).json()
        assert data["total"] == 1
        assert "anomalies" in data
        assert len(data["results"]) == 1

    def test_batch_empty_returns_200(self, client):
        batch = {"readings": []}
        data = client.post("/predict/batch", json=batch).json()
        assert data["total"] == 0
        assert data["anomalies"] == 0
