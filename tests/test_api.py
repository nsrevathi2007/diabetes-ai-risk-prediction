"""API integration tests for the diabetes risk prediction backend."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def sample_patient_payload() -> dict[str, object]:
    return {
        "HighBP": 0,
        "HighChol": 0,
        "CholCheck": 1,
        "BMI": 27.5,
        "Smoker": 0,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": 1,
        "Fruits": 1,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 3,
        "MentHlth": 0,
        "PhysHlth": 0,
        "DiffWalk": 0,
        "Sex": 0,
        "Age": 7,
        "Education": 4,
        "Income": 5,
    }


def test_root_endpoint(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Diabetes Risk Prediction API is running."}


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["llm_available"] is False


def test_model_info_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/model-info")
    payload = response.json()
    assert response.status_code == 200
    assert payload["model_name"] == "Best Available Model"
    assert payload["feature_count"] == 22


def test_prediction_endpoint(client: TestClient) -> None:
    response = client.post("/api/v1/predict", json=sample_patient_payload())
    payload = response.json()
    assert response.status_code == 200
    assert payload["prediction"] in {"Low Diabetes Risk", "High Diabetes Risk"}
    assert 0.0 <= payload["probability"] <= 1.0
    assert payload["risk_level"] in {"Low Risk", "Moderate Risk", "High Risk", "Very High Risk"}


def test_recommendation_endpoint(client: TestClient) -> None:
    response = client.post("/api/v1/recommend", json=sample_patient_payload())
    payload = response.json()
    assert response.status_code == 200
    assert payload["patient_id"] == "patient_01"
    assert payload["prediction_probability"] == response.json()["prediction_probability"]


def test_explanation_endpoint(client: TestClient) -> None:
    response = client.post("/api/v1/explain", json=sample_patient_payload())
    payload = response.json()
    assert response.status_code == 200
    assert payload["prediction"] in {"Low Diabetes Risk", "High Diabetes Risk"}
    assert isinstance(payload["top_risk_factors"], list)
    assert isinstance(payload["top_protective_factors"], list)


def test_full_analysis_endpoint(client: TestClient) -> None:
    response = client.post("/api/v1/full-analysis", json=sample_patient_payload())
    payload = response.json()
    assert response.status_code == 200
    assert payload["prediction"]["prediction"] in {"Low Diabetes Risk", "High Diabetes Risk"}
    assert payload["recommendation"]["patient_id"] == "patient_01"
    assert payload["explanation"]["risk_level"] in {"Low Risk", "Moderate Risk", "High Risk", "Very High Risk"}
    assert "llm_explanation" in payload
    assert payload["llm_explanation"]["patient_id"] == "patient_01"


def test_swagger_and_redoc_endpoints(client: TestClient) -> None:
    docs_response = client.get("/docs")
    redoc_response = client.get("/redoc")
    assert docs_response.status_code == 200
    assert redoc_response.status_code == 200
