from __future__ import annotations

from fastapi.testclient import TestClient

from womens_health_ai.api.main import app
from womens_health_ai.data.dataset import load_dataset
from womens_health_ai.features.engineering import RAW_FEATURES


def test_health_endpoint(trained_metadata):
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_ready"] is True
    assert payload["champion_model"] == trained_metadata["champion_model"]["name"]


def test_model_info_endpoint(trained_metadata):
    with TestClient(app) as client:
        response = client.get("/model-info")

    assert response.status_code == 200
    payload = response.json()
    assert payload["project_name"] == trained_metadata["project_name"]
    assert "metric_rationale" in payload
    assert "sample_payloads" in payload


def test_predict_endpoint(trained_metadata):
    dataset = load_dataset()
    sample_payload = {
        key: float(value)
        for key, value in dataset.loc[0, RAW_FEATURES].to_dict().items()
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=sample_payload)

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_class"] in {"benign", "malignant"}
    assert 0.0 <= payload["malignant_probability"] <= 1.0
    assert payload["model_name"] == trained_metadata["champion_model"]["name"]
    assert len(payload["top_contributors"]) > 0

