import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "models" in data


def test_models_endpoint():
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "qwen3" in data["models"]
    assert "phi4" not in data["models"]


def test_ocr_check_endpoint_success():
    payload = {
        "ocr_text": "INVOICE #1024\nDate: 2026-08-01\nTotal Amount: $450.00",
        "mandatory_fields": ["Invoice Number", "Date", "Total Amount"],
        "target_language": "en",
        "model_name": "qwen3"
    }
    response = client.post("/ocr/check", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "id" in data
    assert "score" in data
    assert data["score"]["overall_score"] >= 80.0
    assert "tool_results" in data
    assert "agent_reasoning" in data


def test_ocr_check_endpoint_validation_error():
    payload = {
        "ocr_text": "",
        "mandatory_fields": ["Invoice Number"]
    }
    response = client.post("/ocr/check", json=payload)
    assert response.status_code == 400


def test_invalid_model_name_error():
    response = client.post("/models/load?model_name=Qw")
    assert response.status_code == 400
    assert "Invalid model_name" in response.json()["detail"]


def test_load_specific_model():
    response = client.post("/models/load?model_name=qwen3")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["status"]["models"]["qwen3"]["loaded"] is True


def test_sample_invoice_endpoint():
    response = client.get("/sample/invoice")
    assert response.status_code == 200
    data = response.json()
    assert "ocr_text" in data
    assert "mandatory_fields" in data
    assert "target_language" in data


def test_favicon_endpoint():
    response = client.get("/favicon.ico")
    assert response.status_code == 204