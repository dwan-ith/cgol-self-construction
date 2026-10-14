"""
Tests for FastAPI endpoints in app.py.
"""

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_api_presets():
    res = client.get("/api/presets")
    assert res.status_code == 200
    data = res.json()
    assert "presets" in data
    assert len(data["presets"]) >= 6


def test_api_state():
    res = client.get("/api/state")
    assert res.status_code == 200
    data = res.json()
    assert "step" in data
    assert "active_cells" in data
    assert "validation" in data


def test_api_step():
    res = client.post("/api/step", json={"steps": 3})
    assert res.status_code == 200
    data = res.json()
    assert data["step"] >= 3


def test_api_step_back():
    res = client.post("/api/step-back")
    assert res.status_code == 200


def test_api_reset():
    res = client.post("/api/reset")
    assert res.status_code == 200
    data = res.json()
    assert data["step"] == 0


def test_api_load_preset():
    res = client.post("/api/load-preset/elbow_shift")
    assert res.status_code == 200
    data = res.json()
    assert data["preset_id"] == "elbow_shift"


def test_api_run_to_validation():
    res = client.post("/api/run-to-validation")
    assert res.status_code == 200
    data = res.json()
    assert data["validation"]["passed"] is True
