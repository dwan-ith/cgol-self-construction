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
    assert len(data["presets"]) >= 8


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


def test_api_presets_include_machines():
    """The vendored von Neumann machines are exposed through the API."""
    ids = {p["id"] for p in client.get("/api/presets").json()["presets"]}
    assert {"vn_counter", "vn_cell_coders", "vn_self_replicator"} <= ids


def test_api_unknown_preset_404():
    assert client.post("/api/load-preset/does_not_exist").status_code == 404


def test_api_step_is_clamped():
    """The step endpoint caps a request at 100 generations."""
    client.post("/api/load-preset/block_synthesis")
    data = client.post("/api/step", json={"steps": 100000}).json()
    assert data["step"] == 100


def test_api_all_presets_run_to_validation():
    """Every preset (Conway, VN primitives, VN machines) proves via the API."""
    from mesa_self_construction import ALL_PRESETS
    for pid in ALL_PRESETS:
        client.post(f"/api/load-preset/{pid}")
        data = client.post("/api/run-to-validation").json()
        assert data["validation"]["passed"], f"{pid}: {data['validation']}"
