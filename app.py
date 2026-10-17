"""
FastAPI application serving the interactive Mesa Self-Construction dashboard.
Provides REST endpoints for stepping, resetting, preset switching, and telemetry.
"""

from pathlib import Path
from threading import Lock
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from mesa_self_construction.model import SelfConstructionModel
from mesa_self_construction.presets import ALL_PRESETS

# Project paths
BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

app = FastAPI(
    title="Mesa Self-Construction Visualizer",
    description="Interactive construction automata: Conway slow-salvo synthesis and the canonical von Neumann 29-state automaton.",
    version="1.0.0",
)

# Global simulation instance (single-user demo; guard with a lock).
current_model = SelfConstructionModel(preset_id="block_synthesis")
model_lock = Lock()


class StepRequest(BaseModel):
    steps: Optional[int] = 1


@app.get("/api/presets")
def get_presets_list() -> Dict[str, Any]:
    """Return all available presets and metadata."""
    presets = []
    for pid, p in ALL_PRESETS.items():
        presets.append({
            "id": p.id,
            "name": p.name,
            "category": p.category,
            "description": p.description,
            "grid_size": list(p.grid_size),
            "expected_population": p.expected_population,
            "validation_step": p.validation_step,
            "is_von_neumann": p.is_von_neumann,
        })
    return {"presets": presets, "active": current_model.preset.id}


@app.get("/api/state")
def get_simulation_state() -> Dict[str, Any]:
    """Return comprehensive state of current simulation."""
    with model_lock:
        return current_model.get_state()


@app.post("/api/step")
def step_simulation(req: StepRequest = StepRequest()) -> Dict[str, Any]:
    """Advance simulation by requested number of steps."""
    num_steps = max(1, min(req.steps or 1, 100))
    with model_lock:
        for _ in range(num_steps):
            current_model.step()
        return current_model.get_state()


@app.post("/api/step-back")
def step_back_simulation() -> Dict[str, Any]:
    """Step backward 1 generation using history buffer."""
    with model_lock:
        success = current_model.step_backward()
        state = current_model.get_state()
    state["step_backward_success"] = success
    return state


@app.post("/api/reset")
def reset_simulation() -> Dict[str, Any]:
    """Reset current preset back to generation 0."""
    with model_lock:
        current_model.reset()
        return current_model.get_state()


@app.post("/api/load-preset/{preset_id}")
def load_preset(preset_id: str) -> Dict[str, Any]:
    """Switch active experiment preset."""
    if preset_id not in ALL_PRESETS:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found.")
    with model_lock:
        current_model.load_preset(preset_id)
        return current_model.get_state()


@app.post("/api/run-to-validation")
def run_to_validation() -> Dict[str, Any]:
    """Fast-forward simulation until target validation step or proof achieved."""
    with model_lock:
        target_step = current_model.preset.validation_step
        max_steps = max(300, target_step)
        steps_taken = 0
        while current_model.current_step < target_step and steps_taken < max_steps:
            current_model.step()
            steps_taken += 1
            if current_model.latest_validation.passed:
                break
        return current_model.get_state()


# Mount static web directory
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    print("Starting Mesa Self-Construction dashboard at http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
