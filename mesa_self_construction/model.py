"""
SelfConstructionModel: A Mesa 3.x agent-based model for self-constructing automata.
Implements Conway's Game of Life slow-salvo construction (EZLiang/cgol-self-construction)
and John von Neumann's 29-state universal constructor.
"""

from typing import Dict, Any, Optional, List, Tuple
import copy
import numpy as np

from mesa import Model
from mesa.discrete_space import OrthogonalMooreGrid, PropertyLayer
from mesa_self_construction.agents import ConstructionCell
from mesa_self_construction.salvo import SalvoSchedule, ScheduledGlider
from mesa_self_construction.presets import ConstructionPreset, get_preset
from mesa_self_construction.validator import validate_target_pattern, ValidationResult


def conway_step_zero_boundary(grid: np.ndarray) -> np.ndarray:
    """Vectorized Conway Life step with zero (dead) borders.

    Edges do NOT wrap: cells outside the grid count as dead. This models
    an infinite vacuum for aimed glider collisions instead of a torus,
    where wrap-around gliders cause artificial self-interference.
    """
    padded = np.pad(grid.astype(np.int32), 1, mode="constant", constant_values=0)
    nbrs = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
        + padded[1:-1, :-2] + padded[1:-1, 2:]
        + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    )
    return ((nbrs == 3) | ((grid == 1) & (nbrs == 2))).astype(np.uint8)


def detect_flying_gliders(grid: np.ndarray) -> Tuple[int, np.ndarray]:
    """Detect isolated in-flight gliders by exact 3x3 template matching.

    Vectorized: all 3x3 windows are compared against all 16 glider
    templates (4 directions x 4 phases) with NumPy broadcasting, and the
    one-cell surrounding ring must be empty (isolated glider in vacuum).
    Returns (count, glider_cell_mask). Only meaningful for binary
    Conway grids; VN grids return (0, empty mask).
    """
    from mesa_self_construction.salvo import GLIDER_PHASES

    h, w = grid.shape
    mask = np.zeros_like(grid, dtype=bool)
    if h < 5 or w < 5:
        return 0, mask
    g = (grid == 1).astype(np.uint8)
    try:
        from numpy.lib.stride_tricks import sliding_window_view
    except ImportError:  # pragma: no cover - numpy>=1.20 always has it
        sliding_window_view = None
    if sliding_window_view is None:
        return 0, mask
    templates = np.stack(
        [t for phases in GLIDER_PHASES.values() for t in phases]
    )  # (16,3,3)
    win3 = sliding_window_view(g, (3, 3))  # (h-2, w-2, 3, 3)
    # Cheap prefilter: only 5-cell windows can be gliders (usually a few).
    cand = np.nonzero(win3.sum(axis=(2, 3)) == 5)
    win5 = sliding_window_view(g, (5, 5))  # (h-4, w-4, 5, 5)
    ring_sum = win5.sum(axis=(2, 3))
    count = 0
    for i, j in zip(cand[0].tolist(), cand[1].tolist()):
        # Need a one-cell ring margin: skip border windows.
        if i < 1 or j < 1 or i > h - 5 or j > w - 5:
            continue
        if ring_sum[i - 1, j - 1] != 5:
            continue
        window = win3[i, j]
        if (templates == window).all(axis=(1, 2)).any():
            mask[i:i + 3, j:j + 3] |= window.astype(bool)
            count += 1
    return count, mask


def classify_components(
    grid_data: np.ndarray,
    target_box: Tuple[int, int, int, int],
    glider_mask: np.ndarray,
    elbow_center: Optional[Tuple[int, int]] = None,
) -> Dict[str, int]:
    """Count live cells by component class for telemetry."""
    rmin, rmax, cmin, cmax = target_box
    h, w = grid_data.shape
    rmin, rmax = max(0, rmin), min(h - 1, rmax)
    cmin, cmax = max(0, cmin), min(w - 1, cmax)
    counts = {"target": 0, "glider": 0, "elbow": 0, "debris": 0}
    live = np.argwhere(grid_data == 1)
    for r, c in live:
        if rmin <= r <= rmax and cmin <= c <= cmax:
            counts["target"] += 1
        elif glider_mask[r, c]:
            counts["glider"] += 1
        elif (
            elbow_center is not None
            and abs(int(r) - elbow_center[0]) <= 3
            and abs(int(c) - elbow_center[1]) <= 3
        ):
            counts["elbow"] += 1
        else:
            counts["debris"] += 1
    return counts


class SelfConstructionModel(Model):
    """
    Simulation model for observable, reproducible self-construction in cellular automata.
    Supports high-performance PropertyLayer execution, discrete FixedAgent exploration,
    timed slow-salvo injection, and automated proof-of-construction verification.
    """

    def __init__(
        self,
        preset_id: str = "block_synthesis",
        mode: str = "property_layer",
        rng=None,
    ):
        super().__init__(rng=rng)
        self.preset: ConstructionPreset = get_preset(preset_id)
        self.mode = mode  # 'property_layer' or 'agent'
        
        self.height, self.width = self.preset.grid_size
        self.is_von_neumann = self.preset.is_von_neumann

        # Initialize Mesa discrete space grid.
        # NOTE: construction presets assume an infinite vacuum, so the
        # grid is non-toroidal. Mesa's bundled Life demo uses torus=True
        # for random soups; wrapping gliders here would cause artificial
        # self-collisions (see audit). Grid dims are (width, height).
        self.grid = OrthogonalMooreGrid(
            (self.width, self.height),
            capacity=1,
            random=self.random,
            torus=False,
        )

        # High-performance grid data representation
        dtype = np.int32 if self.is_von_neumann else np.uint8
        self.grid_data = np.copy(self.preset.initial_matrix).astype(dtype)

        # PropertyLayer for Mesa-native array storage. NOTE: data layout
        # is row-major (height, width), matching grid_data, so pass the
        # array shape as dimensions (Mesa allocates exactly that shape).
        self.state_layer = PropertyLayer(
            "state",
            dimensions=self.grid_data.shape,
            default_value=0,
            dtype=dtype,
        )
        self.state_layer.data = self.grid_data

        # Salvo schedule: per-model deep copy so preset singletons in
        # ALL_PRESETS are never mutated by one model instance.
        self.schedule: SalvoSchedule = copy.deepcopy(self.preset.schedule)

        # Telemetry & tracking
        self.current_step: int = 0
        self.gliders_injected_count: int = 0
        self.gliders_consumed_count: int = 0
        self.gliders_in_flight: int = 0
        self.component_counts: Dict[str, int] = {
            "target": 0,
            "glider": 0,
            "elbow": 0,
            "debris": 0,
        }
        # Von Neumann construction-event counters (29-state system).
        self.vn_constructed_cells: int = 0
        self.vn_killed_cells: int = 0
        self.vn_sensitized_cells: int = 0
        self.stage: str = "Initial Configuration"

        # Inject step 0 gliders if any
        self._inject_due_gliders()

        self.population_history: List[int] = [int(np.count_nonzero(self.grid_data))]
        self.history_buffer: List[np.ndarray] = [np.copy(self.grid_data)]
        self.max_history: int = 200
        # Snapshots of schedule / counter state for exact step_backward.
        self._injected_count_history: List[int] = [self.gliders_injected_count]
        self._consumed_count_history: List[int] = [self.gliders_consumed_count]
        self._schedule_flags_history: List[List[bool]] = [
            [g.injected for g in self.schedule.gliders]
        ]

        # Create discrete FixedAgents if running in agent mode
        self.cell_agent_map: Dict[Tuple[int, int], ConstructionCell] = {}
        if self.mode == "agent" and not self.is_von_neumann:
            self._init_agents()

        self._update_telemetry()

        # Initial validation state
        self.latest_validation: ValidationResult = self._run_validation()
        self.running = True

    def _elbow_center(self) -> Optional[Tuple[int, int]]:
        meta = self.preset.metadata or {}
        for key in ("elbow_initial_pos", "block_origin"):
            if key in meta:
                pos = meta[key]
                return (int(pos[0]), int(pos[1]))
        return None

    def _update_telemetry(self) -> None:
        """Refresh glider-flight / consumption / component telemetry."""
        if self.is_von_neumann:
            self.gliders_in_flight = 0
            self.gliders_consumed_count = 0
            self.component_counts = {
                "target": 0,
                "glider": 0,
                "elbow": 0,
                "debris": 0,
            }
            return
        flying, glider_mask = detect_flying_gliders(
            self.grid_data.astype(np.uint8)
        )
        self.gliders_in_flight = flying
        self.gliders_consumed_count = max(
            0, self.gliders_injected_count - flying
        )
        # Mark scheduled gliders consumed once they are no longer flying
        # and their release step is well in the past (settled collisions).
        # Simple honest rule: a glider counts as consumed when it was
        # injected but no isolated glider remains for it.
        remaining = flying
        for g in self.schedule.gliders:
            if g.injected:
                if remaining > 0:
                    g.active = True
                    g.consumed = False
                    remaining -= 1
                else:
                    g.active = False
                    g.consumed = True
            else:
                g.active = False
                g.consumed = False
        self.component_counts = classify_components(
            self.grid_data.astype(np.uint8),
            self.preset.target_box,
            glider_mask,
            self._elbow_center(),
        )
        if self.mode == "agent" and self.cell_agent_map:
            self._sync_agent_components(glider_mask)

    def _sync_agent_components(self, glider_mask: np.ndarray) -> None:
        rmin, rmax, cmin, cmax = self.preset.target_box
        elbow = self._elbow_center()
        for (r, c), agent in self.cell_agent_map.items():
            agent.state = int(self.grid_data[r, c])
            if agent.state == 0:
                agent.component_type = "empty"
            elif rmin <= r <= rmax and cmin <= c <= cmax:
                agent.component_type = "target"
            elif glider_mask[r, c]:
                agent.component_type = "glider"
            elif (
                elbow is not None
                and abs(r - elbow[0]) <= 3
                and abs(c - elbow[1]) <= 3
            ):
                agent.component_type = "elbow"
            else:
                agent.component_type = "debris"

    def _init_agents(self) -> None:
        """Instantiate discrete FixedAgent cells across the grid."""
        for cell in self.grid.all_cells:
            r, c = cell.coordinate[1], cell.coordinate[0]
            val = int(self.grid_data[r, c])
            agent = ConstructionCell(self, cell, init_state=val)
            self.cell_agent_map[(r, c)] = agent

    def _inject_due_gliders(self) -> None:
        """Inject any gliders whose scheduled release time has arrived.

        Clips against grid borders (no torus wrap, no negative-index
        aliasing): partially off-grid gliders stamp only the on-grid part.
        Overlapping stamps use OR so elbow blocks are preserved.
        """
        due_gliders = self.schedule.gliders_to_inject(self.current_step)
        for g in due_gliders:
            matrix = g.get_matrix()
            gh, gw = matrix.shape
            r0 = max(0, g.row)
            c0 = max(0, g.col)
            r_end = min(self.height, g.row + gh)
            c_end = min(self.width, g.col + gw)
            if r_end <= r0 or c_end <= c0:
                continue  # fully off-grid: skip cleanly
            mr0 = r0 - g.row
            mc0 = c0 - g.col
            mr1 = mr0 + (r_end - r0)
            mc1 = mc0 + (c_end - c0)
            self.grid_data[r0:r_end, c0:c_end] = np.bitwise_or(
                self.grid_data[r0:r_end, c0:c_end],
                matrix[mr0:mr1, mc0:mc1],
            )
            self.gliders_injected_count += 1
            self.stage = f"Injecting {g.label}"

    def step(self) -> None:
        """Advance the simulation by one generation."""
        # 1. Check for incoming gliders due at this step
        self._inject_due_gliders()

        # 2. Compute cellular automaton evolution
        if self.is_von_neumann:
            self._step_von_neumann()
        else:
            if self.mode == "agent":
                self._step_conway_agents()
            else:
                self._step_conway_property_layer()

        self.current_step += 1

        # 3. Record population & buffer state
        pop = int(np.count_nonzero(self.grid_data))
        self.population_history.append(pop)
        if len(self.population_history) > 300:
            self.population_history.pop(0)

        # Buffer history for stepping backwards
        self.history_buffer.append(np.copy(self.grid_data))
        self._injected_count_history.append(self.gliders_injected_count)
        self._consumed_count_history.append(self.gliders_consumed_count)
        self._schedule_flags_history.append(
            [g.injected for g in self.schedule.gliders]
        )
        if len(self.history_buffer) > self.max_history:
            self.history_buffer.pop(0)
            self._injected_count_history.pop(0)
            self._consumed_count_history.pop(0)
            self._schedule_flags_history.pop(0)

        # 4. Update telemetry, stage & validation
        self._update_telemetry()
        self.latest_validation = self._run_validation()
        self._update_stage()

    def _step_conway_property_layer(self) -> None:
        """High-performance Conway Life step with zero borders (no wrap)."""
        self.grid_data = conway_step_zero_boundary(self.grid_data)
        self.state_layer.data = self.grid_data

    def _step_conway_agents(self) -> None:
        """Agent-based step conforming to Mesa discrete space agents."""
        for (r, c), agent in self.cell_agent_map.items():
            agent.state = int(self.grid_data[r, c])

        self.agents.do("determine_state")
        self.agents.do("assume_state")

        for (r, c), agent in self.cell_agent_map.items():
            self.grid_data[r, c] = agent.state

        self.state_layer.data = self.grid_data

    def _step_von_neumann(self) -> None:
        """One synchronous step of the documented 29-state VN system.

        Delegates to von_neumann.step_von_neumann_29 (directional
        transmission, confluent latch/fire, sensitized construct/kill).
        Counts genuine construction events (sensitized -> wired cell).
        """
        from mesa_self_construction.von_neumann import step_von_neumann_29

        next_grid, events = step_von_neumann_29(self.grid_data)
        self.grid_data = next_grid.astype(np.int32)
        self.state_layer.data = self.grid_data
        self.vn_constructed_cells += int(events["constructed"])
        self.vn_killed_cells += int(events["killed"])
        self.vn_sensitized_cells += int(events["sensitized"])

    def _run_validation(self) -> ValidationResult:
        """Run validation against current target expectations."""
        if self.is_von_neumann:
            # Extending constructors legitimately grow: stability not
            # required by default, but genuine sensitized->wired events
            # are. Per-preset metadata may tighten either gate.
            meta = self.preset.metadata or {}
            return validate_target_pattern(
                current_grid=self.grid_data,
                target_box=self.preset.target_box,
                expected_pattern=self.preset.expected_pattern,
                expected_population=self.preset.expected_population,
                step=self.current_step,
                recent_populations=self.population_history,
                gliders_used=self.gliders_injected_count,
                is_von_neumann=True,
                require_settled=bool(meta.get("require_settled", False)),
                constructions=self.vn_constructed_cells,
                min_constructions=int(meta.get("min_constructions", 5)),
            )
        return validate_target_pattern(
            current_grid=self.grid_data,
            target_box=self.preset.target_box,
            expected_pattern=self.preset.expected_pattern,
            expected_population=self.preset.expected_population,
            step=self.current_step,
            recent_populations=self.population_history,
            gliders_used=self.gliders_injected_count,
            is_von_neumann=False,
        )

    def _update_stage(self) -> None:
        """Update readable human-friendly stage description."""
        if self.latest_validation.passed:
            self.stage = "Target Constructed & Verified"
        elif self.latest_validation.match_ratio > 0.6:
            self.stage = "Reaction Settling / Solidification"
        elif self.gliders_injected_count > 0:
            self.stage = "Glider Packet In-Flight / Interacting"
        else:
            self.stage = "Initialized"

    def step_backward(self) -> bool:
        """Step backward in time using history buffer (exact restore)."""
        if len(self.history_buffer) > 1:
            self.history_buffer.pop()  # Remove current
            prev_grid = self.history_buffer[-1]
            self.grid_data = np.copy(prev_grid)
            self.state_layer.data = self.grid_data
            self.current_step = max(0, self.current_step - 1)
            if self.population_history:
                self.population_history.pop()
            self._injected_count_history.pop()
            self._consumed_count_history.pop()
            flags = self._schedule_flags_history.pop()
            for g, f in zip(self.schedule.gliders, flags):
                g.injected = f
                g.active = f
            self.gliders_injected_count = self._injected_count_history[-1]
            self.gliders_consumed_count = self._consumed_count_history[-1]
            self._update_telemetry()
            # _update_telemetry recomputes consumed from flying count;
            # restore exact snapshot for counter continuity.
            self.gliders_consumed_count = self._consumed_count_history[-1]
            self.latest_validation = self._run_validation()
            self._update_stage()
            return True
        return False

    def reset(self) -> None:
        """Reset model to initial configuration."""
        self.current_step = 0
        self.grid_data = np.copy(self.preset.initial_matrix).astype(self.grid_data.dtype)
        self.state_layer.data = self.grid_data
        self.schedule.reset()
        self.gliders_injected_count = 0
        self.gliders_consumed_count = 0
        self.gliders_in_flight = 0
        self.vn_constructed_cells = 0
        self.vn_killed_cells = 0
        self.vn_sensitized_cells = 0
        self.stage = "Reset to Initial State"
        self._inject_due_gliders()
        self.population_history = [int(np.count_nonzero(self.grid_data))]
        self.history_buffer = [np.copy(self.grid_data)]
        self._injected_count_history = [self.gliders_injected_count]
        self._consumed_count_history = [0]
        self._schedule_flags_history = [
            [g.injected for g in self.schedule.gliders]
        ]
        self._update_telemetry()
        self._consumed_count_history = [self.gliders_consumed_count]
        self.latest_validation = self._run_validation()

    def load_preset(self, preset_id: str) -> None:
        """Switch to a different self-construction preset."""
        self.preset = get_preset(preset_id)
        # preset.grid_size is (height, width); Mesa grid takes (width, height).
        self.height, self.width = self.preset.grid_size
        self.is_von_neumann = self.preset.is_von_neumann

        dtype = np.int32 if self.is_von_neumann else np.uint8
        self.grid_data = np.copy(self.preset.initial_matrix).astype(dtype)
        self.grid = OrthogonalMooreGrid(
            (self.width, self.height),
            capacity=1,
            random=self.random,
            torus=False,
        )
        self.state_layer = PropertyLayer(
            "state",
            dimensions=self.grid_data.shape,
            default_value=0,
            dtype=dtype,
        )
        self.state_layer.data = self.grid_data
        self.schedule = copy.deepcopy(self.preset.schedule)
        if self.mode == "agent" and not self.is_von_neumann:
            self.cell_agent_map = {}
            self._init_agents()
        self.reset()

    def get_state(self) -> Dict[str, Any]:
        """Return comprehensive state for API and web visualization."""
        # Active gliders metadata
        gliders_data = [g.to_dict() for g in self.schedule.gliders]

        # Extract active cells sparse coordinates for fast JSON transfer
        active_coords = []
        if self.is_von_neumann:
            indices = np.argwhere(self.grid_data > 0)
            for r, c in indices:
                active_coords.append([int(r), int(c), int(self.grid_data[r, c])])
        else:
            indices = np.argwhere(self.grid_data == 1)
            for r, c in indices:
                active_coords.append([int(r), int(c), 1])

        # Construction-head marker for the visualizer: easternmost
        # sensitized-or-newly-built cell on the arm row (VN), else None.
        vn_head = None
        preset_metadata = dict(self.preset.metadata or {})
        if self.is_von_neumann:
            sens = np.argwhere(
                (self.grid_data >= 21) & (self.grid_data <= 28)
            )
            if len(sens):
                # Easternmost sensitized cell = live construction front.
                tip = sens[np.argmax(sens[:, 1])]
                vn_head = [int(tip[0]), int(tip[1])]

        return {
            "preset_id": self.preset.id,
            "preset_name": self.preset.name,
            "category": self.preset.category,
            "description": self.preset.description,
            "is_von_neumann": self.is_von_neumann,
            "step": self.current_step,
            "width": self.width,
            "height": self.height,
            "stage": self.stage,
            "population": int(np.count_nonzero(self.grid_data)),
            "population_history": self.population_history[-50:],
            "target_box": list(self.preset.target_box),
            "gliders": gliders_data,
            "gliders_injected": self.gliders_injected_count,
            "gliders_consumed": self.gliders_consumed_count,
            "gliders_in_flight": self.gliders_in_flight,
            "component_counts": dict(self.component_counts),
            "vn_constructed": self.vn_constructed_cells,
            "vn_killed": self.vn_killed_cells,
            "vn_sensitized": self.vn_sensitized_cells,
            "vn_head": vn_head,
            "metadata": preset_metadata,
            "active_cells": active_coords,
            "validation": self.latest_validation.to_dict(),
        }
