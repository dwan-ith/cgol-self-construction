"""
Predefined, verified construction experiments.

Conway's Game of Life: slow-salvo glider collisions on zero-border grids
that synthesize a target structure with no lingering debris.

Von Neumann: demonstrations of the canonical 29-state automaton's
construction primitives (ordinary transmission, special/ordinary
interaction, confluent merging) evaluated by the vendored reference
transition table. These are primitives, not a universal constructor
program; see ``von_neumann.py`` for the scope statement.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from mesa_self_construction.salvo import SalvoSchedule, GLIDER_PHASES


@dataclass
class ConstructionPreset:
    """A self-contained construction experiment configuration."""
    id: str
    name: str
    category: str
    description: str
    grid_size: Tuple[int, int]
    target_box: Tuple[int, int, int, int]  # (rmin, rmax, cmin, cmax)
    expected_pattern: np.ndarray
    expected_population: int
    validation_step: int
    initial_matrix: np.ndarray
    schedule: SalvoSchedule
    is_von_neumann: bool = False
    metadata: Dict[str, Any] = None


# ---------------------------------------------------------------------------
# Conway's Game of Life: slow-salvo construction
# ---------------------------------------------------------------------------

def make_block_synthesis_preset() -> ConstructionPreset:
    """
    Preset 1: De Novo 2-Glider Block Synthesis
    Two gliders collide head-on at 180 degrees in empty space (zero
    borders, no torus wrap). Verified recipe: SE phase-1 from (10, 10)
    + NW phase-0 from (24, 29) -> pristine 2x2 Block at (19, 18),
    settled (pop 4) from step 37. Directions verified by period-4
    displacement test (see salvo.GLIDER_PHASES).
    """
    grid = np.zeros((46, 46), dtype=np.uint8)

    schedule = SalvoSchedule(name="2-Glider Head-on Salvo")
    schedule.add_glider(
        step=0, row=10, col=10, direction="SE", phase=1,
        label="Glider Alpha (SE)",
    )
    schedule.add_glider(
        step=0, row=24, col=29, direction="NW", phase=0,
        label="Glider Beta (NW)",
    )

    expected_block = np.array([[1, 1], [1, 1]], dtype=np.uint8)

    return ConstructionPreset(
        id="block_synthesis",
        name="De Novo 2-Glider Block Synthesis",
        category="Conway's Game of Life: Construction",
        description=(
            "Two synchronized gliders colliding head-on in vacuum at 180 deg. "
            "Their destructive and constructive interference settles into a permanent 2x2 Block "
            "with zero lingering debris, proving de novo matter synthesis from pure kinetic glider packets."
        ),
        grid_size=(46, 46),
        target_box=(18, 21, 17, 20),
        expected_pattern=expected_block,
        expected_population=4,
        validation_step=50,
        initial_matrix=grid,
        schedule=schedule,
        is_von_neumann=False,
        metadata={
            "initial_population": 10,
            "final_population": 4,
            "gliders_used": 2,
            "construction_type": "head_on_collision",
            "settle_time": 37,
            "block_origin": (19, 18),
            "boundary": "zero (non-toroidal)",
        },
    )


def make_elbow_shift_preset() -> ConstructionPreset:
    """
    Preset 2: Single-Channel Elbow Move / Construction Arm Shift
    The fundamental primitive of EZLiang/cgol-self-construction and simeks.txt:
    A single glider hits a 2x2 elbow block at lane offset -5.
    The glider is absorbed, and the block cleanly relocates by (dr=-1, dc=-2).
    """
    grid = np.zeros((48, 48), dtype=np.uint8)
    grid[25:27, 25:27] = 1

    schedule = SalvoSchedule(name="Single-Channel Arm Extension Salvo")
    schedule.add_glider(
        step=0, row=10, col=5, direction="SE", phase=0,
        label="Arm Drive Glider (Lane -5)",
    )

    expected_block = np.array([[1, 1], [1, 1]], dtype=np.uint8)

    return ConstructionPreset(
        id="elbow_shift",
        name="Single-Channel Arm Extension (0move-2)",
        category="Conway's Game of Life: Construction Arm",
        description=(
            "A single-channel slow-salvo glider hits an elbow block at lane offset -5. "
            "The glider is absorbed and the block shifts cleanly to (24, 23) with zero debris. "
            "This is the core elbow repositioning primitive used in EZLiang's 'sc' and 'oncoming' compilers."
        ),
        grid_size=(48, 48),
        target_box=(23, 26, 22, 25),
        expected_pattern=expected_block,
        expected_population=4,
        validation_step=82,
        initial_matrix=grid,
        schedule=schedule,
        is_von_neumann=False,
        metadata={
            "elbow_initial_pos": (25, 25),
            "elbow_final_pos": (24, 23),
            "displacement": (-1, -2),
            "gliders_used": 1,
            "compiler_op": "0move-2",
            "settle_time": 71,
        },
    )


def make_two_stage_arm_preset() -> ConstructionPreset:
    """
    Preset 3: Sequential Two-Stage Arm Walking
    Glider 1 repositions the elbow to (24, 23); after settling, Glider 2
    (released at step 75) advances it to (23, 21).
    """
    grid = np.zeros((56, 56), dtype=np.uint8)
    grid[25:27, 25:27] = 1

    schedule = SalvoSchedule(name="Two-Glider Chained Elbow Salvo")
    schedule.add_glider(
        step=0, row=10, col=5, direction="SE", phase=0,
        label="Pulse 1: Reposition Elbow -> (24, 23)",
    )
    schedule.add_glider(
        step=75, row=10, col=4, direction="SE", phase=0,
        label="Pulse 2: Advance Elbow -> (23, 21)",
    )

    expected_block = np.array([[1, 1], [1, 1]], dtype=np.uint8)

    return ConstructionPreset(
        id="two_stage_arm",
        name="Sequential Slow-Salvo Arm Walking",
        category="Conway's Game of Life: Construction Arm",
        description=(
            "A timed sequence of two slow-salvo gliders that incrementally 'walks' the elbow block "
            "across the grid: (25, 25) -> (24, 23) -> (23, 21). "
            "Demonstrates multi-glider algorithmic control where each step settles cleanly before the next arrives."
        ),
        grid_size=(56, 56),
        target_box=(22, 25, 20, 23),
        expected_pattern=expected_block,
        expected_population=4,
        validation_step=150,
        initial_matrix=grid,
        schedule=schedule,
        is_von_neumann=False,
        metadata={
            "stages": [
                {"step": 0, "pos": (25, 25)},
                {"step": 71, "pos": (24, 23)},
                {"step": 146, "pos": (23, 21)},
            ],
            "gliders_used": 2,
            "displacement_total": (-2, -4),
        },
    )


def make_blinker_synthesis_preset() -> ConstructionPreset:
    """
    Preset 4: 2-Glider Blinker (Oscillator) Synthesis
    Verified recipe (zero borders): SE phase-2 from (10, 10) + NW
    phase-0 from (20, 23) -> clean period-2 blinker around rows 19-21,
    cols 16-18 from step ~30.
    """
    grid = np.zeros((46, 46), dtype=np.uint8)

    schedule = SalvoSchedule(name="Blinker Synthesis Salvo")
    schedule.add_glider(
        step=0, row=10, col=10, direction="SE", phase=2, label="Glider A (SE)"
    )
    schedule.add_glider(
        step=0, row=20, col=23, direction="NW", phase=0, label="Glider B (NW)"
    )

    expected_blinker = np.array([[1, 1, 1]], dtype=np.uint8)

    return ConstructionPreset(
        id="blinker_synthesis",
        name="2-Glider Blinker Oscillator Synthesis",
        category="Conway's Game of Life: Construction",
        description=(
            "Two gliders collide to synthesize a dynamic Period-2 Blinker oscillator. "
            "Demonstrates that slow salvo collisions can construct both stationary still lifes and clock oscillators."
        ),
        grid_size=(46, 46),
        target_box=(19, 21, 16, 18),
        expected_pattern=expected_blinker,
        expected_population=3,
        validation_step=41,
        initial_matrix=grid,
        schedule=schedule,
        is_von_neumann=False,
        metadata={"period": 2, "final_population": 3, "gliders_used": 2},
    )


# ---------------------------------------------------------------------------
# Von Neumann: canonical 29-state construction primitives
# ---------------------------------------------------------------------------

def _vn_row_wire_pulse(rows: int, cols: int, row: int, start: int, length: int):
    """Build a horizontal ordinary-east wire of state 9 with a leading 13."""
    grid = np.zeros((rows, cols), dtype=np.int32)
    grid[row, start:start + length] = 9
    grid[row, start] = 13
    return grid


def make_vn_wire_signal_preset() -> ConstructionPreset:
    """
    Preset 5: Ordinary transmission wire (signal in flight).

    A wire of ordinary-east transmission cells (state 9) carries an
    excited pulse (state 13) that advances exactly one cell per step,
    demonstrating the canonical signal carrier. Validated mid-flight
    (dynamic), so settling is not required.
    """
    grid = _vn_row_wire_pulse(8, 24, row=4, start=2, length=16)
    expected = np.full((1, 16), 9, dtype=np.int32)
    expected[0, 10] = 13  # after 10 steps the pulse has advanced to col 12

    return ConstructionPreset(
        id="vn_wire_signal",
        name="Von Neumann Ordinary Transmission Wire",
        category="Von Neumann Automata: Primitives",
        description=(
            "A wire of ordinary transmission states (OTS, state 9) carries an "
            "excited pulse (state 13) that propagates east at exactly one cell "
            "per synchronous step. This is the canonical information channel of "
            "von Neumann's automaton."
        ),
        grid_size=(8, 24),
        target_box=(4, 4, 2, 17),
        expected_pattern=expected,
        expected_population=16,
        validation_step=10,
        initial_matrix=grid,
        schedule=SalvoSchedule(name="VN wire (no salvo)"),
        is_von_neumann=True,
        metadata={
            "require_settled": False,
            "min_constructions": 0,
            "pulse_start": (4, 2),
            "pulse_at_step": (10, 4, 12),
        },
    )


def make_vn_wire_extension_preset() -> ConstructionPreset:
    """
    Preset 6: Canonical wire extension (sensitize -> decode -> new cell).

    A pulse travels the wire and, at the blank tip, sensitizes a quiescent
    cell (state 1). With no further input the sensitized cell walks the
    decoding tree 1 -> 2 -> 4 -> 8 and resolves into a new ordinary-east
    transmission cell (state 9). The arm has thus *constructed* one new
    wired cell from vacuum via the canonical mechanism. Settles to a
    still-life wire (period 1).
    """
    grid = _vn_row_wire_pulse(8, 16, row=4, start=2, length=6)
    expected = np.full((1, 7), 9, dtype=np.int32)

    return ConstructionPreset(
        id="vn_wire_extension",
        name="Von Neumann Wire Extension (Construction)",
        category="Von Neumann Automata: Construction",
        description=(
            "A pulse reaches the blank tip of an ordinary wire, sensitizes it "
            "(state 1), and the canonical sensitization tree 1->2->4->8->9 "
            "resolves it into a new ordinary transmission cell. One new wired "
            "cell is constructed from vacuum; the result is a stable longer wire."
        ),
        grid_size=(8, 16),
        target_box=(4, 4, 2, 8),
        expected_pattern=expected,
        expected_population=7,
        validation_step=12,
        initial_matrix=grid,
        schedule=SalvoSchedule(name="VN wire extension (no salvo)"),
        is_von_neumann=True,
        metadata={
            "require_settled": True,
            "min_constructions": 0,
            "constructed_cell": (4, 8),
            "decode_path": [1, 2, 4, 8, 9],
        },
    )


def make_vn_special_kill_preset() -> ConstructionPreset:
    """
    Preset 7: Special/ordinary interaction (retraction primitive).

    An excited special transmission cell (state 21) fires east and kills
    the adjacent ordinary transmission cell (state 9 -> 0), then settles
    into a quiescent special-east cell (state 17). Excited OTS-kills-STS /
    STS-kills-OTS is the mechanism von Neumann's arm uses to erase cells.
    """
    grid = np.zeros((8, 12), dtype=np.int32)
    grid[4, 2] = 21  # special east, excited
    grid[4, 3] = 9   # ordinary east, quiescent (will be killed)
    expected = np.array([[17, 0]], dtype=np.int32)

    return ConstructionPreset(
        id="vn_special_kill",
        name="Von Neumann Special/Ordinary Kill",
        category="Von Neumann Automata: Primitives",
        description=(
            "An excited special transmission state (state 21) fires at an "
            "adjacent ordinary transmission cell and kills it (9 -> 0), then "
            "settles to quiescent special-east (17). This is the erase/"
            "retraction primitive of the construction arm."
        ),
        grid_size=(8, 12),
        target_box=(4, 4, 2, 3),
        expected_pattern=expected,
        expected_population=1,
        validation_step=3,
        initial_matrix=grid,
        schedule=SalvoSchedule(name="VN kill (no salvo)"),
        is_von_neumann=True,
        metadata={
            "require_settled": True,
            "min_constructions": 0,
            "killed_cell": (4, 3),
        },
    )


def make_vn_confluent_delay_preset() -> ConstructionPreset:
    """
    Preset 8: Confluent OR-junction with one-cycle delay.

    A confluent cell (state 25, C00) receives one excited input and runs
    through C00 -> C01 -> C10 -> C00 (states 25 -> 26 -> 27 -> 25), then
    holds. The intermediate cycle is the delay that lets the confluent
    merge two signals arriving on successive steps.
    """
    grid = np.zeros((8, 12), dtype=np.int32)
    grid[4, 2] = 13  # ordinary east, excited (source)
    grid[4, 3] = 25  # C00
    expected = np.array([[9, 26]], dtype=np.int32)

    return ConstructionPreset(
        id="vn_confluent_delay",
        name="Von Neumann Confluent Delay Junction",
        category="Von Neumann Automata: Primitives",
        description=(
            "A confluent cell (C00, state 25) absorbs an excited input and "
            "runs 25 -> 26 -> 27 -> 25 before holding. The built-in one-cycle "
            "delay is what lets a confluent OR-junction combine two signals "
            "arriving on successive steps."
        ),
        grid_size=(8, 12),
        target_box=(4, 4, 2, 3),
        expected_pattern=expected,
        expected_population=2,
        validation_step=1,
        initial_matrix=grid,
        schedule=SalvoSchedule(name="VN confluent delay (no salvo)"),
        is_von_neumann=True,
        metadata={
            "require_settled": False,
            "min_constructions": 0,
            "cycle": [25, 26, 27, 25],
        },
    )


# ---------------------------------------------------------------------------
# Von Neumann: documented reference machines (vendored Golly JvN29 patterns)
# ---------------------------------------------------------------------------

def _make_machine_preset(
    key: str,
    pid: str,
    name: str,
    description: str,
    validation_step: int,
    min_constructed: int,
    required_constructed_types: Optional[List[int]] = None,
    require_settled: bool = False,
) -> ConstructionPreset:
    """Wrap a vendored Golly JvN29 pattern as a behavior-validated preset.

    These run von Neumann's *actual* circuits under the canonical table; the
    proof is behavioral (cells built out of vacuum, and the documented
    constructed cell types), not a hand-supplied target shape.
    """
    from mesa_self_construction.von_neumann import load_jvn_pattern

    pattern = load_jvn_pattern(key)
    grid = pattern.grid.astype(np.int32)
    h, w = grid.shape
    return ConstructionPreset(
        id=pid,
        name=name,
        category="Von Neumann Automata: Reference Machines",
        description=description,
        grid_size=(h, w),
        target_box=(0, h - 1, 0, w - 1),
        expected_pattern=None,
        expected_population=int(np.count_nonzero(grid > 0)),
        validation_step=validation_step,
        initial_matrix=grid,
        schedule=SalvoSchedule(name=f"{name} (no salvo)"),
        is_von_neumann=True,
        metadata={
            "machine": True,
            "min_constructed": min_constructed,
            "required_constructed_types": list(required_constructed_types or []),
            "require_settled": require_settled,
            "source_pattern": pattern.name or key,
            "reference": "Golly JvN29 pattern collection (Nobili / von Neumann)",
        },
    )


def make_vn_counter_preset() -> ConstructionPreset:
    """Nobili's counting circuit: ten input pulses -> one output pulse + reset.

    As shipped by Golly the circuit runs on its own stored pulses; over time
    it builds a handful of new cells as it cycles. We validate that live,
    growing behavior rather than a static shape.
    """
    return _make_machine_preset(
        key="counter",
        pid="vn_counter",
        name="Von Neumann Counter (Nobili)",
        description=(
            "Renato Nobili's counting circuit in von Neumann's 29-state "
            "automaton: after ten well-separated input pulses it emits a "
            "single output pulse and resets. It is a live, self-clocked "
            "circuit; the proof tracks the cells it builds out of vacuum."
        ),
        validation_step=1000,
        min_constructed=4,
        require_settled=False,
    )


def make_vn_cell_coders_preset() -> ConstructionPreset:
    """The nine signal trains that construct the nine constructible cell types.

    ``cell-coders-demo.rle`` (Nobili) shows, for each constructible type
    (9-12, 17-20, 25), the signal train that builds it and a coder that
    produces the train. Starting from a quiescent grid, running it constructs
    18 cells from vacuum whose types are exactly those nine, twice over.
    """
    return _make_machine_preset(
        key="cell_coders",
        pid="vn_cell_coders",
        name="Von Neumann Cell Coders (construct all 9 types)",
        description=(
            "Nobili's cell-coders demo: nine coders each turn a single excited "
            "pulse into the signal train that constructs one of the nine "
            "constructible cell types (OTS 9-12, STS 17-20, confluent 25). "
            "Running it builds 18 new cells from vacuum covering all nine types."
        ),
        validation_step=80,
        min_constructed=18,
        required_constructed_types=[9, 10, 11, 12, 17, 18, 19, 20, 25],
        require_settled=True,
    )


def make_vn_self_replicator_preset() -> ConstructionPreset:
    """A compact von Neumann self-reproducing machine.

    The machine reads its own description tape and begins constructing a copy;
    the first few hundred generations already build new transmission and
    confluent cells near the construction front. Full replication is on the
    order of 10^5 generations, so the demo validates the *growth* (new cells
    from vacuum), not the completed copy.
    """
    return _make_machine_preset(
        key="self_replicator",
        pid="vn_self_replicator",
        name="Von Neumann Self-Replicator (small variant)",
        description=(
            "A compact self-reproducing machine in von Neumann's 29-state "
            "automaton. It reads its own description tape and starts building "
            "a copy: within a few hundred steps it constructs new excited "
            "transmission and confluent cells. Proof tracks construction from "
            "vacuum; full replication takes ~10^5 generations."
        ),
        validation_step=300,
        min_constructed=8,
        require_settled=False,
    )


ALL_PRESETS: Dict[str, ConstructionPreset] = {
    "block_synthesis": make_block_synthesis_preset(),
    "elbow_shift": make_elbow_shift_preset(),
    "two_stage_arm": make_two_stage_arm_preset(),
    "blinker_synthesis": make_blinker_synthesis_preset(),
    "vn_wire_signal": make_vn_wire_signal_preset(),
    "vn_wire_extension": make_vn_wire_extension_preset(),
    "vn_special_kill": make_vn_special_kill_preset(),
    "vn_confluent_delay": make_vn_confluent_delay_preset(),
    "vn_counter": make_vn_counter_preset(),
    "vn_cell_coders": make_vn_cell_coders_preset(),
    "vn_self_replicator": make_vn_self_replicator_preset(),
}


def get_preset(preset_id: str) -> ConstructionPreset:
    """Retrieve a preset by id, defaulting to block_synthesis if not found."""
    return ALL_PRESETS.get(preset_id, ALL_PRESETS["block_synthesis"])
