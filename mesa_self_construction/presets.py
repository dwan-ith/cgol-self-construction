"""
Predefined, mathematically verified self-construction experiments in Conway's Game of Life
and Von Neumann's 29-state cellular automaton constructor.
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
    # Glider 1: SE from (10, 10), phase 1
    schedule.add_glider(
        step=0,
        row=10,
        col=10,
        direction="SE",
        phase=1,
        label="Glider Alpha (SE)",
    )
    # Glider 2: NW from (24, 29), phase 0
    schedule.add_glider(
        step=0,
        row=24,
        col=29,
        direction="NW",
        phase=0,
        label="Glider Beta (NW)",
    )

    expected_block = np.array([[1, 1], [1, 1]], dtype=np.uint8)

    return ConstructionPreset(
        id="block_synthesis",
        name="De Novo 2-Glider Block Synthesis",
        category="Conway's Game of Life: Construction",
        description=(
            "Two synchronized gliders colliding head-on in vacuum at 180°. "
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
    The glider is absorbed, and the block cleanly relocates by (dr=-1, dc=-2) at step 71.
    """
    grid = np.zeros((48, 48), dtype=np.uint8)
    # Initial elbow block at (25, 25)
    grid[25:27, 25:27] = 1
    
    schedule = SalvoSchedule(name="Single-Channel Arm Extension Salvo")
    schedule.add_glider(
        step=0,
        row=10,
        col=5,
        direction="SE",
        phase=0,
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
    Demonstrates sequential slow salvo control:
    Glider 1 repositions the elbow to (24, 23) at step 71.
    After settling, Glider 2 (released at step 75) hits the relocated elbow,
    advancing it cleanly to (23, 21) at step 146!
    """
    grid = np.zeros((56, 56), dtype=np.uint8)
    grid[25:27, 25:27] = 1  # Initial block at (25, 25)
    
    schedule = SalvoSchedule(name="Two-Glider Chained Elbow Salvo")
    # Glider 1: fires at t=0
    schedule.add_glider(
        step=0,
        row=10,
        col=5,
        direction="SE",
        phase=0,
        label="Pulse 1: Reposition Elbow -> (24, 23)",
    )
    # Glider 2: fires at t=75 (after Glider 1 fully settled at t=71)
    schedule.add_glider(
        step=75,
        row=10,
        col=4,
        direction="SE",
        phase=0,
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
    phase-0 from (20, 23) -> clean period-2 blinker oscillating around
    rows 19-21, cols 16-18 from step ~30. Horizontal phase on odd steps.
    """
    grid = np.zeros((46, 46), dtype=np.uint8)

    schedule = SalvoSchedule(name="Blinker Synthesis Salvo")
    schedule.add_glider(
        step=0,
        row=10,
        col=10,
        direction="SE",
        phase=2,
        label="Glider A (SE)",
    )
    schedule.add_glider(
        step=0,
        row=20,
        col=23,
        direction="NW",
        phase=0,
        label="Glider B (NW)",
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
        metadata={
            "period": 2,
            "final_population": 3,
            "gliders_used": 2,
        },
    )


def make_von_neumann_preset() -> ConstructionPreset:
    """
    Preset 5: Von Neumann 29-state constructor demonstration
    (tape loop + confluent latch + extending arm).

    Uses the full documented 29-state system in
    ``von_neumann.step_von_neumann_29``: ordinary/special transmission
    (ground+excited, directional), confluent ground/latched/firing
    (one-tick delay memory), and sensitized construct/kill. The arm tip
    sensitizes quiescent cells which solidify into wired transmission
    cells as pulses arrive: the arm genuinely extends eastward
    (16+ counted sensitized->wired events by step 50, easternmost wired
    cell advancing past col 31 while keeping a 12+ cell clear margin to
    the grid edge, so finite boundaries introduce no artificial
    interactions during validation).

    Pulse phase on the arm has period 3 (confluent latch delay); the
    expected 1x6 segment below is the phase at steps with step % 3 == 2
    (e.g. step 50). Population grows while extending, so validation
    requires construction events rather than 8-generation stability
    (see validator.require_settled=False, min_constructions=5).

    Explicitly out of scope: decoder/universal-constructor programs and
    self-description tapes (no stored machine description is read; see
    instead the vn_description_reader preset, which builds output cells
    from a stored description row).
    """
    grid = np.zeros((44, 64), dtype=np.int32)

    # Looped instruction tape
    grid[12, 8:21] = 1
    grid[12, 21] = 17
    grid[13:19, 21] = 4
    grid[18, 8:21] = 3
    grid[13:18, 8] = 2

    # Extending Construction Arm
    grid[12, 22:31] = 1
    grid[12, 31] = 21

    # Initial instruction excitation pulse
    grid[12, 10] = 5
    grid[12, 14] = 5

    schedule = SalvoSchedule(name="Von Neumann Instruction Stream")

    return ConstructionPreset(
        id="von_neumann_constructor",
        name="Von Neumann 29-State Constructor Demonstration",
        category="Von Neumann Automata: Construction",
        description=(
            "Full 29-state von Neumann system (directional ordinary/special "
            "transmission, confluent latch with delay, sensitized "
            "construct/kill). A looped tape circulates pulses through a "
            "confluent junction; the arm tip sensitizes quiescent cells "
            "which solidify into wired cells, extending the arm eastward. "
            "Construction events are counted, not claimed as reproduction: "
            "decoder programs and self-description tapes are out of scope."
        ),
        grid_size=(44, 64),
        target_box=(12, 12, 24, 29),
        expected_pattern=np.array([[5, 1, 1, 5, 1, 1]], dtype=np.int32),
        expected_population=65,
        validation_step=50,
        initial_matrix=grid,
        schedule=schedule,
        is_von_neumann=True,
        metadata={
            "states_count": 29,
            "scope": "29-state transitions + arm extension; no universal-construction program",
            "architecture": "Tape loop + confluent latch + extending arm",
            "construction_arm_length": 9,
            "confluent_junction": (12, 21),
            "head_coordinate": (12, 31),
            "arm_period": 3,
            "arm_phase": "step % 3 == 2 reads [5,1,1,5,1,1] on cols 24-29",
        },
    )


def make_description_reader_preset() -> ConstructionPreset:
    """
    Preset 6: Von Neumann description reader (build from a stored tape).

    Three independent reader lines (rows 6, 10, 14) each hold one
    description cell (ordinary-E = bit O, special-E = bit S) at col 6,
    fed by a wired line (cols 2-5) carrying two seed pulses. The first
    pulse sensitizes the blank build slot (col 7); the second solidifies
    it into a wired cell whose kind matches the description cell. The
    stored description [O, S, O] is thus copied into built cells
    [1, 9, 1] at col 7 -- construction driven by a stored description,
    the core mechanism von Neumann's universal constructor scales up
    (a full decoder/program tape remains out of scope).

    Settles to total pop 18; validated with the strict settled gate plus
    min_constructions=3.
    """
    grid = np.zeros((20, 44), dtype=np.int32)
    desc = {6: 1, 10: 9, 14: 1}  # row -> description cell state
    for row, kind in desc.items():
        wire = 1 if kind == 1 else 9
        pulse = 5 if kind == 1 else 13
        grid[row, 2:6] = wire
        grid[row, 2] = pulse
        grid[row, 4] = pulse
        grid[row, 6] = kind
        # col 7 left blank as the build slot.

    schedule = SalvoSchedule(name="Description Reader (no glider salvo)")

    expected = np.array([[1], [0], [0], [0], [9], [0], [0], [0], [1]], dtype=np.int32)

    return ConstructionPreset(
        id="vn_description_reader",
        name="Von Neumann Description Reader (Tape-Driven Build)",
        category="Von Neumann Automata: Construction",
        description=(
            "Three reader lines copy a stored 3-bit description [O, S, O] "
            "into built transmission cells [1, 9, 1]. First pulse "
            "sensitizes each blank build slot; second pulse solidifies it "
            "with the described kind. Demonstrates description-driven "
            "construction, the mechanism at the heart of universal "
            "construction."
        ),
        grid_size=(20, 44),
        target_box=(6, 14, 7, 7),
        expected_pattern=expected,
        expected_population=18,
        validation_step=16,
        initial_matrix=grid,
        schedule=schedule,
        is_von_neumann=True,
        metadata={
            "states_count": 29,
            "description_bits": ["O", "S", "O"],
            "description_cells": [(6, 6, 1), (10, 6, 9), (14, 6, 1)],
            "output_cells": [(6, 7, 1), (10, 7, 9), (14, 7, 1)],
            "min_constructions": 3,
            "require_settled": True,
        },
    )


ALL_PRESETS: Dict[str, ConstructionPreset] = {
    "block_synthesis": make_block_synthesis_preset(),
    "elbow_shift": make_elbow_shift_preset(),
    "two_stage_arm": make_two_stage_arm_preset(),
    "blinker_synthesis": make_blinker_synthesis_preset(),
    "von_neumann_constructor": make_von_neumann_preset(),
    "vn_description_reader": make_description_reader_preset(),
}


def get_preset(preset_id: str) -> ConstructionPreset:
    """Retrieve a preset by id, defaulting to block_synthesis if not found."""
    return ALL_PRESETS.get(preset_id, ALL_PRESETS["block_synthesis"])
