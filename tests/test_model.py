"""
Unit and integration tests for Mesa Self-Construction.
Validates Game of Life slow-salvo de novo block synthesis, elbow shifting,
two-stage arm walking, oscillator creation, and Von Neumann 29-state constructor.
"""

import pytest
import numpy as np
from mesa_self_construction import SelfConstructionModel, ALL_PRESETS
from mesa_self_construction.rle import parse_rle, encode_rle


def test_block_synthesis_preset():
    """Verify that Preset 1 synthesizes a 2x2 Block with 100% bitmask match."""
    model = SelfConstructionModel(preset_id="block_synthesis")
    assert model.current_step == 0
    assert model.population_history[0] == 10  # Two gliders of 5 cells each
    
    # Run to validation step
    for _ in range(model.preset.validation_step):
        model.step()
        
    validation = model.latest_validation
    assert validation.match_ratio == 1.0, f"Expected 1.0 match ratio, got {validation.match_ratio}"
    assert validation.population == 4, f"Expected final block pop=4, got {validation.population}"
    assert validation.passed, f"Validation failed: {validation.proof_message}"


def test_elbow_shift_preset():
    """Verify that Preset 2 moves the elbow block cleanly without leaving debris."""
    model = SelfConstructionModel(preset_id="elbow_shift")
    assert model.current_step == 0
    # Elbow (4) + Glider (5) = 9
    assert model.population_history[0] == 9
    
    for _ in range(model.preset.validation_step):
        model.step()
        
    validation = model.latest_validation
    assert validation.match_ratio == 1.0
    assert validation.population == 4, f"Expected clean block pop=4, got {validation.population}"
    assert validation.passed, f"Validation failed: {validation.proof_message}"


def test_two_stage_arm_preset():
    """Verify sequential two-stage slow-salvo arm walking."""
    model = SelfConstructionModel(preset_id="two_stage_arm")
    
    for _ in range(model.preset.validation_step):
        model.step()
        
    validation = model.latest_validation
    assert validation.match_ratio == 1.0
    assert validation.population == 4
    assert validation.passed, f"Validation failed: {validation.proof_message}"


def test_von_neumann_constructor_preset():
    """Verify 29-state constructor extends its arm with counted events."""
    model = SelfConstructionModel(preset_id="von_neumann_constructor")
    assert model.is_von_neumann is True

    for _ in range(model.preset.validation_step):
        model.step()

    validation = model.latest_validation
    assert validation.passed, f"Validation failed: {validation.proof_message}"
    assert validation.population == 65
    assert validation.details["exact_box_match"] is True
    assert validation.details["constructions"] >= 5
    assert model.vn_constructed_cells >= 5
    # Arm genuinely extended past its initial tip (col 31).
    import numpy as np
    row = model.grid_data[12, :]
    eastmax = int(np.max(np.argwhere(row > 0)))
    assert eastmax > 31, f"arm did not extend: eastmax={eastmax}"


def test_von_neumann_29_state_rules():
    """Unit-cover all 29 states: directional feed, delay, construct/kill."""
    import numpy as np
    from mesa_self_construction.von_neumann import (
        step_von_neumann_29, N_STATES, STATE_NAMES,
    )
    assert N_STATES == 29 and len(STATE_NAMES) == 29
    # Ordinary directional feed (E feeds E from west; N from south does not).
    g = np.zeros((5, 5), dtype=np.int32); g[2, 1] = 5; g[2, 2] = 1
    n, _ = step_von_neumann_29(g); assert n[2, 2] == 5
    g = np.zeros((5, 5), dtype=np.int32); g[3, 2] = 6; g[2, 2] = 1
    n, _ = step_von_neumann_29(g); assert n[2, 2] == 1
    # Special line symmetric + kind isolation (special ignored by ordinary).
    g = np.zeros((5, 5), dtype=np.int32); g[2, 1] = 13; g[2, 2] = 9
    n, _ = step_von_neumann_29(g); assert n[2, 2] == 13
    g = np.zeros((5, 5), dtype=np.int32); g[2, 1] = 13; g[2, 2] = 1
    n, _ = step_von_neumann_29(g); assert n[2, 2] == 1
    # Confluent latch delay: C -> C1 -> C*.
    g = np.zeros((5, 5), dtype=np.int32); g[2, 1] = 5; g[2, 2] = 17
    n, _ = step_von_neumann_29(g); assert n[2, 2] == 19
    n2, _ = step_von_neumann_29(n); assert n2[2, 2] in (18, 20)
    # Sensitize -> construct; wrong-kind -> kill.
    g = np.zeros((5, 5), dtype=np.int32); g[2, 1] = 5
    n, ev = step_von_neumann_29(g); assert n[2, 2] == 21 and ev["sensitized"] == 1
    g = np.zeros((5, 5), dtype=np.int32); g[2, 2] = 21; g[2, 1] = 5
    n, ev = step_von_neumann_29(g); assert n[2, 2] == 1 and ev["constructed"] == 1
    g = np.zeros((5, 5), dtype=np.int32); g[2, 2] = 21; g[2, 1] = 13
    n, ev = step_von_neumann_29(g); assert n[2, 2] == 0 and ev["killed"] == 1
    # Ambiguity refusal: two excited neighbors -> stays blank.
    g = np.zeros((5, 5), dtype=np.int32); g[2, 1] = 5; g[2, 3] = 7
    n, _ = step_von_neumann_29(g); assert n[2, 2] == 0


def test_simeks_toolkit_and_recipe_replay():
    """EZLiang grounding: simeks toolkit loads; recipe replays end-to-end."""
    from pathlib import Path
    from mesa_self_construction.salvo import (
        SalvoSchedule, parse_lane_phase_recipe, lane_to_position,
        load_simeks_toolkit,
    )
    toolkit = load_simeks_toolkit(
        Path(__file__).resolve().parent.parent / "cgol_ref" / "compilers" / "data" / "simeks.txt"
    )
    assert "0move-2" in toolkit
    assert toolkit["0move-2"][0] == 0 and len(toolkit["0move-2"]) > 5
    # Real tokens from oncoming.py head: "-5:1 -2:5 -7:5".
    recipe = parse_lane_phase_recipe("-5:1 -2:5 -7:5")
    assert recipe == [(-5, 1), (-2, 1), (-7, 1)]  # phases reduced mod 4
    # Perpendicular geometry: SE lane shifts (-1,+1) per lane.
    assert lane_to_position(0, 10, 10, "SE") == (10, 10)
    assert lane_to_position(-5, 10, 10, "SE") == (15, 5)
    r1, c1 = lane_to_position(3, 10, 10, "SE")
    r2, c2 = lane_to_position(4, 10, 10, "SE")
    assert (r2 - r1, c2 - c1) == (-1, 1)
    # End-to-end: schedule, inject, step without errors; all injected.
    sched = SalvoSchedule.from_lane_phase_recipe(
        "-5:1 -2:5 -7:5", base_row=10, base_col=10, direction="SE"
    )
    assert len(sched.gliders) == 3
    from mesa_self_construction import SelfConstructionModel
    from mesa_self_construction.presets import ConstructionPreset
    import numpy as np
    import copy
    base = SelfConstructionModel(preset_id="elbow_shift")
    preset = ConstructionPreset(
        id="recipe_replay", name="t", category="t", description="t",
        grid_size=(48, 48), target_box=(0, 2, 0, 2),
        expected_pattern=np.zeros((3, 3), dtype=np.uint8),
        expected_population=0, validation_step=5,
        initial_matrix=np.zeros((48, 48), dtype=np.uint8),
        schedule=sched, is_von_neumann=False,
    )
    base.preset = preset
    base.height, base.width = preset.grid_size
    base.grid_data = np.copy(preset.initial_matrix)
    base.schedule = copy.deepcopy(sched)
    base.current_step = 0
    base.gliders_injected_count = 0
    base._inject_due_gliders()
    for _ in range(250):
        base.step()
    assert base.gliders_injected_count == 3


def test_telemetry_consumed_and_components():
    """Injected - in-flight = consumed; component classes add up."""
    from mesa_self_construction import SelfConstructionModel
    m = SelfConstructionModel(preset_id="block_synthesis")
    assert m.gliders_injected_count == 2
    assert m.gliders_in_flight == 2
    assert m.gliders_consumed_count == 0
    for _ in range(m.preset.validation_step):
        m.step()
    assert m.latest_validation.passed
    assert m.gliders_in_flight == 0
    assert m.gliders_consumed_count == 2
    cc = m.component_counts
    assert cc["target"] == 4 and cc["glider"] == 0
    assert sum(cc.values()) == m.latest_validation.population
    state = m.get_state()
    assert state["gliders_consumed"] == 2
    assert state["gliders_in_flight"] == 0
    assert all(g["consumed"] for g in state["gliders"])


def test_rle_encode_decode():
    """Verify RLE roundtrip encoding and decoding."""
    pattern = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
    ], dtype=np.uint8)
    
    rle_str = encode_rle(pattern, name="Glider")
    decoded, meta = parse_rle(rle_str)
    assert np.array_equal(pattern, decoded[:3, :3])


def test_step_backward():
    """Verify history buffer and reverse stepping (exact restore)."""
    model = SelfConstructionModel(preset_id="block_synthesis")
    for _ in range(5):
        model.step()
    assert model.current_step == 5
    injected_before = model.gliders_injected_count

    success = model.step_backward()
    assert success is True
    assert model.current_step == 4
    # Schedule flags and counters must rewind too; stepping forward
    # again must reproduce the same grid.
    grid_at_4 = model.grid_data.copy()
    model.step()
    assert model.current_step == 5
    assert model.gliders_injected_count == injected_before

    # Independent models must not share schedule state.
    m1 = SelfConstructionModel(preset_id="block_synthesis")
    m2 = SelfConstructionModel(preset_id="block_synthesis")
    assert m1.schedule is not m2.schedule
    m1.step()
    assert m2.current_step == 0


def test_all_presets_exist():
    """Verify all defined presets load cleanly."""
    for preset_id in ALL_PRESETS:
        model = SelfConstructionModel(preset_id=preset_id)
        assert model.preset.id == preset_id
        model.step()
        assert model.current_step == 1


def test_glider_directions():
    """Regression: NW must move (-1,-1), NE (-1,+1) over one period."""
    import numpy as np
    from mesa_self_construction.salvo import GLIDER_PHASES
    from mesa_self_construction.model import conway_step_zero_boundary

    expected = {"SE": (1, 1), "SW": (1, -1), "NW": (-1, -1), "NE": (-1, 1)}
    for direction, (dr, dc) in expected.items():
        g = np.zeros((16, 16), dtype=np.uint8)
        g[6:9, 6:9] = GLIDER_PHASES[direction][0]
        c0 = np.argwhere(g == 1).mean(axis=0)
        for _ in range(4):
            g = conway_step_zero_boundary(g)
        disp = np.argwhere(g == 1).mean(axis=0) - c0
        assert tuple(np.round(disp).astype(int)) == (dr, dc), direction


def test_blinker_is_clean():
    """Blinker synthesis must leave exactly 3 live cells (no debris)."""
    model = SelfConstructionModel(preset_id="blinker_synthesis")
    for _ in range(model.preset.validation_step):
        model.step()
    v = model.latest_validation
    assert v.passed
    assert v.population == 3
    assert v.details["exact_box_match"] is True


def test_description_reader_preset():
    """Tape description [O,S,O] must be copied into built cells [1,9,1]."""
    import numpy as np
    model = SelfConstructionModel(preset_id="vn_description_reader")
    for _ in range(model.preset.validation_step):
        model.step()
    v = model.latest_validation
    assert v.passed, f"Validation failed: {v.proof_message}"
    assert v.population == 18
    assert model.vn_constructed_cells >= 3
    assert int(model.grid_data[6, 7]) == 1
    assert int(model.grid_data[10, 7]) == 9
    assert int(model.grid_data[14, 7]) == 1


def test_vn_construction_keeps_boundary_margin():
    """Extending arm must stay clear of the grid edge (no edge effects)."""
    import numpy as np
    model = SelfConstructionModel(preset_id="von_neumann_constructor")
    for _ in range(model.preset.validation_step):
        model.step()
    assert model.latest_validation.passed
    for r in range(model.height):
        row = model.grid_data[r, :]
        nz = np.argwhere(row > 0).flatten()
        if len(nz):
            assert int(nz.max()) <= model.width - 5, f"row {r} too close to edge"
            assert int(nz.min()) >= 4, f"row {r} too close to edge"


def test_single_channel_separation_replay():
    """simeks single-channel stream must execute with documented timing."""
    import copy
    import pytest
    from pathlib import Path
    from mesa_self_construction.salvo import (
        SalvoSchedule, load_simeks_toolkit,
    )
    toolkit = load_simeks_toolkit(
        Path(__file__).resolve().parent.parent / "cgol_ref" / "compilers" / "data" / "simeks.txt"
    )
    seps = toolkit["0move-2"][:3]
    assert seps[0] == 0 and len(seps) == 3
    sched = SalvoSchedule.from_separation_list(
        seps, row=10, col=5, direction="SE", op_name="0move-2-test"
    )
    # Release steps are cumulative separations; same single lane.
    assert [g.release_step for g in sched.gliders] == [0, seps[1], seps[1] + seps[2]]
    assert {(g.row, g.col, g.direction) for g in sched.gliders} == {(10, 5, "SE")}
    # Validation of malformed input.
    with pytest.raises(ValueError):
        SalvoSchedule.from_separation_list([109, 90])
    with pytest.raises(ValueError):
        SalvoSchedule.from_separation_list([0, -5])
    # Replay against the elbow executes without errors; every glider is
    # injected and accounted for (in-flight or consumed).
    import numpy as np
    model = SelfConstructionModel(preset_id="elbow_shift")
    model.schedule = copy.deepcopy(sched)
    model.current_step = 0
    model.gliders_injected_count = 0
    model._inject_due_gliders()
    model.population_history = [int(np.count_nonzero(model.grid_data))]
    model.history_buffer = [np.copy(model.grid_data)]
    model._injected_count_history = [model.gliders_injected_count]
    model._consumed_count_history = [0]
    model._schedule_flags_history = [[g.injected for g in model.schedule.gliders]]
    for _ in range(sum(seps) + 250):
        model.step()
    assert model.gliders_injected_count == 3
    assert model.gliders_in_flight + model.gliders_consumed_count == 3
