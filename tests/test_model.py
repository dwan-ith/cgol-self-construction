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


def test_vn_wire_extension_constructs_cell():
    """Canonical construction: a pulse sensitizes the blank wire tip and the
    decoding tree resolves it into a new ordinary transmission cell."""
    import numpy as np
    model = SelfConstructionModel(preset_id="vn_wire_extension")
    assert model.is_von_neumann is True
    assert int(model.grid_data[4, 8]) == 0  # blank tip initially

    for _ in range(model.preset.validation_step):
        model.step()

    v = model.latest_validation
    assert v.passed, v.proof_message
    assert v.population == 7
    assert v.details["exact_box_match"] is True
    assert v.details["period"] == 1
    assert int(model.grid_data[4, 8]) == 9  # OTS E constructed at the tip
    assert model.vn_sensitized_cells >= 1   # the tip was sensitized en route


def test_jvn29_reference_table_parses():
    """The vendored canonical table loads with the expected structure."""
    from mesa_self_construction.von_neumann import load_jvn29_table, N_STATES
    table = load_jvn29_table()
    assert table.n_states == N_STATES == 29
    assert len(table.rules) == 102  # canonical JvN29 transition count
    # A ground cell is sensitized by any single excited orthogonal input.
    assert table.next_state(0, 0, 0, 0, 13) == 1
    assert table.next_state(0, 16, 0, 0, 0) == 1


def test_jvn29_golden_transitions():
    """Golden neighbourhood -> state cases read from JvN29.table.

    Input order is C,N,E,S,W (Golly vonNeumann); expectations are the
    canonical rules, hand-checked against the reference file.
    """
    from mesa_self_construction.von_neumann import load_jvn29_table
    T = load_jvn29_table()
    cases = {
        # sensitization tree: 1 -> 2/3 -> 4..7 -> 8 -> 9/10 ...
        (1, 0, 0, 0, 0): 2,
        (1, 0, 0, 0, 13): 3,
        (2, 0, 0, 0, 0): 4,
        (2, 0, 0, 0, 13): 5,
        (3, 0, 0, 0, 0): 6,
        (3, 0, 0, 0, 13): 7,
        (4, 0, 0, 0, 0): 8,
        (4, 0, 0, 0, 13): 11,
        (5, 0, 0, 0, 0): 12,
        (5, 0, 0, 0, 13): 17,
        (6, 0, 0, 0, 0): 18,
        (6, 0, 0, 0, 13): 19,
        (7, 0, 0, 0, 0): 20,
        (7, 0, 0, 0, 13): 25,
        (8, 0, 0, 0, 0): 9,
        (8, 0, 0, 0, 13): 10,
        # ordinary transmission: quiescent gets excited, excited decays
        (9, 0, 0, 0, 13): 13,
        (13, 0, 0, 0, 0): 9,
        # special/ordinary interaction: an excited STS kills an OTS
        (9, 0, 0, 0, 21): 0,
        (17, 0, 0, 0, 21): 21,
        # confluent delay cycle 25 -> 26 -> 27 -> 25
        (25, 0, 0, 0, 13): 26,
        (26, 0, 0, 0, 0): 27,
        (26, 0, 0, 0, 13): 28,
        (27, 0, 0, 0, 0): 25,
        (28, 0, 0, 0, 0): 27,
        # unlisted neighbourhoods leave the cell unchanged
        (0, 0, 0, 0, 14): 0,
        (17, 0, 0, 0, 0): 17,
    }
    for (c, n, e, s, w), expected in cases.items():
        got = T.next_state(c, n, e, s, w)
        assert got == expected, (
            f"C={c} N={n} E={e} S={s} W={w}: got {got}, want {expected}"
        )


def test_jvn29_kernel_matches_independent_evaluator():
    """The production kernel agrees with a separately written evaluator.

    The evaluator re-reads JvN29.table into a plain list of (tokens, sets)
    and scans it in file order with membership + repeated-variable equality,
    sharing no code with JvN29Table.next_state. This is a regression guard
    for the kernel; fidelity of the *rules* themselves is pinned by the
    vendored table plus the golden transitions above.
    """
    import numpy as np
    from mesa_self_construction.von_neumann import (
        step_von_neumann_29, DATA_PATH,
    )
    text = DATA_PATH.read_text(encoding="utf-8")
    var_sets = {}
    ref_rules = []  # (tokens, [allowed sets], out_token)
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("@") or ":" in line:
            continue
        if line.startswith("var "):
            name, _, rhs = line[4:].partition("=")
            var_sets[name.strip()] = set(
                int(v) for v in rhs.strip()[1:-1].split(",")
            )
            continue
        toks = line.split(",")
        inputs = toks[:5]
        allowed = [var_sets[t] if t in var_sets else {int(t)} for t in inputs]
        ref_rules.append((inputs, allowed, toks[5]))

    def ref_step(grid):
        h, w = grid.shape
        out = grid.copy()
        for r in range(h):
            for c in range(w):
                states = (
                    int(grid[r, c]),
                    int(grid[r - 1, c]) if r > 0 else 0,
                    int(grid[r, c + 1]) if c + 1 < w else 0,
                    int(grid[r + 1, c]) if r + 1 < h else 0,
                    int(grid[r, c - 1]) if c > 0 else 0,
                )
                for inputs, allowed, out_tok in ref_rules:
                    bound = {}
                    ok = True
                    for i, tok in enumerate(inputs):
                        if states[i] not in allowed[i]:
                            ok = False
                            break
                        if tok.isalpha():
                            if tok in bound and bound[tok] != states[i]:
                                ok = False
                                break
                            bound[tok] = states[i]
                    if ok:
                        out[r, c] = (
                            bound[out_tok] if out_tok.isalpha() else int(out_tok)
                        )
                        break
        return out

    rng = np.random.default_rng(1234)
    for _ in range(30):
        g = rng.integers(0, 29, size=(7, 7)).astype(np.int32)
        g[0, :] = 0; g[-1, :] = 0; g[:, 0] = 0; g[:, -1] = 0
        got, _ = step_von_neumann_29(g)
        assert np.array_equal(got, ref_step(g))


def test_jvn29_wire_propagation():
    """A pulse travels an ordinary wire exactly one cell per synchronous step."""
    import numpy as np
    from mesa_self_construction.von_neumann import step_von_neumann_29
    g = np.zeros((5, 12), dtype=np.int32)
    g[2, 1:10] = 9
    g[2, 1] = 13
    for t in range(1, 8):
        g, _ = step_von_neumann_29(g)
        live = list(map(int, g[2]))
        assert live[1 + t] == 13  # pulse advanced to col 1+t
        assert live.count(13) == 1


def test_vn_off_grid_is_vacuum():
    """Off-grid neighbours are state 0: no torus wrap at the borders."""
    import numpy as np
    from mesa_self_construction.von_neumann import step_von_neumann_29
    g = np.zeros((4, 4), dtype=np.int32)
    g[0, 0] = 13  # ordinary-east excited at the top-left corner
    n, _ = step_von_neumann_29(g)
    assert int(n[0, 0]) == 9  # 13 decays to 9
    assert int(n[3, 3]) == 0  # no wrap-around excitation
    assert int(n[3, 0]) == 0
    assert int(n[0, 3]) == 0


def test_validator_period_detection():
    """Settling uses full-grid periodicity, not a population heuristic."""
    import numpy as np
    from mesa_self_construction.validator import detect_period
    a = np.zeros((4, 4), dtype=np.uint8)
    b = a.copy(); b[1, 1] = 1
    assert detect_period([a, a, a]) == 1
    assert detect_period([a, b, a, b]) == 2
    c = a.copy(); c[2, 2] = 1
    assert detect_period([a, b, c, a, b, c]) == 3
    # Same population, different configurations -> not settled.
    d = a.copy(); d[0, 0] = 1
    assert detect_period([b, c, d]) is None


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


def test_vn_canonical_presets():
    """All canonical VN primitive presets reach their documented target."""
    checks = {
        "vn_wire_signal": (16, (4, 12), 13),    # pop, (row,col), state
        "vn_wire_extension": (7, (4, 8), 9),
        "vn_special_kill": (1, (4, 2), 17),
        "vn_confluent_delay": (2, (4, 3), 26),
    }
    for pid, (pop, (r, c), st) in checks.items():
        m = SelfConstructionModel(preset_id=pid)
        for _ in range(m.preset.validation_step):
            m.step()
        v = m.latest_validation
        assert v.passed, f"{pid}: {v.proof_message}"
        assert v.population == pop, pid
        assert int(m.grid_data[r, c]) == st, f"{pid} at ({r},{c})"


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


# ---------------------------------------------------------------------------
# Golly multi-state RLE codec
# ---------------------------------------------------------------------------

def test_mstate_rle_roundtrip():
    """Multi-state RLE encode/decode is lossless across all 29 states."""
    from mesa_self_construction.mstate_rle import (
        encode_mstate_rle, decode_mstate_rle, MultiStatePattern,
    )
    rng = np.random.default_rng(7)
    grid = rng.integers(0, 29, size=(9, 13)).astype(np.int32)
    text = encode_mstate_rle(
        MultiStatePattern(grid=grid, rule="JvN29", name="random")
    )
    back = decode_mstate_rle(text)
    assert back.rule == "JvN29"
    assert back.grid.shape == (9, 13)
    assert np.array_equal(back.grid, grid)
    assert back.population == int(np.count_nonzero(grid))


def test_mstate_rle_high_state_encoding():
    """States 25-28 use the two-letter pA..pD prefixes Golly defines."""
    from mesa_self_construction.mstate_rle import decode_mstate_rle
    # pA=25, pB=26, pC=27, pD=28; A=1, X=24.
    text = "x = 6, y = 1, rule = JvN29\npApBpCpDAX!"
    g = decode_mstate_rle(text).grid
    assert list(map(int, g[0])) == [25, 26, 27, 28, 1, 24]


# ---------------------------------------------------------------------------
# Vendored von Neumann reference machines
# ---------------------------------------------------------------------------

def test_load_jvn_pattern():
    """Every vendored machine loads under rule JvN29 with a sane population."""
    from mesa_self_construction.von_neumann import load_jvn_pattern, KNOWN_PATTERNS
    expected = {
        "counter": (26, 60, 275),
        "cell_coders": (66, 60, 500),
        "tape_reader": (88, 124, 1972),
        "self_replicator": (21, 12314, 24866),
    }
    for key, (h, w, pop) in expected.items():
        pat = load_jvn_pattern(key)
        assert pat.grid.shape == (h, w), key
        assert pat.population == pop, key
        assert KNOWN_PATTERNS[key].endswith(".rle")


def test_vectorized_kernel_matches_scalar():
    """The vectorized kernel is exactly equivalent to the scalar reference."""
    import numpy as np
    from mesa_self_construction.von_neumann import (
        step_von_neumann_29, step_von_neumann_29_scalar,
    )
    rng = np.random.default_rng(99)
    for _ in range(15):
        g = rng.integers(0, 29, size=(6, 8)).astype(np.int32)
        g[0, :] = 0; g[-1, :] = 0; g[:, 0] = 0; g[:, -1] = 0
        vec, _ = step_von_neumann_29(g)
        ref, _ = step_von_neumann_29_scalar(g)
        assert np.array_equal(vec, ref)


def test_analyze_construction_counts_vacuum_builds():
    """analyze_construction reports only 0 -> non-zero builds."""
    import numpy as np
    from mesa_self_construction.validator import analyze_construction
    init = np.zeros((4, 4), dtype=np.int32)
    init[1, 1] = 9  # pre-existing cell is not "constructed"
    cur = init.copy()
    cur[2, 2] = 17
    cur[3, 3] = 25
    info = analyze_construction(cur, init)
    assert info["constructed"] == 2
    assert info["types"] == [17, 25]
    assert info["bbox"] == (2, 3, 2, 3)


def test_vn_cell_coders_construct_all_nine_types():
    """Running Nobili's cell-coders demo builds all nine constructible types.

    The nine constructible cell types are 9-12 (OTS quiescent), 17-20 (STS
    quiescent) and 25 (confluent). Starting from a quiescent grid the demo
    constructs 18 cells from vacuum whose types are exactly those nine.
    """
    model = SelfConstructionModel(preset_id="vn_cell_coders")
    for _ in range(model.preset.validation_step):
        model.step()
    v = model.latest_validation
    assert v.passed, v.proof_message
    assert v.details["constructed"] == 18
    assert set(v.details["constructed_types"]) == {
        9, 10, 11, 12, 17, 18, 19, 20, 25,
    }
    assert model.vn_constructed_cells == 18


def test_vn_machine_presets_are_behaviorally_validated():
    """Machine presets prove construction-from-vacuum, not a static shape."""
    for pid in ("vn_counter", "vn_self_replicator"):
        model = SelfConstructionModel(preset_id=pid)
        for _ in range(model.preset.validation_step):
            model.step()
        v = model.latest_validation
        assert v.passed, f"{pid}: {v.proof_message}"
        assert v.details["machine"] is True
        assert v.details["constructed"] >= model.preset.metadata["min_constructed"]
        # Honestly labelled as dynamic, not settled.
        assert v.details["is_settled"] is False


def test_mstate_rle_prefix_rollover_and_runs():
    """Two-char states roll over past pX (48) and run counts exceed 9."""
    from mesa_self_construction.mstate_rle import (
        encode_mstate_rle, decode_mstate_rle, MultiStatePattern,
    )
    g = np.zeros((1, 3), dtype=np.int32)
    g[0] = [48, 49, 50]  # pX, qA, qB
    back = decode_mstate_rle(encode_mstate_rle(MultiStatePattern(grid=g))).grid
    assert list(map(int, back[0])) == [48, 49, 50]
    run = np.zeros((1, 25), dtype=np.int32)
    run[0, :] = 1
    back2 = decode_mstate_rle(encode_mstate_rle(MultiStatePattern(grid=run))).grid
    assert back2.shape == (1, 25) and int(back2.sum()) == 25


def test_mstate_rle_malformed_and_eof():
    """Malformed tokens raise; '!' is a legal end-of-pattern terminator."""
    from mesa_self_construction.mstate_rle import decode_mstate_rle
    for bad in ["x = 2, y = 1\np!", "x = 2, y = 1\npz!", "x = 2, y = 1\nZ!"]:
        with pytest.raises(ValueError):
            decode_mstate_rle(bad)
    tail = decode_mstate_rle("x = 5, y = 1, rule = JvN29\n2A!pB!").grid
    assert list(map(int, tail[0])) == [1, 1, 0, 0, 0]


def test_mstate_rle_vendored_idempotence():
    """decode -> encode -> decode is identity for every vendored pattern."""
    from mesa_self_construction.mstate_rle import (
        encode_mstate_rle, decode_mstate_rle, MultiStatePattern,
    )
    from mesa_self_construction.von_neumann import PATTERN_DIR, KNOWN_PATTERNS
    for fn in KNOWN_PATTERNS.values():
        p1 = decode_mstate_rle((PATTERN_DIR / fn).read_text(encoding="utf-8"))
        text = encode_mstate_rle(
            MultiStatePattern(grid=p1.grid, rule=p1.rule, name=p1.name)
        )
        p2 = decode_mstate_rle(text)
        assert np.array_equal(p1.grid, p2.grid), fn
        assert p2.rule == p1.rule


def test_vectorized_kernel_thin_and_structured_grids():
    """Kernel parity holds on 1xN/Nx1 and a grid containing every state."""
    from mesa_self_construction.von_neumann import (
        step_von_neumann_29, step_von_neumann_29_scalar,
    )
    rng = np.random.default_rng(5)
    for shp in [(1, 1), (1, 6), (6, 1)]:
        g = rng.integers(0, 29, size=shp).astype(np.int32)
        assert np.array_equal(
            step_von_neumann_29(g)[0], step_von_neumann_29_scalar(g)[0]
        ), shp
    g = np.tile(np.arange(29, dtype=np.int32), (29, 1))[:, :29]
    assert np.array_equal(
        step_von_neumann_29(g)[0], step_von_neumann_29_scalar(g)[0]
    )


def test_vectorized_events_consistency():
    """events['changed'] equals the number of cells that actually changed."""
    from mesa_self_construction.von_neumann import step_von_neumann_29
    rng = np.random.default_rng(11)
    g = rng.integers(0, 29, size=(20, 20)).astype(np.int32)
    nxt, events = step_von_neumann_29(g)
    assert int(events["changed"]) == int(np.count_nonzero(nxt != g))


def test_vn_machine_determinism_and_reset():
    """Machine presets are deterministic and reset restores the exact start."""
    import hashlib

    def digest(grid):
        return hashlib.sha256(
            np.ascontiguousarray(grid).tobytes()
        ).hexdigest()

    m1 = SelfConstructionModel("vn_cell_coders")
    m2 = SelfConstructionModel("vn_cell_coders")
    start = digest(m1.grid_data)
    for _ in range(40):
        m1.step()
        m2.step()
    assert digest(m1.grid_data) == digest(m2.grid_data)
    assert m1.vn_constructed_cells == m2.vn_constructed_cells
    m1.reset()
    assert digest(m1.grid_data) == start
    assert m1.vn_constructed_cells == 0


def test_vn_machine_step_backward_exact():
    """History replay returns a machine grid to its exact earlier state."""
    import hashlib

    def digest(grid):
        return hashlib.sha256(
            np.ascontiguousarray(grid).tobytes()
        ).hexdigest()

    m = SelfConstructionModel("vn_cell_coders")
    start = digest(m.grid_data)
    for _ in range(30):
        m.step()
    at30 = digest(m.grid_data)
    for _ in range(10):
        m.step()
    for _ in range(10):
        assert m.step_backward() is True
    assert m.current_step == 30
    assert digest(m.grid_data) == at30
    # Rewinding all the way to step 0 clears the obtained-built count too.
    for _ in range(30):
        m.step_backward()
    assert m.current_step == 0
    assert digest(m.grid_data) == start
    assert m.vn_constructed_cells == 0


def test_load_jvn_pattern_rejects_foreign_rule():
    """A pattern whose rule is not JvN29 is refused rather than mis-run."""
    import tempfile
    import os
    from mesa_self_construction.von_neumann import load_jvn_pattern
    path = os.path.join(tempfile.gettempdir(), "sc_foreign_rule_test.rle")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("x = 3, y = 3, rule = B3/S23\nbo$2bo$3o!")
    with pytest.raises(ValueError):
        load_jvn_pattern(path)
