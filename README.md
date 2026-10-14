# Mesa Self-Construction

> **Executable, observable, and reproducible self-constructing cellular automata in Mesa.**
> Conway slow-salvo syntheses run on zero-border (non-toroidal) grids with
> strict exact-box validation; Von Neumann entries run a documented
> **29-state transition system** with counted sensitized→wired construction
> events, including a tape-description reader. Grounded in the slow-salvo
> construction tooling of
> [`EZLiang/cgol-self-construction`](https://github.com/EZLiang/cgol-self-construction)
> (vendored under `cgol_ref/`, MIT, see `LICENSE`).

---

## 🌟 Overview

Conventional Conway's Game of Life examples demonstrate how simple deterministic rules generate complex emergence. **Mesa Self-Construction** takes a fundamental leap forward into **algorithmic construction and self-reproduction**:

> *Can a system made from identical, locally interacting components construct another instance of itself or execute instructions to build arbitrary target machines?*

This project bridges two monumental paradigms in cellular automata theory:
1. **Conway's Game of Life Self-Construction (2 States, 8 Neighbors)**:
   - Uses **slow salvos** (sequences of timed glider collision packets) to operate single-channel construction arms (`EZLiang` compilers: `sc` and `oncoming`).
   - Gliders collide with an **elbow block** to shift the arm (`0move`), emit secondary gliders, and synthesize target structures with zero lingering debris.
2. **John von Neumann's Self-Reproducing Automaton (29 States, 5 Neighbors)**:
   - Implements the documented 29-state transition system (directional ordinary/special transmission, confluent latch+fire, sensitized construct/kill) with synchronous updates.
   - A looped instruction tape feeds a confluent junction and an extending arm that solidifies sensitized cells into daughter machinery (counted events, edge-margin verified).
   - A description-reader preset copies a stored 3-bit tape into built cells — the read-and-build mechanism universal construction scales up (full decoder programs remain out of scope).

---

## 🚀 Key Features

- **Built on Mesa 3.x (`mesa.Model`, `PropertyLayer`, and `FixedAgent`)**:
  - Zero-border Conway stepping (no torus wrap) for aimed glider collisions.
  - Dual execution backends: vectorized NumPy `PropertyLayer` path (default)
    and discrete `FixedAgent` path (`mode="agent"`, Conway only, parity-tested).
  - Per-model schedule copies; exact history-restore step-back.
  - Live telemetry: injected / in-flight (template-matched) / consumed
    glider counts plus per-cell component classes (`target`, `glider`,
    `elbow`, `debris`, `empty`) synced to agents and `/api/state`.
- **EZLiang grounding (`salvo.py`)**:
  - `parse_lane_phase_recipe` / `lane_to_position` implement the
    `oncoming.py` token format with perpendicular lane geometry
    (documented, tested against the real recipe head `-5:1 -2:5 -7:5`).
  - `load_simeks_toolkit` parses `cgol_ref/compilers/data/simeks.txt`
    (including multi-token ops like `2xO 7move3`); `0move-2` entry asserted.
  - `from_separation_list` replays single-channel separation streams with
    exact release timing and glider accounting (channel-calibration caveat
    documented: uncalibrated elbows execute but don't reproduce named vectors).
- **6 Verified Construction Experiments** (all require exact target-box
  contents, matching population, and settling/construction gates):
1. `block_synthesis`: De Novo 2-Glider Block Synthesis (SE + NW head-on -> 2x2 still life).
2. `elbow_shift`: Single-Channel Arm Nudge (glider + elbow block -> clean relocate).
3. `two_stage_arm`: Sequential Slow-Salvo Arm Walking (two timed pulses).
4. `blinker_synthesis`: 2-Glider Oscillator Synthesis (clean period-2 blinker, pop 3).
5. `von_neumann_constructor`: Von Neumann 29-State Constructor Demo — tape loop + confluent latch feed an arm that genuinely extends (16 counted construction events by step 50, 27-cell edge margin).
6. `vn_description_reader`: Von Neumann Description Reader — stored tape [O,S,O] copied into built cells [1,9,1] (3 constructions, settled).
- **Automated Proof-of-Construction Engine (`validator.py`)**:
  - Requires exact target-box contents (no debris), matching population
    (total + in-box), and 8-generation settling before `PROVEN` — or, for
    genuinely extending constructors, counted construction events
    (`min_constructions`) instead of stability.
  - Reports `exact_box_match`, `pop_matches`, `match_found`,
    `constructions` separately so near-misses stay `IN PROGRESS` / `INTERMEDIATE`.
- **Interactive Web Dashboard & Visualizer**:
  - Dark glassmorphic canvas interface with smooth pan and zoom.
  - Live glider telemetry, trajectory ray indicators, active cell glows, and population sparklines.
  - Hover any cell to inspect its live state (Conway alive/dead; VN 29-state name).
  - VN construction-front (◇ BUILD FRONT) and confluent-junction markers.
  - Step-forward, step-backward (history replay), fast-forward to proof, and speed control.

---

## 🏗️ Architecture

```
mesa-self-construction/
├── mesa_self_construction/         # Core Python simulation package
│   ├── __init__.py                 # Exports model, agents, presets, validator
│   ├── model.py                    # Mesa Model with PropertyLayer & Agent modes
│   ├── agents.py                   # Discrete space FixedAgent cell implementation
│   ├── presets.py                  # Verified experiment presets (Conway & Von Neumann)
│   ├── salvo.py                    # Slow-salvo + single-channel scheduler, simeks toolkit
│   ├── von_neumann.py              # 29-state table + synchronous transition function
│   ├── validator.py                # Automated bitmask verification & proof engine
│   └── rle.py                      # Life RLE parser and serializer
├── web/                            # Interactive Frontend
│   ├── index.html                  # Semantic dashboard structure
│   ├── index.css                   # Modern dark design system
│   └── app.js                      # HTML5 Canvas renderer, pan/zoom, HUD telemetry
├── tests/                          # Automated Pytest suite
│   ├── test_model.py               # Unit tests for all presets, RLE, and reverse stepping
│   └── test_api.py                 # REST API endpoint verification
├── app.py                          # FastAPI application & server
├── run_experiments.py              # Headless CLI benchmark & validation runner
└── pyproject.toml                  # Packaging & dependencies
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install mesa fastapi uvicorn numpy scipy pydantic pytest httpx
```

### 2. Launch the Interactive Visualizer
```bash
python app.py
```
Open your browser at **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

### 3. Run Automated CLI Benchmarks
```bash
python run_experiments.py
```

### 4. Run Pytest Suite
```bash
python -m pytest tests/ -v
```

---

## 🔬 Benchmark Results

All 6 experiments pass strict validation (exact box + population + settling/constructions).
Timings are machine-dependent; rerun with `python run_experiments.py`
(which also reports generations, cell-updates, Mcell-updates/s, glider
accounting, and construction-event counts per experiment):

| ID | Category | Target Validation Step | Pattern Match | Proof Status |
|:---|:---|:---:|:---:|:---:|
| `block_synthesis` | Conway's Life | Step 50 | **100.0% exact** | **PROVEN** |
| `elbow_shift` | Conway's Life Arm | Step 82 | **100.0% exact** | **PROVEN** |
| `two_stage_arm` | Conway's Life Arm | Step 150 | **100.0% exact** | **PROVEN** |
| `blinker_synthesis` | Conway's Life | Step 41 | **100.0% exact** | **PROVEN** |
| `von_neumann_constructor`| Von Neumann 29-State | Step 50 | **100.0% exact** | **PROVEN (16 constructions)** |
| `vn_description_reader`| Von Neumann 29-State | Step 16 | **100.0% exact** | **PROVEN (3 constructions)** |

---

## 📖 Theoretical Comparison

| Dimension | Conway's Game of Life Self-Construction | Von Neumann 29-State (this repo) |
|:---|:---|:---|
| **Cell States** | 2 (0: Dead, 1: Alive) | 29, documented in `von_neumann.py` (ordinary/special transmission, confluent latch/fire, sensitized construct/kill) |
| **Neighborhood** | 8 cells (Moore) | 5 cells (Orthogonal Von Neumann) |
| **Construction Arm** | Glider slow-salvo hitting an elbow block | Wired arm extended by sensitized→cell construction events (counted) |
| **Information Carrier** | Glider separations and phase offsets | Directional excitation pulses (ordinary/special kinds) |
| **Stored Description** | Implicit in salvo timing (this repo) | Explicit 3-bit tape in `vn_description_reader`, copied into built cells |
| **Key Reference** | `EZLiang/cgol-self-construction` lane-phase scheduling + simeks single-channel replay; full `sc` calibration out of scope | Von Neumann (1966) / Burks — transitions + description-driven build implemented; full decoder/self-reproduction programs out of scope |

### Self-reproduction vs. repeated shapes

A pattern that merely generates repeated shapes (e.g., a glider gun
emitting gliders, or a pulse train traveling down a wire) is **not**
self-construction: the output is a transient signal, and no new machine
is left behind. This repo counts construction only when quiescent matter
is converted into persistent wired machinery:

- Conway side: two gliders annihilate into a still-life block that
  persists with zero debris (`block_synthesis`), or an elbow block is
  relocated and persists (`elbow_shift`, `two_stage_arm`).
- Von Neumann side: sensitized cells solidify into transmission cells
  that persist and relay further signals (`vn_constructed_cells`), and
  in `vn_description_reader` the built cells match a stored description
  bit-for-bit. Mere pulse circulation without solidification scores zero
  constructions and cannot pass.

---

## 🧪 Reproducibility

Every experiment is a frozen initial configuration plus a deterministic
synchronous update (no randomness after setup). To reproduce:

```bash
python -m pytest tests/ -v   # 22 checks: rules, recipes, presets, API
python run_experiments.py     # headless re-run of all 6 validations
```

| Check | What it pins |
|:---|:---|
| `test_glider_directions` | All 4 glider orientations move one diagonal cell per 4-step period on zero-border grids |
| `test_simeks_toolkit_and_recipe_replay` | `simeks.txt` parses (incl. `0move-2`); real `oncoming.py` tokens `-5:1 -2:5 -7:5` schedule and inject end-to-end |
| `test_single_channel_separation_replay` | simeks separation gaps become cumulative release steps on one lane; replay executes with exact glider accounting |
| `test_von_neumann_29_state_rules` | Directional feed, kind isolation, confluent `C→C1→C*` delay, sensitize→construct/kill, ambiguity refusal |
| Per-preset tests | Exact target-box contents, population, settling/construction gates at the documented validation step |
| `test_vn_construction_keeps_boundary_margin` | Arm stays ≥4 cells from every edge, so boundaries never interact |

Initial configurations live in `mesa_self_construction/presets.py`
(`make_*_preset`), the 29-state table in
`mesa_self_construction/von_neumann.py`, and lane geometry in
`mesa_self_construction/salvo.py` (`lane_to_position`).

---

## 📜 License

MIT License — see `LICENSE`. `cgol_ref/` remains under EZLiang's MIT
(see `cgol_ref/LICENSE` and `LICENSE`). Developed for the Mesa
agent-based modeling community.
