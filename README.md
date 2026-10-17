# Mesa Self-Construction

> **Executable, observable, and reproducible construction automata in Mesa.**
> Conway slow-salvo syntheses run on zero-border (non-toroidal) grids with
> strict exact-box validation. The von Neumann side is a faithful
> implementation of the **canonical 29-state automaton**, driven by the
> normative `@TABLE` transition data from Golly's `JvN29.rule` (vendored
> verbatim in `mesa_self_construction/data/JvN29.table` for auditability).
> Conway tooling is grounded in
> [`EZLiang/cgol-self-construction`](https://github.com/EZLiang/cgol-self-construction)
> (vendored under `cgol_ref/`, MIT, see `LICENSE`).

---

## Overview

Conventional Conway's Game of Life examples demonstrate how simple
deterministic rules generate complex emergence. **Mesa Self-Construction**
takes a step toward **algorithmic construction**:

> *Can a system made from identical, locally interacting components build
> persistent machinery out of quiescent matter?*

This project bridges two classic paradigms in cellular-automata theory:

1. **Conway's Game of Life (2 states, 8 neighbours)**
   - Uses **slow salvos** (timed glider packets) to operate a
     single-channel construction arm (`EZLiang` compilers `sc` / `oncoming`).
   - Gliders collide with an **elbow block** to nudge the arm and synthesize
     target structures with zero lingering debris.
2. **John von Neumann's self-reproducing automaton (29 states, 5 neighbours)**
   - The **full canonical state set**: an 8-state sensitization tree, ordinary
     (OTS) and special (STS) transmission states in quiescent/excited form,
     and the confluent delay organ — with synchronous, zero-border updates.
   - Rules are **loaded from a vendored copy of the reference transition
     table**, not hand-written, so the primitives execute the published
     automaton exactly.
   - Because the engine is faithful, it also **loads and runs von Neumann's
     own documented machines** (vendored, unmodified, from Golly's `JvN29`
     pattern collection): a counting circuit, the **cell coders that construct
     all nine constructible cell types**, a tape reader, and a **small
     self-replicator**. These are evaluated by *behaviour* — cells built out
     of vacuum — not by a hand-supplied target shape.

**Scope note.** On the Conway side this repository demonstrates *construction
primitives* and *pattern synthesis*: timed gliders extend an arm, relocate an
elbow, and assemble still lifes and oscillators with zero debris. On the von
Neumann side it executes the canonical automaton and runs von Neumann's
original circuits — including a genuine self-reproducing machine — as
vendored reference patterns (their construction program is von Neumann's, not
ours). What is *synthesised here* from scratch remains the building blocks
(pulse propagation, tip construction, erasure, delay, glider syntheses); a
full self-description/decoder tape authored in this repository is out of
scope.

---

## Key Features

- **Built on Mesa 3.x (`mesa.Model`, `PropertyLayer`, and `FixedAgent`)**
  - Zero-border Conway stepping (no torus wrap) for aimed glider collisions.
  - Dual execution backends: vectorized NumPy `PropertyLayer` path (default)
    and a discrete `FixedAgent` path (`mode="agent"`, Conway only,
    parity-tested).
  - Per-model schedule copies; exact history-restore step-back.
  - Live telemetry: injected / in-flight (template-matched) / consumed glider
    counts plus per-cell component classes (`target`, `glider`, `elbow`,
    `debris`, `empty`) synced to agents and `/api/state`.
- **Canonical von Neumann 29-state automaton (`von_neumann.py`)**
  - A parser + synchronous interpreter for `data/JvN29.table`: `C,N,E,S,W`
    input order, `symmetries:none`, **first matching rule wins**, off-grid
    neighbours read as the ground state (no torus wrap).
  - Canonical state grouping: `0` ground; `1–8` sensitization tree;
    `9–12` OTS quiescent; `13–16` OTS* excited; `17–20` STS quiescent;
    `21–24` STS* excited; `25–28` confluent (`C00→C01→C10→C00`).
  - `step_von_neumann_29` returns the next grid plus classified events
    (`changed` / `sensitized` / `killed`). A vectorized kernel packs each
    cell's five-neighbourhood into one integer key, evaluates every *distinct*
    neighbourhood exactly once, and maps it back (6–14× faster than the scalar
    reference); `step_von_neumann_29_scalar` is retained and parity-tested.
  - `load_jvn_pattern(name)` loads a vendored machine (`counter`,
    `cell_coders`, `tape_reader`, `self_replicator`) via the Golly
    **multi-state RLE** codec in `mstate_rle.py` (`A`–`X` = 1–24, `pA`–`pD` =
    25–28). `analyze_construction` / `validate_machine` then prove a machine
    by the cells it builds out of vacuum and the documented cell types.
- **EZLiang grounding (`salvo.py`)**
  - `parse_lane_phase_recipe` / `lane_to_position` implement the `oncoming.py`
    token format with perpendicular lane geometry (tested against the real
    recipe head `-5:1 -2:5 -7:5`).
  - `load_simeks_toolkit` parses `cgol_ref/compilers/data/simeks.txt`
    (including multi-token ops like `2xO 7move3`); the `0move-2` entry is
    asserted.
  - `from_separation_list` replays single-channel separation streams with
    exact release timing and glider accounting (channel-calibration caveat
    documented).
- **11 verified experiments** (exact target-box contents, matching population,
  and settling/period gates, or behavioural construction-from-vacuum proof):
  1. `block_synthesis` — De Novo 2-Glider Block Synthesis (SE + NW head-on → 2×2 still life).
  2. `elbow_shift` — Single-Channel Arm Nudge (glider + elbow block → clean relocate).
  3. `two_stage_arm` — Sequential Slow-Salvo Arm Walking (two timed pulses).
  4. `blinker_synthesis` — 2-Glider Oscillator Synthesis (clean period-2 blinker).
  5. `vn_wire_signal` — a pulse propagates along an ordinary wire (OTS = 9), one cell per tick.
  6. `vn_wire_extension` — a pulse sensitizes the blank tip and the tree builds a new OTS cell.
  7. `vn_special_kill` — an excited special cell (STS* = 21) erases a quiescent ordinary cell.
  8. `vn_confluent_delay` — the confluent organ holds a signal for a fixed number of ticks.
  9. `vn_counter` — Nobili's counting circuit, run from the vendored pattern (builds 8 cells from vacuum).
  10. `vn_cell_coders` — the nine coders construct **all nine constructible cell types**: 18 cells built from vacuum (types 9–12, 17–20, 25).
  11. `vn_self_replicator` — von Neumann's small self-reproducing machine; within a few hundred steps it constructs new excited-transmission and confluent cells (growth, not the finished copy, is proven).
- **Automated proof-of-construction engine (`validator.py`)**
  - Settling is decided by **full-grid period detection**
    (`detect_period`), not a population proxy: a still life is period 1 and a
    blinker is period 2; equal population with different cells is not
    settled.
  - `PROVEN` requires exact target-box contents, matching population, and a
    bounded period — or, for the deliberately dynamic VN primitives, counted
    construction events (`min_constructions`).
  - Reports `exact_box_match`, `pop_matches`, `match_found`, `period`, and
    `constructions` separately, so near-misses stay `IN PROGRESS` /
    `INTERMEDIATE`.
- **Interactive web dashboard & visualizer**
  - Dark glassmorphic canvas interface with smooth pan and zoom.
  - Live glider telemetry, trajectory ray indicators, active cell glows, and
    population sparklines.
  - Hover any cell to inspect its live state (Conway alive/dead; canonical VN
    state name).
  - VN construction-front marker.
  - Step-forward, step-backward (history replay), fast-forward to proof, and
    speed control.

---

## Architecture

```
mesa-self-construction/
├── mesa_self_construction/         # Core Python simulation package
│   ├── __init__.py                 # Exports model, agents, presets, validator, JvN29 API
│   ├── data/
│   │   ├── JvN29.table             # Canonical von Neumann transition table (vendored)
│   │   └── patterns/               # Vendored Golly JvN29 machines (counter, coders, …)
│   ├── model.py                    # Mesa Model with PropertyLayer & Agent modes
│   ├── agents.py                   # Discrete-space FixedAgent cell implementation
│   ├── presets.py                  # Verified experiment presets (Conway, VN primitives, machines)
│   ├── salvo.py                    # Slow-salvo + single-channel scheduler, simeks toolkit
│   ├── von_neumann.py              # JvN29 parser + vectorized interpreter + pattern loader
│   ├── mstate_rle.py               # Golly multi-state RLE codec (states 0–28)
│   ├── validator.py                # Exact-box verification, period & construction analysis
│   └── rle.py                      # Life RLE parser and serializer
├── web/                            # Interactive frontend
│   ├── index.html                  # Semantic dashboard structure
│   ├── index.css                   # Modern dark design system
│   └── app.js                      # HTML5 Canvas renderer, pan/zoom, HUD telemetry
├── tests/                          # Automated pytest suite
│   ├── test_model.py               # Presets, JvN29 golden/behavioural tests, RLE, step-back
│   └── test_api.py                 # REST API endpoint verification
├── app.py                          # FastAPI application & server
├── run_experiments.py              # Headless CLI benchmark & validation runner
└── pyproject.toml                  # Packaging & dependencies
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install mesa fastapi uvicorn numpy scipy pydantic pytest httpx
```

### 2. Launch the interactive visualizer
```bash
python app.py
```
Open your browser at **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

### 3. Run automated CLI benchmarks
```bash
python run_experiments.py
```

### 4. Run the pytest suite
```bash
python -m pytest tests/ -v
```

---

## Benchmark Results

All 11 experiments pass strict validation (exact box + population +
period/construction gate, or behavioural construction-from-vacuum for the
machines). Timings are machine-dependent; rerun with
`python run_experiments.py` (which also reports generations, cell-updates,
Mcell-updates/s, glider accounting, VN event counts, and cells built from
vacuum):

| ID | Category | Validation Step | Outcome | Proof Status |
|:---|:---|:---:|:---:|:---:|
| `block_synthesis` | Conway's Life | 50 | 100% exact | **PROVEN** (period 1) |
| `elbow_shift` | Conway's Life Arm | 82 | 100% exact | **PROVEN** (period 1) |
| `two_stage_arm` | Conway's Life Arm | 150 | 100% exact | **PROVEN** (period 1) |
| `blinker_synthesis` | Conway's Life | 41 | 100% exact | **PROVEN** (period 2) |
| `vn_wire_signal` | von Neumann | 10 | 100% exact | **PROVEN** (dynamic primitive) |
| `vn_wire_extension` | von Neumann | 12 | 100% exact | **PROVEN** (period 1) |
| `vn_special_kill` | von Neumann | 3 | 100% exact | **PROVEN** (period 1) |
| `vn_confluent_delay` | von Neumann | 1 | 100% exact | **PROVEN** (dynamic primitive) |
| `vn_counter` | VN machine | 1000 | 8 cells built | **PROVEN** (dynamic machine) |
| `vn_cell_coders` | VN machine | 80 | 18 cells, all 9 types | **PROVEN** (period 1) |
| `vn_self_replicator` | VN machine | 300 | 8 cells built | **PROVEN** (dynamic machine) |

---

## Theoretical Comparison

| Dimension | Conway's Game of Life | von Neumann 29-state (this repo) |
|:---|:---|:---|
| **Cell states** | 2 (dead/alive) | 29 (ground, sensitization tree, OTS/OTS*, STS/STS*, confluent) |
| **Neighbourhood** | 8 (Moore) | 5 (orthogonal von Neumann) |
| **Rule source** | Conway's B3/S23 | vendored `data/JvN29.table` (Golly `JvN29.rule`) |
| **Construction arm** | glider slow-salvo hitting an elbow block | wire tip built by the sensitization tree; cells erased by STS/OTS interaction |
| **Information carrier** | glider separations and phase offsets | directional excitation pulses (ordinary/special kinds) |
| **Stored description** | implicit in salvo timing | von Neumann's vendored machines carry a description tape (run as-is); a tape authored in this repo is out of scope |
| **Key reference** | `EZLiang/cgol-self-construction` lane-phase scheduling + simeks single-channel replay | von Neumann (1966) / Burks — canonical transitions implemented; decoder and self-reproduction programs out of scope |

### Pattern synthesis is not self-reproduction

A pattern that merely emits repeated shapes (a glider gun, or a pulse
travelling down a wire) leaves no new machine behind. This repository counts
*construction* only when quiescent matter becomes persistent machinery:

- **Conway side:** two gliders annihilate into a still-life block that
  persists with zero debris (`block_synthesis`), or an elbow block is
  relocated and persists (`elbow_shift`, `two_stage_arm`). The blinker
  (`blinker_synthesis`) is a persistent oscillator synthesised de novo.
- **von Neumann side:** `vn_wire_extension` turns a blank tip into a new
  transmission cell that persists; `vn_wire_signal` and `vn_confluent_delay`
  are shown as deliberately dynamic primitives (a travelling pulse, a delay
  organ) and are labelled as such. `vn_cell_coders` builds **all nine
  constructible cell types** from vacuum (18 new cells), and
  `vn_self_replicator` runs von Neumann's own self-reproducing machine, whose
  construction of new excited-transmission and confluent cells we observe.

The distinctions this repo cares about are: (a) *our* synthesised
constructions are primitives, not a self-describing program; and (b) running
von Neumann's vendored machine is evidence the engine is faithful, not a claim
that we authored a self-reproducer.

---

## Reproducibility

Every experiment is a frozen initial configuration plus a deterministic
synchronous update (no randomness after setup). To reproduce:

```bash
python -m pytest tests/ -q   # 45 checks
python run_experiments.py    # headless re-run of all 11 validations
```

| Check | What it pins |
|:---|:---|
| `test_glider_directions` | All 4 glider orientations move one diagonal cell per 4-step period on zero-border grids |
| `test_simeks_toolkit_and_recipe_replay` | `simeks.txt` parses (incl. `0move-2`); real `oncoming.py` tokens `-5:1 -2:5 -7:5` schedule and inject end-to-end |
| `test_single_channel_separation_replay` | simeks separation gaps become cumulative release steps on one lane; replay executes with exact glider accounting |
| `test_jvn29_reference_table_parses` | Vendored table loads with 102 transitions / 29 states; ground sensitized by a single excited input |
| `test_jvn29_golden_transitions` | Hand-checked neighbourhood→state cases: sensitization tree, OTS propagation/decay, STS erase, confluent cycle, unlisted⇒unchanged |
| `test_jvn29_kernel_matches_independent_evaluator` | Production kernel agrees with a separately written rule evaluator over random grids |
| `test_jvn29_wire_propagation` / `test_vn_off_grid_is_vacuum` | Pulse advances one cell per tick; off-grid neighbours are ground (no torus wrap) |
| `test_validator_period_detection` | Full-grid periodicity: period 1/2/3 recognised; equal population with different cells is not settled |
| `test_mstate_rle_roundtrip` / `_high_state_encoding` / `_prefix_rollover_and_runs` | Golly multi-state RLE is lossless over all 29 states; `pA`–`pD` = 25–28 and `pX`/`qA` roll over; run counts exceed 9 |
| `test_mstate_rle_malformed_and_eof` | Malformed tokens raise; `!` is a legal end-of-pattern terminator |
| `test_mstate_rle_vendored_idempotence` | `decode → encode → decode` is the identity for every vendored pattern |
| `test_vectorized_kernel_matches_scalar` / `_thin_and_structured_grids` / `_events_consistency` | Vectorized JvN29 kernel is exactly equivalent to the scalar reference (incl. 1×N/N×1 and all-state grids); `events['changed']` is exact |
| `test_load_jvn_pattern` / `_rejects_foreign_rule` | Every vendored machine loads under rule `JvN29`; a non-`JvN29` pattern is refused |
| `test_analyze_construction_counts_vacuum_builds` | `analyze_construction` counts only genuine `0 → non-zero` builds |
| `test_vn_cell_coders_construct_all_nine_types` | Running the cell-coders demo builds exactly 18 cells from vacuum covering all nine constructible types |
| `test_vn_machine_presets_are_behaviorally_validated` | Machine presets prove construction-from-vacuum and are honestly labelled dynamic (not settled) |
| `test_vn_machine_determinism_and_reset` / `_step_backward_exact` | Machines are bit-for-bit deterministic; reset and history replay restore exact grids (and the built-cell count) |
| `test_api_*` (incl. `_unknown_preset_404`, `_step_is_clamped`, `_all_presets_run_to_validation`) | REST surface: all 11 presets load/step/reset and prove; unknown preset is 404; step is capped at 100 |
| Per-preset tests | Exact target-box contents, population, period/construction gates at the documented step |

Initial configurations live in `mesa_self_construction/presets.py`
(`make_*_preset`), the canonical transition table in
`mesa_self_construction/data/JvN29.table` (loaded by `von_neumann.py`), and
lane geometry in `mesa_self_construction/salvo.py` (`lane_to_position`).

---

## License

MIT License — see `LICENSE`.

- `cgol_ref/` is a vendored copy of
  [`EZLiang/cgol-self-construction`](https://github.com/EZLiang/cgol-self-construction)
  (MIT); see `cgol_ref/LICENSE`.
- `mesa_self_construction/data/JvN29.table` reproduces the normative `@TABLE`
  data from Golly's `JvN29.rule` for von Neumann's 29-state automaton,
  retrieved via [`EZLiang/vonNeumannTuringMachine`](https://github.com/EZLiang/vonNeumannTuringMachine)
  (MIT), itself derived from von Neumann's *Theory of Self-Reproducing
  Automata* (A. W. Burks, ed., 1966).
- `mesa_self_construction/data/patterns/*.rle` are vendored, unmodified from
  Golly's standard pattern collection (`Patterns/Self-Rep/JvN/`), authored by
  Renato Nobili and others; see `data/patterns/PROVENANCE.md`.

Built for the Mesa agent-based modeling community.
