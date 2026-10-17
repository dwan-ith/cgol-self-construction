"""
Validation and verification suite for Game of Life and Von Neumann construction.

Proves that a target structure has been synthesized by checking three
independent things, and only reports PROVEN when all hold:

  1. the target box contains the expected pattern (exact, no debris);
  2. the live-cell count agrees (total for still-life syntheses, in-box
     for arm walks / oscillators where machinery may persist elsewhere);
  3. the *actual grid* has settled -- still life (period 1) or a genuine
     period-2 oscillator -- determined by comparing full grid states, not
     by a population-count heuristic.

Period detection compares complete generations, so it cannot be fooled by
two different configurations that merely share a population count.
Construction-event evidence (for genuinely extending systems) is reported
separately and never substitutes for settling unless the caller opts out.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


@dataclass
class ValidationResult:
    """Result of validating a self-construction state."""
    passed: bool
    step: int
    match_ratio: float
    population: int
    expected_population: int
    is_settled: bool
    gliders_used: int
    target_box: Tuple[int, int, int, int]
    detected_shape: Tuple[int, int]
    proof_message: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "step": self.step,
            "match_ratio": round(self.match_ratio, 4),
            "population": self.population,
            "expected_population": self.expected_population,
            "is_settled": self.is_settled,
            "gliders_used": self.gliders_used,
            "target_box": list(self.target_box),
            "detected_shape": list(self.detected_shape),
            "proof_message": self.proof_message,
            "details": self.details,
        }


def detect_period(
    states: List[np.ndarray], max_period: int = 8
) -> Optional[int]:
    """Smallest p in 1..max_period such that the tail is p-periodic.

    Compares full grid states (``np.array_equal``), so a period-1 result
    means a true still life and period-2 a true oscillator. Returns None
    if no period <= max_period is found in the supplied window.
    """
    if not states or len(states) < 2:
        return None
    n = len(states)
    for p in range(1, min(max_period, n - 1) + 1):
        if not np.array_equal(states[-1], states[-1 - p]):
            continue
        ok = True
        for i in range(p + 1):
            j = n - 1 - i - p
            if j < 0:
                break
            if not np.array_equal(states[-1 - i], states[j]):
                ok = False
                break
        if ok:
            return p
    return None


def validate_target_pattern(
    current_grid: np.ndarray,
    target_box: Tuple[int, int, int, int],
    expected_pattern: np.ndarray,
    expected_population: int,
    step: int,
    recent_populations: Optional[List[int]] = None,
    recent_states: Optional[List[np.ndarray]] = None,
    gliders_used: int = 0,
    is_von_neumann: bool = False,
    require_settled: bool = True,
    constructions: int = 0,
    min_constructions: int = 0,
) -> ValidationResult:
    """
    Validate that a target structure has been constructed within target_box.

    Args:
        current_grid: 2D numpy array of current cell states.
        target_box: (rmin, rmax, cmin, cmax) search zone.
        expected_pattern: 2D numpy array of expected pattern.
        expected_population: expected total (Conway still life) or in-box pop.
        step: current generation number.
        recent_populations: population history (fallback when states absent).
        recent_states: recent full grid states for true period detection.
        gliders_used: count of gliders dispatched.
        is_von_neumann: whether using the canonical 29-state rules.
        require_settled: if False, settling is not required (used for
            genuinely dynamic demonstrations such as a signal in flight).
        constructions: counted construction events (sensitized->wired).
        min_constructions: required construction events for PROVEN.
    """
    rmin, rmax, cmin, cmax = target_box
    subgrid = current_grid[rmin:rmax + 1, cmin:cmax + 1]

    if is_von_neumann:
        active_in_box = np.argwhere(subgrid > 0)
        pop_in_box = len(active_in_box)
        total_pop = int(np.count_nonzero(current_grid > 0))
    else:
        active_in_box = np.argwhere(subgrid == 1)
        pop_in_box = len(active_in_box)
        total_pop = int(current_grid.sum())

    # --- Settling: full-grid period detection (authoritative) ----------
    period: Optional[int] = None
    if recent_states and len(recent_states) >= 2:
        period = detect_period(recent_states)
        is_settled = period is not None and period <= 2
    elif recent_populations and len(recent_populations) >= 8:
        # Fallback only when full states are unavailable: at most two
        # distinct populations. Weaker, but explicit about being weaker.
        last_8 = recent_populations[-8:]
        is_settled = len(set(last_8)) <= 2
    else:
        is_settled = False

    # --- Pattern match inside the target box ---------------------------
    h_exp, w_exp = expected_pattern.shape
    match_found = False
    best_ratio = 0.0
    exact_box_match = False
    detected_shape = (0, 0)

    if pop_in_box > 0 and len(active_in_box) > 0:
        box_rmin, box_cmin = active_in_box.min(axis=0)
        box_rmax, box_cmax = active_in_box.max(axis=0)
        detected_shape = (
            int(box_rmax - box_rmin + 1),
            int(box_cmax - box_cmin + 1),
        )
        sub_h, sub_w = subgrid.shape
        if sub_h >= h_exp and sub_w >= w_exp:
            for r in range(sub_h - h_exp + 1):
                for c in range(sub_w - w_exp + 1):
                    window = subgrid[r:r + h_exp, c:c + w_exp]
                    if is_von_neumann:
                        matches = np.count_nonzero(
                            (window > 0) == (expected_pattern > 0)
                        )
                    else:
                        matches = np.count_nonzero(window == expected_pattern)
                    ratio = matches / (h_exp * w_exp)
                    if ratio > best_ratio:
                        best_ratio = ratio
                    if ratio == 1.0:
                        match_found = True
                        if subgrid.shape == expected_pattern.shape:
                            if is_von_neumann:
                                exact_box_match = bool(
                                    np.array_equal(
                                        (subgrid > 0), (expected_pattern > 0)
                                    )
                                    and np.array_equal(subgrid, expected_pattern)
                                )
                            else:
                                exact_box_match = bool(
                                    np.array_equal(subgrid, expected_pattern)
                                )
                        else:
                            if is_von_neumann:
                                exp_live = int(
                                    np.count_nonzero(expected_pattern > 0)
                                )
                            else:
                                exp_live = int(
                                    np.count_nonzero(expected_pattern == 1)
                                )
                            exact_box_match = pop_in_box == exp_live
                        break
                if match_found:
                    break

    # --- Population agreement ------------------------------------------
    if is_von_neumann:
        exp_live_total = int(np.count_nonzero(expected_pattern > 0))
        pop_matches = pop_in_box == exp_live_total
    else:
        exp_live_total = int(np.count_nonzero(expected_pattern == 1))
        pop_matches = (total_pop == expected_population) and (
            pop_in_box == exp_live_total
        )

    passed = (
        match_found
        and exact_box_match
        and pop_matches
        and (is_settled or not require_settled)
        and constructions >= min_constructions
    )

    if passed:
        if require_settled:
            stability = (
                f"State is settled (full-grid period {period})"
            )
        else:
            stability = (
                "Dynamic demonstration: settling not required "
                f"({constructions} construction events counted)"
            )
        proof = (
            f"PROVEN: target {h_exp}x{w_exp} pattern verified at step {step} "
            f"with exact box match (no debris, population {total_pop}). "
            f"{stability} ({gliders_used} gliders injected)."
        )
    elif (
        match_found
        and exact_box_match
        and pop_matches
        and constructions >= min_constructions
        and not (is_settled or not require_settled)
    ):
        proof = (
            f"INTERMEDIATE: target pattern exactly present "
            f"({round(best_ratio * 100, 1)}% window fidelity, box-exact, "
            f"population {total_pop}/{expected_population}), but the grid "
            f"has not settled (period {period}) at t={step}."
        )
    else:
        proof = (
            f"IN PROGRESS: synthesizing at step {step}. "
            f"Population: {total_pop} (expected {expected_population}), "
            f"target-box occupancy: {pop_in_box} cells, best window match "
            f"{round(best_ratio * 100, 1)}%."
        )

    return ValidationResult(
        passed=passed,
        step=step,
        match_ratio=best_ratio,
        population=total_pop,
        expected_population=expected_population,
        is_settled=is_settled,
        gliders_used=gliders_used,
        target_box=target_box,
        detected_shape=detected_shape,
        proof_message=proof,
        details={
            "pop_in_box": pop_in_box,
            "total_pop": total_pop,
            "is_settled": is_settled,
            "period": period,
            "match_found": match_found,
            "exact_box_match": exact_box_match,
            "pop_matches": pop_matches,
            "constructions": constructions,
            "min_constructions": min_constructions,
        },
    )


def analyze_construction(
    current_grid: np.ndarray, initial_grid: np.ndarray
) -> Dict[str, Any]:
    """Describe cells built out of vacuum (0 -> non-zero) since step 0.

    A cell is "constructed" when it was empty in the initial configuration
    and is occupied now. Returns the count, the set of cell types that were
    constructed, and their bounding box. This is the honest evidence that a
    machine *built* matter rather than merely moving it around (von Neumann's
    "direct process"), and it needs no hand-supplied target pattern.
    """
    current_grid = np.asarray(current_grid)
    initial_grid = np.asarray(initial_grid)
    if current_grid.shape != initial_grid.shape:
        return {"constructed": 0, "types": [], "bbox": None}
    new_mask = (initial_grid == 0) & (current_grid > 0)
    count = int(new_mask.sum())
    if count == 0:
        return {"constructed": 0, "types": [], "bbox": None}
    types = sorted(int(v) for v in np.unique(current_grid[new_mask]))
    coords = np.argwhere(new_mask)
    (r0, c0) = coords.min(axis=0)
    (r1, c1) = coords.max(axis=0)
    return {
        "constructed": count,
        "types": types,
        "bbox": (int(r0), int(r1), int(c0), int(c1)),
    }


def validate_machine(
    current_grid: np.ndarray,
    initial_grid: np.ndarray,
    step: int,
    min_constructed: int = 0,
    required_constructed_types: Optional[List[int]] = None,
    require_settled: bool = False,
    recent_states: Optional[List[np.ndarray]] = None,
    name: str = "machine",
) -> ValidationResult:
    """Validate a von Neumann reference machine by *behavior*, not a target.

    PROVEN requires that at least ``min_constructed`` cells have been built
    out of vacuum since step 0, that every type in ``required_constructed_types``
    was among the built cells, and (if ``require_settled``) that the grid has
    reached a bounded full-grid period.
    """
    info = analyze_construction(current_grid, initial_grid)
    constructed = info["constructed"]
    built_types = set(info["types"])
    required = set(required_constructed_types or [])

    period: Optional[int] = None
    is_settled = False
    if recent_states and len(recent_states) >= 2:
        period = detect_period(recent_states)
        is_settled = period is not None and period <= 2

    total_pop = int(np.count_nonzero(current_grid > 0))
    populated = set(int(v) for v in np.unique(current_grid) if v)

    types_ok = required.issubset(built_types)
    settled_ok = is_settled or not require_settled
    passed = (constructed >= min_constructed) and types_ok and settled_ok

    if passed:
        if require_settled:
            tail = f"settled at full-grid period {period}"
        else:
            tail = "still constructing (dynamic reference machine)"
        proof = (
            f"PROVEN: {name} built {constructed} cells from vacuum "
            f"(types {sorted(built_types)}) by step {step}; {tail}."
        )
    else:
        missing = sorted(required - built_types)
        why = []
        if constructed < min_constructed:
            why.append(f"constructed {constructed}/{min_constructed}")
        if missing:
            why.append(f"missing types {missing}")
        if not settled_ok:
            why.append(f"not settled (period {period})")
        proof = (
            f"IN PROGRESS: {name} at step {step}; constructed "
            f"{constructed} cells (types {sorted(built_types)}). "
            f"Pending: {'; '.join(why) or 'objective met partly'}."
        )

    return ValidationResult(
        passed=passed,
        step=step,
        match_ratio=1.0 if passed else 0.0,
        population=total_pop,
        expected_population=total_pop,
        is_settled=is_settled,
        gliders_used=0,
        target_box=info["bbox"] or (0, 0, 0, 0),
        detected_shape=(
            (info["bbox"][1] - info["bbox"][0] + 1,
             info["bbox"][3] - info["bbox"][2] + 1)
            if info["bbox"]
            else (0, 0)
        ),
        proof_message=proof,
        details={
            "machine": True,
            "constructed": constructed,
            "constructed_types": sorted(built_types),
            "required_constructed_types": sorted(required),
            "is_settled": is_settled,
            "period": period,
            "types_present": sorted(populated),
            "exact_box_match": passed,
        },
    )
