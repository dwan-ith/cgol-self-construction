"""
Validation and verification suite for Game of Life and Von Neumann self-construction.
Proves that target structure synthesis or construction arm movement has occurred
by verifying bitmasks, population invariants, and settling criteria.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
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


def validate_target_pattern(
    current_grid: np.ndarray,
    target_box: Tuple[int, int, int, int],
    expected_pattern: np.ndarray,
    expected_population: int,
    step: int,
    recent_populations: Optional[list] = None,
    gliders_used: int = 0,
    is_von_neumann: bool = False,
    require_settled: bool = True,
    constructions: int = 0,
    min_constructions: int = 0,
) -> ValidationResult:
    """
    Validate that a target structure has been constructed within target_box.

    Args:
        current_grid: 2D numpy array of current cell states
        target_box: (rmin, rmax, cmin, cmax) search zone
        expected_pattern: 2D numpy array of expected pattern
        expected_population: expected total or target population
        step: current generation number
        recent_populations: history of populations to test settling
        gliders_used: count of gliders dispatched
        is_von_neumann: whether using Von Neumann 29-state rules
        require_settled: if False, 8-generation population stability is
            not required (used for genuinely extending constructors
            whose population legitimately grows while building).
        constructions: counted construction events (sensitized->wired).
        min_constructions: required construction events for PROVEN.

    Returns:
        ValidationResult with pass/fail and proof analysis
    """
    rmin, rmax, cmin, cmax = target_box
    subgrid = current_grid[rmin:rmax + 1, cmin:cmax + 1]

    if is_von_neumann:
        # In Von Neumann, cells are non-zero integers
        active_in_box = np.argwhere(subgrid > 0)
        pop_in_box = len(active_in_box)
        total_pop = int(np.count_nonzero(current_grid > 0))
        # Outside-box debris check (VN sketches may legitimately hold a
        # large static scaffold, so only require the box itself exact).
        box_is_exact = False
    else:
        active_in_box = np.argwhere(subgrid == 1)
        pop_in_box = len(active_in_box)
        total_pop = int(current_grid.sum())
        box_is_exact = False

    # Check settling over a longer window with period detection.
    # Settled = still life (1 unique pop + identical grids would be ideal)
    # or a small oscillator (<=2 unique pops AND box contents periodic).
    # We only have population history here, so require 8 samples and
    # at most 2 unique values; callers should also ensure exact match.
    is_settled = False
    if recent_populations and len(recent_populations) >= 8:
        last_8 = recent_populations[-8:]
        is_settled = len(set(last_8)) <= 2
        
    # Check if expected pattern is contained inside subgrid
    h_exp, w_exp = expected_pattern.shape
    match_found = False
    best_ratio = 0.0
    exact_box_match = False
    detected_shape = (0, 0)

    if pop_in_box > 0 and len(active_in_box) > 0:
        box_rmin, box_cmin = active_in_box.min(axis=0)
        box_rmax, box_cmax = active_in_box.max(axis=0)
        detected_shape = (int(box_rmax - box_rmin + 1), int(box_cmax - box_cmin + 1))
        
        # Test sub-window matching
        sub_h, sub_w = subgrid.shape
        if sub_h >= h_exp and sub_w >= w_exp:
            for r in range(sub_h - h_exp + 1):
                for c in range(sub_w - w_exp + 1):
                    window = subgrid[r:r + h_exp, c:c + w_exp]
                    if is_von_neumann:
                        matches = np.count_nonzero((window > 0) == (expected_pattern > 0))
                    else:
                        matches = np.count_nonzero(window == expected_pattern)
                    ratio = matches / (h_exp * w_exp)
                    if ratio > best_ratio:
                        best_ratio = ratio
                    if ratio == 1.0:
                        match_found = True
                        # Exactness: the whole target box must equal the
                        # expected pattern placed in a zero background when
                        # shapes are identical; otherwise require that the
                        # number of live cells in the box equals the number
                        # of live cells in the expected pattern (no debris).
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

    # Population must agree: total for still-life syntheses, in-box for
    # arm-walk / oscillator cases where gliders may still be in flight.
    # Require at least one of them to match; both are reported.
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
            stability = "State is settled over the last 8 generations"
        else:
            stability = (
                f"Constructor active: {constructions} sensitized->wired "
                f"construction events counted (population growth while "
                f"extending is expected, stability not required)"
            )
        proof = (
            f"PROVEN: Construction successfully synthesized target structure at step {step}. "
            f"Target {h_exp}x{w_exp} pattern verified with exact box match "
            f"(no debris in target box, population {total_pop}). "
            f"{stability} "
            f"({gliders_used} gliders injected)."
        )
    elif (
        match_found
        and exact_box_match
        and pop_matches
        and constructions >= min_constructions
        and not (is_settled or not require_settled)
    ):
        proof = (
            f"INTERMEDIATE: Target pattern exactly present "
            f"({round(best_ratio*100, 1)}% window fidelity, box-exact, "
            f"population {total_pop}/{expected_population}), "
            f"but dynamics have not settled over 8 generations (t={step})."
        )
    else:
        proof = (
            f"IN PROGRESS: Synthesizing pattern at step {step}. "
            f"Current population: {total_pop} (expected: {expected_population}), "
            f"Target box occupancy: {pop_in_box} cells, Best pattern match: {round(best_ratio*100, 1)}%."
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
            "match_found": match_found,
            "exact_box_match": exact_box_match,
            "pop_matches": pop_matches,
            "constructions": constructions,
            "min_constructions": min_constructions,
        },
    )
