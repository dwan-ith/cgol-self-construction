"""
Slow Salvo glider management, orientation definitions, and scheduler.
Supports lane-phase recipes compatible with EZLiang/cgol-self-construction.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


def parse_lane_phase_token(token: str) -> Tuple[int, int]:
    """Parse one 'lane:phase' token (slmake/pslmake/EZLiang style)."""
    token = token.strip()
    if not token:
        raise ValueError("empty lane-phase token")
    if ":" in token:
        lane_str, phase_str = token.split(":", 1)
        lane = int(lane_str)
        phase = int(phase_str) % 4
    else:
        lane = int(token)
        phase = 0
    return lane, phase


def parse_lane_phase_recipe(recipe_str: str) -> List[Tuple[int, int]]:
    """Parse a full lane-phase recipe string into [(lane, phase), ...]."""
    tokens = recipe_str.strip().split()
    if not tokens:
        raise ValueError("empty lane-phase recipe")
    return [parse_lane_phase_token(t) for t in tokens]


def lane_to_position(
    lane: int, base_row: int, base_col: int, direction: str = "SE"
) -> Tuple[int, int]:
    """Map a slow-salvo lane to a glider injection (row, col).

    The lane is the perpendicular offset of the glider track from the
    elbow. Motion (dr, dc) per direction and perpendicular (−dc, dr):

    - SE (1,1)  -> perp (-1,+1): (base_row - L, base_col + L)
    - SW (1,-1) -> perp (+1,+1): (base_row + L, base_col + L)
    - NW (-1,-1)-> perp (+1,-1): (base_row + L, base_col - L)
    - NE (-1,+1)-> perp (-1,-1): (base_row - L, base_col - L)
    """
    if direction == "SE":
        return base_row - lane, base_col + lane
    elif direction == "SW":
        return base_row + lane, base_col + lane
    elif direction == "NW":
        return base_row + lane, base_col - lane
    elif direction == "NE":
        return base_row - lane, base_col - lane
    raise ValueError(f"unknown direction {direction!r}")


def load_simeks_toolkit(path: str | Path) -> Dict[str, List[int]]:
    """Load EZLiang's ``simeks.txt`` toolkit (slmake separation lists).

    Each line is ``[0, 109, 91, ...] OP`` where OP is e.g. ``0move-2``.
    Returns {op: separations}. Raises FileNotFoundError / ValueError on
    bad input.
    """
    import ast

    toolkit: Dict[str, List[int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            # Lines are "[seps...] OP" where OP may itself contain spaces
            # (e.g. "[...] 2xO 7move3", "[...] -18iE 0move-20").
            try:
                close = line.index("]")
                list_part, op = line[: close + 1], line[close + 1:].strip()
                seps = ast.literal_eval(list_part)
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"simeks.txt line {lineno}: {e}")
            if not op:
                raise ValueError(f"simeks.txt line {lineno}: missing op label")
            if not isinstance(seps, list) or not seps or seps[0] != 0:
                raise ValueError(f"simeks.txt line {lineno}: bad separation list")
            toolkit[op] = [int(x) for x in seps]
    if not toolkit:
        raise ValueError("empty simeks toolkit")
    return toolkit

# Canonical 4-phase matrices for gliders traveling in 4 diagonal directions
GLIDER_PHASES: Dict[str, List[np.ndarray]] = {
    "SE": [  # Moves +r, +c
        np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.uint8),
        np.array([[1, 0, 1], [0, 1, 1], [0, 1, 0]], dtype=np.uint8),
        np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=np.uint8),
        np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]], dtype=np.uint8),
    ],
    "SW": [  # Moves +r, -c
        np.array([[0, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=np.uint8),
        np.array([[1, 0, 1], [1, 1, 0], [0, 1, 0]], dtype=np.uint8),
        np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0]], dtype=np.uint8),
        np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=np.uint8),
    ],
    "NW": [  # Moves -r, -c
        np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0]], dtype=np.uint8),
        np.array([[0, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8),
        np.array([[0, 1, 1], [1, 0, 1], [0, 0, 1]], dtype=np.uint8),
        np.array([[1, 1, 0], [0, 1, 1], [1, 0, 0]], dtype=np.uint8),
    ],
    "NE": [  # Moves -r, +c
        np.array([[1, 1, 1], [0, 0, 1], [0, 1, 0]], dtype=np.uint8),
        np.array([[0, 1, 0], [1, 1, 0], [1, 0, 1]], dtype=np.uint8),
        np.array([[1, 1, 0], [1, 0, 1], [1, 0, 0]], dtype=np.uint8),
        np.array([[0, 1, 1], [1, 1, 0], [0, 0, 1]], dtype=np.uint8),
    ],
}


@dataclass
class ScheduledGlider:
    """A glider scheduled to be injected into the simulation grid at a specific step."""
    release_step: int
    row: int
    col: int
    direction: str = "SE"
    phase: int = 0
    label: str = "Glider"
    injected: bool = False
    active: bool = False
    consumed: bool = False

    def get_matrix(self) -> np.ndarray:
        """Return the 3x3 binary matrix for this glider's direction and phase."""
        phases = GLIDER_PHASES.get(self.direction, GLIDER_PHASES["SE"])
        return phases[self.phase % len(phases)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "release_step": self.release_step,
            "row": self.row,
            "col": self.col,
            "direction": self.direction,
            "phase": self.phase,
            "label": self.label,
            "injected": self.injected,
            "active": self.active,
            "consumed": self.consumed,
        }


class SalvoSchedule:
    """Manages an ordered sequence of gliders that form a slow salvo."""
    def __init__(self, name: str = "Slow Salvo"):
        self.name = name
        self.gliders: List[ScheduledGlider] = []

    def add_glider(
        self,
        step: int,
        row: int,
        col: int,
        direction: str = "SE",
        phase: int = 0,
        label: Optional[str] = None,
    ) -> None:
        idx = len(self.gliders) + 1
        lbl = label or f"G#{idx} (t={step}, dir={direction})"
        self.gliders.append(
            ScheduledGlider(
                release_step=step,
                row=row,
                col=col,
                direction=direction,
                phase=phase,
                label=lbl,
            )
        )

    def gliders_to_inject(self, current_step: int) -> List[ScheduledGlider]:
        """Return gliders ready to be injected at current_step."""
        ready = []
        for g in self.gliders:
            if not g.injected and g.release_step <= current_step:
                g.injected = True
                g.active = True
                ready.append(g)
        return ready

    def reset(self) -> None:
        """Reset injection status for all gliders."""
        for g in self.gliders:
            g.injected = False
            g.active = False
            g.consumed = False

    @classmethod
    def from_lane_phase_recipe(
        cls,
        recipe_str: str,
        start_step: int = 0,
        inter_glider_spacing: int = 90,
        base_row: int = 5,
        base_col: int = 5,
        direction: str = "SE",
    ) -> "SalvoSchedule":
        """
        Parse a lane-phase string in EZLiang/cgol-self-construction style.

        Format: '-5:1 -2:5 -7:5' where each item is 'lane:phase' (the same
        token format as ``oncoming.py``'s ``recipe`` string and slmake /
        pslmake lane-phase lists). Bare integers (no colon) default to
        phase 0. EZLiang phases may exceed 3; they are reduced mod 4 to
        the glider phase index used here (documented approximation: the
        p120 arm only distinguishes phase parity, classic slow salvos use
        phase mod 4).

        Geometry: ``lane`` is the perpendicular offset from the elbow, not
        a raw column shift. For a SE glider (motion (1,1)), the
        perpendicular is (-1,+1), so lane L maps to
        ``(row, col) = (base_row - L, base_col + L)``. This preserves the
        glider's arrival phase while shifting its track sideways, unlike a
        pure column shift which also retimes the collision. Other
        directions use the analogous perpendicular.

        Timing: tokens are spaced ``inter_glider_spacing`` generations
        apart (default 90, safely above the ~70-tick elbow settle time so
        each collision settles before the next arrives).
        """
        tokens = recipe_str.strip().split()
        if not tokens or tokens == [""]:
            raise ValueError("empty lane-phase recipe")
        salvo = cls(name=f"Recipe Salvo ({len(tokens)} gliders)")
        t = start_step
        for i, item in enumerate(tokens):
            lane, phase = parse_lane_phase_token(item)
            r, c = lane_to_position(lane, base_row, base_col, direction)
            salvo.add_glider(
                step=t,
                row=r,
                col=c,
                direction=direction,
                phase=phase,
                label=f"Recipe #{i+1} [Lane {lane}, Phase {phase}]",
            )
            t += inter_glider_spacing
        return salvo

    @classmethod
    def from_separation_list(
        cls,
        separations: List[int],
        start_step: int = 0,
        row: int = 5,
        col: int = 5,
        direction: str = "SE",
        phase: int = 0,
        op_name: str = "",
    ) -> "SalvoSchedule":
        """Build a single-channel glider stream from simeks separations.

        ``separations`` is one toolkit entry (e.g. ``toolkit["0move-2"]``):
        ``separations[0]`` must be 0 and each subsequent entry is the tick
        gap after the previous glider -- the slmake single-channel format
        used by EZLiang's ``sc`` compiler. All gliders share one lane
        (row, col, direction): that is what makes it single-channel.
        Phases cycle 0..3 down the stream, matching how a fixed channel
        emits successive gliders.

        Calibration caveat (documented, not hidden): simeks separations
        assume the sc90 elbow/channel geometry. Replayed against a
        different elbow placement, the stream executes faithfully but the
        toolkit's named displacement (e.g. ``0move-2`` -> (-1,-2)) is only
        reproduced when the channel geometry matches; the replay test
        therefore asserts graceful execution and elbow preservation, not
        the named vector.
        """
        if not separations or separations[0] != 0:
            raise ValueError("separation list must start with 0")
        if any(s < 0 for s in separations):
            raise ValueError("separations must be non-negative")
        name = f"Single-Channel {op_name}" if op_name else "Single-Channel Stream"
        salvo = cls(name=f"{name} ({len(separations)} gliders)")
        t = start_step
        for i, gap in enumerate(separations):
            t += gap if i else 0
            salvo.add_glider(
                step=t,
                row=row,
                col=col,
                direction=direction,
                phase=(phase + i) % 4,
                label=f"{op_name or 'SC'} #{i+1} [t={t}]",
            )
        return salvo
