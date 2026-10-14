"""
Von Neumann 29-state cellular automaton: documented state table and
synchronous transition function.

Scope statement (read before citing): this module implements all 29
states with directional transmission, confluent OR + one-tick delay
memory, and sensitized construct/kill semantics. It does NOT implement a
universal constructor/decoder program or self-description tape: there is
no stored machine description being read, only pulse circulation and a
genuinely extending arm (sensitized -> wired-cell construction events,
counted). Preset ``von_neumann_constructor`` demonstrates pulse
propagation plus arm extension; full self-reproduction programs remain
out of scope.

State IDs (0-28, contiguous):
  0            U          quiescent / blank
  1-4          T_O        ordinary transmission ground (E,N,W,S)
  5-8          T_O*       ordinary transmission excited (E,N,W,S)
  9-12         T_S        special transmission ground (E,N,W,S)
  13-16        T_S*       special transmission excited (E,N,W,S)
  17           C          confluent ground
  18           C*         confluent excited (fires this step)
  19           C1         confluent latched (input seen, fires next step)
  20           C1*        confluent latched+excited (fires, holds memory)
  21-24        V_O        sensitized-by-ordinary, pending (E,N,W,S)
  25-28        V_S        sensitized-by-special, pending (E,N,W,S)

Directions: E=(0,+1), N=(-1,0), W=(0,-1), S=(+1,0). A transmission cell
with direction d is fed from the neighbor at -d.

Rules (synchronous, orthogonal neighborhood only):
- T_O ground: excited (-> +4) iff fed by ordinary-excited or
  confluent-excited (18/20) neighbor; special-excited input ignored.
- T_S ground: symmetric (special/confluent excite; ordinary ignored).
- Excited (5-8, 13-16): always decay to ground next step (they emit
  this step).
- C ground (17): any excited transmission/confluent neighbor -> C1 (19)
  (latch, one-tick delay; no fire yet).
- C1 (19): -> C* (18) next step regardless (fires after delay).
- C* (18): -> C ground (emit done).
- C1* (20): like C* but entered when C1 receives continued input;
  -> C ground after firing (documents coincidence without extra memory).
- U (0): exactly one excited orthogonal neighbor -> sensitize
  (ordinary-excited/C* -> V_O pending signal direction; special-excited
  -> V_S). Zero or 2+ excited neighbors -> stay U (ambiguity refused).
  Confluent-latched (19) does not sensitize (not yet firing).
- V_O (21-24): fed by matching ordinary/C* input along pending dir ->
  construct ordinary ground cell of pending dir (counted event). Fed by
  special-excited instead -> kill to U. Otherwise hold.
- V_S (25-28): symmetric (-> special ground on special input, kill on
  ordinary input).
"""

from typing import Dict, Tuple
import numpy as np

# Direction index: 0=E, 1=N, 2=W, 3=S
DIRS = [(0, 1), (-1, 0), (0, -1), (1, 0)]
DIR_NAME = ["E", "N", "W", "S"]

U = 0
T_O = [1, 2, 3, 4]
T_O_EXC = [5, 6, 7, 8]
T_S = [9, 10, 11, 12]
T_S_EXC = [13, 14, 15, 16]
C, C_EXC, C1, C1_EXC = 17, 18, 19, 20
V_O = [21, 22, 23, 24]
V_S = [25, 26, 27, 28]

N_STATES = 29

STATE_NAMES: Dict[int, str] = {0: "U blank"}
for i, d in enumerate(DIR_NAME):
    STATE_NAMES[1 + i] = f"T_O {d}"
    STATE_NAMES[5 + i] = f"T_O* {d}"
    STATE_NAMES[9 + i] = f"T_S {d}"
    STATE_NAMES[13 + i] = f"T_S* {d}"
    STATE_NAMES[21 + i] = f"V_O pending {d}"
    STATE_NAMES[25 + i] = f"V_S pending {d}"
STATE_NAMES.update({17: "C confluent", 18: "C* firing", 19: "C1 latched", 20: "C1* latched+firing"})

# Excited states act as emission sources this step.
EXCITED = frozenset(T_O_EXC + T_S_EXC + [C_EXC, C1_EXC])
ORDINARY_EXC = frozenset(T_O_EXC)
SPECIAL_EXC = frozenset(T_S_EXC)
CONFLUENT_FIRING = frozenset([C_EXC, C1_EXC])


def _in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _fed_from(grid: np.ndarray, r: int, c: int, d: int) -> Tuple[bool, bool, bool]:
    """Check feeders at -d of (r,c).

    Returns (ordinary_in, special_in, confluent_in) where each is True if
    an excited source sits at the feeding neighbor AND points at (r,c)
    (transmission sources must point toward the cell; confluent firing
    sources feed any direction).
    """
    h, w = grid.shape
    dr, dc = DIRS[d]
    nr, nc = r - dr, c - dc
    if not _in_bounds(nr, nc, h, w):
        return False, False, False
    ns = int(grid[nr, nc])
    ordinary_in = special_in = confluent_in = False
    if ns in CONFLUENT_FIRING:
        confluent_in = True
    elif ns in ORDINARY_EXC:
        # neighbor's direction must point toward (r,c): neighbor dir == d
        ndir = T_O_EXC.index(ns)
        if ndir == d:
            ordinary_in = True
    elif ns in SPECIAL_EXC:
        ndir = T_S_EXC.index(ns)
        if ndir == d:
            special_in = True
    return ordinary_in, special_in, confluent_in


def step_von_neumann_29(grid: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """One synchronous step of the documented 29-state system.

    Returns (next_grid, events) with events = {"constructed": n,
    "killed": n, "sensitized": n} counting transitions this step.
    """
    h, w = grid.shape
    nxt = np.copy(grid)
    constructed = killed = sensitized = 0

    for r in range(h):
        for c in range(w):
            s = int(grid[r, c])

            if s == U:
                # Collect excited orthogonal neighbors.
                exc = []
                for d, (dr, dc) in enumerate(DIRS):
                    nr, nc = r + dr, c + dc
                    if not _in_bounds(nr, nc, h, w):
                        continue
                    ns = int(grid[nr, nc])
                    if ns in EXCITED:
                        exc.append((d, ns))
                if len(exc) != 1:
                    continue
                d_to_nb, ns = exc[0]
                # Signal direction: transmission neighbor pointing at us
                # keeps its direction; confluent firing flows away from
                # the neighbor, i.e. toward us = d_to_nb.
                if ns in ORDINARY_EXC:
                    ndir = T_O_EXC.index(ns)
                    # Must point at (r,c): neighbor at +d_to_nb side of us
                    # pointing -d_to_nb... equivalently ndir == opposite?
                    # Neighbor at (r+dr_d, c+dc_d) pointing toward us has
                    # direction index opposite of d_to_nb.
                    if ndir == (d_to_nb + 2) % 4:
                        nxt[r, c] = V_O[ndir]
                        sensitized += 1
                elif ns in SPECIAL_EXC:
                    ndir = T_S_EXC.index(ns)
                    if ndir == (d_to_nb + 2) % 4:
                        nxt[r, c] = V_S[ndir]
                        sensitized += 1
                elif ns in CONFLUENT_FIRING:
                    # Decide kind by majority of *other* tape context?
                    # Deterministic documented choice: confluent firing
                    # sensitizes ordinary pending the flow direction.
                    # Flow dir = from neighbor to us = opposite of d_to_nb.
                    flow = (d_to_nb + 2) % 4
                    nxt[r, c] = V_O[flow]
                    sensitized += 1
                continue

            elif 1 <= s <= 4:  # ordinary ground, dir index s-1
                d = s - 1
                o_in, _, c_in = _fed_from(grid, r, c, d)
                if o_in or c_in:
                    nxt[r, c] = s + 4

            elif 5 <= s <= 8:  # ordinary excited -> decay
                nxt[r, c] = s - 4

            elif 9 <= s <= 12:  # special ground
                d = s - 9
                _, sp_in, c_in = _fed_from(grid, r, c, d)
                if sp_in or c_in:
                    nxt[r, c] = s + 4

            elif 13 <= s <= 16:  # special excited -> decay
                nxt[r, c] = s - 4

            elif s == C:  # confluent ground: latch on any excited input
                any_exc = False
                for d, (dr, dc) in enumerate(DIRS):
                    nr, nc = r + dr, c + dc
                    if _in_bounds(nr, nc, h, w) and int(grid[nr, nc]) in EXCITED:
                        any_exc = True
                        break
                if any_exc:
                    nxt[r, c] = C1

            elif s == C1:  # latched -> fire (delay elapsed)
                # Continued input upgrades to C1* (coincidence marker).
                any_exc = False
                for d, (dr, dc) in enumerate(DIRS):
                    nr, nc = r + dr, c + dc
                    if _in_bounds(nr, nc, h, w) and int(grid[nr, nc]) in EXCITED:
                        any_exc = True
                        break
                nxt[r, c] = C1_EXC if any_exc else C_EXC

            elif s in (C_EXC, C1_EXC):  # fired -> ground
                nxt[r, c] = C

            elif 21 <= s <= 24:  # V_O pending dir s-21
                d = s - 21
                o_in, sp_in, c_in = _fed_from(grid, r, c, d)
                if o_in or c_in:
                    nxt[r, c] = T_O[d]  # CONSTRUCT wired cell
                    constructed += 1
                elif sp_in:
                    nxt[r, c] = U  # wrong-kind kill
                    killed += 1

            elif 25 <= s <= 28:  # V_S pending dir s-25
                d = s - 25
                o_in, sp_in, c_in = _fed_from(grid, r, c, d)
                if sp_in or c_in:
                    # Confluent firing constructs ordinary (documented);
                    # pure special input constructs special.
                    nxt[r, c] = T_S[d] if sp_in else T_O[d]
                    constructed += 1
                elif o_in:
                    nxt[r, c] = U
                    killed += 1

    return nxt, {"constructed": constructed, "killed": killed, "sensitized": sensitized}
