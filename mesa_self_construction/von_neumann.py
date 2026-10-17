"""
Von Neumann 29-state cellular automaton: a faithful interpreter for the
canonical JvN29 transition table.

Fidelity statement (read before citing)
---------------------------------------
This module does NOT hand-code an approximation of von Neumann's rules.
It parses the normative transition table vendored verbatim in
``data/JvN29.table`` (the ``@TABLE`` section of Golly's ``JvN29.rule``,
as distributed by EZLiang/vonNeumannTuringMachine) and evaluates it with
Golly's documented semantics:

  * neighborhood ``vonNeumann`` with input order ``C, N, E, S, W``
    (center, north, east, south, west);
  * ``symmetries:none`` (no rotation/reflection expansion);
  * transitions are checked top-to-bottom and the **first** matching rule
    is applied; unlisted neighbourhoods leave the center cell unchanged;
  * cells outside the finite grid are state 0 (an infinite vacuum), the
    same convention Golly uses for an unbounded universe.

Because the behaviour is defined by the reference table itself, the
correctness claim is auditable: the data file can be diffed against the
upstream rule, and ``tests`` re-derive transitions directly from it.

Canonical state numbering (the five orthogonal groups):
  0        U     ground / quiescent
  1..8     S     sensitized (transition) states -- an 8-leaf binary
                 decoding tree. Each of the first three update cycles
                 contributes one input bit (present/absent); the leaf
                 determines the transmission state a quiescent cell is
                 sensitized into. This tree is the mechanism that decodes
                 a serial bit-stream into a constructed cell type.
  9..12    OTS   ordinary transmission, quiescent (E,N,W,S)
  13..16   OTS*  ordinary transmission, excited (E,N,W,S)
  17..20   STS   special transmission, quiescent (E,N,W,S)
  21..24   STS*  special transmission, excited (E,N,W,S)
  25..28   C     confluent states C00, C01, C10, C11 (OR-junction with a
                 built-in one-cycle delay, so two signals can be merged)

Scope: this is the canonical rule engine and the primitives built from it
(signal propagation, special-vs-ordinary interaction, confluent merging).
It is NOT a universal constructor program: the full decoder and the
self-description tape that von Neumann's machine reads are not supplied.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple, Union
import numpy as np

_DATA_DIR = Path(__file__).resolve().parent / "data"

DATA_PATH = _DATA_DIR / "JvN29.table"

# Von Neumann's documented machines, vendored from Golly's pattern
# collection (see data/patterns/PROVENANCE.md).
PATTERN_DIR = _DATA_DIR / "patterns"
KNOWN_PATTERNS = {
    "counter": "counter-demo.rle",
    "cell_coders": "cell-coders-demo.rle",
    "tape_reader": "read-arm-demo.rle",
    "self_replicator": "small-JvN-self-replicator.rle",
}

N_STATES = 29

# Direction index for the N,E,S,W neighbour slots (Golly vonNeumann order).
NEIGHBOR_ORDER = ("N", "E", "S", "W")

# Canonical names. Sensitized-state labels are derived from the tree they
# implement (documented in the module docstring); transmission directions
# are derived from the table (a state fed from the west points east, etc.).
STATE_NAMES: Dict[int, str] = {
    0: "U (ground)",
    1: "S (sensitized)",
    2: "S0",
    3: "S1",
    4: "S00",
    5: "S01",
    6: "S10",
    7: "S11",
    8: "S000",
    9: "OTS E",
    10: "OTS N",
    11: "OTS W",
    12: "OTS S",
    13: "OTS* E",
    14: "OTS* N",
    15: "OTS* W",
    16: "OTS* S",
    17: "STS E",
    18: "STS N",
    19: "STS W",
    20: "STS S",
    21: "STS* E",
    22: "STS* N",
    23: "STS* W",
    24: "STS* S",
    25: "C00 (confluent)",
    26: "C01",
    27: "C10",
    28: "C11",
}

# State groups (canonical).
U = 0
SENSITIZED = frozenset(range(1, 9))
OTS = frozenset(range(9, 13))
OTS_EXC = frozenset(range(13, 17))
STS = frozenset(range(17, 21))
STS_EXC = frozenset(range(21, 25))
CONFLUENT = frozenset(range(25, 29))
TRANSMISSION = OTS | OTS_EXC | STS | STS_EXC
EXCITED = OTS_EXC | STS_EXC | CONFLUENT


@dataclass
class _CompiledRule:
    """One table transition compiled into matchable form.

    The extra fields are derived once at parse time so the hot loop of the
    vectorized stepper only does array comparisons:

    * ``constraint_values`` -- each position's allowed states as an ndarray
      (for :func:`numpy.isin`), avoiding per-step set reconstruction;
    * ``equal_pairs`` -- position pairs whose tokens are the *same* variable
      (repeated-variable equality is part of matching);
    * ``output_pos`` -- position index holding the output value when the
      output is a variable (``None`` for a literal output).
    """

    tokens: Tuple[Union[int, str], ...]  # 5 input tokens (int or var name)
    constraints: Tuple[FrozenSet[int], ...]  # allowed states per position
    output: Union[int, str]  # literal state or variable name
    line: str  # original text (for debugging / golden tests)
    constraint_values: Tuple[np.ndarray, ...] = ()
    equal_pairs: Tuple[Tuple[int, int], ...] = ()
    output_pos: Optional[int] = None


@dataclass
class JvN29Table:
    """The parsed canonical table plus a memoised transition lookup."""

    n_states: int
    rules: List[_CompiledRule]
    rules_by_center: Dict[int, List[_CompiledRule]] = field(default_factory=dict)
    _cache: Dict[Tuple[int, int, int, int, int], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rules_by_center = {s: [] for s in range(self.n_states)}
        for rule in self.rules:
            for s in rule.constraints[0]:
                self.rules_by_center[s].append(rule)

    def next_state(self, center: int, n: int, e: int, s: int, w: int) -> int:
        """Apply the table (first-match) to one neighbourhood."""
        key = (center, n, e, s, w)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        nxt = center
        for rule in self.rules_by_center.get(center, ()):
            assigned: Dict[str, int] = {}
            states = (center, n, e, s, w)
            ok = True
            for tok, cons, val in zip(
                rule.tokens, rule.constraints, states
            ):
                if val not in cons:
                    ok = False
                    break
                if isinstance(tok, str):
                    prev = assigned.get(tok)
                    if prev is not None and prev != val:
                        ok = False
                        break
                    assigned[tok] = val
            if ok:
                nxt = (
                    assigned[rule.output]
                    if isinstance(rule.output, str)
                    else rule.output
                )
                break
        self._cache[key] = nxt
        return nxt


def parse_jvn29_table(text: str) -> JvN29Table:
    """Parse a Golly @TABLE (JvN29 subset) into a :class:`JvN29Table`.

    Supports ``n_states``/``neighborhood``/``symmetries`` descriptors,
    ``var name={...}`` definitions (including variables that reference
    earlier variables) and ``C,N,E,S,W,C'`` transition lines.
    """
    variables: Dict[str, FrozenSet[int]] = {}
    raw_rules: List[Tuple[Tuple[str, ...], str]] = []
    n_states = N_STATES
    neighborhood = "vonNeumann"
    symmetries = "none"
    seen_content = False

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("@"):
            continue
        low = line.lower()
        if low.startswith("n_states"):
            n_states = int(line.split(":", 1)[1])
            continue
        if low.startswith("neighborhood"):
            neighborhood = line.split(":", 1)[1].strip()
            continue
        if low.startswith("symmetries"):
            symmetries = line.split(":", 1)[1].strip()
            continue
        if low.startswith("var "):
            name, _, rhs = line[4:].partition("=")
            values = _expand_var_set(rhs.strip(), variables)
            variables[name.strip()] = frozenset(values)
            continue
        seen_content = True
        tokens = tuple(t.strip() for t in line.split(","))
        if len(tokens) != 6:
            raise ValueError(f"malformed transition line: {raw!r}")
        raw_rules.append((tokens[:5], tokens[5]))

    if not seen_content or not raw_rules:
        raise ValueError("no transitions found in table text")
    if neighborhood != "vonNeumann":
        raise ValueError(f"unsupported neighborhood: {neighborhood}")
    if symmetries != "none":
        raise ValueError(f"unsupported symmetries: {symmetries}")

    compiled: List[_CompiledRule] = []
    for inputs, out in raw_rules:
        constraints = []
        tokens: List[Union[int, str]] = []
        for tok in inputs:
            if tok in variables:
                constraints.append(variables[tok])
                tokens.append(tok)
            else:
                val = int(tok)
                constraints.append(frozenset((val,)))
                tokens.append(val)
        output: Union[int, str] = out if out in variables else int(out)
        constraint_values = tuple(
            np.fromiter(sorted(c), dtype=np.int32) for c in constraints
        )
        equal_pairs: List[Tuple[int, int]] = []
        for i in range(5):
            if not isinstance(tokens[i], str):
                continue
            for j in range(i + 1, 5):
                if tokens[j] == tokens[i]:
                    equal_pairs.append((i, j))
        output_pos: Optional[int] = None
        if isinstance(output, str):
            for i, tok in enumerate(tokens):
                if tok == output:
                    output_pos = i
                    break
            if output_pos is None:
                raise ValueError(
                    f"output variable {output!r} not among inputs: {inputs}"
                )
        compiled.append(
            _CompiledRule(
                tokens=tuple(tokens),
                constraints=tuple(constraints),
                output=output,
                line=",".join(inputs) + "," + out,
                constraint_values=constraint_values,
                equal_pairs=tuple(equal_pairs),
                output_pos=output_pos,
            )
        )

    return JvN29Table(n_states=n_states, rules=compiled)


def _expand_var_set(rhs: str, variables: Dict[str, FrozenSet[int]]) -> List[int]:
    """Expand a ``{...}`` set, allowing earlier variable names inside."""
    rhs = rhs.strip()
    if not (rhs.startswith("{") and rhs.endswith("}")):
        raise ValueError(f"bad variable definition: {rhs!r}")
    out: List[int] = []
    for item in rhs[1:-1].split(","):
        item = item.strip()
        if not item:
            continue
        if item in variables:
            out.extend(sorted(variables[item]))
        else:
            out.append(int(item))
    return out


_DEFAULT_TABLE: Optional[JvN29Table] = None


def load_jvn29_table(
    path: Union[str, Path, None] = None,
) -> JvN29Table:
    """Load (and cache) the canonical vendored JvN29 table."""
    global _DEFAULT_TABLE
    if path is None:
        if _DEFAULT_TABLE is None:
            _DEFAULT_TABLE = parse_jvn29_table(
                DATA_PATH.read_text(encoding="utf-8")
            )
        return _DEFAULT_TABLE
    return parse_jvn29_table(Path(path).read_text(encoding="utf-8"))


def load_jvn_pattern(name: str):
    """Load a vendored von Neumann machine pattern by short name.

    ``name`` is a key of :data:`KNOWN_PATTERNS` (``"counter"``,
    ``"cell_coders"``, ``"tape_reader"``, ``"self_replicator"``) or a path
    to a Golly multi-state RLE file. Returns a
    :class:`~mesa_self_construction.mstate_rle.MultiStatePattern`.
    """
    from mesa_self_construction.mstate_rle import decode_mstate_rle

    if name in KNOWN_PATTERNS:
        path = PATTERN_DIR / KNOWN_PATTERNS[name]
    else:
        path = Path(name)
    text = path.read_text(encoding="utf-8")
    pattern = decode_mstate_rle(text)
    if not pattern.rule.upper().startswith("JVN29"):
        raise ValueError(
            f"pattern {path.name!r} uses rule {pattern.rule!r}, expected JvN29"
        )
    return pattern


def _shift_views(grid: np.ndarray):
    """Return the (C, N, E, S, W) neighbour arrays with 0 borders.

    Off-grid neighbours are state 0 (infinite vacuum), so the arrays are the
    central grid shifted with a constant-zero pad rather than a wrap.
    """
    n = np.zeros_like(grid)
    n[1:, :] = grid[:-1, :]
    e = np.zeros_like(grid)
    e[:, :-1] = grid[:, 1:]
    s = np.zeros_like(grid)
    s[:-1, :] = grid[1:, :]
    w = np.zeros_like(grid)
    w[:, 1:] = grid[:, :-1]
    return grid, n, e, s, w


def step_von_neumann_29(
    grid: np.ndarray, table: Optional[JvN29Table] = None
) -> Tuple[np.ndarray, Dict[str, int]]:
    """One synchronous step of the canonical JvN29 automaton (vectorized).

    Evaluates the reference transition table with NumPy over the whole grid
    at once, applying rules in file order to cells not yet matched (Golly's
    documented first-match semantics). Off-grid neighbours are state 0.

    Returns ``(next_grid, events)`` where ``events`` reports ``changed``
    cells, fresh ``sensitized`` cells (0 -> 1) and ``killed`` cells
    (non-zero -> 0) for telemetry and validation.

    ``step_von_neumann_29_scalar`` is retained as an independent reference
    and is cross-checked against this in the test-suite.
    """
    table = table or load_jvn29_table()
    grid = np.asarray(grid, dtype=np.int32)
    arrays = _shift_views(grid)
    c, n, e, s, w = arrays

    # Encode each cell's (C,N,E,S,W) neighbourhood as one integer, then
    # evaluate each *distinct* neighbourhood once. Real CA grids visit far
    # fewer neighbourhoods than cells, and the per-table memo makes repeats
    # across steps free -- so this is both exact and fast.
    key = ((((c.astype(np.int64) * 29 + n) * 29 + e) * 29 + s) * 29 + w)
    uniq, inverse = np.unique(key.ravel(), return_inverse=True)
    memo = table._cache
    lookup = np.empty(uniq.shape[0], dtype=np.int32)
    for i in range(uniq.shape[0]):
        u = int(uniq[i])
        if u in memo:
            lookup[i] = memo[u]
            continue
        ww = u % 29; u //= 29
        ss = u % 29; u //= 29
        ee = u % 29; u //= 29
        nn = u % 29; u //= 29
        cc = u
        val = table.next_state(cc, nn, ee, ss, ww)
        memo[(cc, nn, ee, ss, ww)] = val
        lookup[i] = val
    nxt = lookup[inverse].reshape(grid.shape).astype(np.int32)

    diff = nxt != grid
    changed = int(diff.sum())
    center = grid
    sensitized = int(((center == 0) & (nxt == 1)).sum())
    killed = int(((center != 0) & (nxt == 0)).sum())
    return nxt, {
        "changed": changed,
        "sensitized": sensitized,
        "killed": killed,
    }


def step_von_neumann_29_scalar(
    grid: np.ndarray, table: Optional[JvN29Table] = None
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Reference (pure-Python) stepper, kept for equivalence testing."""
    table = table or load_jvn29_table()
    grid = np.asarray(grid)
    h, w = grid.shape
    nxt = np.copy(grid)
    changed = sensitized = killed = 0
    for r in range(h):
        for c in range(w):
            center = int(grid[r, c])
            n = int(grid[r - 1, c]) if r > 0 else 0
            e = int(grid[r, c + 1]) if c + 1 < w else 0
            s = int(grid[r + 1, c]) if r + 1 < h else 0
            wp = int(grid[r, c - 1]) if c > 0 else 0
            new = table.next_state(center, n, e, s, wp)
            if new != center:
                nxt[r, c] = new
                changed += 1
                if center == 0 and new == 1:
                    sensitized += 1
                elif center != 0 and new == 0:
                    killed += 1
    return nxt, {
        "changed": changed,
        "sensitized": sensitized,
        "killed": killed,
    }
