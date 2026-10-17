"""
Multi-state RLE codec (Golly format).

The binary `rle.py` module only understands Life's two states. Von Neumann's
29-state machines are distributed as *multi-state* RLE, where a cell's value
is encoded by a letter, not a bit. This module implements the real Golly
encoding so the canonical JvN29 patterns can be loaded and saved verbatim:

  * ``.`` and ``b``           -> state 0
  * ``A`` .. ``X``            -> states 1 .. 24
  * ``p`` .. ``y`` + ``A..X`` -> states 25 .. 24 + 24*25  (two characters)

The two-character form packs 24 states per "row": ``pA`` = 25, ``pB`` = 26,
``pC`` = 27, ``pD`` = 28, ... ``pX`` = 48, ``qA`` = 49, and so on. Run
counts (``3A``), end-of-row (``$``) and end-of-pattern (``!``) behave as in
the binary format. See https://golly.sourceforge.io/Help/formats.html
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

_ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWX"  # states 1..24
_ALPHA_INDEX = {ch: i + 1 for i, ch in enumerate(_ALPHA)}
# Prefix characters for the two-char (>=25) forms: p=0, q=1, ...
_PREFIXES = "pqrstuvwxy"
_PREFIX_INDEX = {ch: i for i, ch in enumerate(_PREFIXES)}


@dataclass
class MultiStatePattern:
    """A decoded multi-state pattern plus its RLE metadata."""

    grid: np.ndarray  # (height, width) int32
    rule: str = ""
    name: str = ""
    comments: List[str] = field(default_factory=list)
    width: int = 0
    height: int = 0

    @property
    def population(self) -> int:
        return int(np.count_nonzero(self.grid))

    def bounding_box(self) -> Tuple[int, int, int, int]:
        coords = np.argwhere(self.grid > 0)
        if len(coords) == 0:
            return (0, -1, 0, -1)
        (r0, c0) = coords.min(axis=0)
        (r1, c1) = coords.max(axis=0)
        return (int(r0), int(r1), int(c0), int(c1))

    def states_present(self) -> Dict[int, int]:
        vals, counts = np.unique(self.grid, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts) if v != 0}


def _state_char(state: int) -> str:
    if state < 1 or state > 24 + 24 * len(_PREFIXES):
        raise ValueError(f"state {state} not encodable")
    if state <= 24:
        return _ALPHA[state - 1]
    state -= 25
    return _PREFIXES[state // 24] + _ALPHA[state % 24]


def decode_mstate_rle(text: str) -> MultiStatePattern:
    """Parse a Golly multi-state RLE string into a :class:`MultiStatePattern`."""
    comments: List[str] = []
    body: List[str] = []
    name = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            comments.append(line)
            if line[:2] in ("#N", "#n"):
                name = line[2:].strip()
            continue
        body.append(line)
    if not body:
        return MultiStatePattern(np.zeros((0, 0), np.int32), comments=comments)

    header = body[0]
    dims: Dict[str, str] = {}
    for part in header.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            dims[k.strip().lower()] = v.strip()
    width = int(dims.get("x", 0))
    height = int(dims.get("y", 0))
    rule = dims.get("rule", "")
    data = "".join(body[1:])

    rows: List[List[int]] = []
    current: List[int] = []
    count = 0
    i = 0
    n = len(data)
    while i < n:
        ch = data[i]
        if ch.isdigit():
            count = count * 10 + int(ch)
            i += 1
            continue
        run = count if count > 0 else 1
        count = 0
        if ch == "!":
            break
        if ch == "$":
            rows.append(current)
            current = []
            for _ in range(run - 1):
                rows.append([])
        elif ch in (".", "b"):
            current.extend([0] * run)
        elif ch in _ALPHA_INDEX:
            current.extend([_ALPHA_INDEX[ch]] * run)
        elif ch in _PREFIX_INDEX:
            i += 1
            if i >= n or data[i] not in _ALPHA_INDEX:
                raise ValueError(f"bad multi-state pair at offset {i}")
            state = 25 + _PREFIX_INDEX[ch] * 24 + (_ALPHA_INDEX[data[i]] - 1)
            current.extend([state] * run)
        else:
            raise ValueError(f"unexpected character {ch!r} at offset {i}")
        i += 1
    if current:
        rows.append(current)

    cols = max(max((len(r) for r in rows), default=0), width)
    total_rows = max(len(rows), height)
    grid = np.zeros((max(total_rows, 1), max(cols, 1)), dtype=np.int32)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            if val:
                grid[r, c] = val
    return MultiStatePattern(
        grid=grid, rule=rule, name=name, comments=comments,
        width=width, height=height,
    )


def _runs(row: np.ndarray) -> List[str]:
    out: List[str] = []
    n = len(row)
    c = 0
    while c < n:
        val = int(row[c])
        run = 1
        while c + run < n and int(row[c + run]) == val:
            run += 1
        if val == 0:
            token = "."
        else:
            token = _state_char(val)
        out.append(f"{run}{token}" if run > 1 else token)
        c += run
    # Trim a trailing run of dead cells (RLE omits it implicitly).
    return out


def encode_mstate_rle(
    pattern: MultiStatePattern,
    line_length: int = 70,
) -> str:
    """Serialise a :class:`MultiStatePattern` back to Golly multi-state RLE.

    Trailing dead cells on every row are omitted, exactly as Golly does, so
    decoding our output reproduces the pattern up to a right/bottom trim of
    empty space.
    """
    grid = pattern.grid
    h, w = grid.shape
    lines: List[str] = []
    lines.extend(pattern.comments)
    if pattern.name and not any(
        c.startswith("#N") for c in pattern.comments
    ):
        lines.append(f"#N {pattern.name}")
    rule = pattern.rule or "JvN29"
    lines.append(f"x = {w}, y = {h}, rule = {rule}")

    tokens: List[str] = []
    for r in range(h):
        row = grid[r]
        last = np.nonzero(row)[0]
        if len(last) == 0:
            if r < h - 1:
                tokens.append("$")
            continue
        end = int(last[-1]) + 1
        tokens.extend(_runs(row[:end]))
        if r < h - 1:
            tokens.append("$")
    tokens.append("!")
    raw = "".join(tokens)

    for i in range(0, len(raw), line_length):
        lines.append(raw[i:i + line_length])
    return "\n".join(lines)
