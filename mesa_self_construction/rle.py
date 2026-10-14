"""
RLE (Run Length Encoded) parser and serializer for Conway's Game of Life.
Supports standard Life RLE format, bounding box extraction, and pattern manipulation.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np


def parse_rle(rle_text: str, origin: Tuple[int, int] = (0, 0)) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Parse an RLE formatted string into a 2D numpy array and metadata.

    Args:
        rle_text: RLE string (can include comments starting with #)
        origin: (row, col) offset of the pattern's top-left corner in the
            returned matrix (implemented as zero padding).

    Returns:
        (pattern_array, metadata_dict)
    """
    lines = [l.strip() for l in rle_text.strip().split("\n") if l.strip()]
    metadata: Dict[str, Any] = {"comments": [], "rule": "B3/S23", "width": 0, "height": 0}
    
    header_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("#"):
            metadata["comments"].append(line)
        else:
            header_idx = i
            break
            
    if header_idx == -1:
        return np.zeros((0, 0), dtype=np.uint8), metadata
        
    header_line = lines[header_idx]
    # parse header: x = 11, y = 11, rule = B3/S23
    parts = [p.strip() for p in header_line.split(",")]
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            k = k.strip().lower()
            v = v.strip()
            if k == "x":
                metadata["width"] = int(v)
            elif k == "y":
                metadata["height"] = int(v)
            elif k == "rule":
                metadata["rule"] = v
                
    w = metadata.get("width", 0)
    h = metadata.get("height", 0)
    
    data_lines = "".join(lines[header_idx + 1:])
    
    rows: List[List[int]] = []
    current_row: List[int] = []
    count = 0
    
    for ch in data_lines:
        if ch == "!":
            break
        elif ch.isdigit():
            count = count * 10 + int(ch)
        elif ch == "b":  # Dead cell
            run = count if count > 0 else 1
            current_row.extend([0] * run)
            count = 0
        elif ch == "o":  # Live cell
            run = count if count > 0 else 1
            current_row.extend([1] * run)
            count = 0
        elif ch == "$":  # End of row
            run = count if count > 0 else 1
            rows.append(current_row)
            for _ in range(run - 1):
                rows.append([])
            current_row = []
            count = 0
            
    if current_row:
        rows.append(current_row)
        
    # Determine actual dimensions
    max_cols = max((len(r) for r in rows), default=0)
    cols = max(w, max_cols)
    total_rows = max(h, len(rows))
    
    matrix = np.zeros((total_rows, cols), dtype=np.uint8)
    for r_idx, r in enumerate(rows):
        for c_idx, val in enumerate(r):
            if val:
                matrix[r_idx, c_idx] = 1

    if origin != (0, 0):
        orow, ocol = int(origin[0]), int(origin[1])
        if orow < 0 or ocol < 0:
            raise ValueError("origin must be non-negative (row, col)")
        padded = np.zeros(
            (matrix.shape[0] + orow, matrix.shape[1] + ocol), dtype=np.uint8
        )
        padded[orow:orow + matrix.shape[0], ocol:ocol + matrix.shape[1]] = matrix
        matrix = padded

    return matrix, metadata


def encode_rle(matrix: np.ndarray, name: str = "Pattern") -> str:
    """Encode a 2D binary numpy array into an RLE string."""
    h, w = matrix.shape
    lines = [f"#N {name}", f"x = {w}, y = {h}, rule = B3/S23"]
    
    rle_data = []
    for r in range(h):
        row = matrix[r]
        # Run-length encode row
        c = 0
        while c < w:
            val = row[c]
            run = 1
            while c + run < w and row[c + run] == val:
                run += 1
            ch = "o" if val == 1 else "b"
            if run > 1:
                rle_data.append(f"{run}{ch}")
            else:
                rle_data.append(ch)
            c += run
        if r < h - 1:
            rle_data.append("$")
            
    rle_data.append("!")
    raw_str = "".join(rle_data)
    
    # Wrap at 70 characters
    wrapped = []
    for i in range(0, len(raw_str), 70):
        wrapped.append(raw_str[i:i + 70])
        
    lines.extend(wrapped)
    return "\n".join(lines)


def crop_to_bounding_box(matrix: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop matrix to active live cells bounding box.
    Returns (cropped_array, (rmin, rmax, cmin, cmax)).
    """
    coords = np.argwhere(matrix == 1)
    if len(coords) == 0:
        return np.zeros((0, 0), dtype=np.uint8), (0, 0, 0, 0)
    rmin, cmin = coords.min(axis=0)
    rmax, cmax = coords.max(axis=0)
    return matrix[rmin:rmax + 1, cmin:cmax + 1], (int(rmin), int(rmax), int(cmin), int(cmax))
