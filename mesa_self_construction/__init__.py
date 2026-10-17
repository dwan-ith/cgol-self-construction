"""
Mesa Self-Construction: Observable, Reproducible Self-Constructing Automata in Mesa.
Based on EZLiang/cgol-self-construction, slow-salvo compilers, and John von Neumann's 29-state automaton.
"""

from mesa_self_construction.model import SelfConstructionModel
from mesa_self_construction.agents import ConstructionCell
from mesa_self_construction.salvo import (
    SalvoSchedule,
    ScheduledGlider,
    parse_lane_phase_token,
    parse_lane_phase_recipe,
    lane_to_position,
    load_simeks_toolkit,
)
from mesa_self_construction.presets import ALL_PRESETS, get_preset
from mesa_self_construction.validator import (
    validate_target_pattern,
    validate_machine,
    analyze_construction,
    ValidationResult,
    detect_period,
)
from mesa_self_construction.mstate_rle import (
    decode_mstate_rle,
    encode_mstate_rle,
    MultiStatePattern,
)
from mesa_self_construction import von_neumann
from mesa_self_construction.von_neumann import (
    step_von_neumann_29,
    step_von_neumann_29_scalar,
    load_jvn29_table,
    parse_jvn29_table,
    load_jvn_pattern,
    KNOWN_PATTERNS,
    STATE_NAMES as VN_STATE_NAMES,
)

__version__ = "1.0.0"

__all__ = [
    "SelfConstructionModel",
    "ConstructionCell",
    "SalvoSchedule",
    "ScheduledGlider",
    "parse_lane_phase_token",
    "parse_lane_phase_recipe",
    "lane_to_position",
    "load_simeks_toolkit",
    "ALL_PRESETS",
    "get_preset",
    "validate_target_pattern",
    "validate_machine",
    "analyze_construction",
    "ValidationResult",
    "detect_period",
    "decode_mstate_rle",
    "encode_mstate_rle",
    "MultiStatePattern",
    "von_neumann",
    "step_von_neumann_29",
    "step_von_neumann_29_scalar",
    "load_jvn29_table",
    "parse_jvn29_table",
    "load_jvn_pattern",
    "KNOWN_PATTERNS",
    "VN_STATE_NAMES",
]
