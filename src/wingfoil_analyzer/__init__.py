"""Heuristic wingfoil activity analysis."""

from .analysis import (
    add_surf,
    courses,
    crashes,
    events,
    foil_onsets,
    recover_transitions,
    segment,
    surf_metrics,
    wrap,
)
from .causal_crashes import CrashConfig, CrashDetector, detect_crashes
from .hsmm import HSMMModel, State, states_and_edges
from .hsmm import decode as decode_hsmm
from .loading import load
from .phase_refinement import refine_phases

__all__ = [
    "add_surf",
    "CrashConfig",
    "CrashDetector",
    "crashes",
    "courses",
    "events",
    "detect_crashes",
    "decode_hsmm",
    "foil_onsets",
    "load",
    "HSMMModel",
    "refine_phases",
    "recover_transitions",
    "segment",
    "State",
    "states_and_edges",
    "surf_metrics",
    "wrap",
]
