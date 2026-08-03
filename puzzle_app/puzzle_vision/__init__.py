"""# 拼图视觉求解库 — A4纸拼图设备的视觉处理模块
Vision-only solver for the A4 puzzle device.

Public API
----------
- ``PuzzleVisionPipeline`` — full detect → solve → plan pipeline
- ``interfaces`` — protocol definitions (IPaperDetector, IPieceDetector, …)
- ``detector`` — backward-compatible facade for paper + piece detection
- ``solver`` — backward-compatible facade for all solver modes
- ``config`` — configuration loading and management
- ``geometry`` — low-level geometric primitives
"""

from .pipeline import PuzzleVisionPipeline

# Convenience re-exports
from .detector import (
    DetectionError,
    PaperView,
    PieceObservation,
    detect_pieces,
    find_a4_corners,
    find_divider,
    foreground_mask,
    order_quad,
    rectify_paper,
)
from .solver import (
    AssemblyCandidate,
    SolveError,
    UnknownPuzzleSolver,
    solve_card,
    solve_fixed,
    solve_taught,
    solve_unknown,
)

__all__ = [
    # Pipeline
    "PuzzleVisionPipeline",
    # Detector symbols
    "DetectionError",
    "PaperView",
    "PieceObservation",
    "detect_pieces",
    "find_a4_corners",
    "find_divider",
    "foreground_mask",
    "order_quad",
    "rectify_paper",
    # Solver symbols
    "AssemblyCandidate",
    "SolveError",
    "UnknownPuzzleSolver",
    "solve_card",
    "solve_fixed",
    "solve_taught",
    "solve_unknown",
]
