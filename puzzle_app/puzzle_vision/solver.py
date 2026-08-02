"""Public API for puzzle solving (backward-compatible facade).

All symbols are re-exported from their canonical sub-modules so existing
``from puzzle_vision.solver import ...`` statements continue to work.
New code should import directly from the sub-modules:
``puzzle_vision.solver_base``, ``solver_fixed``, ``solver_taught``,
``solver_card``, ``solver_unknown``.
"""

# 外观门面 — 所有符号从 solver_base / solver_fixed / solver_taught / solver_card / solver_unknown 重新导出

from .solver_base import (
    AssemblyCandidate,
    SolveError,
    UnknownPuzzleSolver,
    _polygon_intersection_area,
    _sample_polygon,
    _shape_alignment,
    _validate_fixed_template,
)
from .solver_card import solve_card
from .solver_fixed import solve_fixed
from .solver_taught import solve_taught
from .solver_unknown import solve_unknown

__all__ = [
    "AssemblyCandidate",
    "SolveError",
    "UnknownPuzzleSolver",
    "_polygon_intersection_area",
    "_sample_polygon",
    "_shape_alignment",
    "_validate_fixed_template",
    "solve_card",
    "solve_fixed",
    "solve_taught",
    "solve_unknown",
]
