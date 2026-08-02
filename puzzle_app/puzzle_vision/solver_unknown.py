"""Unknown-puzzle solver — thin wrapper around ``UnknownPuzzleSolver``.

This is the default autonomous solver for pieces whose target layout is not
known in advance.  It delegates entirely to the rigid edge-docking search in
``solver_base.UnknownPuzzleSolver``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .detector import PieceObservation
from .solver_base import UnknownPuzzleSolver


def solve_unknown(
    observations: list[PieceObservation],
    unknown_cfg: dict[str, Any],
    rectified_image: np.ndarray | None,
    pixels_per_mm: float,
    use_texture: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return UnknownPuzzleSolver(
        observations,
        unknown_cfg,
        rectified_image=rectified_image,
        pixels_per_mm=pixels_per_mm,
        use_texture=use_texture,
    ).solve()
