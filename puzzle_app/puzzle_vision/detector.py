"""Public API for paper and piece detection (backward-compatible facade).

All symbols are re-exported from their canonical sub-modules so existing
``from puzzle_vision.detector import ...`` statements continue to work.
New code should import directly from the sub-modules:
``puzzle_vision.paper_detection`` and ``puzzle_vision.piece_detection``.
"""

from .paper_detection import (
    DetectionError,
    PaperView,
    find_a4_corners,
    find_divider,
    order_quad,
    rectify_paper,
)
from .piece_detection import (
    PieceObservation,
    detect_pieces,
    foreground_mask,
)

__all__ = [
    "DetectionError",
    "PaperView",
    "PieceObservation",
    "detect_pieces",
    "find_a4_corners",
    "find_divider",
    "foreground_mask",
    "order_quad",
    "rectify_paper",
]
