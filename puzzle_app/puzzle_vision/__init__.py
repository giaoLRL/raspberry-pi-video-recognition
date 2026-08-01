"""Vision-only solver for the A4 puzzle device."""

try:
    from .pipeline import PuzzleVisionPipeline
    __all__ = ["PuzzleVisionPipeline"]
except ImportError:
    __all__ = []