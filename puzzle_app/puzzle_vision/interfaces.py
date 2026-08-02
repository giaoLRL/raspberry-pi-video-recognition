"""Abstract interfaces for the puzzle vision pipeline.

Each protocol defines the contract that a component must fulfil.  Concrete
implementations are provided by the same package; the protocols exist so that
callers can depend on the interface rather than a specific implementation, and
so that unit tests can substitute mock detectors and solvers.
"""

# 抽象接口定义（Protocol）：供调用方依赖注入和单元测试mock使用

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Forward references for type hints (resolved at runtime via typing.TYPE_CHECKING)
# ---------------------------------------------------------------------------

class PaperView(Protocol):
    """Structural protocol matching detector.PaperView."""
    # PaperView结构协议（供前向引用）
    image: np.ndarray
    pixels_per_mm: float
    width_mm: float
    height_mm: float
    divider_y_mm: float
    divider_width_mm: float
    divider_contrast_lab: float
    homography: np.ndarray
    corners_px: np.ndarray


class PieceObservation(Protocol):
    """Structural protocol matching detector.PieceObservation."""
    # PieceObservation结构协议（供前向引用）
    id: str
    polygon_mm: np.ndarray
    centroid_mm: np.ndarray
    pickup_mm: np.ndarray
    area_mm2: float
    perimeter_mm: float
    edge_lengths_mm: np.ndarray
    contour_px: np.ndarray


# ---------------------------------------------------------------------------
# Component interfaces
# ---------------------------------------------------------------------------

@runtime_checkable
class IPaperDetector(Protocol):
    """Detect and rectify the A4 puzzle sheet in a camera frame."""
    # A4纸检测器接口：检测纸张位置并校正

    def find_a4_corners(self, frame: np.ndarray) -> np.ndarray:
        """Return the four corner points (4×2) of the A4 sheet in image pixels."""
        ...

    def rectify_paper(
        self,
        frame: np.ndarray,
        corners: np.ndarray | None = None,
    ) -> PaperView:
        """Rectify the A4 sheet to a top-down view and locate the divider line."""
        ...


@runtime_checkable
class IPieceDetector(Protocol):
    """Segment puzzle pieces from a rectified paper image."""
    # 拼图块检测器接口：从校正图像中分割出拼图块

    def detect_pieces(
        self,
        paper: PaperView,
        paper_cfg: dict[str, Any],
        segmentation_cfg: dict[str, Any],
        background_rectified: np.ndarray | None = None,
        source_region: str = "upper",
    ) -> tuple[list[PieceObservation], np.ndarray, str, str]:
        """Return (observations, mask, detected_region, selected_mode)."""
        ...


@runtime_checkable
class IPuzzleSolver(Protocol):
    """Assemble detected pieces into a target layout."""
    # 拼图求解器接口：将检测到的拼图块组装成目标布局

    def solve(
        self,
        observations: list[PieceObservation],
        config: dict[str, Any],
        rectified_image: np.ndarray | None = None,
        pixels_per_mm: float = 4.0,
        use_texture: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Return (plan, solver_info)."""
        ...


@runtime_checkable
class IPuzzlePipeline(Protocol):
    """Complete vision pipeline: detect → solve → plan."""
    # 完整流水线接口：检测→求解→输出方案

    def process_frame(self, frame: np.ndarray) -> dict[str, Any]:
        """Run detection and solving on a single camera frame.

        Returns a dictionary with keys such as ``plan``, ``solver_info``,
        ``observations``, ``paper_view``, ``mask``.
        """
        ...

    def solve(
        self,
        observations: list[PieceObservation],
        mode: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Dispatch to the appropriate solver for the given *mode*."""
        ...
