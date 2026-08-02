"""Vision pipeline — detect paper, segment pieces, and solve in one call.

``PuzzleVisionPipeline`` coordinates the full detect → solve → plan flow,
exposing a single ``process_frame`` entry point suitable for both interactive
use and headless robot control.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .paper_detection import DetectionError, find_a4_corners, rectify_paper
from .piece_detection import detect_pieces
from .solver import (
    SolveError,
    solve_card,
    solve_fixed,
    solve_taught,
    solve_unknown,
)


# 完整视觉流水线：纸张检测 → 拼图块分割 → 求解 → 输出方案
class PuzzleVisionPipeline:
    """Complete vision pipeline: paper detection → piece segmentation → solve.

    Parameters
    ----------
    config : dict
        Full configuration dictionary with keys ``paper``, ``segmentation``,
        ``fixed``, ``unknown``, and optionally ``taught_layouts``.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    # 定位A4纸四角
    def find_paper(self, frame: np.ndarray) -> np.ndarray:
        """Locate and return the four corners of the A4 sheet."""
        return find_a4_corners(frame, self.config["paper"])

    # 将A4纸校正为俯视图PaperView
    def rectify(self, frame: np.ndarray, cached_corners: np.ndarray | None = None):
        """Return a ``PaperView`` (top-down rectified A4)."""
        from .paper_detection import PaperView
        return rectify_paper(frame, self.config["paper"], cached_corners)

    # 从校正图中分割出拼图块
    def find_pieces(
        self,
        paper,
        background_rectified: np.ndarray | None = None,
        source_region: str = "upper",
    ):
        """Return ``(observations, mask, region, mode)`` for *paper*."""
        return detect_pieces(
            paper,
            self.config["paper"],
            self.config["segmentation"],
            background_rectified,
            source_region,
        )

    # ------------------------------------------------------------------
    # Solve dispatch
    # ------------------------------------------------------------------

    # 根据模式分派到对应求解器（fixed/card/taught/unknown）
    def solve(
        self,
        observations,
        mode: str,
        rectified_image: np.ndarray | None = None,
        pixels_per_mm: float = 4.0,
        taught_layout: dict[str, Any] | None = None,
        use_texture: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Dispatch to the appropriate solver for the given *mode*.

        Parameters
        ----------
        observations : list[PieceObservation]
            Detected puzzle pieces.
        mode : str
            One of ``"fixed"``, ``"card"``, ``"taught"``, ``"unknown"``.
        rectified_image : np.ndarray, optional
            Top-down paper image (used for texture scoring in card mode).
        pixels_per_mm : float
            Pixels per millimetre in *rectified_image*.
        taught_layout : dict, optional
            Pre-taught layout data (required when *mode* is ``"taught"``).
        use_texture : bool
            Enable texture-based seam scoring.

        Returns
        -------
        (plan, solver_info) — *plan* is a list of per-piece motion dicts;
        *solver_info* holds diagnostic data.
        """
        if mode == "fixed":
            plan, info = solve_fixed(observations, dict(self.config["fixed"]))
            info["solver_method"] = "fixed"
            return plan, info

        unknown_cfg = dict(self.config["unknown"])

        # Target zone defaults to the lower half of A4.
        if "target_zone_mm" not in unknown_cfg:
            pw = float(self.config["paper"].get("width_mm", 210.0))
            ph = float(self.config["paper"].get("height_mm", 297.0))
            unknown_cfg["target_zone_mm"] = [0.0, ph * 0.5, pw, ph]

        if mode == "card":
            unknown_cfg["target_orientation"] = "portrait"
            plan, info = solve_card(
                observations,
                unknown_cfg,
                rectified_image,
                pixels_per_mm,
            )
            info["solver_method"] = "card"
            return plan, info

        if mode == "taught":
            if taught_layout is None:
                raise SolveError("Taught layout is required for 'taught' mode")
            unknown_cfg["target_orientation"] = "landscape"
            plan, info = solve_taught(observations, taught_layout, unknown_cfg)
            info["solver_method"] = "taught"
            return plan, info

        if mode == "unknown":
            unknown_cfg["target_orientation"] = "landscape"
            plan, info = solve_unknown(
                observations,
                unknown_cfg,
                rectified_image,
                pixels_per_mm,
                use_texture=use_texture,
            )
            info["solver_method"] = "unknown"
            return plan, info

        raise ValueError(f"Unsupported solve mode: {mode}")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    # 单帧完整处理入口：检测→分割→求解，返回方案字典
    def process_frame(
        self,
        frame: np.ndarray,
        mode: str = "unknown",
        source_region: str = "upper",
        cached_corners: np.ndarray | None = None,
        background_rectified: np.ndarray | None = None,
        taught_layout: dict[str, Any] | None = None,
        use_texture: bool = False,
    ) -> dict[str, Any]:
        """Run detection and solving on a single camera frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR camera image.
        mode : str
            Solve mode: ``"fixed"``, ``"card"``, ``"taught"``, or
            ``"unknown"`` (default).
        source_region : str
            ``"upper"``, ``"lower"``, or ``"auto"`` — which half of the A4
            sheet contains the puzzle pieces.
        cached_corners : np.ndarray, optional
            Pre-computed A4 corners (skips paper detection).
        background_rectified : np.ndarray, optional
            Empty-sheet reference for background subtraction.
        taught_layout : dict, optional
            Taught layout data (required for ``mode="taught"``).
        use_texture : bool
            Enable texture-based seam scoring in the solver.

        Returns
        -------
        dict with keys:
            ``plan`` — list of per-piece motion dicts (pick_mm, place_mm,
                rotate_deg, …)
            ``solver_info`` — diagnostic metadata
            ``observations`` — list of ``PieceObservation`` values
            ``paper_view`` — ``PaperView`` (top-down rectified image)
            ``mask`` — binary foreground mask
            ``detected_region`` — which half-sheet was used
            ``selected_mode`` — the colour-segmentation mode selected
            ``corners_px`` — A4 corner coordinates in image pixels
            ``elapsed_sec`` — total processing time
        """
        started = time.perf_counter()
        result: dict[str, Any] = {}

        # 1. Paper detection
        corners = cached_corners
        if corners is None:
            corners = self.find_paper(frame)
        result["corners_px"] = np.round(corners, 2).tolist()

        # 2. Rectify
        paper = self.rectify(frame, corners)

        # 3. Piece detection
        observations, mask, detected_region, selected_mode = self.find_pieces(
            paper, background_rectified, source_region
        )

        # 4. Solve
        ppm = paper.pixels_per_mm
        plan, solver_info = self.solve(
            observations,
            mode,
            rectified_image=paper.image,
            pixels_per_mm=ppm,
            taught_layout=taught_layout,
            use_texture=use_texture,
        )

        result.update(
            {
                "plan": plan,
                "solver_info": solver_info,
                "observations": observations,
                "paper_view": paper,
                "mask": mask,
                "detected_region": detected_region,
                "selected_mode": selected_mode,
                "elapsed_sec": round(time.perf_counter() - started, 4),
            }
        )
        return result
