"""A4 paper detection — locate and rectify the puzzle sheet in a camera frame.

This module handles the competition task's mandatory A4 sheet: a portrait sheet
with a horizontal black divider at half height.  It provides several detection
strategies (divider-line, central-colour, long-line contour) and a combined
``find_a4_corners`` entry point that tries them in order.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .geometry import (
    edge_lengths,
    normalize_winding,
    polygon_area,
    polygon_centroid,
    safe_interior_point,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class DetectionError(RuntimeError):
    """Raised when the A4 sheet or its divider cannot be found."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PaperView:
    """A rectified top-down view of the A4 puzzle sheet."""

    image: np.ndarray
    homography: np.ndarray
    corners_px: np.ndarray
    pixels_per_mm: float
    width_mm: float
    height_mm: float
    divider_y_mm: float
    divider_width_mm: float
    divider_contrast_lab: float

    def pixels_to_mm(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64) / self.pixels_per_mm


# ---------------------------------------------------------------------------
# Quadrilateral ordering helpers
# ---------------------------------------------------------------------------

def order_quad(points: np.ndarray) -> np.ndarray:
    """Order four points as [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.asarray(points, dtype=np.float32)
    result = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    result[0] = pts[np.argmin(sums)]
    result[2] = pts[np.argmax(sums)]
    result[1] = pts[np.argmin(diffs)]
    result[3] = pts[np.argmax(diffs)]
    return result


def _rotate_corner_order(corners: np.ndarray, quadrants: int) -> np.ndarray:
    """Rotate the four-corner ordering by *quadrants* × 90° CW."""
    q = quadrants % 4
    if q == 0:
        return corners
    return np.roll(corners, -q, axis=0)


# ---------------------------------------------------------------------------
# Boundary contrast / divider validation
# ---------------------------------------------------------------------------

def _quad_boundary_colour_contrast(
    lab_image: np.ndarray,
    quad: np.ndarray,
) -> float:
    """Measure the local Lab colour step across a candidate sheet boundary."""

    height, width = lab_image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255)
    radius = max(2, int(round(min(height, width) * 0.008)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    inside = cv2.subtract(mask, cv2.erode(mask, kernel))
    outside = cv2.subtract(cv2.dilate(mask, kernel), mask)
    inside_samples = lab_image[inside > 0]
    outside_samples = lab_image[outside > 0]
    if len(inside_samples) < 40 or len(outside_samples) < 40:
        return 0.0
    inside_colour = np.median(inside_samples, axis=0)
    outside_colour = np.median(outside_samples, axis=0)
    return float(np.linalg.norm(inside_colour - outside_colour))


def _quick_candidate_has_divider(
    image: np.ndarray,
    corners: np.ndarray,
    paper_cfg: dict[str, Any],
) -> bool:
    """Return True when *corners* enclose the mandatory A4 mid-sheet divider."""

    width = 210
    height = 297
    destination = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(
        np.asarray(corners, dtype=np.float32), destination
    )
    preview = cv2.warpPerspective(image, homography, (width, height))
    try:
        find_divider(preview, paper_cfg, 1.0)
        return True
    except DetectionError:
        return False


# ---------------------------------------------------------------------------
# Detection strategies
# ---------------------------------------------------------------------------

def _find_a4_from_divider_line(
    image: np.ndarray,
    paper_cfg: dict[str, Any],
) -> np.ndarray | None:
    """Recover the complete A4 from its mandatory mid-sheet divider.

    With a near-normal camera, the divider endpoints give the sheet width and
    the known A4 aspect ratio gives the two outer halves.  This remains stable
    when a dark corner of coloured paper blends into the table.
    """

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(gray))
    low = int(max(16, 0.45 * median))
    high = int(min(230, max(low + 28, 1.35 * median)))
    edges = cv2.Canny(gray, low, high)
    minimum_dimension = min(height, width)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 360.0,
        threshold=max(24, int(minimum_dimension * 0.03)),
        minLineLength=max(90, int(minimum_dimension * 0.17)),
        maxLineGap=max(28, int(minimum_dimension * 0.08)),
    )
    if lines is None:
        return None

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    aspect = float(paper_cfg["height_mm"]) / float(paper_cfg["width_mm"])
    image_area = float(width * height)
    maximum_area_ratio = float(
        paper_cfg.get("maximum_image_area_ratio", 0.78)
    )
    candidates: list[tuple[float, np.ndarray]] = []
    for raw in lines[:, 0]:
        first = np.asarray(raw[:2], dtype=np.float64)
        second = np.asarray(raw[2:], dtype=np.float64)
        if first[0] > second[0]:
            first, second = second, first
        vector = second - first
        length = float(np.linalg.norm(vector))
        if length <= 1.0 or abs(vector[1]) > 0.16 * abs(vector[0]):
            continue
        tangent = vector / length
        downward = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
        if downward[1] < 0:
            downward = -downward
        half_height = 0.5 * length * aspect
        quad = np.asarray(
            [
                first - downward * half_height,
                second - downward * half_height,
                second + downward * half_height,
                first + downward * half_height,
            ],
            dtype=np.float32,
        )
        allowance = max(3.0, minimum_dimension * 0.008)
        if (
            np.min(quad[:, 0]) < -allowance
            or np.max(quad[:, 0]) > width + allowance
            or np.min(quad[:, 1]) < -allowance
            or np.max(quad[:, 1]) > height + allowance
        ):
            continue
        area_ratio = abs(cv2.contourArea(quad)) / image_area
        if (
            area_ratio < float(
                paper_cfg.get("minimum_image_area_ratio", 0.045)
            )
            or area_ratio > maximum_area_ratio
        ):
            continue
        boundary_contrast = _quad_boundary_colour_contrast(lab, quad)
        if boundary_contrast < 5.0:
            continue

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255)
        inset = max(3, int(round(minimum_dimension * 0.018)))
        mask = cv2.erode(
            mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * inset + 1, 2 * inset + 1)
            ),
        )
        # Exclude the divider itself before measuring paper uniformity.
        centre_y = int(round(0.25 * np.sum(quad[:, 1])))
        divider_margin = max(2, int(round(minimum_dimension * 0.012)))
        mask[
            max(0, centre_y - divider_margin) :
            min(height, centre_y + divider_margin + 1)
        ] = 0
        samples = lab[mask > 0]
        if len(samples) < 300:
            continue
        paper_colour = np.median(samples, axis=0)
        dispersion = float(
            np.median(np.linalg.norm(samples - paper_colour, axis=1))
        )
        centre = np.mean(quad, axis=0)
        centre_error = float(
            np.linalg.norm(
                centre - np.asarray([width * 0.5, height * 0.5])
            )
            / max(math.hypot(width, height), 1.0)
        )
        score = (
            0.9 * area_ratio
            + 0.025 * min(boundary_contrast, 70.0)
            - 0.014 * min(dispersion, 60.0)
            - 0.12 * centre_error
        )
        candidates.append((score, quad))

    for _, candidate in sorted(
        candidates, key=lambda item: item[0], reverse=True
    ):
        if _quick_candidate_has_divider(image, candidate, paper_cfg):
            return candidate
    return None


def _find_a4_from_central_colour(
    image: np.ndarray,
    paper_cfg: dict[str, Any],
) -> np.ndarray | None:
    """Find the large, nearly uniform sheet that occupies the image centre.

    In the competition setup the A4 is always present, approximately centred,
    and its lower half contains no pieces.  Sampling several central/lower
    patches therefore gives a much stronger and faster prior than requiring a
    perfect grayscale border.  This works for white, blue, red, or other paper
    colours and bridges the black divider before contour extraction.
    """

    height, width = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    sample_colours: list[np.ndarray] = []
    paper_colour = paper_cfg.get("color_bgr")
    if paper_colour is not None:
        pixel = np.asarray(paper_colour, dtype=np.uint8).reshape(1, 1, 3)
        sample_colours.append(
            cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)
        )

    patch_half_w = max(8, int(round(width * 0.055)))
    patch_half_h = max(8, int(round(height * 0.045)))
    for x_fraction, y_fraction in (
        (0.50, 0.62),
        (0.50, 0.52),
        (0.42, 0.62),
        (0.58, 0.62),
        (0.50, 0.40),
    ):
        cx = int(round(width * x_fraction))
        cy = int(round(height * y_fraction))
        patch = lab[
            max(0, cy - patch_half_h) : min(height, cy + patch_half_h + 1),
            max(0, cx - patch_half_w) : min(width, cx + patch_half_w + 1),
        ]
        if patch.size:
            sample_colours.append(np.median(patch.reshape(-1, 3), axis=0))

    image_area = float(width * height)
    target_ratio = float(paper_cfg["height_mm"]) / float(paper_cfg["width_mm"])
    minimum_area_ratio = float(paper_cfg.get("minimum_image_area_ratio", 0.045))
    open_size = max(3, int(round(min(height, width) * 0.005)) | 1)
    divider_bridge_x = max(5, int(round(min(height, width) * 0.010)) | 1)
    divider_bridge_y = max(9, int(round(min(height, width) * 0.018)) | 1)
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_size, open_size)
    )
    bridge_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (divider_bridge_x, divider_bridge_y)
    )
    candidates: list[tuple[float, np.ndarray]] = []
    seen_colours: set[tuple[int, int, int]] = set()
    for colour in sample_colours:
        colour_key = tuple(np.rint(colour / 4.0).astype(int).tolist())
        if colour_key in seen_colours:
            continue
        seen_colours.add(colour_key)
        distance_modes: list[tuple[str, np.ndarray, tuple[float, ...]]] = [
            ("lab", np.linalg.norm(lab - colour, axis=2), (22.0, 32.0, 44.0))
        ]
        chroma_strength = float(
            np.linalg.norm(colour[1:] - np.asarray([128.0, 128.0]))
        )
        if chroma_strength >= 7.0:
            distance_modes.insert(
                0,
                (
                    "chroma",
                    np.linalg.norm(lab[:, :, 1:] - colour[1:], axis=2),
                    (8.0, 12.0, 18.0, 26.0),
                ),
            )
        for distance_mode, distance, tolerances in distance_modes:
            for tolerance in tolerances:
                mask = np.where(distance <= tolerance, 255, 0).astype(np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_CLOSE, bridge_kernel, iterations=2
                )
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    area = abs(cv2.contourArea(contour))
                    if area < image_area * minimum_area_ratio:
                        continue
                    hull = cv2.convexHull(contour)
                    hull_perimeter = cv2.arcLength(hull, True)
                    quad: np.ndarray | None = None
                    for epsilon in (
                        0.008,
                        0.012,
                        0.018,
                        0.025,
                        0.035,
                        0.05,
                        0.075,
                    ):
                        approx = cv2.approxPolyDP(
                            hull, epsilon * hull_perimeter, True
                        )
                        if len(approx) == 4 and cv2.isContourConvex(approx):
                            quad = order_quad(approx.reshape(4, 2))
                            break
                    if quad is None:
                        continue
                    if (
                        np.min(quad[:, 0]) < -2.0
                        or np.max(quad[:, 0]) > width + 2.0
                        or np.min(quad[:, 1]) < -2.0
                        or np.max(quad[:, 1]) > height + 2.0
                    ):
                        continue
                    boundary_counts = (
                        int(np.count_nonzero(quad[:, 0] <= 1.0)),
                        int(
                            np.count_nonzero(
                                quad[:, 0] >= width - 2.0
                            )
                        ),
                        int(np.count_nonzero(quad[:, 1] <= 1.0)),
                        int(
                            np.count_nonzero(
                                quad[:, 1] >= height - 2.0
                            )
                        ),
                    )
                    if max(boundary_counts) >= 2:
                        continue
                    horizontal = 0.5 * (
                        np.linalg.norm(quad[1] - quad[0])
                        + np.linalg.norm(quad[2] - quad[3])
                    )
                    vertical = 0.5 * (
                        np.linalg.norm(quad[3] - quad[0])
                        + np.linalg.norm(quad[2] - quad[1])
                    )
                    if min(horizontal, vertical) < min(height, width) * 0.12:
                        continue
                    ratio = max(horizontal, vertical) / max(
                        min(horizontal, vertical), 1.0
                    )
                    aspect_error = abs(ratio - target_ratio) / target_ratio
                    if aspect_error > float(
                        paper_cfg.get(
                            "maximum_colour_aspect_error_ratio", 0.22
                        )
                    ):
                        continue
                    if (
                        np.min(quad[:, 0]) <= 1.0
                        and np.max(quad[:, 0]) >= width - 2.0
                        and np.min(quad[:, 1]) <= 1.0
                        and np.max(quad[:, 1]) >= height - 2.0
                    ):
                        continue
                    quad_area = abs(cv2.contourArea(quad))
                    if quad_area / image_area > float(
                        paper_cfg.get(
                            "maximum_image_area_ratio", 0.78
                        )
                    ):
                        continue
                    rectangularity = area / max(quad_area, 1.0)
                    centre = np.mean(quad, axis=0)
                    centre_error = float(
                        np.linalg.norm(
                            centre
                            - np.asarray([width * 0.5, height * 0.5])
                        )
                        / max(math.hypot(width, height), 1.0)
                    )
                    score = (
                        3.0 * quad_area / image_area
                        + 0.35 * min(rectangularity, 1.0)
                        - 0.8 * aspect_error
                        - 0.25 * centre_error
                        - 0.002 * tolerance
                        + (0.08 if distance_mode == "chroma" else 0.0)
                    )
                    candidates.append((score, quad))
    if not candidates:
        return None
    unique: dict[tuple[int, ...], tuple[float, np.ndarray]] = {}
    for score, candidate in candidates:
        key = tuple(np.rint(candidate.reshape(-1) / 6.0).astype(int).tolist())
        previous = unique.get(key)
        if previous is None or score > previous[0]:
            unique[key] = (score, candidate)
    candidates = []
    for score, candidate in unique.values():
        boundary_contrast = _quad_boundary_colour_contrast(lab, candidate)
        candidates.append(
            (
                score + 0.018 * min(boundary_contrast, 70.0),
                candidate,
            )
        )
    for _, candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
        candidate = candidate.copy()
        top = np.linalg.norm(candidate[1] - candidate[0])
        left = np.linalg.norm(candidate[3] - candidate[0])
        if top > left:
            candidate = np.roll(candidate, -1, axis=0)
        if _quick_candidate_has_divider(image, candidate, paper_cfg):
            return candidate
    return None


def _find_a4_from_long_lines(
    image: np.ndarray,
    edges: np.ndarray,
    paper_cfg: dict[str, Any],
) -> np.ndarray | None:
    """Recover an A4 quadrilateral when clutter interrupts one outer edge.

    Two long side lines plus either the top or bottom line are sufficient
    because the A4 aspect ratio is known.  This is especially useful when a
    laptop or mechanism sits close to one paper edge.
    """

    height, width = image.shape[:2]
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(45, int(min(height, width) * 0.04)),
        minLineLength=max(120, int(min(height, width) * 0.18)),
        maxLineGap=max(30, int(min(height, width) * 0.045)),
    )
    if lines is None:
        return None

    vertical: list[tuple[float, float, float, float, float]] = []
    horizontal: list[tuple[float, float, float]] = []
    middle_y = height * 0.5
    middle_x = width * 0.5
    for raw in lines[:, 0]:
        x1, y1, x2, y2 = map(float, raw)
        dx, dy = x2 - x1, y2 - y1
        length = float(math.hypot(dx, dy))
        if abs(dy) >= 2.0 * abs(dx) and abs(dy) > 1.0:
            slope = dx / dy
            intercept = x1 - slope * y1
            x_middle = intercept + slope * middle_y
            vertical.append((x_middle, length, slope, intercept, min(y1, y2)))
        elif abs(dx) >= 2.0 * abs(dy) and abs(dx) > 1.0:
            slope = dy / dx
            intercept = y1 - slope * x1
            y_middle = intercept + slope * middle_x
            horizontal.append((y_middle, length, slope))
    if len(vertical) < 2 or not horizontal:
        return None

    def deduplicate(
        values: list[tuple[Any, ...]], coordinate_index: int
    ) -> list[tuple[Any, ...]]:
        selected: dict[int, tuple[Any, ...]] = {}
        for item in values:
            key = int(round(float(item[coordinate_index]) / 18.0))
            if key not in selected or float(item[1]) > float(selected[key][1]):
                selected[key] = item
        return list(selected.values())

    vertical = deduplicate(vertical, 0)
    horizontal = deduplicate(horizontal, 0)
    target_ratio = float(paper_cfg["height_mm"]) / float(paper_cfg["width_mm"])
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    colour_hint_lab: np.ndarray | None = None
    if paper_cfg.get("color_bgr") is not None:
        pixel = np.asarray(paper_cfg["color_bgr"], dtype=np.uint8).reshape(1, 1, 3)
        colour_hint_lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB).reshape(3)

    candidates: list[tuple[float, np.ndarray]] = []
    for left, right in itertools.combinations(vertical, 2):
        if left[0] > right[0]:
            left, right = right, left
        if (
            abs(left[2]) > 0.065
            or abs(right[2]) > 0.065
            or abs(left[2] - right[2]) > 0.055
        ):
            continue
        paper_width = right[0] - left[0]
        if paper_width < width * 0.18:
            continue
        paper_height = paper_width * target_ratio
        side_evidence = min(1.0, (left[1] + right[1]) / max(height, 1.0))
        for y_line, line_length, _ in horizontal:
            for top, bottom in (
                (y_line, y_line + paper_height),
                (y_line - paper_height, y_line),
            ):
                if top < -5 or bottom > height + 5:
                    continue

                def x_at(item: tuple[float, ...], y: float) -> float:
                    return item[3] + item[2] * y

                quad = np.asarray(
                    [
                        [x_at(left, top), top],
                        [x_at(right, top), top],
                        [x_at(right, bottom), bottom],
                        [x_at(left, bottom), bottom],
                    ],
                    dtype=np.float32,
                )
                frame_margin = max(4.0, min(height, width) * 0.008)
                if (
                    np.min(quad[:, 0]) < frame_margin
                    or np.max(quad[:, 0]) > width - frame_margin
                    or top < frame_margin
                    or bottom > height - frame_margin
                ):
                    continue
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255)
                inset = max(5, int(round(min(height, width) * 0.015)))
                mask = cv2.erode(
                    mask,
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (2 * inset + 1, 2 * inset + 1)
                    ),
                )
                samples = lab[mask > 0]
                if len(samples) < 1000:
                    continue
                median = np.median(samples, axis=0)
                dispersion = float(
                    np.median(np.linalg.norm(samples - median, axis=1))
                )
                colour_error = (
                    float(np.linalg.norm(median - colour_hint_lab))
                    if colour_hint_lab is not None
                    else 0.0
                )
                line_evidence = min(1.0, line_length / max(paper_width, 1.0))
                area_score = cv2.contourArea(quad) / float(width * height)
                score = (
                    area_score
                    + 0.18 * side_evidence
                    + 0.12 * line_evidence
                    - 0.018 * dispersion
                    - 0.004 * colour_error
                )
                candidates.append((score, quad))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_a4_corners(frame: np.ndarray, paper_cfg: dict[str, Any]) -> np.ndarray:
    """Detect and return the four corners of the A4 sheet in *frame*.

    Several detection strategies are tried in order:
    1. Divider-line based (most reliable when the black divider is visible)
    2. Central-colour based (works with any paper colour)
    3. Long-line contour based (recovers partial / cluttered views)

    Returns a (4×2) float32 array ordered [TL, TR, BR, BL].
    Raises ``DetectionError`` when nothing plausible is found.
    """
    override = paper_cfg.get("corner_override_px")
    if override is not None:
        corners = np.asarray(override, dtype=np.float32)
        if corners.shape != (4, 2):
            raise DetectionError(
                "paper.corner_override_px must contain four [x,y] points"
            )
        return _rotate_corner_order(
            order_quad(corners), paper_cfg["rotation_quadrants"]
        )

    height, width = frame.shape[:2]
    maximum_dimension = float(paper_cfg.get("detection_max_dimension", 960))
    scale = min(1.0, maximum_dimension / max(height, width))
    small = cv2.resize(
        frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    divider_candidate = _find_a4_from_divider_line(small, paper_cfg)
    divider_area_ratio = (
        abs(cv2.contourArea(divider_candidate))
        / float(max(small.shape[0] * small.shape[1], 1))
        if divider_candidate is not None
        else 0.0
    )
    colour_candidate = None
    if divider_candidate is None or divider_area_ratio < 0.20:
        colour_candidate = _find_a4_from_central_colour(small, paper_cfg)
    if divider_candidate is not None and (
        colour_candidate is None
        or abs(cv2.contourArea(divider_candidate))
        >= 0.70 * abs(cv2.contourArea(colour_candidate))
    ):
        corners = divider_candidate / scale
        return _rotate_corner_order(
            corners.astype(np.float32), paper_cfg["rotation_quadrants"]
        )
    if colour_candidate is not None:
        corners = colour_candidate / scale
        return _rotate_corner_order(
            corners.astype(np.float32), paper_cfg["rotation_quadrants"]
        )

    fallback_dimension = float(
        paper_cfg.get("fallback_detection_max_dimension", 1600)
    )
    fallback_scale = min(1.0, fallback_dimension / max(height, width))
    if fallback_scale > scale + 1e-3:
        scale = fallback_scale
        small = cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(gray))
    low = int(max(20, 0.55 * median))
    high = int(min(240, max(low + 30, 1.45 * median)))
    edges = cv2.Canny(gray, low, high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    image_area = float(small.shape[0] * small.shape[1])
    target_ratio = float(paper_cfg["height_mm"]) / float(paper_cfg["width_mm"])
    paper_colour = paper_cfg.get("color_bgr")
    colour_hint_lab: np.ndarray | None = None
    small_lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    if paper_colour is not None:
        colour_pixel = (
            np.asarray(paper_colour, dtype=np.uint8).reshape(1, 1, 3)
        )
        colour_hint_lab = (
            cv2.cvtColor(colour_pixel, cv2.COLOR_BGR2LAB)
            .reshape(3)
            .astype(np.float32)
        )
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = abs(cv2.contourArea(contour))
        if area < image_area * 0.12:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = order_quad(approx.reshape(4, 2))
        horizontal = 0.5 * (
            np.linalg.norm(quad[1] - quad[0])
            + np.linalg.norm(quad[2] - quad[3])
        )
        vertical = 0.5 * (
            np.linalg.norm(quad[3] - quad[0])
            + np.linalg.norm(quad[2] - quad[1])
        )
        if min(horizontal, vertical) < 1:
            continue
        ratio = max(horizontal, vertical) / min(horizontal, vertical)
        aspect_error = abs(ratio - target_ratio) / target_ratio
        if aspect_error > 0.24:
            continue
        rectangularity = area / max(horizontal * vertical, 1.0)
        score = area / image_area - 0.8 * aspect_error + 0.1 * rectangularity
        boundary_contrast = _quad_boundary_colour_contrast(small_lab, quad)
        score += 0.018 * min(boundary_contrast, 70.0)
        if colour_hint_lab is not None:
            colour_mask = np.zeros(small.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(
                colour_mask, np.rint(quad).astype(np.int32), 255
            )
            inset = max(3, int(round(min(small.shape[:2]) * 0.01)))
            colour_mask = cv2.erode(
                colour_mask,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * inset + 1, 2 * inset + 1)
                ),
            )
            samples = small_lab[colour_mask > 0]
            if len(samples):
                observed_colour = np.median(samples, axis=0)
                colour_error = float(
                    np.linalg.norm(observed_colour - colour_hint_lab)
                )
                score -= float(paper_cfg.get("color_hint_weight", 0.8)) * min(
                    colour_error / 100.0, 1.5
                )
        candidates.append((score, quad))

    corners = None
    if candidates:
        for _, candidate in sorted(
            candidates, key=lambda item: item[0], reverse=True
        ):
            candidate_for_check = candidate.copy()
            candidate_top = np.linalg.norm(
                candidate_for_check[1] - candidate_for_check[0]
            )
            candidate_left = np.linalg.norm(
                candidate_for_check[3] - candidate_for_check[0]
            )
            if candidate_top > candidate_left:
                candidate_for_check = np.roll(
                    candidate_for_check, -1, axis=0
                )
            if _quick_candidate_has_divider(
                small, candidate_for_check, paper_cfg
            ):
                corners = candidate_for_check
                break
    if corners is None:
        corners = _find_a4_from_long_lines(small, edges, paper_cfg)
        if corners is None:
            raise DetectionError(
                "A4 sheet was not found. Improve border contrast or set "
                "paper.corner_override_px in config.json."
            )
    corners = corners / scale

    top = np.linalg.norm(corners[1] - corners[0])
    left = np.linalg.norm(corners[3] - corners[0])
    if top > left:
        corners = np.roll(corners, -1, axis=0)
    return _rotate_corner_order(
        corners.astype(np.float32), paper_cfg["rotation_quadrants"]
    )


def rectify_paper(
    frame: np.ndarray,
    paper_cfg: dict[str, Any],
    cached_corners: np.ndarray | None = None,
) -> PaperView:
    """Rectify the A4 sheet to a top-down ``PaperView``."""
    corners = (
        np.asarray(cached_corners, dtype=np.float32).copy()
        if cached_corners is not None
        else find_a4_corners(frame, paper_cfg)
    )
    ppm = float(paper_cfg["pixels_per_mm"])
    out_w = int(round(float(paper_cfg["width_mm"]) * ppm))
    out_h = int(round(float(paper_cfg["height_mm"]) * ppm))
    destination = np.array(
        [[0, 0], [out_w - 1, 0],
         [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(corners, destination)
    image = cv2.warpPerspective(frame, homography, (out_w, out_h))
    divider_y_mm, divider_width_mm, divider_contrast = find_divider(
        image, paper_cfg, ppm
    )
    return PaperView(
        image=image,
        homography=homography,
        corners_px=corners,
        pixels_per_mm=ppm,
        width_mm=float(paper_cfg["width_mm"]),
        height_mm=float(paper_cfg["height_mm"]),
        divider_y_mm=divider_y_mm,
        divider_width_mm=divider_width_mm,
        divider_contrast_lab=divider_contrast,
    )


def find_divider(
    rectified: np.ndarray,
    paper_cfg: dict[str, Any],
    pixels_per_mm: float,
) -> tuple[float, float, float]:
    """Locate and validate the horizontal upper/lower divider on the A4 sheet.

    The competition sheet has one obvious solid line near half height.  Median
    Lab colour per row makes the test insensitive to the A4/table colours and
    suppresses local dirt, wrinkles, and pieces that cover only part of a row.

    Returns ``(divider_y_mm, divider_width_mm, peak_lab_contrast)``.
    """

    ppm = float(pixels_per_mm)
    expected = int(round(float(paper_cfg["divider_y_mm"]) * ppm))
    half_range = max(
        1,
        int(round(float(paper_cfg["divider_search_half_range_mm"]) * ppm)),
    )
    maximum_width = max(
        1,
        int(round(float(paper_cfg["divider_max_width_mm"]) * ppm)),
    )
    minimum_contrast = float(paper_cfg["divider_min_contrast_lab"])
    x_margin = max(2, int(round(8.0 * ppm)))
    if rectified.shape[1] <= 2 * x_margin + 20:
        raise DetectionError("Rectified A4 image is too narrow to validate divider")

    lab = cv2.cvtColor(rectified, cv2.COLOR_BGR2LAB).astype(np.float32)
    row_colour = np.median(lab[:, x_margin:-x_margin], axis=1)
    start = max(maximum_width + 2, expected - half_range)
    stop = min(len(row_colour) - maximum_width - 2, expected + half_range + 1)
    if stop <= start:
        raise DetectionError("Divider search window is outside the rectified A4")

    gap = maximum_width + max(2, int(round(1.0 * ppm)))
    flank = max(3, int(round(3.0 * ppm)))
    scores = np.zeros(stop - start, dtype=np.float32)
    for offset, y in enumerate(range(start, stop)):
        upper = row_colour[max(0, y - gap - flank) : y - gap]
        lower = row_colour[
            y + gap + 1 : min(len(row_colour), y + gap + flank + 1)
        ]
        if len(upper) == 0 or len(lower) == 0:
            continue
        baseline = np.median(np.vstack((upper, lower)), axis=0)
        scores[offset] = float(np.linalg.norm(row_colour[y] - baseline))

    peak_offset = int(np.argmax(scores))
    peak_score = float(scores[peak_offset])
    if peak_score < minimum_contrast:
        raise DetectionError(
            "A4 sheet was found, but the required horizontal divider was not. "
            f"Peak Lab contrast {peak_score:.1f} is below "
            f"{minimum_contrast:.1f}."
        )

    band_threshold = max(minimum_contrast, peak_score * 0.45)
    left = peak_offset
    right = peak_offset
    while left > 0 and scores[left - 1] >= band_threshold:
        left -= 1
    while right + 1 < len(scores) and scores[right + 1] >= band_threshold:
        right += 1
    detected_width_px = right - left + 1

    band_scores = scores[left : right + 1].astype(np.float64)
    band_rows = np.arange(
        start + left, start + right + 1, dtype=np.float64
    )
    centre_px = float(np.average(band_rows, weights=band_scores))
    return (
        centre_px / ppm,
        detected_width_px / ppm,
        peak_score,
    )
