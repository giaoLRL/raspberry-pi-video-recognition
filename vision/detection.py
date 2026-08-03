#!/usr/bin/env python3
"""Piece detection ? A4 paper detection + puzzle piece segmentation.

Extracted from main.py. Dependencies injected via setup().
"""

import math
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from puzzle_vision.detector import PieceObservation
from puzzle_vision.geometry import safe_interior_point, polygon_centroid as solver_polygon_centroid

_ctx = {}

# ?? Constants (from main.py) ??
WARP_WIDTH = 840
WARP_HEIGHT = 1188
A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
PIXELS_PER_MM_X = WARP_WIDTH / A4_WIDTH_MM
PIXELS_PER_MM_Y = WARP_HEIGHT / A4_HEIGHT_MM
PIXELS_PER_CM = (WARP_WIDTH / A4_WIDTH_MM + WARP_HEIGHT / A4_HEIGHT_MM) * 0.5 * 10.0
MIN_SIDES = 3
MAX_SIDES = 5
POLYGON_EPSILON_RATIO = 0.015
MIN_PIECE_SHORT_SIDE_PX = 24.0
MIN_FILL_RATIO = 0.20
MIN_CONTOUR_AREA_PX = 80
MIN_CONTOUR_WIDTH = 20
MIN_CONTOUR_HEIGHT = 20
ORDERED_PIECE_IDS = ["piece_1", "piece_2", "piece_3", "piece_4"]

def setup(*, calib_file=None, pixels_per_mm=None,
          min_piece_area_px=None, max_piece_area_px=None,
          log_print=None):
    _ctx["CALIBRATION_FILE"] = calib_file
    _ctx["PIXELS_PER_MM"] = pixels_per_mm
    _ctx["MIN_PIECE_AREA_PX"] = min_piece_area_px
    _ctx["MAX_PIECE_AREA_PX"] = max_piece_area_px
    _ctx["log_print"] = log_print


@dataclass
class DetectedPiece:

    piece_id: int

    center_x: float

    center_y: float

    center_x_image: float

    center_y_image: float

    pickup_x_image: float  # safe_interior_point in warp px

    pickup_y_image: float

    area_px: float

    vertices: list

    contour: np.ndarray

    polygon: np.ndarray


@dataclass

class RecoveryResult:

    requested_mode: str

    selected_mode: str

    plan: list[dict[str, Any]]

    solver_info: dict[str, Any]

    observations: list[PieceObservation]

    pattern_score: float

    solve_time_sec: float

    attempts: list[dict[str, Any]]


# ============================================================

# Calibration helpers (original)

# ============================================================

def order_points(pts: np.ndarray) -> np.ndarray:

    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]

    rect[2] = pts[np.argmax(s)]

    d = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(d)]

    rect[3] = pts[np.argmax(d)]

    return rect


def load_corners() -> Optional[np.ndarray]:

    if not _ctx['CALIBRATION_FILE'].exists():

        return None

    try:

        data = json.loads(_ctx['CALIBRATION_FILE'].read_text(encoding="utf-8"))

        corners = np.asarray(data["corners"], dtype=np.float32)

        if corners.shape == (4, 2):

            corners = order_points(corners)

        return corners

    except Exception:

        return None


def build_matrices(corners: np.ndarray):

    # 自动检测A4方向：WARP坐标系按竖向A4设计(840宽=210mm, 1188高=297mm)。

    # 若画面中A4顶边(宽) > 左边(高)，说明A4横放，需旋转corners使短边对应WARP_WIDTH。

    w_top = float(np.linalg.norm(corners[1] - corners[0]))

    h_left = float(np.linalg.norm(corners[3] - corners[0]))

    if w_top > h_left:

        corners = np.roll(corners, -1, axis=0)

    dst = np.array([[0, 0], [WARP_WIDTH - 1, 0],

                    [WARP_WIDTH - 1, WARP_HEIGHT - 1], [0, WARP_HEIGHT - 1]],

                   dtype=np.float32)

    c2w = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    c2w = c2w.astype(np.float32)

    w2c = cv2.getPerspectiveTransform(dst, corners.astype(np.float32))

    return c2w, w2c


def _a4_otsu_binary(frame):
    """Shared Otsu binary for A4 detection (dark foreground, light background)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_big, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed


def _a4_otsu_binary_light(board_mode=False):
    """Return threshold function for the given board color mode.
    board_mode=False: light paper on dark background (BINARY)
    board_mode=True: dark board on light background (BINARY_INV)
    """
    mode = cv2.THRESH_BINARY_INV if board_mode else cv2.THRESH_BINARY
    def _inner(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, mode + cv2.THRESH_OTSU)
        kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_big, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel, iterations=1)
        return closed
    return _inner


def _try_a4_from_closed(closed, h, w):
    """Extract A4 contour from a closed binary image."""
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    min_area = w * h * 0.05
    max_area = w * h * 0.95
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(c, True)
        for eps in [0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                pts = approx.reshape(4, 2).astype(np.float32)
                ordered = order_points(pts)
                wa = np.linalg.norm(ordered[1] - ordered[0])
                ha = np.linalg.norm(ordered[3] - ordered[0])
                if wa < 10 or ha < 10:
                    continue
                ratio = max(wa, ha) / max(min(wa, ha), 1.0)
                if 1.30 < ratio < 1.55:
                    candidates.append((area, ordered))
                    break
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return None


def auto_detect_a4(frame):
    """Detect A4 paper/board using Otsu thresholding.
    Tries both light-paper-on-dark (BINARY) and dark-board-on-light (BINARY_INV) modes.
    """
    h, w = frame.shape[:2]

    # Try both threshold modes
    for board_mode in (False, True):
        thresh_fn = _a4_otsu_binary_light(board_mode)
        closed = thresh_fn(frame)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    candidates = []

    min_area = w * h * 0.05

    max_area = w * h * 0.95

    for c in contours:

        area = cv2.contourArea(c)

        if area < min_area or area > max_area:

            continue

        peri = cv2.arcLength(c, True)

        for eps in [0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:

            approx = cv2.approxPolyDP(c, eps * peri, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):

                pts = approx.reshape(4, 2).astype(np.float32)

                ordered = order_points(pts)

                wa = np.linalg.norm(ordered[1] - ordered[0])

                ha = np.linalg.norm(ordered[3] - ordered[0])

                if wa < 10 or ha < 10:

                    continue

                ratio = max(wa, ha) / max(min(wa, ha), 1.0)

                if 1.30 < ratio < 1.55:

                    candidates.append((area, ordered))

                break

    if candidates:

        candidates.sort(key=lambda x: x[0], reverse=True)

        print(f"[A4 OTSU] found quad area={candidates[0][0]:.0f}px ratio={candidates[0][0]/(w*h)*100:.0f}%", flush=True)

        return candidates[0][1]


    return None


# ============================================================

def auto_detect_a4_partial(frame):
    """Fallback: find approximate A4 corners when paper is partially out of frame.

    Uses convex hull of largest dark region + minAreaRect to guess corners.
    """
    try:
        h, w = frame.shape[:2]
        # Try both threshold modes for partial detection
        for board_mode in (False, True):
            thresh_fn = _a4_otsu_binary_light(board_mode)
            closed = thresh_fn(frame)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area < w * h * 0.03:
                continue
            hull = cv2.convexHull(largest)
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            box = box.astype(np.float32)
            box_ordered = order_points(box)
            wa = np.linalg.norm(box_ordered[1] - box_ordered[0])
            ha = np.linalg.norm(box_ordered[3] - box_ordered[0])
            if wa < 20 or ha < 20:
                continue
            ratio = max(wa, ha) / max(min(wa, ha), 1.0)
            if 1.1 < ratio < 1.8:
                return box_ordered
        return None
    except Exception:
        return None

def create_piece_mask(calibrated_region: np.ndarray) -> np.ndarray:

    hsv = cv2.cvtColor(calibrated_region, cv2.COLOR_BGR2HSV)

    value_channel = hsv[:, :, 2]

    blurred = cv2.GaussianBlur(value_channel, (5, 5), 0)

    threshold_value, mask = cv2.threshold(

        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if threshold_value < 30:

        _, mask = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    return mask


def calculate_area_center(contour: np.ndarray) -> Optional[tuple[int, int]]:

    moments = cv2.moments(contour)

    if abs(moments["m00"]) < 1e-8:

        return None

    center_x = int(round(moments["m10"] / moments["m00"]))

    center_y = int(round(moments["m01"] / moments["m00"]))

    return center_x, center_y


def approximate_straight_polygon(contour: np.ndarray) -> Optional[np.ndarray]:

    hull = cv2.convexHull(contour)

    perimeter = cv2.arcLength(hull, closed=True)

    if perimeter <= 0:

        return None

    ratios = [POLYGON_EPSILON_RATIO, 0.010, 0.012, 0.018, 0.020, 0.025, 0.030, 0.040]

    candidates: list[tuple[float, np.ndarray]] = []

    hull_area = max(cv2.contourArea(hull), 1.0)

    for ratio in ratios:

        polygon = cv2.approxPolyDP(hull, ratio * perimeter, closed=True)

        side_count = len(polygon)

        if not MIN_SIDES <= side_count <= MAX_SIDES:

            continue

        if not cv2.isContourConvex(polygon):

            continue

        polygon_area = cv2.contourArea(polygon)

        area_error = abs(hull_area - polygon_area) / hull_area

        candidates.append((area_error, polygon))

    if not candidates:

        return None

    candidates.sort(key=lambda item: item[0])

    return candidates[0][1]


def convert_polygon_vertices_to_output(polygon: np.ndarray) -> list[tuple[int, int]]:

    return [(int(round(p[0][0])), int(round(p[0][1]))) for p in polygon]


def detect_pieces(calibrated_region: np.ndarray) -> tuple[list[DetectedPiece], np.ndarray]:

    mask = create_piece_mask(calibrated_region)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[dict] = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < _ctx['MIN_PIECE_AREA_PX'] or area > _ctx['MAX_PIECE_AREA_PX']:

            continue

        rotated_rect = cv2.minAreaRect(contour)

        rect_width, rect_height = rotated_rect[1]

        short_side = min(rect_width, rect_height)

        if short_side < MIN_PIECE_SHORT_SIDE_PX:

            continue

        fill_ratio = area / (rect_width * rect_height) if rect_width * rect_height > 0 else 0.0

        if fill_ratio < MIN_FILL_RATIO:

            continue

        polygon = approximate_straight_polygon(contour)

        if polygon is None:

            continue

        center = calculate_area_center(contour)

        if center is None:

            continue

        vertices = convert_polygon_vertices_to_output(polygon)

        if len(vertices) != len(polygon):

            continue

        candidates.append({

            "area": area, "center": center, "vertices": vertices,

            "contour": contour, "polygon": polygon,

        })

    candidates.sort(key=lambda item: (item["area"], item["center"][1], item["center"][0]))

    if candidates:

        parts = []

        for c in candidates:

            rr = cv2.minAreaRect(c["contour"])

            w_mm = rr[1][0] / _ctx['PIXELS_PER_MM']

            h_mm = rr[1][1] / _ctx['PIXELS_PER_MM']

            parts.append(f"{c['area']:.0f}px²({c['area']/PIXELS_PER_CM**2:.1f}cm², {w_mm:.1f}x{h_mm:.1f}mm)")

        print(f"[DETECT] {len(candidates)} pieces: {' | '.join(parts)}", flush=True)

    pieces: list[DetectedPiece] = []

    for index, candidate in enumerate(candidates, start=1):

        center_x_image, center_y_image = candidate["center"]

        center_x_output = (WARP_HEIGHT - 1) - center_y_image

        center_y_output = (WARP_WIDTH - 1) - center_x_image

        # Compute safe_interior_point (pole of inaccessibility) for pickup
        poly_px = np.asarray(candidate["polygon"], dtype=np.float64).reshape(-1, 2)
        poly_mm = poly_px / _ctx['PIXELS_PER_MM']
        try:
            pickup_mm = safe_interior_point(poly_mm, resolution_mm=0.5)
        except (cv2.error, ValueError, RuntimeError):
            pickup_mm = solver_polygon_centroid(poly_mm)
        pickup_x_image = float(pickup_mm[0] * _ctx['PIXELS_PER_MM'])
        pickup_y_image = float(pickup_mm[1] * _ctx['PIXELS_PER_MM'])

        pieces.append(DetectedPiece(

            piece_id=index,

            center_x=center_x_output,

            center_y=center_y_output,

            center_x_image=center_x_image,

            center_y_image=center_y_image,

            pickup_x_image=pickup_x_image,

            pickup_y_image=pickup_y_image,

            area_px=candidate["area"],

            vertices=candidate["vertices"],

            contour=candidate["contour"],

            polygon=candidate["polygon"],

        ))

    return pieces, mask


        # ============================================================

        # Bridge: pick DetectedPiece -> puzzle_vision PieceObservation (original)

        # ============================================================


