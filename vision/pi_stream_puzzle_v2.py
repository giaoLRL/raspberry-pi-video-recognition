import json, math, sys, time

from copy import deepcopy

from dataclasses import dataclass

from pathlib import Path

from typing import Any, Optional


import cv2

import numpy as np

import http.server

import socketserver

import threading

import queue


# Add project path

PROJECT_DIR = Path("/home/man/puzzle_app")
ROBOT_DIR = Path("/home/man/puzzle_robot_project")

for d in (str(ROBOT_DIR), str(PROJECT_DIR)):
    if d not in sys.path:
        sys.path.insert(0, d)


from puzzle_vision.config import load_config

from puzzle_vision.detector import PieceObservation

from puzzle_vision.geometry import (

    edge_lengths as solver_edge_lengths,

    normalize_winding as solver_normalize_winding,

    polygon_area as solver_polygon_area,

    polygon_centroid as solver_polygon_centroid,

    rotation_matrix_row,

    transform_points,

    safe_interior_point,

)

from puzzle_vision.solver import SolveError, solve_card, solve_fixed, solve_taught, solve_unknown

from serial_protocol import send_and_wait_done, start_listener   # bidirectional serial
from coords import image_to_arm, solver_to_arm, arm_to_warp, arm_distance
from tjc_display import open_tjc, draw_state, arm_to_screen  # TJC serial screen


# ============================================================

# Parameters (identical to original pick_base_rdk_solver.py)

# ============================================================

CAMERA_INDEX = 0

WARP_WIDTH = 840

WARP_HEIGHT = 1188


A4_WIDTH_CM = 21.0

A4_HEIGHT_CM = 29.7

A4_WIDTH_MM = 210.0

A4_HEIGHT_MM = 297.0

PIXELS_PER_MM_X = WARP_WIDTH / A4_WIDTH_MM

PIXELS_PER_MM_Y = WARP_HEIGHT / A4_HEIGHT_MM

PIXELS_PER_MM = (PIXELS_PER_MM_X + PIXELS_PER_MM_Y) * 0.5

PIXELS_PER_CM = PIXELS_PER_MM * 10.0


MIN_PIECE_AREA_CM2 = 3.0

MAX_PIECE_AREA_CM2 = 115.0

MIN_PIECE_AREA_PX = int(round(MIN_PIECE_AREA_CM2 * PIXELS_PER_CM ** 2))

MAX_PIECE_AREA_PX = int(round(MAX_PIECE_AREA_CM2 * PIXELS_PER_CM ** 2))


MIN_SIDES = 3

MAX_SIDES = 5

POLYGON_EPSILON_RATIO = 0.015

MIN_PIECE_SHORT_SIDE_PX = 24.0

MIN_FILL_RATIO = 0.20  # 轮廓面积/外接矩形面积，过滤空心大框


CALIBRATION_FILE = Path("/home/man/puzzle_app") / "a4_corners.json"

SOLVER_CONFIG_FILE = Path("/home/man/puzzle_app") / "config.json"

TAUGHT_LAYOUT_FILE = Path("/home/man/puzzle_app") / "taught_layout.json"

MAX_DETECTED_PIECES = 4


RECOVERY_MODES = ("auto", "fixed", "unknown-white", "unknown-pattern")

RECOVERY_MODE_NAMES = {

    "auto": "AUTO",

    "fixed": "Fixed",

    "unknown-white": "White",

    "unknown-pattern": "Pattern",

}


AUTO_FIXED_STRONG_ERROR_MM = 7.0

AUTO_PATTERN_SCORE_THRESHOLD = 0.012


PORT = 8080
# Coord origin: BR corner (bottom-right). X: leftward (BR->BL). Y: upward (BR->TR).

# Coordinate system origin: on BL-TL line (left edge = long side = 297mm),

# 1/4 distance from BL toward TL

ORIGIN_FRACTION = 0.25

# Coordinate system origin: on bottom edge of A4,

# measured from BR corner toward BL corner,

# distance = A4_long_side / 4 = 297mm / 4 = 74.25mm

ORIGIN_FRACTION_OF_LONG_EDGE = 0.25

JPEG_QUALITY = 80

STREAM_FPS = 15


# ============================================================

# Data classes (original)

# ============================================================

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

    if not CALIBRATION_FILE.exists():

        return None

    try:

        data = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))

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

    w2c = cv2.getPerspectiveTransform(dst, corners.astype(np.float32))

    return c2w, w2c


def auto_detect_a4(frame):

    """Detect A4 board using Otsu thresholding (dark board, white dividing line)."""

    h, w = frame.shape[:2]


    # ── A4 detection: Otsu grayscale threshold ──

    # Otsu + BINARY_INV: dark regions become white (foreground), bright -> black.

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # Bridge the white dividing line and close small holes.

    # 15x15 close x1 first, then 7x7 fill fine holes.

    kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))

    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_big, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel, iterations=1)


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

# Piece detection (original pick_base)

# ============================================================

def create_piece_mask(calibrated_region: np.ndarray) -> np.ndarray:

    hsv = cv2.cvtColor(calibrated_region, cv2.COLOR_BGR2HSV)

    value_channel = hsv[:, :, 2]

    blurred = cv2.GaussianBlur(value_channel, (5, 5), 0)

    threshold_value, mask = cv2.threshold(

        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if threshold_value < 30:

        _, mask = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

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

        if area < MIN_PIECE_AREA_PX or area > MAX_PIECE_AREA_PX:

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

            w_mm = rr[1][0] / PIXELS_PER_MM

            h_mm = rr[1][1] / PIXELS_PER_MM

            parts.append(f"{c['area']:.0f}px²({c['area']/PIXELS_PER_CM**2:.1f}cm², {w_mm:.1f}x{h_mm:.1f}mm)")

        print(f"[DETECT] {len(candidates)} pieces: {' | '.join(parts)}", flush=True)

    pieces: list[DetectedPiece] = []

    for index, candidate in enumerate(candidates, start=1):

        center_x_image, center_y_image = candidate["center"]

        center_x_output = (WARP_HEIGHT - 1) - center_y_image

        center_y_output = (WARP_WIDTH - 1) - center_x_image

        # Compute safe_interior_point (pole of inaccessibility) for pickup
        poly_px = np.asarray(candidate["polygon"], dtype=np.float64).reshape(-1, 2)
        poly_mm = poly_px / PIXELS_PER_MM
        try:
            pickup_mm = safe_interior_point(poly_mm, resolution_mm=0.5)
        except (cv2.error, ValueError, RuntimeError):
            pickup_mm = solver_polygon_centroid(poly_mm)
        pickup_x_image = float(pickup_mm[0] * PIXELS_PER_MM)
        pickup_y_image = float(pickup_mm[1] * PIXELS_PER_MM)

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

def detected_pieces_to_solver_observations(

    pieces: list[DetectedPiece],

) -> list[PieceObservation]:

    observations: list[PieceObservation] = []

    for piece in pieces:

        polygon_px = np.asarray(piece.polygon, dtype=np.float64).reshape(-1, 2)

        polygon_mm = solver_normalize_winding(polygon_px / PIXELS_PER_MM)

        centroid_mm = solver_polygon_centroid(polygon_mm)

        try:

            pickup_mm = safe_interior_point(polygon_mm, resolution_mm=0.5)

        except (cv2.error, ValueError, RuntimeError):

            pickup_mm = centroid_mm.copy()

        lengths_mm = solver_edge_lengths(polygon_mm)

        area_mm2 = solver_polygon_area(polygon_mm)

        perimeter_mm = float(np.sum(lengths_mm))

        observations.append(PieceObservation(

            id=f"piece_{piece.piece_id}",

            polygon_mm=polygon_mm,

            contour_px=np.asarray(piece.contour, dtype=np.int32),

            centroid_mm=centroid_mm,

            pickup_mm=pickup_mm,

            area_mm2=float(area_mm2),

            perimeter_mm=perimeter_mm,

            edge_lengths_mm=lengths_mm,

        ))

    return observations


# ============================================================

# Pattern score estimation (original)

# ============================================================

def estimate_pattern_score(

    calibrated_region: np.ndarray, pieces: list[DetectedPiece],

) -> float:

    gray = cv2.cvtColor(calibrated_region, cv2.COLOR_BGR2GRAY)

    scores: list[float] = []

    for piece in pieces:

        mask = np.zeros(gray.shape, dtype=np.uint8)

        polygon = np.asarray(piece.polygon, dtype=np.int32).reshape(-1, 2)

        cv2.fillPoly(mask, [polygon], 255)

        kernel_size = max(5, int(round(2.0 * PIXELS_PER_MM)) | 1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        inner = cv2.erode(mask, kernel, iterations=1)

        valid = inner > 0

        count = int(np.count_nonzero(valid))

        if count < 100:

            continue

        smooth = cv2.GaussianBlur(gray, (0, 0), 2.0)

        residual = cv2.absdiff(gray, smooth)

        high_frequency = float(np.mean(residual[valid] >= 13))

        edges_c = cv2.Canny(gray, 55, 145)

        edge_density = float(np.mean(edges_c[valid] > 0))

        values = gray[valid].astype(np.float32)

        p10, p90 = np.percentile(values, [10.0, 90.0])

        contrast = min(1.0, max(0.0, float(p90 - p10) / 90.0))

        scores.append(0.50 * high_frequency + 0.35 * edge_density + 0.15 * contrast)

    return float(np.median(scores)) if scores else 0.0


# ============================================================

# Solver dispatch (original)

# ============================================================

def load_solver_assets() -> tuple[dict[str, Any], Optional[dict[str, Any]]]:

    config = load_config(str(SOLVER_CONFIG_FILE) if SOLVER_CONFIG_FILE.exists() else None)

    config["paper"]["pixels_per_mm"] = PIXELS_PER_MM

    config["paper"]["width_mm"] = A4_WIDTH_MM

    config["paper"]["height_mm"] = A4_HEIGHT_MM

    config["paper"]["divider_y_mm"] = A4_HEIGHT_MM * 0.5

    # target_zone in solver coords = lower half of A4 (below midline)
    config["unknown"]["target_zone_mm"] = [0.0, A4_HEIGHT_MM * 0.5, A4_WIDTH_MM, A4_HEIGHT_MM]

    config["unknown"]["taught_layout_path"] = str(TAUGHT_LAYOUT_FILE)

    taught_layout: Optional[dict[str, Any]] = None

    if TAUGHT_LAYOUT_FILE.exists():

        try:

            taught_layout = json.loads(TAUGHT_LAYOUT_FILE.read_text(encoding="utf-8"))

        except (OSError, ValueError, TypeError, json.JSONDecodeError):

            taught_layout = None

    return config, taught_layout


def _plan_to_arm(plan: list[dict[str, Any]], info: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert solver-mm coordinates to arm-mm in-place, so rendering uses arm directly."""
    for item in plan:
        if "place_mm" in item:
            ax, ay = solver_to_arm(float(item["place_mm"][0]), float(item["place_mm"][1]))
            item["place_mm"] = [round(ax, 3), round(ay, 3)]
        if "target_polygon_mm" in item:
            item["target_polygon_mm"] = [
                [round(solver_to_arm(float(x), float(y))[0], 3),
                 round(solver_to_arm(float(x), float(y))[1], 3)]
                for x, y in item["target_polygon_mm"]
            ]
        if "measured_target_polygon_mm" in item:
            item["measured_target_polygon_mm"] = [
                [round(solver_to_arm(float(x), float(y))[0], 3),
                 round(solver_to_arm(float(x), float(y))[1], 3)]
                for x, y in item["measured_target_polygon_mm"]
            ]
    if info.get("target_origin_mm"):
        ax, ay = solver_to_arm(float(info["target_origin_mm"][0]), float(info["target_origin_mm"][1]))
        info["target_origin_mm"] = [round(ax, 3), round(ay, 3)]
    return plan, info


def run_one_solver_mode(

    mode: str, observations: list[PieceObservation],

    calibrated_region: np.ndarray, config: dict[str, Any],

    taught_layout: Optional[dict[str, Any]],

    pieces: list[DetectedPiece],

) -> tuple[list[dict[str, Any]], dict[str, Any]]:

    if mode == "fixed":

        plan, info = solve_fixed(observations, deepcopy(config["fixed"]))
        return _plan_to_arm(plan, info)

    unknown_cfg = deepcopy(config["unknown"])

    # Target zone = lower half of A4 (below midline), solver coords
    unknown_cfg["target_zone_mm"] = [0.0, A4_HEIGHT_MM * 0.5, A4_WIDTH_MM, A4_HEIGHT_MM]

    if mode == "unknown-pattern":

        unknown_cfg["target_orientation"] = "portrait"

        plan, info = solve_card(observations, unknown_cfg, calibrated_region, PIXELS_PER_MM)
        return _plan_to_arm(plan, info)

    if mode != "unknown-white":

        raise ValueError(f"Unsupported mode: {mode}")

    unknown_cfg["target_orientation"] = "landscape"

    taught_error: Optional[str] = None

    if (bool(unknown_cfg.get("use_taught_layout", False))

            and taught_layout is not None

            and int(taught_layout.get("piece_count", 0)) == len(observations)):

        try:

            plan, info = solve_taught(observations, taught_layout, unknown_cfg)

            if bool(info.get("solution_accepted", False)):

                info["solver_path"] = "taught_layout"

                return _plan_to_arm(plan, info)

            raise SolveError("taught layout rejected")

        except SolveError as exc:

            taught_error = str(exc)

    plan, info = solve_unknown(observations, unknown_cfg, calibrated_region, PIXELS_PER_MM, use_texture=False)

    info["solver_path"] = "unknown_geometry"

    if taught_error is not None:

        info["taught_layout_fallback_reason"] = taught_error

    return _plan_to_arm(plan, info)


# _apply_fast_solver_budget removed — solver now runs with full config budget


def _fmt(val: Any) -> str:

    """Safe float formatter for solver_info values."""

    if val is None:

        return "?"

    try:

        return f"{float(val):.3f}"

    except (ValueError, TypeError):

        return str(val)


def _solver_attempt_record(mode, accepted, elapsed, info=None, error=None):

    record = {"mode": mode, "accepted": bool(accepted), "time_sec": round(float(elapsed), 4)}

    if info is not None:

        record["quality"] = info.get("solution_quality")

        record["fill_ratio"] = info.get("fill_ratio")

        record["geometry_score"] = info.get("geometry_score")

        record["max_match_error_mm"] = info.get("max_match_error_mm")

    if error:

        record["error"] = error

    return record


def solve_with_pick_recognition(

    pieces: list[DetectedPiece], calibrated_region: np.ndarray, requested_mode: str,

) -> RecoveryResult:

    if requested_mode not in RECOVERY_MODES:

        raise RuntimeError(f"Unknown mode: {requested_mode}")

    if not 1 <= len(pieces) <= MAX_DETECTED_PIECES:

        raise RuntimeError(f"Recovery requires 1-{MAX_DETECTED_PIECES} pieces, got {len(pieces)}")

    observations = detected_pieces_to_solver_observations(pieces)

    config, taught_layout = load_solver_assets()

    pattern_score = estimate_pattern_score(calibrated_region, pieces)

    attempts: list[dict[str, Any]] = []

    successful: dict[str, tuple] = {}

    total_started = time.perf_counter()

    def attempt(mode):

        if mode in successful or any(item["mode"] == mode for item in attempts):

            return

        started = time.perf_counter()

        try:

            plan, info = run_one_solver_mode(mode, observations, calibrated_region, config, taught_layout, pieces)

            elapsed = time.perf_counter() - started

            accepted = bool(info.get("solution_accepted", False))

            attempts.append(_solver_attempt_record(mode, accepted, elapsed, info=info))

            if accepted:

                successful[mode] = (plan, info, elapsed)

        except (SolveError, RuntimeError, ValueError, cv2.error) as exc:

            elapsed = time.perf_counter() - started

            attempts.append(_solver_attempt_record(mode, False, elapsed, error=str(exc)))

    if requested_mode != "auto":

        attempt(requested_mode)

        if requested_mode not in successful:

            detail = next((item.get("error") for item in attempts if item["mode"] == requested_mode), None)

            raise RuntimeError(f"{RECOVERY_MODE_NAMES[requested_mode]} failed" + (f": {detail}" if detail else ": rejected"))

        plan, info, _ = successful[requested_mode]

        return RecoveryResult(requested_mode, requested_mode, plan, info, observations, pattern_score,

                              time.perf_counter() - total_started, attempts)

    # AUTO mode — match standard D:\nnn\main.py logic

    if len(pieces) == 4:

        attempt("fixed")

        fixed = successful.get("fixed")

        if fixed is not None:

            plan, info, _ = fixed

            # Strict auto checks for fixed (matching standard)

            max_err = float(info.get("max_match_error_mm", 1e9))

            total_area_mm2 = sum(float(o.area_mm2) for o in observations)

            fixed_width = float(config["fixed"].get("target_size_mm", [100, 60])[0])

            fixed_height = float(config["fixed"].get("target_size_mm", [100, 60])[1])

            area_ratio = total_area_mm2 / max(fixed_width * fixed_height, 1e-9)

            assignment_cost = float(info.get("assignment_cost", 1e9))

            if max_err <= 8.0 and 0.88 <= area_ratio <= 1.12 and assignment_cost <= 20.0:

                return RecoveryResult("auto", "fixed", plan, info, observations, pattern_score,

                                      time.perf_counter() - total_started, attempts)

            else:

                log_print(f"auto: fixed rejected (err={max_err:.1f}mm area_ratio={area_ratio:.3f} cost={assignment_cost:.1f})")

    # One lightweight classification, then run preferred path only

    is_pattern = pattern_score >= AUTO_PATTERN_SCORE_THRESHOLD

    if is_pattern and 2 <= len(pieces) <= 4:

        attempt("unknown-pattern")

        card = successful.get("unknown-pattern")

        if card is not None:

            plan, info, _ = card

            return RecoveryResult("auto", "unknown-pattern", plan, info, observations, pattern_score,

                                  time.perf_counter() - total_started, attempts)

    # White mode (preferred if not pattern, or fallback if pattern failed)

    attempt("unknown-white")

    white = successful.get("unknown-white")

    if white is not None:

        plan, info, _ = white

        return RecoveryResult("auto", "unknown-white", plan, info, observations, pattern_score,

                              time.perf_counter() - total_started, attempts)

    # Last resort: try pattern if white failed and not yet attempted

    if not is_pattern and 2 <= len(pieces) <= 4:

        attempt("unknown-pattern")

        card = successful.get("unknown-pattern")

        if card is not None:

            plan, info, _ = card

            return RecoveryResult("auto", "unknown-pattern", plan, info, observations, pattern_score,

                                  time.perf_counter() - total_started, attempts)

    # Final fallback: fixed (even if strict check failed)

    fixed = successful.get("fixed")

    if fixed is not None:

        plan, info, _ = fixed

        return RecoveryResult("auto", "fixed", plan, info, observations, pattern_score,

                              time.perf_counter() - total_started, attempts)

    details = "; ".join(

        f"{RECOVERY_MODE_NAMES.get(item['mode'], item['mode'])}: {item.get('error', 'rejected')}"

        for item in attempts)

    raise RuntimeError("All recovery modes failed: " + details)


# ============================================================

# Warp and overlay drawing

# ============================================================

def warp_to_camera(pts, mat):

    r = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)

    return cv2.perspectiveTransform(r, mat).reshape(-1, 2)


def draw_overlay(frame, corners, pieces, reconst, w2c, area_mode=0):

    out = frame.copy()

    if corners is not None:

        cv2.polylines(out, [np.round(corners).astype(np.int32)], True, (255, 255, 0), 2, cv2.LINE_AA)

    # Draw detected pieces
    piece_pickup_camera = {}  # piece_id -> (cx, cy) in camera pixels
    if w2c is not None:

        for p in pieces:

            # Convert pickup point (warp px) to arm-mm coords
            px_mm, py_mm = image_to_arm(p.pickup_x_image, p.pickup_y_image)


            # Original contour (thin grey)

            ccam = warp_to_camera(p.contour, w2c)

            cv2.drawContours(out, [np.round(ccam).astype(np.int32)], -1, (128, 128, 128), 1, cv2.LINE_AA)

            # Simplified polygon (yellow)

            pcam = warp_to_camera(p.polygon, w2c)

            cv2.polylines(out, [np.round(pcam).astype(np.int32)], True, (0, 255, 255), 3, cv2.LINE_AA)

            # Pickup point = safe_interior_point (red filled circle)

            ccam = warp_to_camera(np.array([[[p.pickup_x_image, p.pickup_y_image]]], dtype=np.float32), w2c)[0]

            cxi, cyi = int(ccam[0]), int(ccam[1])

            cv2.circle(out, (cxi, cyi), 7, (0, 0, 255), -1)

            cv2.circle(out, (cxi, cyi), 10, (0, 0, 255), 2, cv2.LINE_AA)

            # Save pickup camera coords for connecting-line reference
            piece_pickup_camera[p.piece_id] = (cxi, cyi)

            # ID label

            cv2.putText(out, f"ID:{p.piece_id}",

                        (cxi + 14, cyi - 18),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            # Pickup coordinate in mm (white text with dark background for readability)

            coord_str = f"({px_mm:.1f}, {py_mm:.1f})mm"

            (tw, th), _ = cv2.getTextSize(coord_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            cv2.rectangle(out,

                          (cxi - tw // 2 - 4, cyi + 16 - th - 2),

                          (cxi + tw // 2 + 4, cyi + 16 + 2),

                          (0, 0, 0), -1, cv2.LINE_AA)

            cv2.putText(out, coord_str,

                        (cxi - tw // 2, cyi + 16),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Log to console

            area_mm2 = p.area_px / (PIXELS_PER_MM ** 2)

            nv = len(p.polygon)

            print(f"[PICKUP] ID:{p.piece_id} pos=({px_mm:.1f},{py_mm:.1f})mm area={area_mm2:.0f}mm2 sides={nv}", flush=True)

    cv2.putText(out, f"Pieces: {len(pieces)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Area reference boxes

    if area_mode in (1, 2) and w2c is not None:

        area_px = MIN_PIECE_AREA_PX if area_mode == 1 else MAX_PIECE_AREA_PX

        label = "MIN AREA" if area_mode == 1 else "MAX AREA"

        color = (0, 255, 0) if area_mode == 1 else (255, 0, 255)

        side = int(np.sqrt(area_px))

        cx_w, cy_w = WARP_WIDTH // 2, WARP_HEIGHT // 2

        square_warp = np.array([[cx_w - side // 2, cy_w - side // 2],

                                [cx_w + side // 2, cy_w - side // 2],

                                [cx_w + side // 2, cy_w + side // 2],

                                [cx_w - side // 2, cy_w + side // 2]], dtype=np.float32)

        square_cam = np.round(warp_to_camera(square_warp, w2c)).astype(np.int32)

        cv2.polylines(out, [square_cam], True, color, 3, cv2.LINE_AA)

        cv2.putText(out, f"{label}: {area_px}px", (square_cam[0][0], square_cam[0][1] - 8),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # Reconstruction plan

    if reconst is not None and reconst.plan and w2c is not None:

        tpts = []

        for idx, item in enumerate(reconst.plan, start=1):

            pid = item.get("piece_id", idx)

            # 绿色：目标放置位置（arm mm → warp → camera）
            poly_arm = np.asarray(item["target_polygon_mm"], dtype=np.float32).reshape(-1, 2)
            poly_warp = np.array([[arm_to_warp(float(x), float(y)) for x, y in poly_arm]], dtype=np.float32).reshape(-1, 2)
            pcam = np.round(warp_to_camera(poly_warp, w2c)).astype(np.int32)

            tpts.append(pcam)

            cv2.polylines(out, [pcam], True, (0, 255, 0), 3, cv2.LINE_AA)

            # 碎片ID标签（绿色）

            label_pos = tuple(pcam[0])

            cv2.putText(out, f"#{pid}", (label_pos[0] - 5, label_pos[1] - 8),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            # 红色1px：同源数据验证

            cv2.polylines(out, [pcam], True, (0, 0, 255), 1, cv2.LINE_AA)

            # ── Place point (safe_interior at target) + connecting line ──
            # Distinct BGR colors for up to 6 pieces
            LINE_PALETTE = [
                (255, 0, 255),   # Magenta
                (255, 255, 0),   # Cyan
                (0, 140, 255),   # Orange
                (0, 255, 128),   # Lime green
                (0, 255, 255),   # Yellow
                (128, 0, 255),   # Pink
            ]
            line_color = LINE_PALETTE[(idx - 1) % len(LINE_PALETTE)]

            # place_mm is the safe_interior_point at target, already in arm-mm
            place_arm = np.array(item["place_mm"], dtype=np.float64)  # [x_mm, y_mm]
            tc_wx, tc_wy = arm_to_warp(float(place_arm[0]), float(place_arm[1]))
            tc_cam = warp_to_camera(np.array([[[tc_wx, tc_wy]]], dtype=np.float32), w2c)[0]
            tc_cx, tc_cy = int(round(tc_cam[0])), int(round(tc_cam[1]))

            # Find matching original pickup point and draw connecting line
            if isinstance(pid, str) and pid.startswith("piece_"):
                piece_num = int(pid.split("_")[1])
            else:
                piece_num = int(pid) if pid is not None else idx
            if piece_num in piece_pickup_camera:
                orig_cx, orig_cy = piece_pickup_camera[piece_num]
                # Color-coded line: original pickup → target place
                cv2.line(out, (orig_cx, orig_cy), (tc_cx, tc_cy), line_color, 2, cv2.LINE_AA)

            # Target place marker (color matches the connecting line)
            cv2.circle(out, (tc_cx, tc_cy), 7, line_color, -1)
            cv2.circle(out, (tc_cx, tc_cy), 11, line_color, 2, cv2.LINE_AA)

            # Target place coordinate label (color matches the marker)
            tcoord_str = f"({place_arm[0]:.1f},{place_arm[1]:.1f})mm"
            (tw, th), _ = cv2.getTextSize(tcoord_str, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 2)
            cv2.rectangle(out,
                          (tc_cx - tw // 2 - 4, tc_cy + 14 - th - 2),
                          (tc_cx + tw // 2 + 4, tc_cy + 14 + 2),
                          (0, 0, 0), -1, cv2.LINE_AA)
            cv2.putText(out, tcoord_str,
                        (tc_cx - tw // 2, tc_cy + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, line_color, 2, cv2.LINE_AA)

            # Log target place point
            print(f"[TARGET] {pid}: place=({place_arm[0]:.1f},{place_arm[1]:.1f})mm", flush=True)

        if tpts:

            # 蓝框：目标矩形（arm mm → warp → camera）
            info = reconst.solver_info

            target_origin = info.get("target_origin_mm")

            target_size = info.get("target_size_mm")

            if target_origin is not None and target_size is not None:

                arm_ox = float(target_origin[0])
                arm_oy = float(target_origin[1])
                w = float(target_size[0])
                h = float(target_size[1])
                # target_origin in arm = top-right corner (solver TL flipped to arm)
                rect_arm = np.array([
                    [arm_ox, arm_oy],           # top-right
                    [arm_ox - w, arm_oy],       # top-left
                    [arm_ox - w, arm_oy + h],   # bottom-left
                    [arm_ox, arm_oy + h],       # bottom-right
                ], dtype=np.float64)

                box_pts = []
                for ax, ay in rect_arm:
                    wx, wy = arm_to_warp(float(ax), float(ay))
                    pc = warp_to_camera(np.array([[[wx, wy]]], dtype=np.float32), w2c)[0]
                    box_pts.append([int(round(pc[0])), int(round(pc[1]))])
                box = np.array(box_pts, dtype=np.int32)

            else:

                merged = np.vstack(tpts)

                box = np.round(cv2.boxPoints(cv2.minAreaRect(merged.astype(np.float32)))).astype(np.int32)

            cv2.polylines(out, [box], True, (255, 0, 0), 2, cv2.LINE_AA)

            if not hasattr(draw_overlay, "_last_box_hash"):

                draw_overlay._last_box_hash = None

            box_hash = hash(box.tobytes())

            if box_hash != draw_overlay._last_box_hash:

                draw_overlay._last_box_hash = box_hash

                corners_str = ", ".join(f"({x},{y})" for x, y in box)

                if target_size is not None and target_origin is not None:

                    sz = np.asarray(target_size, dtype=np.float64)
                    log_print(f"BlueBox: [{corners_str}] size={sz[0]:.1f}x{sz[1]:.1f}mm arm=({float(target_origin[0]):.1f},{float(target_origin[1]):.1f})")

                else:

                    log_print(f"BlueBox: [{corners_str}] (minAreaRect fallback)")

        cv2.putText(out, f"Restored: {reconst.selected_mode}", (20, 70),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2, cv2.LINE_AA)

    # ============================================================
    # ARM COORDINATE SYSTEM (authoritative for all display rendering)
    #
    #   Origin: A4 right edge + 75mm down => warp pixel (839, 300)
    #   X axis: leftward  (0 at BR edge, ~210 at TL edge)
    #   Y axis: downward   (0 at 75mm from top, ~222 at bottom edge)
    #
    #   TL = arm(209.8, -75.0)    TR = arm(0.0, -75.0)
    #   BL = arm(209.8, 221.8)    BR = arm(0.0, 221.8)
    #
    #   Conversions (coords.py):
    #     image_to_arm(wx, wy)   : warp px -> arm mm
    #     arm_to_warp(ax, ay)    : arm mm  -> warp px
    #
    #   Rendering: arm mm -> arm_to_warp -> warp_to_camera -> screen
    # ============================================================
    if corners is not None and w2c is not None:

        # --- Origin in warp ---

        origin_wx = WARP_WIDTH - 1  # BR right edge (pixel 839)

        origin_warp = np.array([origin_wx, 300.0])  # TR + 75mm down

        origin_cam = warp_to_camera(np.array([[origin_warp]], dtype=np.float32), w2c)[0]

        ox, oy = int(round(origin_cam[0])), int(round(origin_cam[1]))


        def phys_to_cam(x_mm, y_mm):

            wx, wy = arm_to_warp(x_mm, y_mm)


            pcam = warp_to_camera(np.array([[[wx, wy]]], dtype=np.float32), w2c)[0]

            return int(round(pcam[0])), int(round(pcam[1]))


        X_MIN_MM = 0.0       # at BR (origin)
        X_MAX_MM = 210.0     # A4 width (X leftward)
        Y_MAX_MM = 297.0     # A4 height (Y downward from origin)

        # --- Draw X axis (red) ---

        x0_cam = phys_to_cam(X_MIN_MM, 0.0)

        x1_cam = phys_to_cam(X_MAX_MM, 0.0)

        cv2.line(out, x0_cam, x1_cam, (0, 0, 255), 2, cv2.LINE_AA)

        # Arrow head at top

        temp1 = phys_to_cam(X_MAX_MM - 10.0, -6.0)

        temp2 = phys_to_cam(X_MAX_MM - 10.0, 6.0)

        cv2.line(out, x1_cam, temp1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.line(out, x1_cam, temp2, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(out, "X", (x1_cam[0] + 8, x1_cam[1] - 8),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


        # --- Draw Y axis (green) ---

        y0_cam = phys_to_cam(0.0, 0.0)

        y1_cam = phys_to_cam(0.0, Y_MAX_MM)

        cv2.line(out, y0_cam, y1_cam, (0, 255, 0), 2, cv2.LINE_AA)

        # Arrow head at right

        temp1 = phys_to_cam(-6.0, Y_MAX_MM - 10.0)

        temp2 = phys_to_cam(6.0, Y_MAX_MM - 10.0)

        cv2.line(out, y1_cam, temp1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.line(out, y1_cam, temp2, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(out, "Y", (y1_cam[0] + 8, y1_cam[1] + 4),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


        # --- X-axis tick marks and labels ---

        # X-axis ticks: 0 to 210mm, step=20mm

        x_step = 20.0

        for x_mm in np.arange(X_MIN_MM, X_MAX_MM + 0.1, x_step):

            t0 = phys_to_cam(x_mm, -5.0)

            t1 = phys_to_cam(x_mm, 5.0)

            cv2.line(out, t0, t1, (0, 0, 255), 1, cv2.LINE_AA)

            # Label: place to left of axis

            tlbl = phys_to_cam(x_mm, -10.0)

            cv2.putText(out, f"{x_mm:.0f}",

                        (tlbl[0] - 30, tlbl[1] + 4),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)


        # --- Y-axis tick marks and labels ---

        y_step = 20.0

        for y_mm in np.arange(0.0, Y_MAX_MM + 0.1, y_step):

            t0 = phys_to_cam(-5.0, y_mm)

            t1 = phys_to_cam(5.0, y_mm)

            cv2.line(out, t0, t1, (0, 255, 0), 1, cv2.LINE_AA)

            # Label below axis

            tlbl = phys_to_cam(-10.0, y_mm)

            cv2.putText(out, f"{y_mm:.0f}",

                        (tlbl[0] - 28, tlbl[1] + 8),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)


        # --- Origin marker ---

        cv2.circle(out, (ox, oy), 8, (0, 255, 255), -1, cv2.LINE_AA)

        cv2.circle(out, (ox, oy), 14, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(out, "O(0,0)@75mm",

                    (ox + 18, oy - 14),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)


        # --- Corners with labels ---

        warp_corners_all = np.array([

            [0, 0], [WARP_WIDTH - 1, 0],

            [WARP_WIDTH - 1, WARP_HEIGHT - 1], [0, WARP_HEIGHT - 1],

        ], dtype=np.float32)

        cc_all = warp_to_camera(warp_corners_all, w2c)

        tl_c, tr_c, br_c, bl_c = cc_all


        # Arm coords of A4 corners
        _arm_tl = image_to_arm(0.0, 0.0)
        _arm_tr = image_to_arm(float(WARP_WIDTH - 1), 0.0)
        _arm_bl = image_to_arm(0.0, float(WARP_HEIGHT - 1))
        _arm_br = image_to_arm(float(WARP_WIDTH - 1), float(WARP_HEIGHT - 1))

        corner_defs = [
            ("TL", tl_c, (0, 255, 0),    24, -10, f"({_arm_tl[0]:.1f},{_arm_tl[1]:.1f})"),
            ("TR", tr_c, (255, 160, 0),  -52, -10, f"({_arm_tr[0]:.1f},{_arm_tr[1]:.1f})"),
            ("BL", bl_c, (0, 255, 200),   24, 22,  f"({_arm_bl[0]:.1f},{_arm_bl[1]:.1f})"),
            ("BR", br_c, (255, 100, 255), -52, 22,  f"({_arm_br[0]:.1f},{_arm_br[1]:.1f})"),
        ]

        for label, pt, color, dx, dy, coord_str in corner_defs:

            px, py = int(round(pt[0])), int(round(pt[1]))

            cv2.circle(out, (px, py), 8, color, 2, cv2.LINE_AA)

            cv2.circle(out, (px, py), 2, color, -1, cv2.LINE_AA)

            cv2.putText(out, f"{label}{coord_str}",

                        (px + dx, py + dy),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


        # --- Grid lines (subtle) ---

        for x_mm in np.arange(X_MIN_MM, X_MAX_MM + 0.1, x_step):

            if abs(x_mm) < 0.01:

                continue  # skip origin, Y axis already drawn

            g0 = phys_to_cam(x_mm, 0.0)

            g1 = phys_to_cam(x_mm, Y_MAX_MM)

            cv2.line(out, g0, g1, (60, 60, 60), 1, cv2.LINE_AA)


        for y_mm in np.arange(y_step, Y_MAX_MM + 0.1, y_step):

            g0 = phys_to_cam(X_MIN_MM, y_mm)

            g1 = phys_to_cam(X_MAX_MM, y_mm)

            cv2.line(out, g0, g1, (60, 60, 60), 1, cv2.LINE_AA)


        # --- Assembly target zone: lower half of A4 (mid-line to bottom) ---
        # A4 lower half in solver mm: [0, 148.5, 210, 297]
        # In arm mm: X=0(BR edge) to X=210(TL edge), Y=73.5(mid) to Y=221.8(bottom)
        _zone_x1 = 0.0
        _zone_x2 = (WARP_WIDTH - 1) / PIXELS_PER_MM                  # ~209.75
        _zone_y1 = A4_HEIGHT_MM * 0.5 - 300.0 / PIXELS_PER_MM       # 148.5 - 75 = 73.5
        _zone_y2 = (WARP_HEIGHT - 1 - 300.0) / PIXELS_PER_MM         # (1187-300)/4 = 221.75

        zone_corners_phys = [(_zone_x1, _zone_y1), (_zone_x2, _zone_y1),
                             (_zone_x2, _zone_y2), (_zone_x1, _zone_y2)]

        zone_pts_cam = [phys_to_cam(x, y) for x, y in zone_corners_phys]

        # Draw filled translucent zone

        zone_overlay = out.copy()

        cv2.fillPoly(zone_overlay, [np.array(zone_pts_cam, dtype=np.int32)], (200, 150, 50))

        out = cv2.addWeighted(out, 0.75, zone_overlay, 0.25, 0)

        # Cyan border

        cv2.polylines(out, [np.array(zone_pts_cam, dtype=np.int32)], True, (255, 200, 0), 3, cv2.LINE_AA)

        # Corner labels

        zone_labels = [f"Z1(0,{_zone_y1:.0f})", f"Z2({_zone_x2:.0f},{_zone_y1:.0f})",
                       f"Z3({_zone_x2:.0f},{_zone_y2:.0f})", f"Z4(0,{_zone_y2:.0f})"]

        for zlbl, (zx, zy) in zip(zone_labels, zone_pts_cam):

            cv2.circle(out, (zx, zy), 6, (255, 200, 0), -1, cv2.LINE_AA)

            cv2.putText(out, zlbl, (zx + 8, zy - 8),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 200, 0), 1, cv2.LINE_AA)

        # Center label

        zcx, zcy = phys_to_cam((_zone_x1 + _zone_x2) * 0.5, (_zone_y1 + _zone_y2) * 0.5)

        cv2.putText(out, "ASSEMBLY ZONE",

                    (zcx - 60, zcy),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)


    # A4 label

    if corners is not None:

        pts = np.round(corners).astype(np.int32)

        cv2.putText(out, "A4", (pts[0][0] + 5, pts[0][1] - 8),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

    else:

        cv2.putText(out, "Searching A4...", (20, 100),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # Render log messages

    with SharedState.lock:

        logs = list(SharedState.log_lines)

    if logs:

        fh = out.shape[0]

        line_h = 16

        margin = 8

        log_h = len(logs) * line_h + margin * 2

        overlay = out.copy()

        cv2.rectangle(overlay, (0, fh - log_h), (520, fh), (0, 0, 0), -1)

        out = cv2.addWeighted(out, 0.6, overlay, 0.4, 0)

        for i, line in enumerate(logs):

            text = line if len(line) <= 90 else line[:87] + "..."

            cv2.putText(out, text, (margin + 4, fh - margin - (len(logs) - 1 - i) * line_h),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    return out


# ============================================================

# Web UI and shared state

# ============================================================

# ============================================================

class SharedState:

    frame = None

    raw_frame = None

    status = {"pieces": 0, "mode": "idle", "fps": 0, "show_target": True, "area_mode": 0, "last_action": "Ready", "selected_mode": "AUTO"}

    lock = threading.Lock()

    action_queue = queue.Queue()

    show_target = True

    area_display_mode = 0

    current_mode = "auto"  # M/0/1/2/3 select mode, P executes with this mode

    last_action_msg = "Ready"

    log_lines = []

    max_log_lines = 12

    recognition_active = False  # OFF by default, toggle via R button

    # Freeze state

    frozen = False

    freeze_data = None  # dict: pieces, solve_info, etc.

    # Calibration data for web crosshair (arm-mm coordinate display)
    calib_data = None  # dict: corners, c2w, ppm, origin_wx, origin_wy, has_calib

    # Auto-freeze stability tracking

    _stability_counter = 0

    _last_piece_centroids = []  # list of (cx,cy) tuples per frame

    _STABILITY_FRAMES = 5       # consecutive stable frames to trigger

    _STABILITY_THRESHOLD_MM = 1.5  # max centroid movement between frames


def log_print(msg):

    print(msg)

    with SharedState.lock:

        SharedState.log_lines.append(str(msg))

        if len(SharedState.log_lines) > SharedState.max_log_lines:

            SharedState.log_lines.pop(0)


MODE_LABELS = {"auto": "AUTO", "fixed": "FIXED", "unknown-white": "WHITE", "unknown-pattern": "PATTERN"}

MODE_CYCLE = ["auto", "fixed", "unknown-white", "unknown-pattern"]


def handle_action(cmd):

    """Handle action from web UI. M/0/1/2/3 change mode, P triggers solve."""

    SharedState.action_queue.put(cmd)

    if cmd == "M":

        idx = MODE_CYCLE.index(SharedState.current_mode)

        SharedState.current_mode = MODE_CYCLE[(idx + 1) % len(MODE_CYCLE)]

        SharedState.last_action_msg = f"Mode: {MODE_LABELS[SharedState.current_mode]}"

        return f"OK Mode -> {MODE_LABELS[SharedState.current_mode]}"

    elif cmd in ("0", "1", "2", "3"):

        mode_idx = int(cmd)

        if mode_idx == 0:

            SharedState.current_mode = "auto"

        elif mode_idx == 1:

            SharedState.current_mode = "fixed"

        elif mode_idx == 2:

            SharedState.current_mode = "unknown-white"

        else:

            SharedState.current_mode = "unknown-pattern"

        SharedState.last_action_msg = f"Mode: {MODE_LABELS[SharedState.current_mode]}"

        return f"OK Mode -> {MODE_LABELS[SharedState.current_mode]}"

    elif cmd == "R":

        SharedState.recognition_active = not SharedState.recognition_active

        state_str = "ON" if SharedState.recognition_active else "OFF"

        SharedState.last_action_msg = f"Recognition {state_str}"

        return f"OK Recognition -> {state_str}"

    elif cmd == "F":

        if not SharedState.recognition_active:

            return "OK: Turn on recognition first"

        SharedState.frozen = not SharedState.frozen

        if SharedState.frozen:

            SharedState.last_action_msg = "FROZEN"

            log_print("F: FREEZE - detection locked")

        else:

            SharedState.frozen = False

            SharedState._stability_counter = 0

            SharedState._last_piece_centroids = []

            SharedState.freeze_data = None

            SharedState.last_action_msg = "UNFROZEN"

            log_print("F: UNFREEZE - detection resumed")

        state_str = "FROZEN" if SharedState.frozen else "READY"

        return f"OK Freeze -> {state_str}"

    names = {"P": "Recover", "T": "Overlay toggled", "D": "Debug", "A": "Area box", "S": "Saved", "R": "Recognition", "F": "Freeze"}

    SharedState.last_action_msg = names.get(cmd, cmd)

    return "OK: " + SharedState.last_action_msg


HTML = """<!DOCTYPE html><html><head><title>Puzzle Recognition</title>

<meta charset=utf-8>

<style>

*{margin:0;padding:0;box-sizing:border-box}

body{background:#111;color:#fff;font:14px/1.5 monospace}

.top{display:flex;height:calc(100vh - 140px)}

.video{flex:1;display:flex;align-items:center;justify-content:center;position:relative;cursor:crosshair}

.video img{max-width:100%;max-height:100%;display:block}

.video canvas{position:absolute;top:0;left:0;pointer-events:none;z-index:10}

.crosshair-tip{position:absolute;pointer-events:none;z-index:20;color:#0f0;font:bold 12px monospace;background:rgba(0,0,0,0.8);padding:3px 7px;border-radius:4px;border:1px solid #0f0;white-space:nowrap;display:none}

.ctrl{position:fixed;bottom:0;left:0;right:0;background:rgba(0,0,0,0.95);padding:10px;display:flex;flex-wrap:wrap;gap:6px;justify-content:center;z-index:100;border-top:2px solid #333}

.ctrl button{padding:10px 14px;font-size:13px;font-weight:bold;border:2px solid #555;border-radius:6px;cursor:pointer;color:#fff;transition:all 0.15s;min-width:55px}

.ctrl button:hover{opacity:0.85;transform:scale(1.03)}

.ctrl button:active{transform:scale(0.96)}

.ctrl .sep{width:2px;background:#444;margin:0 4px}

.btn-num{background:#607D8B;border-color:#90A4AE}

.btn-t{background:#009688;border-color:#4DB6AC}

.btn-d{background:#795548;border-color:#A1887F}

.btn-a{background:#3F51B5;border-color:#7986CB}

.btn-s{background:#FF9800;border-color:#FFB74D}

.btn-r-on{background:#4CAF50;border-color:#81C784}

.btn-r-off{background:#F44336;border-color:#E57373}

.btn-f-on{background:#E91E63;border-color:#F06292;animation:pulse 1.2s infinite}

.btn-f-off{background:#607D8B;border-color:#90A4AE}

@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}

.mode-bar{position:fixed;bottom:88px;left:50%;transform:translateX(-50%);color:#ff0;font:bold 15px monospace;background:rgba(0,0,0,0.85);padding:6px 20px;border-radius:6px 6px 0 0;z-index:50}

.info{position:fixed;top:10px;left:10px;color:#0f0;font:14px monospace;background:rgba(0,0,0,0.75);padding:8px 14px;border-radius:6px;z-index:50;pointer-events:none}

</style></head><body>

<div class=info id=info>Loading...</div>

<div class=top><div class=video id=video_container><img id=stream src=/stream><canvas id=crosshair></canvas><div class=crosshair-tip id=coord_tip></div></div></div>

<div class=mode-bar id=mode_bar>Mode: AUTO</div>

<div class=ctrl>

<button class=btn-num onclick=act('1')>1</button>

<button class=btn-num onclick=act('2')>2</button>

<button class=btn-num onclick=act('3')>3</button>

<span class=sep></span>

<button class=btn-t onclick=act('T')>T</button>

<button class=btn-d onclick=act('D')>D</button>

<button class=btn-a onclick=act('A')>A</button>

<button class=btn-s onclick=act('S')>S</button>

<span class=sep></span>

<button class=btn-r-off id=btn_r onclick=act('R')>R:OFF</button>

<span class=sep></span>

<button class=btn-f-off id=btn_f onclick=act('F')>F:OFF</button>

</div>

<script>

function act(cmd){fetch('/action?cmd='+cmd).then(r=>r.text()).then(t=>{

 if(cmd=='M'||cmd=='0'||cmd=='1'||cmd=='2'||cmd=='3') document.getElementById('mode_bar').innerHTML=('Mode: '+t.split('->')[1])||t;

 if(cmd=='R'){var b=document.getElementById('btn_r');var on=t.includes('ON');b.className=on?'btn-r-on':'btn-r-off';b.textContent=on?'R:ON':'R:OFF';}

 if(cmd=='F'){var b=document.getElementById('btn_f');var on=t.includes('FROZEN');b.className=on?'btn-f-on':'btn-f-off';b.textContent=on?'F:FROZEN':'F:OFF';}

})}

// ---- Crosshair: mouse hover shows ARM-mm coords (A4 area only) ----
(function(){
 var img=document.getElementById('stream'),
     canvas=document.getElementById('crosshair'),
     tip=document.getElementById('coord_tip'),
     container=document.getElementById('video_container'),
     ctx=canvas.getContext('2d');

 var calib=null; // {has_calib, corners, c2w, ppm, origin_wx, origin_wy}

 // Fetch calibration data every 3s
 function fetchCalib(){
   fetch('/calib_data').then(function(r){return r.json()}).then(function(d){
     calib=d.has_calib?d:null;
   }).catch(function(){calib=null;});
 }
 fetchCalib();
 setInterval(fetchCalib,3000);

 function syncCanvas(){
   var r=img.getBoundingClientRect(),
       cr=container.getBoundingClientRect();
   var ox=r.left-cr.left, oy=r.top-cr.top;
   canvas.style.left=ox+'px';
   canvas.style.top=oy+'px';
   canvas.width=r.width;
   canvas.height=r.height;
 }
 img.addEventListener('load',syncCanvas);
 window.addEventListener('resize',syncCanvas);
 setInterval(syncCanvas,2000);

 function hideCrosshair(){
   ctx.clearRect(0,0,canvas.width,canvas.height);
   tip.style.display='none';
 }

 // Perspective transform: camera-img-px -> warp-px
 function transformPoint(x,y,matrix){
   var u=matrix[0][0]*x+matrix[0][1]*y+matrix[0][2];
   var v=matrix[1][0]*x+matrix[1][1]*y+matrix[1][2];
   var w=matrix[2][0]*x+matrix[2][1]*y+matrix[2][2];
   return [u/w, v/w];
 }

 // Warp-px -> arm-mm (matching coords.py image_to_arm)
 function warpToArm(wx,wy,ppm,ox,oy){
   return [(ox-wx)/ppm, (wy-oy)/ppm];
 }

 // Point in polygon (ray-casting)
 function pointInPolygon(px,py,poly){
   var inside=false;
   for(var i=0,j=poly.length-1;i<poly.length;j=i++){
     var xi=poly[i][0], yi=poly[i][1];
     var xj=poly[j][0], yj=poly[j][1];
     if((yi>py)!==(yj>py) && px<(xj-xi)*(py-yi)/(yj-yi)+xi) inside=!inside;
   }
   return inside;
 }

 container.addEventListener('mousemove',function(e){
   var r=img.getBoundingClientRect();
   var dx=e.clientX-r.left, dy=e.clientY-r.top;
   if(dx<0||dy<0||dx>r.width||dy>r.height){hideCrosshair();return;}

   var scaleX=img.naturalWidth/r.width, scaleY=img.naturalHeight/r.height;
   var px=dx*scaleX, py=dy*scaleY;

   // Check calibration + A4 boundary
   var insideA4=false, armX=0, armY=0;
   if(calib && calib.corners){
     insideA4=pointInPolygon(px,py,calib.corners);
     if(insideA4){
       var wp=transformPoint(px,py,calib.c2w);
       var arm=warpToArm(wp[0],wp[1],calib.ppm,calib.origin_wx,calib.origin_wy);
       armX=arm[0]; armY=arm[1];
     }
   }

   if(!insideA4){hideCrosshair();return;}

   syncCanvas();
   ctx.clearRect(0,0,canvas.width,canvas.height);

   // Dashed crosshair lines (green = valid ARM coords)
   ctx.strokeStyle='rgba(0,255,0,0.7)';
   ctx.lineWidth=1;
   ctx.setLineDash([5,5]);

   ctx.beginPath();
   ctx.moveTo(0,dy);
   ctx.lineTo(canvas.width,dy);
   ctx.stroke();

   ctx.beginPath();
   ctx.moveTo(dx,0);
   ctx.lineTo(dx,canvas.height);
   ctx.stroke();

   ctx.setLineDash([]);

   // Solid center dot
   ctx.fillStyle='#0f0';
   ctx.beginPath();
   ctx.arc(dx,dy,4,0,Math.PI*2);
   ctx.fill();

   // Coordinate tip in arm mm
   tip.textContent='('+armX.toFixed(1)+', '+armY.toFixed(1)+') mm';
   var cr=container.getBoundingClientRect();
   var tx=e.clientX-cr.left+16, ty=e.clientY-cr.top+16;
   if(tx+130>cr.width) tx=e.clientX-cr.left-140;
   if(ty+26>cr.height) ty=e.clientY-cr.top-30;
   tip.style.left=tx+'px';
   tip.style.top=ty+'px';
   tip.style.display='block';
 });

 container.addEventListener('mouseleave',hideCrosshair);
})();

setInterval(function(){fetch('/status').then(r=>r.json()).then(d=>{

 document.getElementById('info').innerHTML=(d.frozen?'[FROZEN] ':'')+'Pieces: '+d.pieces+' | Mode: '+d.mode+' | FPS: '+d.fps+' | '+d.last_action;

 document.getElementById('mode_bar').innerHTML='Mode: '+d.selected_mode + (d.recognition ? ' | REC:ON' : ' | REC:OFF');

 var b=document.getElementById('btn_r');if(d.recognition){b.className='btn-r-on';b.textContent='R:ON';}else{b.className='btn-r-off';b.textContent='R:OFF';}

 var fb=document.getElementById('btn_f');if(d.frozen){fb.className='btn-f-on';fb.textContent='F:FROZEN';}else{fb.className='btn-f-off';fb.textContent='F:OFF';}

})},1000)

</script></body></html>"""


CALIB_HTML = """<!DOCTYPE html><html><head><title>Calibrate</title>

<meta charset=utf-8>

<style>body{margin:0;background:#000;text-align:center;font:14px monospace;color:#fff}

img{max-width:100vw;max-height:85vh;cursor:crosshair}

.info{color:#0f0;padding:10px}

button{padding:10px 20px;margin:5px;font-size:16px;cursor:pointer;background:#333;color:#fff;border:2px solid #555;border-radius:4px}

#coords{color:#ff0;margin:10px}

</style></head><body><div class=info>Click 4 corners of A4 paper (auto-sorted)</div>

<div id=coords></div>

<img id=calib_img src=/raw_frame>

<div><button onclick=save()>Save</button> <button onclick=reset()>Reset</button></div>

<script>

var pts=[];

document.getElementById('calib_img').onclick=function(e){

 var r=this.getBoundingClientRect();

 var x=e.clientX-r.left, y=e.clientY-r.top;

 var sx=this.naturalWidth/r.width, sy=this.naturalHeight/r.height;

 var px=Math.round(x*sx), py=Math.round(y*sy);

 if(pts.length<4){pts.push([px,py]);

  document.getElementById('coords').innerHTML='Pt'+pts.length+':('+px+','+py+')|'+JSON.stringify(pts);}

 if(pts.length==4) document.getElementById('coords').innerHTML='Ready: '+JSON.stringify(pts);

};

function save(){

 if(pts.length!=4){alert('Need 4 corners');return;}

 fetch('/save_calib',{method:'POST',body:JSON.stringify(pts)}).then(r=>r.text()).then(t=>alert(t));

}

function reset(){pts=[];document.getElementById('coords').innerHTML='';}

</script></body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):

    def do_GET(self):

        if self.path == "/":

            self.send_response(200)

            self.send_header("Content-type", "text/html; charset=utf-8")

            self.end_headers()

            self.wfile.write(HTML.encode())

        elif self.path.startswith("/action"):

            cmd = "auto"

            if "?" in self.path:

                qs = self.path.split("?", 1)[1]

                for kv in qs.split("&"):

                    if "=" in kv and kv.split("=")[0] == "cmd":

                        cmd = kv.split("=")[1]

            self.send_response(200)

            self.send_header("Content-type", "text/plain; charset=utf-8")

            self.end_headers()

            self.wfile.write(handle_action(cmd).encode())

        elif self.path == "/calibrate":

            self.send_response(200)

            self.send_header("Content-type", "text/html; charset=utf-8")

            self.end_headers()

            self.wfile.write(CALIB_HTML.encode())

        elif self.path == "/raw_frame":

            self.send_response(200)

            self.send_header("Content-type", "image/jpeg")

            self.end_headers()

            with SharedState.lock:

                f = SharedState.raw_frame.copy() if SharedState.raw_frame is not None else None

            if f is not None:

                _, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90])

                self.wfile.write(buf.tobytes())

        elif self.path == "/stream":

            self.send_response(200)

            self.send_header("Content-type", "multipart/x-mixed-replace; boundary=frame")

            self.send_header("Cache-Control", "no-cache")

            self.end_headers()

            while True:

                with SharedState.lock:

                    f = SharedState.frame.copy() if SharedState.frame is not None else None

                if f is not None:

                    ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

                    if ok:

                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

                time.sleep(1.0 / STREAM_FPS)

        elif self.path == "/freeze_data":

            self.send_response(200)

            self.send_header("Content-type", "application/json")

            self.send_header("Access-Control-Allow-Origin", "*")

            self.end_headers()

            with SharedState.lock:

                fd = SharedState.freeze_data if SharedState.freeze_data is not None else {"frozen": False, "pieces": [], "message": "No freeze data available"}

            self.wfile.write(json.dumps(fd).encode())

        elif self.path == "/calib_data":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with SharedState.lock:
                cd = SharedState.calib_data if SharedState.calib_data is not None else {"has_calib": False}
            self.wfile.write(json.dumps(cd).encode())

        elif self.path == "/status":

            self.send_response(200)

            self.send_header("Content-type", "application/json")

            self.end_headers()

            with SharedState.lock:

                s = json.dumps(SharedState.status)

            self.wfile.write(s.encode())


    def do_POST(self):

        if self.path == "/save_calib":

            length = int(self.headers.get('Content-Length', 0))

            body = self.rfile.read(length)

            try:

                pts_raw = json.loads(body)

                if len(pts_raw) != 4:

                    self.send_error(400, "Need 4 corners")

                    return

                pts_sorted = order_points(np.array(pts_raw, dtype=np.float32)).tolist()

                with open(str(CALIBRATION_FILE), 'w') as f:

                    json.dump({"corners": pts_sorted}, f, indent=2)

                log_print(f"Calibration saved (sorted)")

                self.send_response(200)

                self.end_headers()

                self.wfile.write(b"OK")

            except Exception as e:

                self.send_error(400, str(e))


class TS(socketserver.ThreadingMixIn, http.server.HTTPServer):

    allow_reuse_address = True

    daemon_threads = True


# ============================================================

# Main loop

# ============================================================

def _build_tjc_state(pieces, reconst):
    """Convert vision data to arm-mm format for TJC display."""
    pieces_arm = []
    for pp in pieces:
        pick_arm = list(image_to_arm(pp.pickup_x_image, pp.pickup_y_image))
        # Convert polygon from warp px → arm-mm
        poly_arm = []
        poly = pp.polygon
        if poly is not None and len(poly) > 0:
            # polygon shape: (N, 1, 2) or (N, 2)
            arr = np.asarray(poly, dtype=np.float64)
            if arr.ndim == 3:
                arr = arr.reshape(-1, 2)
            for vx, vy in arr:
                ax, ay = image_to_arm(float(vx), float(vy))
                poly_arm.append([round(ax, 2), round(ay, 2)])
        pieces_arm.append({
            "id": f"piece_{pp.piece_id}",
            "pick_mm": [round(pick_arm[0], 2), round(pick_arm[1], 2)],
            "polygon_arm_mm": poly_arm,
        })

    plan_items = []
    if reconst is not None and reconst.plan:
        for item in reconst.plan:
            plan_items.append({
                "piece_id": str(item.get("piece_id", "")),
                "place_mm": item.get("place_mm", [0, 0]),
                "rotate_deg": item.get("rotate_deg", 0),
                "target_polygon_mm": item.get("target_polygon_mm"),
            })

    info = {
        "mode": reconst.selected_mode if reconst else "--",
        "fill_ratio": reconst.solver_info.get("fill_ratio", 0) if reconst else 0,
        "pieces_count": len(pieces),
    }
    if reconst and reconst.plan:
        info["assembly_order"] = [str(it["piece_id"]) for it in reconst.plan]

    return pieces_arm, plan_items, info


def main():

    log_print("=== Pi Puzzle Stream ===")


    # Always use LAB auto-detection (ignore saved calibration)

    corners = None

    c2w, w2c = None, None

    log_print("Camera mode - recognition OFF, waiting for toggle")


    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():

        log_print("ERROR: Cannot open camera")

        sys.exit(1)

    time.sleep(1)

    log_print("Camera ready")

    # ── TJC serial screen ──
    tjc = open_tjc()
    if tjc:
        log_print("TJC screen connected")
    else:
        log_print("TJC screen NOT available")


    latest_pieces = []

    latest_reconst = None

    _tjc_counter = [0]

    _tjc_last_frozen = [False]

    latest_calib = None

    last_detect = 0

    last_solve = 0

    last_a4_detect = 0

    last_calib_check = 0

    solving = False

    solve_queue = queue.Queue()

    frame_count = 0

    fps_start = time.time()

    a4_detect_interval = 1.5

    use_auto_detect = True


    def solve_worker(pieces_snap, calib_snap, mode="auto"):

        nonlocal solving

        try:

            # Debug: print input polygon shapes with coordinates

            for p in pieces_snap:

                poly = np.asarray(p.polygon, dtype=np.float64).reshape(-1, 2)

                poly_mm = poly / PIXELS_PER_MM

                edges_mm = [np.linalg.norm(poly_mm[i] - poly_mm[(i+1)%len(poly_mm)]) for i in range(len(poly_mm))]

                perimeter_mm = sum(edges_mm)

                coords = ", ".join(f"({x:.1f},{y:.1f})" for x, y in poly_mm)

                edges_str = ", ".join(f"{e:.1f}" for e in edges_mm)

                print(f"[IN] p{p.piece_id}: edges=[{edges_str}] perim={perimeter_mm:.1f}mm coords=[{coords}]", flush=True)

            result = solve_with_pick_recognition(pieces_snap, calib_snap, mode)

            solve_queue.put(result)

            if result.plan:

                for item in result.plan:

                    poly_mm = np.asarray(item["target_polygon_mm"], dtype=np.float64)

                    edges_mm = [np.linalg.norm(poly_mm[i] - poly_mm[(i+1)%len(poly_mm)]) for i in range(len(poly_mm))]

                    perimeter_mm = sum(edges_mm)

                    coords = ", ".join(f"({x:.1f},{y:.1f})" for x, y in poly_mm)

                    edges_str = ", ".join(f"{e:.1f}" for e in edges_mm)

                    print(f"[OUT] {item['piece_id']}: edges=[{edges_str}] perim={perimeter_mm:.1f}mm coords=[{coords}]", flush=True)

                    # Diagnostic: compare measured vs beautified (skip if vertex counts differ)

                    if "measured_target_polygon_mm" in item:

                        meas = np.asarray(item["measured_target_polygon_mm"], dtype=np.float64)

                        if meas.shape == poly_mm.shape:

                            diff = np.max(np.abs(meas - poly_mm))

                            print(f"[DIAG] {item['piece_id']}: measured_vs_display max_diff={diff:.4f}mm", flush=True)

            log_print(f"Solve OK: {result.selected_mode} fill={_fmt(result.solver_info.get('fill_ratio'))} geom={_fmt(result.solver_info.get('geometry_score'))}")

        except Exception as e:

            import traceback

            traceback.print_exc()

            print(f"[TRACEBACK] {traceback.format_exc()}", flush=True)

            solve_queue.put(None)

            log_print(f"Solve FAILED: {e}")

        finally:

            solving = False


    server = TS(("0.0.0.0", PORT), Handler)

    threading.Thread(target=server.serve_forever, daemon=True).start()

    log_print(f"Stream ready: http://192.168.31.93:{PORT}")


    # ---- serial listener: arm can send $DONE to unfreeze ----

    def _unfreeze_from_arm():

        with SharedState.lock:

            was_frozen = SharedState.frozen

            SharedState.frozen = False

            SharedState._stability_counter = 0

            SharedState._last_piece_centroids = []

            SharedState.freeze_data = None

            SharedState.last_action_msg = "UNFROZEN"

        if was_frozen:

            log_print("F: UNFREEZE (arm $DONE received)")


    start_listener(_unfreeze_from_arm)

    log_print("Serial listener started, waiting for $DONE from arm")

    # ----------------------------------------------------------


    try:

        while True:

            ret, frame = cap.read()

            if not ret:

                time.sleep(0.01)

                continue

            frame_count += 1

            now = time.time()


            # Process web button actions

            while not SharedState.action_queue.empty():

                action = SharedState.action_queue.get_nowait()

                if action == "P":

                    # Execute recovery with current mode (only when recognition is ON)

                    if not SharedState.recognition_active:

                        SharedState.last_action_msg = "Turn on recognition first"

                        log_print("P: ignored - recognition is OFF")

                    elif not solving and latest_pieces and latest_calib is not None:

                        solving = True

                        threading.Thread(target=solve_worker,

                                         args=(list(latest_pieces), latest_calib.copy(), SharedState.current_mode),

                                         daemon=True).start()

                        SharedState.last_action_msg = f"Recover ({MODE_LABELS[SharedState.current_mode]})"

                        log_print(f"P: recover {MODE_LABELS[SharedState.current_mode]}")

                    else:

                        SharedState.last_action_msg = "Busy or no pieces"

                elif action == "T":

                    SharedState.show_target = not SharedState.show_target

                    log_print(f"T: overlay {'ON' if SharedState.show_target else 'OFF'}")

                elif action == "A":

                    SharedState.area_display_mode = (SharedState.area_display_mode + 1) % 3

                    labels = ["OFF", "MIN", "MAX"]

                    log_print(f"A: area box {labels[SharedState.area_display_mode]}")

                elif action == "D":

                    r = latest_reconst

                    if r:

                        log_print(f"D: mode={r.selected_mode} plan={len(r.plan)} score={r.pattern_score:.4f} time={r.solve_time_sec:.2f}s")

                    else:

                        log_print(f"D: pieces={len(latest_pieces)} no reconst")

                elif action == "S":

                    with SharedState.lock:

                        f = SharedState.frame

                    if f is not None:

                        cv2.imwrite("/home/man/puzzle_app/saved_frame.jpg", f)

                        log_print("S: frame saved")

                # M, 0, 1, 2, 3 are handled in handle_action (mode change only, no queue needed)

                # but they still come through queue for logging:

                elif action == "R":

                    # If turning OFF recognition, also unfreeze

                    if not SharedState.recognition_active:

                        SharedState.frozen = False

                        SharedState._stability_counter = 0

                        SharedState._last_piece_centroids = []

                        SharedState.freeze_data = None

                elif action in ("M", "0", "1", "2", "3"):

                    log_print(f"{action}: mode -> {MODE_LABELS[SharedState.current_mode]}")


            # Always run A4 detection (blob bbox)

            if not SharedState.frozen and now - last_a4_detect > a4_detect_interval:

                detected = auto_detect_a4(frame)

                if detected is not None:

                    corners = detected

                    c2w, w2c = build_matrices(corners)

                    # Sync calibration to SharedState for web crosshair (arm-mm)
                    with SharedState.lock:
                        SharedState.calib_data = {
                            "has_calib": True,
                            "corners": corners.tolist(),
                            "c2w": c2w.tolist(),
                            "ppm": round(PIXELS_PER_MM, 4),
                            "origin_wx": WARP_WIDTH - 1,
                            "origin_wy": 300.0,
                        }

                last_a4_detect = now


            # Detection (only when recognition is active AND not frozen)

            if SharedState.recognition_active and not SharedState.frozen:

                if corners is not None and c2w is not None:

                    if now - last_detect >= 0.25:

                        calib = cv2.warpPerspective(frame, c2w, (WARP_WIDTH, WARP_HEIGHT))

                        latest_calib = calib

                        try:

                            pieces, mask = detect_pieces(calib)

                            latest_pieces = pieces

                            if pieces:

                                areas_mm2 = [p.area_px / (PIXELS_PER_MM ** 2) for p in pieces]

                                log_print(f"Detected {len(pieces)} pieces: areas={[round(a) for a in areas_mm2]} mm2 sides={[len(p.polygon) for p in pieces]}")

                        except Exception as e:

                            log_print(f"Detection error: {e}")

                        last_detect = now


                        # Auto-freeze stability check

                        if pieces and latest_reconst is not None and latest_reconst.plan and len(pieces) == len(latest_reconst.plan):

                            # Compute centroids in arm mm

                            centroids = []

                            for p in pieces:

                                px_mm, py_mm = image_to_arm(p.center_x_image, p.center_y_image)


                                centroids.append((px_mm, py_mm))

                            # Compare with previous frame

                            stable = True

                            if SharedState._last_piece_centroids and len(SharedState._last_piece_centroids) == len(centroids):

                                for (cx, cy), (lx, ly) in zip(centroids, SharedState._last_piece_centroids):

                                    if math.hypot(cx - lx, cy - ly) > SharedState._STABILITY_THRESHOLD_MM:

                                        stable = False

                                        break

                            else:

                                stable = False

                            if stable:

                                SharedState._stability_counter += 1

                            else:

                                SharedState._stability_counter = 0

                            SharedState._last_piece_centroids = centroids

                            # Trigger freeze

                            if SharedState._stability_counter >= SharedState._STABILITY_FRAMES:

                                SharedState.frozen = True

                                SharedState.last_action_msg = "FROZEN"

                                SharedState._stability_counter = 0

                                log_print("F: AUTO-FREEZE triggered (stable detection)")

                                # Build freeze data

                                freeze_pieces = []

                                for pp in latest_pieces:

                                    px_mm, py_mm = image_to_arm(pp.pickup_x_image, pp.pickup_y_image)

                                    pid_str = f"piece_{pp.piece_id}"

                                    # Convert polygon vertices to arm-mm for serial screen display

                                    poly_arm = []

                                    for v in pp.polygon:

                                        vx = float(v[0][0])

                                        vy = float(v[0][1])

                                        ax, ay = image_to_arm(vx, vy)

                                        poly_arm.append([round(ax, 2), round(ay, 2)])

                                    freeze_pieces.append({

                                        "id": pid_str,

                                        "pick_mm": [round(px_mm, 2), round(py_mm, 2)],

                                        "area_mm2": round(pp.area_px / (PIXELS_PER_MM**2), 1),

                                        "vertices": int(len(pp.polygon)),

                                        "rotate_deg": 0.0,

                                        "place_mm": [0.0, 0.0],

                                        "polygon_arm_mm": poly_arm,

                                    })

                                # Fill rotate_deg and place_mm from reconstruction plan

                                if latest_reconst is not None and latest_reconst.plan:

                                    for fp in freeze_pieces:

                                        for item in latest_reconst.plan:

                                            if item.get("piece_id") == fp["id"]:

                                                fp["rotate_deg"] = round(item.get("rotate_deg", 0.0), 2)

                                                # place_mm from solver mm → physical mm

                                                # place_mm is already in arm mm (converted by _plan_to_arm)
                                                place_mm_arm = item.get("place_mm", [0.0, 0.0])
                                                fp["place_mm"] = [
                                                    round(float(place_mm_arm[0]), 2),
                                                    round(float(place_mm_arm[1]), 2)
                                                ]

                                                break

                                    # Assembly order = plan order (topological)

                                    assembly_order = [str(item["piece_id"]) for item in latest_reconst.plan]

                                    # Pickup order = sorted by distance from origin

                                    pickup_order = sorted(freeze_pieces, key=lambda p: arm_distance(p["pick_mm"][0], p["pick_mm"][1]))

                                    pickup_order_ids = [p["id"] for p in pickup_order]

                                else:

                                    assembly_order = [p["id"] for p in freeze_pieces]

                                    pickup_order_ids = [p["id"] for p in freeze_pieces]

                                # A4 paper corners in arm-mm (counter-clockwise from top-right)

                                # arm-mm: X← leftward, Y↓ downward. A4 is 210×297mm.
                                a4_corners_arm = [

                                    [210.0, -75.0],   # top-left in arm-mm (max X, min Y)

                                    [0.0, -75.0],     # top-right in arm-mm

                                    [0.0, 222.0],     # bottom-right in arm-mm

                                    [210.0, 222.0],   # bottom-left in arm-mm

                                ]

                                SharedState.freeze_data = {

                                    "frozen": True,

                                    "pieces": freeze_pieces,

                                    "pickup_order": pickup_order_ids,

                                    "assembly_order": assembly_order,

                                    "solve_info": {

                                        "mode": latest_reconst.selected_mode if latest_reconst else "unknown",

                                        "fill_ratio": round(latest_reconst.solver_info.get("fill_ratio", 0.0), 4) if latest_reconst else 0.0,

                                    },

                                    "a4_corners_arm_mm": a4_corners_arm,

                                }

                                # Write freeze.json

                                try:

                                    with open("/home/man/puzzle_app/freeze.json", "w") as jf:

                                        json.dump(SharedState.freeze_data, jf, indent=2)

                                    log_print("F: freeze.json written")

                                    # Also output via serial port for robotic arm

                                    threading.Thread(target=send_and_wait_done,

                                                     args=(SharedState.freeze_data,),

                                                     daemon=True).start()

                                except Exception as e:

                                    log_print(f"F: failed to write freeze.json: {e}")


                        if not solving and len(latest_pieces) >= 1 and now - last_solve > 2.0:

                            solving = True

                            last_solve = now

                            threading.Thread(target=solve_worker,

                                             args=(list(latest_pieces), calib.copy(), "auto"),

                                             daemon=True).start()


                # Check solve result

                try:

                    result = solve_queue.get_nowait()

                    if latest_reconst is None or result != latest_reconst:

                        # New solve result → reset stability counter

                        SharedState._stability_counter = 0

                        SharedState._last_piece_centroids = []

                    latest_reconst = result

                except queue.Empty:

                    pass


                # Draw overlay

                plan_to_draw = latest_reconst if SharedState.show_target and latest_reconst is not None and latest_reconst.plan else None

                display = draw_overlay(frame, corners, latest_pieces, plan_to_draw, w2c, SharedState.area_display_mode)

            elif SharedState.frozen:

                # FROZEN: draw last overlay with FROZEN indicator

                plan_to_draw = latest_reconst if SharedState.show_target and latest_reconst is not None and latest_reconst.plan else None

                display = draw_overlay(frame, corners, latest_pieces, plan_to_draw, w2c, SharedState.area_display_mode)

                # Superimpose FROZEN banner

                fh = display.shape[0]

                fw = display.shape[1]

                # Red border pulse effect

                cv2.rectangle(display, (4, 4), (fw-4, fh-4), (0, 0, 255), 8)

                # FROZEN text in center-top

                txt = "FROZEN"

                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)

                tx, ty = (fw - tw)//2, fh//4

                # Dark background box

                cv2.rectangle(display, (tx-20, ty-th-10), (tx+tw+20, ty+15), (0,0,0), -1)

                cv2.putText(display, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5, cv2.LINE_AA)

            else:

                # Recognition OFF: pure camera, no processing at all

                latest_pieces = []

                latest_reconst = None

                display = frame


            # ── Update TJC serial screen (every 4th frame to avoid bottleneck) ──
            if tjc:
                _tjc_counter[0] += 1
                if _tjc_counter[0] % 4 == 0 or SharedState.frozen != _tjc_last_frozen[0]:
                    _tjc_last_frozen[0] = SharedState.frozen
                    try:
                        pa, pl, inf = _build_tjc_state(latest_pieces, latest_reconst)
                        draw_state(tjc, pa, pl, inf,
                                   frozen=SharedState.frozen,
                                   recognition=SharedState.recognition_active)
                    except Exception as e:
                        log_print(f"TJC error: {e}")

            # Display locally via cv2.imshow (original method)

            cv2.imshow("Camera", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('1'):

                SharedState.action_queue.put("1")

            elif key == ord('2'):

                SharedState.action_queue.put("2")

            elif key == ord('3'):

                SharedState.action_queue.put("3")

            elif key == ord('t'):

                SharedState.action_queue.put("T")

            elif key == ord('r'):

                SharedState.action_queue.put("R")

            elif key == ord('d'):

                SharedState.action_queue.put("D")

            elif key == ord('a'):

                SharedState.action_queue.put("A")

            elif key == ord('s'):

                SharedState.action_queue.put("S")

            elif key == 27 or key == ord('q'):  # ESC or Q

                raise KeyboardInterrupt


            # FPS

            elapsed = now - fps_start

            if elapsed >= 1.0:

                fps_val = frame_count / elapsed

                frame_count = 0

                fps_start = now

                mode_str = "idle"

                if solving:

                    mode_str = "solving"

                elif latest_reconst is not None:

                    mode_str = latest_reconst.selected_mode

                with SharedState.lock:

                    SharedState.status = {

                        "pieces": len(latest_pieces), "mode": mode_str, "fps": round(fps_val, 1),

                        "show_target": SharedState.show_target, "area_mode": SharedState.area_display_mode,

                        "last_action": SharedState.last_action_msg,

                        "selected_mode": MODE_LABELS[SharedState.current_mode],

                        "recognition": SharedState.recognition_active,

                        "frozen": SharedState.frozen,

                    }


            with SharedState.lock:

                SharedState.frame = display

                SharedState.raw_frame = frame


            time.sleep(0.01)

    except KeyboardInterrupt:

        pass

    finally:

        cap.release()

        cv2.destroyAllWindows()

        log_print("Stopped")


if __name__ == "__main__":

    main()