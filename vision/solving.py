#!/usr/bin/env python3
"""Puzzle solving ? multi-mode solver with auto-recovery.

Extracted from main.py. Dependencies injected via setup().
"""

import json
import time
from typing import Any, Optional

import cv2
import numpy as np

from puzzle_vision.config import load_config
from puzzle_vision.detector import PieceObservation
from puzzle_vision.geometry import (
    safe_interior_point,
    edge_lengths as solver_edge_lengths,
    normalize_winding as solver_normalize_winding,
    polygon_area as solver_polygon_area,
    polygon_centroid as solver_polygon_centroid,
    rotation_matrix_row,
)
from puzzle_vision.solver import SolveError, solve_card, solve_fixed, solve_taught, solve_unknown

from vision.detection import DetectedPiece, RecoveryResult

_ctx = {}

def setup(*, solve_queue=None, image_to_arm=None, solver_to_arm=None,
          recovery_modes=None, recovery_mode_names=None,
          auto_fixed_strong_error_mm=None, auto_pattern_score_threshold=None,
          max_detected_pieces=None,
          _build_tjc_state=None, send_and_wait_done=None, mode_labels=None,
          shared_state=None, log_print=None,
          solver_config_file=None, taught_layout_file=None,
          load_corners=None, build_matrices=None, auto_detect_a4=None,
          detect_pieces=None,
          pixels_per_mm=None):
    _ctx["image_to_arm"] = image_to_arm
    _ctx["solver_to_arm"] = solver_to_arm
    _ctx["SharedState"] = shared_state
    _ctx["log_print"] = log_print
    _ctx["SOLVER_CONFIG_FILE"] = solver_config_file
    _ctx["TAUGHT_LAYOUT_FILE"] = taught_layout_file
    _ctx["load_corners"] = load_corners
    _ctx["build_matrices"] = build_matrices
    _ctx["auto_detect_a4"] = auto_detect_a4
    _ctx["detect_pieces"] = detect_pieces
    _ctx["PIXELS_PER_MM"] = pixels_per_mm
    _ctx["solve_queue"] = solve_queue
    _ctx["RECOVERY_MODES"] = recovery_modes
    _ctx["RECOVERY_MODE_NAMES"] = recovery_mode_names
    _ctx["MAX_DETECTED_PIECES"] = max_detected_pieces
    _ctx["AUTO_PATTERN_SCORE_THRESHOLD"] = auto_pattern_score_threshold
    _ctx["AUTO_FIXED_STRONG_ERROR_MM"] = auto_fixed_strong_error_mm


def detected_pieces_to_solver_observations(

    pieces: list[DetectedPiece],

) -> list[PieceObservation]:

    observations: list[PieceObservation] = []

    for piece in pieces:

        polygon_px = np.asarray(piece.polygon, dtype=np.float64).reshape(-1, 2)

        polygon_mm = solver_normalize_winding(polygon_px / _ctx['PIXELS_PER_MM'])

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

        kernel_size = max(5, int(round(2.0 * _ctx['PIXELS_PER_MM'])) | 1)

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

    config = load_config(str(_ctx['SOLVER_CONFIG_FILE']) if _ctx['SOLVER_CONFIG_FILE'].exists() else None)

    config["paper"]["pixels_per_mm"] = _ctx['PIXELS_PER_MM']

    config["paper"]["width_mm"] = A4_WIDTH_MM

    config["paper"]["height_mm"] = A4_HEIGHT_MM

    config["paper"]["divider_y_mm"] = A4_HEIGHT_MM * 0.5

    # target_zone in solver coords = lower half of A4 (below midline)
    config["unknown"]["target_zone_mm"] = [0.0, A4_HEIGHT_MM * 0.5, A4_WIDTH_MM, A4_HEIGHT_MM]

    config["unknown"]["taught_layout_path"] = str(_ctx['TAUGHT_LAYOUT_FILE'])

    taught_layout: Optional[dict[str, Any]] = None

    if _ctx['TAUGHT_LAYOUT_FILE'].exists():

        try:

            taught_layout = json.loads(_ctx['TAUGHT_LAYOUT_FILE'].read_text(encoding="utf-8"))

        except (OSError, ValueError, TypeError, json.JSONDecodeError):

            taught_layout = None

    return config, taught_layout


def _plan_to_arm(plan: list[dict[str, Any]], info: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert solver-mm coordinates to arm-mm in-place, so rendering uses arm directly."""
    for item in plan:
        if "place_mm" in item:
            ax, ay = _ctx['solver_to_arm'](float(item["place_mm"][0]), float(item["place_mm"][1]))
            item["place_mm"] = [round(ax, 3), round(ay, 3)]
        if "target_polygon_mm" in item:
            item["target_polygon_mm"] = [
                [round(_ctx['solver_to_arm'](float(x), float(y))[0], 3),
                 round(_ctx['solver_to_arm'](float(x), float(y))[1], 3)]
                for x, y in item["target_polygon_mm"]
            ]
        if "measured_target_polygon_mm" in item:
            item["measured_target_polygon_mm"] = [
                [round(_ctx['solver_to_arm'](float(x), float(y))[0], 3),
                 round(_ctx['solver_to_arm'](float(x), float(y))[1], 3)]
                for x, y in item["measured_target_polygon_mm"]
            ]
    if info.get("target_origin_mm"):
        ax, ay = _ctx['solver_to_arm'](float(info["target_origin_mm"][0]), float(info["target_origin_mm"][1]))
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

        plan, info = solve_card(observations, unknown_cfg, calibrated_region, _ctx['PIXELS_PER_MM'])
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

    plan, info = solve_unknown(observations, unknown_cfg, calibrated_region, _ctx['PIXELS_PER_MM'], use_texture=False)

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

    if requested_mode not in _ctx['RECOVERY_MODES']:

        raise RuntimeError(f"Unknown mode: {requested_mode}")

    if not 1 <= len(pieces) <= _ctx['MAX_DETECTED_PIECES']:

        raise RuntimeError(f"Recovery requires 1-{_ctx['MAX_DETECTED_PIECES']} pieces, got {len(pieces)}")

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

            raise RuntimeError(f"{_ctx['RECOVERY_MODE_NAMES'][requested_mode]} failed" + (f": {detail}" if detail else ": rejected"))

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

                _ctx['log_print'](f"auto: fixed rejected (err={max_err:.1f}mm area_ratio={area_ratio:.3f} cost={assignment_cost:.1f})")

    # One lightweight classification, then run preferred path only

    is_pattern = pattern_score >= _ctx['AUTO_PATTERN_SCORE_THRESHOLD']

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

        f"{_ctx['RECOVERY_MODE_NAMES'].get(item['mode'], item['mode'])}: {item.get('error', 'rejected')}"

        for item in attempts)

    raise RuntimeError("All recovery modes failed: " + details)


# ============================================================

# Warp and overlay drawing

# ============================================================




# ?? Worker function ??
def solve_worker(pieces_snap, calib_snap, mode="auto"):

    try:

        # Debug: print input polygon shapes with coordinates

        for p in pieces_snap:

            poly = np.asarray(p.polygon, dtype=np.float64).reshape(-1, 2)

            poly_mm = poly / _ctx['PIXELS_PER_MM']

            edges_mm = [np.linalg.norm(poly_mm[i] - poly_mm[(i+1)%len(poly_mm)]) for i in range(len(poly_mm))]

            perimeter_mm = sum(edges_mm)

            coords = ", ".join(f"({x:.1f},{y:.1f})" for x, y in poly_mm)

            edges_str = ", ".join(f"{e:.1f}" for e in edges_mm)

            print(f"[IN] p{p.piece_id}: edges=[{edges_str}] perim={perimeter_mm:.1f}mm coords=[{coords}]", flush=True)

        result = solve_with_pick_recognition(pieces_snap, calib_snap, mode)

        _ctx['solve_queue'].put(result)

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

        _ctx['log_print'](f"Solve OK: {result.selected_mode} fill={_fmt(result.solver_info.get('fill_ratio'))} geom={_fmt(result.solver_info.get('geometry_score'))}")

    except Exception as e:

        import traceback

        traceback.print_exc()

        print(f"[TRACEBACK] {traceback.format_exc()}", flush=True)

        _ctx['solve_queue'].put(None)

        _ctx['log_print'](f"Solve FAILED: {e}")

    finally:

        _ctx['SharedState'].solving = False


