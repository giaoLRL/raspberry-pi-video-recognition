#!/usr/bin/env python3
"""Puzzle Robot — main entry point. Orchestrates all subsystems."""

import json, math, sys, time
import threading
import queue
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# Add project paths
_root = Path(__file__).resolve().parent
_puzzle_app = _root / "puzzle_app"
_vision = _root / "vision"
for d in (str(_vision), str(_puzzle_app)):
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

from serial_protocol import send_and_wait_done, start_listener
from system_check import arm_check, pi_check, StatusReporter, make_unfreeze_callback

from core import SharedState, log_print, MODE_LABELS, MODE_CYCLE
from actions import setup as setup_actions, handle_action
from vision.coords import image_to_arm, solver_to_arm, arm_to_warp, arm_distance
from vision.detection import (
    DetectedPiece, RecoveryResult,
    setup as setup_detect,
    order_points, load_corners, build_matrices,
    auto_detect_a4, auto_detect_a4_partial,
    create_piece_mask, detect_pieces,
)
from vision.solving import (
    setup as setup_solve,
    solve_with_pick_recognition, _fmt, solve_worker,
)
from vision.rendering import setup as setup_render, draw_overlay
from web.server import HTML, CALIB_HTML, setup as setup_web, Handler, TS
from tjc_display import (
    open_tjc, draw_state, arm_to_screen,
    _build_tjc_state, setup_tjc_state,
)


# ============================================================
# Parameters
# ============================================================
CAMERA_INDEX = 0
WARP_WIDTH = 840
WARP_HEIGHT = 1188

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
MIN_FILL_RATIO = 0.20

CALIBRATION_FILE = _puzzle_app / "a4_corners.json"
SOLVER_CONFIG_FILE = _puzzle_app / "config.json"
TAUGHT_LAYOUT_FILE = _puzzle_app / "taught_layout.json"
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
ORIGIN_FRACTION = 0.25
JPEG_QUALITY = 80
STREAM_FPS = 15


# ============================================================
# Plan to arm-mm conversion
# ============================================================
def _plan_to_arm(plan, solver_info):
    items = []
    for p in plan:
        poly = np.asarray(p["target_polygon_mm"], dtype=np.float64)
        # Compute centroid of target polygon and convert to arm coordinates
        centroid = np.mean(poly, axis=0)
        place_mm = solver_to_arm(float(centroid[0]), float(centroid[1]))
        items.append({
            "piece_id": p["piece_id"],
            "place_mm": [round(place_mm[0], 2), round(place_mm[1], 2)],
            "rotate_deg": p.get("rotate_deg", 0),
            "target_polygon_mm": poly.tolist(),
        })
    return items


def main():
    _cached_w2c = None

    log_print("=== Puzzle Robot Starting ===")

    # ---- Camera ----
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        log_print("ERROR: Cannot open camera")
        sys.exit(1)

    # ---- Load calibration ----
    try:
        start_listener(_unfreeze_from_arm)
        log_print("Serial listener started")
    except Exception as e:
        log_print(f"Serial listener skipped: {e}")

    # ---- TJC screen ----
    tjc = open_tjc()
    if tjc:
        log_print("TJC screen connected")
    else:
        log_print("TJC screen not available")

    # ---- Solve queue ----
    solve_queue = queue.Queue()

    # ---- Wire all modules ----
    setup_detect(
        calib_file=CALIBRATION_FILE,
        pixels_per_mm=PIXELS_PER_MM,
        min_piece_area_px=MIN_PIECE_AREA_PX,
        max_piece_area_px=MAX_PIECE_AREA_PX,
        log_print=log_print,
    )

    # ---- Load calibration ----
    corners = load_corners()
    _cached_w2c = None
    if corners is not None:
        _cached_c2w, w2c = build_matrices(corners)
        SharedState.calib_data = {
            "has_calib": True,
            "corners": corners.tolist(),
        }
        log_print("Calibration loaded")
    else:
        log_print("No calibration ? will auto-detect A4")
    setup_solve(
        solve_queue=solve_queue,
        image_to_arm=image_to_arm,
        solver_to_arm=solver_to_arm,
        recovery_modes=RECOVERY_MODES,
        recovery_mode_names=RECOVERY_MODE_NAMES,
        auto_fixed_strong_error_mm=AUTO_FIXED_STRONG_ERROR_MM,
        auto_pattern_score_threshold=AUTO_PATTERN_SCORE_THRESHOLD,
        max_detected_pieces=MAX_DETECTED_PIECES,
        send_and_wait_done=send_and_wait_done,
        mode_labels=MODE_LABELS,
        shared_state=SharedState,
        log_print=log_print,
        solver_config_file=SOLVER_CONFIG_FILE,
        taught_layout_file=TAUGHT_LAYOUT_FILE,
        load_corners=load_corners,
        build_matrices=build_matrices,
        auto_detect_a4=auto_detect_a4,
        detect_pieces=detect_pieces,
        pixels_per_mm=PIXELS_PER_MM,
    )
    setup_render(
        image_to_arm=image_to_arm,
        arm_to_warp=arm_to_warp,
        pixels_per_mm=PIXELS_PER_MM,
        min_piece_area_px=MIN_PIECE_AREA_PX,
        max_piece_area_px=MAX_PIECE_AREA_PX,
        log_print=log_print,
        shared_state=SharedState,
    )
    setup_actions(
        shared_state=SharedState,
        log_print=log_print,
        mode_labels=MODE_LABELS,
        unfreeze_callback=None,
        auto_detect_a4=auto_detect_a4,
        build_matrices=build_matrices,
        image_to_arm=image_to_arm,
    )
    setup_web(
        shared_state=SharedState,
        handle_action=handle_action,
        order_points=order_points,
        log_print=log_print,
        calib_file=CALIBRATION_FILE,
        jpeg_quality=JPEG_QUALITY,
        stream_fps=STREAM_FPS,
    )
    setup_tjc_state(
        shared_state=SharedState,
        mode_labels=MODE_LABELS,
        image_to_arm=image_to_arm,
    )

    # ---- Start HTTP server ----
    server = TS(("0.0.0.0", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log_print(f"HTTP server on port {PORT}")

    # ---- Main loop ----
    frame_count = 0
    fps_val = 0.0
    fps_timer = time.perf_counter()
    fps_frame = 0
    last_detect = 0.0
    detect_interval = 0.25
    latest_pieces = []
    _cached_w2c = None
    _cached_corners = None

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        now = time.perf_counter()
        frame_count += 1
        fps_frame += 1
        if now - fps_timer >= 1.0:
            fps_val = fps_frame / (now - fps_timer)
            fps_timer = now
            fps_frame = 0

        # ---- A4 detection ----
        corners, w2c = None, None
        if frame_count % 8 == 0:
            raw_corners = auto_detect_a4(frame)
            if raw_corners is not None:
                corners = order_points(np.array(raw_corners, dtype=np.float32))
                c2w, w2c = build_matrices(corners)
                _cached_corners = corners
                _cached_w2c = w2c
                SharedState.a4_corners_arm = corners.tolist()
                with SharedState.lock:
                    SharedState.calib_data = {
                        "has_calib": True,
                        "corners": corners.tolist(),
                        "c2w": _cached_w2c,
                    }
        if corners is None and _cached_w2c is not None:
            w2c = _cached_w2c
            corners = _cached_corners

        # ---- Piece detection ----
        if SharedState.recognition_active and w2c is not None and now - last_detect > detect_interval:
            try:
                warped = cv2.warpPerspective(frame, w2c, (WARP_WIDTH, WARP_HEIGHT)); pieces, _ = detect_pieces(warped)
                if pieces:
                    latest_pieces = pieces
                    last_detect = now

                    # Auto-freeze stability check
                    centroids = []
                    for p in pieces:
                        ax, ay = image_to_arm(p.center_x_image, p.center_y_image)
                        centroids.append((round(ax, 2), round(ay, 2)))
                    if centroids == SharedState._last_piece_centroids:
                        SharedState._stability_counter += 1
                    else:
                        SharedState._stability_counter = 0
                        SharedState._last_piece_centroids = centroids

                    if SharedState._stability_counter >= 8 and not SharedState.frozen:
                        SharedState.frozen = True
                        SharedState.plan_to_draw = None
                        log_print("FROZEN")
                        while not solve_queue.empty():
                            try:
                                solve_queue.get_nowait()
                            except queue.Empty:
                                pass
                        SharedState.solving = True
                        threading.Thread(
                            target=solve_worker,
                            args=(list(pieces), w2c, SharedState.current_mode),
                            daemon=True,
                        ).start()
            except Exception as e:
                log_print(f"Detection error: {e}")
                last_detect = now

        # ---- Solve result ----
        try:
            result = solve_queue.get_nowait()
            if result is not None and hasattr(result, "plan") and result.plan:
                plan_items = _plan_to_arm(result.plan, result.solver_info)
                with SharedState.lock:
                    SharedState.plan_to_draw = plan_items
                    SharedState.freeze_data = {
                        "frozen": True,
                        "pieces": plan_items,
                        "message": f"Solved: {result.selected_mode}",
                    }
                log_print(f"Plan ready: {len(plan_items)} pieces")
            elif result is not None:
                SharedState.frozen = False
                SharedState._stability_counter = 0
                log_print("Solve rejected, resuming detection")
        except queue.Empty:
            pass

        # ---- Overlay ----
        with SharedState.lock:
            plan_to_draw = (
                list(SharedState.plan_to_draw) if SharedState.plan_to_draw else None
            )
        if corners is not None and w2c is not None:
            display = draw_overlay(
                frame, corners, latest_pieces, plan_to_draw,
                w2c, SharedState.area_display_mode,
            )
        else:
            display = frame.copy()
        with SharedState.lock:
            SharedState.frame = display
            SharedState.raw_frame = frame

        # ---- TJC display ----
        if tjc and frame_count % 4 == 0:
            try:
                cam_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                cam_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                tjc_pieces_arm, tjc_plan, tjc_info, a4_rot, cam_frame = _build_tjc_state(
                    latest_pieces, None,
                    fps=fps_val,
                    last_action=SharedState.last_action_msg,
                    selected_mode=MODE_LABELS.get(SharedState.current_mode, "AUTO"),
                    a4_corners_camera=corners,
                    cam_w=int(cam_w) if cam_w > 0 else 1920,
                    cam_h=int(cam_h) if cam_h > 0 else 1080,
                )
                # Override plan items with solved plan if available
                if plan_to_draw:
                    tjc_plan = [
                        {
                            "piece_id": str(it.get("piece_id", "")),
                            "place_mm": it.get("place_mm", [0, 0]),
                            "rotate_deg": it.get("rotate_deg", 0),
                            "target_polygon_mm": it.get("target_polygon_mm"),
                        }
                        for it in plan_to_draw
                    ]

                draw_state(
                    tjc,
                    pieces_arm=tjc_pieces_arm,
                    plan_items=tjc_plan,
                    info=tjc_info,
                    frozen=SharedState.frozen,
                    recognition=SharedState.recognition_active,
                    arm_status=SharedState.arm_status,
                    pi_status=SharedState.pi_status,
                    task_status=SharedState.task_status,
                    project_loaded=True,
                    a4_rotation_deg=a4_rot,
                    cam_frame_arm=cam_frame,
                )

                # Poll TJC buttons
                msgs = tjc.poll_messages()
                for msg in msgs:
                    if msg and len(msg) > 1:
                        handle_action(msg)
            except Exception as e:
                import traceback; log_print(f"TJC error: {e}"); traceback.print_exc()

        # ---- Handle queued HTTP actions ----
        while not SharedState.action_queue.empty():
            try:
                action = SharedState.action_queue.get_nowait()
                handle_action(action)
            except queue.Empty:
                break

        # ---- Update status ----
        with SharedState.lock:
            SharedState.status["pieces"] = len(latest_pieces)
            SharedState.status["fps"] = round(fps_val, 1)
            SharedState.status["frozen"] = SharedState.frozen
            SharedState.status["recognition"] = SharedState.recognition_active


def _unfreeze_from_arm():
    with SharedState.lock:
        was_frozen = SharedState.frozen
        SharedState.frozen = False
        SharedState._stability_counter = 0
    if was_frozen:
        log_print("Unfrozen by arm $DONE")


if __name__ == "__main__":
    main()
