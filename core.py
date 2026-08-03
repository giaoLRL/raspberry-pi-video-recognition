#!/usr/bin/env python3
"""Core shared state and logging for the puzzle robot system."""

import threading
import queue
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

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

    # TJC HMI status text (updated by system_check, drawn by draw_state)
    arm_status = "Arm Ready"
    pi_status = "Pi Ready"
    task_status = "Waiting"

    freeze_data = None
    solving = False
    plan_to_draw = None
    a4_corners_arm = None
    a4_rotation_deg = 0.0
    cam_frame_arm = None  # dict: pieces, solve_info, etc.

    # System button toggle state (press to start, press again to stop)
    _arm_check_running = False
    _arm_check_cancel = None  # threading.Event or None
    _pi_check_running = False
    _pi_check_cancel = None
    _arm_pos_running = False

    # Calibration data for web crosshair (arm-mm coordinate display)
    calib_data = None  # dict: corners, c2w, ppm, origin_wx, origin_wy, has_calib
    # Auto-freeze stability tracking

    _stability_counter = 0

    _last_piece_centroids = []  # list of (cx,cy) tuples per frame

    _STABILITY_FRAMES = 3       # consecutive stable frames to trigger

    _STABILITY_THRESHOLD_MM = 1.5  # max centroid movement between frames




def log_print(msg):

    print(msg)

    with SharedState.lock:

        SharedState.log_lines.append(str(msg))

        if len(SharedState.log_lines) > SharedState.max_log_lines:

            SharedState.log_lines.pop(0)


MODE_LABELS = {"auto": "AUTO", "fixed": "FIXED", "unknown-white": "WHITE", "unknown-pattern": "PATTERN"}

MODE_CYCLE = ["auto", "fixed", "unknown-white", "unknown-pattern"]



