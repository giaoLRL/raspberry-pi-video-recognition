#!/usr/bin/env python3
"""Action handlers for TJC button commands, web UI, and system checks."""

import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from core import MODE_CYCLE
from system_check import StatusReporter, arm_check, pi_check

_ctx = {}

def setup(*, shared_state, log_print, mode_labels, unfreeze_callback,
          auto_detect_a4, build_matrices, image_to_arm):
    _ctx["SharedState"] = shared_state
    _ctx["log_print"] = log_print
    _ctx["MODE_LABELS"] = mode_labels
    _ctx["unfreeze_callback"] = unfreeze_callback
    _ctx["auto_detect_a4"] = auto_detect_a4
    _ctx["build_matrices"] = build_matrices
    _ctx["image_to_arm"] = image_to_arm

MONITOR_URL_DATA = "http://127.0.0.1:8081/data"

def handle_action(cmd):
    """Handle action from web UI or TJC. M/0/1/2/3 change mode, R toggles recognition, etc."""

    Shared = _ctx["SharedState"]
    if cmd == "M":
        idx = MODE_CYCLE.index(Shared.current_mode)
        Shared.current_mode = MODE_CYCLE[(idx + 1) % len(MODE_CYCLE)]
        Shared.last_action_msg = f"Mode: {_ctx['MODE_LABELS'][Shared.current_mode]}"
        return f"OK Mode -> {_ctx['MODE_LABELS'][Shared.current_mode]}"

    elif cmd in ("0", "1", "2", "3"):
        mode_idx = int(cmd)
        mapping = {0: "auto", 1: "fixed", 2: "unknown-white", 3: "unknown-pattern"}
        Shared.current_mode = mapping.get(mode_idx, "auto")
        Shared.last_action_msg = f"Mode: {_ctx['MODE_LABELS'][Shared.current_mode]}"
        return f"OK Mode -> {_ctx['MODE_LABELS'][Shared.current_mode]}"

    elif cmd == "R":
        with open("/tmp/action_debug.log", "a") as f:
            f.write(f"R called: before={Shared.recognition_active}, id(Shared)={id(Shared)}, id(SharedState)={id(_ctx['SharedState'])}\n")
        Shared.recognition_active = not Shared.recognition_active
        with open("/tmp/action_debug.log", "a") as f:
            f.write(f"R result: after={Shared.recognition_active}\n")
        state_str = "ON" if Shared.recognition_active else "OFF"
        Shared.last_action_msg = f"Recognition {state_str}"
        return f"OK Recognition -> {state_str}"

    elif cmd in ("M1", "M2", "M3"):
        mode_idx = int(cmd[1])
        mapping = {1: "fixed", 2: "unknown-white", 3: "unknown-pattern"}
        Shared.current_mode = mapping.get(mode_idx, "fixed")
        Shared.recognition_active = True
        Shared.last_action_msg = f"Mode: {_ctx['MODE_LABELS'][Shared.current_mode]}"
        _ctx["log_print"](f"TJC {cmd}: mode -> {_ctx['MODE_LABELS'][Shared.current_mode]}")
        return f"OK Mode -> {_ctx['MODE_LABELS'][Shared.current_mode]}"

    elif cmd == "ARM_CHECK":
        if Shared._arm_check_running:
            _ctx["log_print"]("ARM_CHECK: cancelling...")
            if Shared._arm_check_cancel:
                Shared._arm_check_cancel.set()
            Shared._arm_check_running = False
            Shared.arm_status = "Arm Ready"
            Shared.last_action_msg = "Arm check stopped"
            return "OK Arm check stopped"
        else:
            _ctx["log_print"]("ARM_CHECK: starting arm self-test")
            Shared.last_action_msg = "Arm self-check"
            Shared.arm_status = "Checking..."
            Shared._arm_check_running = True
            cancel = threading.Event()
            Shared._arm_check_cancel = cancel

            def _do_arm_check():
                try:
                    reporter = StatusReporter(
                        set_arm_status=lambda s: setattr(Shared, "arm_status", s),
                        set_pi_status=lambda s: setattr(Shared, "pi_status", s),
                        set_task_status=lambda s: setattr(Shared, "task_status", s),
                        log_fn=_ctx["log_print"],
                    )
                    result = arm_check(reporter, cancel_event=cancel)
                    if not cancel.is_set():
                        _ctx["log_print"](f"ARM_CHECK result: {result}")
                except Exception as e:
                    if not cancel.is_set():
                        _ctx["log_print"](f"ARM_CHECK error: {e}")
                        Shared.arm_status = "Error"
                finally:
                    Shared._arm_check_running = False
            threading.Thread(target=_do_arm_check, daemon=True).start()
            return "OK Arm check started"

    elif cmd == "PI_CHECK":
        if Shared._pi_check_running:
            _ctx["log_print"]("PI_CHECK: cancelling...")
            if Shared._pi_check_cancel:
                Shared._pi_check_cancel.set()
            Shared._pi_check_running = False
            Shared.pi_status = "Pi Ready"
            Shared.last_action_msg = "Pi check stopped"
            return "OK Pi check stopped"
        else:
            _ctx["log_print"]("PI_CHECK: starting Pi diagnostics")
            Shared.last_action_msg = "Pi self-check"
            Shared.pi_status = "Checking..."
            Shared._pi_check_running = True
            cancel = threading.Event()
            Shared._pi_check_cancel = cancel

            def _do_pi_check():
                try:
                    reporter = StatusReporter(
                        set_arm_status=lambda s: setattr(Shared, "arm_status", s),
                        set_pi_status=lambda s: setattr(Shared, "pi_status", s),
                        set_task_status=lambda s: setattr(Shared, "task_status", s),
                        log_fn=_ctx["log_print"],
                    )
                    result = pi_check(reporter, cancel_event=cancel)
                    if not cancel.is_set():
                        _ctx["log_print"](f"PI_CHECK result: {result}")
                except Exception as e:
                    if not cancel.is_set():
                        _ctx["log_print"](f"PI_CHECK error: {e}")
                        Shared.pi_status = "Error"
                finally:
                    Shared._pi_check_running = False
            threading.Thread(target=_do_pi_check, daemon=True).start()
            return "OK Pi check started"

    elif cmd == "ARM_POS_TEST":
        if Shared._arm_pos_running:
            _ctx["log_print"]("ARM_POS_TEST: stopping")
            Shared._arm_pos_running = False
            Shared.arm_status = "Arm Ready"
            Shared.last_action_msg = "Arm pos test stopped"
            return "OK Arm pos test stopped"
        else:
            _ctx["log_print"]("ARM_POS_TEST: starting position test")
            Shared.last_action_msg = "Arm pos test"
            Shared.arm_status = "Moving..."
            Shared._arm_pos_running = True
            return "OK Arm pos test started"

    elif cmd == "F":
        if not Shared.recognition_active:
            return "OK: Turn on recognition first"
        Shared.frozen = not Shared.frozen
        if Shared.frozen:
            Shared.last_action_msg = "FROZEN"
            _ctx["log_print"]("F: FREEZE - detection locked")
        else:
            Shared.frozen = False
            Shared._stability_counter = 0
            Shared._last_piece_centroids = []
            Shared.freeze_data = None
            Shared.last_action_msg = "UNFROZEN"
            _ctx["log_print"]("F: UNFREEZE - detection resumed")
        state_str = "FROZEN" if Shared.frozen else "READY"
        return f"OK Freeze -> {state_str}"

    elif cmd == "RESTART":
        _ctx["log_print"]("RESTART: restarting process...")
        Shared.last_action_msg = "Restarting..."
        import os, sys
        os.execv(sys.executable, [sys.executable] + sys.argv)
        return "OK Restarting"

    names = {"P": "Recover", "T": "Overlay toggled", "D": "Debug", "A": "Area box", "S": "Saved"}
    Shared.last_action_msg = names.get(cmd, cmd)
    return "OK: " + Shared.last_action_msg
