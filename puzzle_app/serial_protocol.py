#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serial Protocol Module — Bidirectional (TX + RX)
================================================
Physical:  Pi 5  → GPIO 14 TX (Pin 8) + GPIO 15 RX (Pin 10) + GND (Pin 6)
Port:     /dev/serial0 → /dev/ttyAMA10, 115200 8N1, 3.3V TTL

------ Pi → Arm (TX, send-on-freeze) ------

  $BEGIN,N=<count>,MODE=<solve_mode>\r\n
  $PIECE,<id>,<pick_x>,<pick_y>,<place_x>,<place_y>,<rotate_deg>\r\n
    … (N lines in pickup_order = distance from origin, nearest first)
  $ASSEMBLY,<id1>,<id2>,...\r\n
  $END\r\n

------ Arm → Pi (RX, continuous listener) ------

  $DONE\r\n       机械臂全部操作完成 → 解冻 + 等待下一轮识别

------ Flow ------

  freeze → send data → wait for $DONE → unfreeze → (auto-detect resumes)
"""

import json
import threading
import time
import urllib.request

# ---- config ----
SERIAL_PORT = "/dev/serial0"
SERIAL_BAUDRATE = 115200
SERIAL_TIMEOUT = 1.0          # readline 超时 (秒)
RECONNECT_DELAY = 2.0         # 串口重连间隔

# ---- shared state (injected by caller) ----
_listener_thread: threading.Thread | None = None
_listener_running = False
_serial_lock = threading.Lock()   # 防止同时读写
_unfreeze_callback = None          # callable, 接收到 $DONE 时调用
MONITOR_URL = "http://127.0.0.1:8081/log"   # serial_monitor web endpoint


# ================================================================
# Protocol formatter
# ================================================================


def _report_to_monitor(direction: str, text: str) -> None:
    """Post a log entry to the serial monitor web UI (port 8081).
    Silently ignores errors when the monitor is not running."""
    try:
        entry = {"dir": direction, "text": text}
        data = json.dumps(entry).encode("utf-8")
        req = urllib.request.Request(
            MONITOR_URL, data=data,
            headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=1)
    except Exception:
        pass  # monitor not running — no problem


def format_protocol(freeze_data: dict) -> str:
    """Convert freeze_data dict → serial protocol string (CRLF terminated)."""
    lines = []
    pieces = freeze_data.get("pieces", [])
    solve_info = freeze_data.get("solve_info", {})

    mode = solve_info.get("mode", "unknown")
    lines.append(f"$BEGIN,N={len(pieces)},MODE={mode}")

    pickup_order = freeze_data.get("pickup_order", [])
    piece_map = {p["id"]: p for p in pieces}

    for pid in pickup_order:
        p = piece_map.get(pid)
        if p is None:
            continue
        px, py = p.get("pick_mm", [0.0, 0.0])
        tx, ty = p.get("place_mm", [0.0, 0.0])
        rot = p.get("rotate_deg", 0.0)
        lines.append(
            f"$PIECE,{pid},{px:.2f},{py:.2f},{tx:.2f},{ty:.2f},{rot:.2f}"
        )

    assembly = freeze_data.get("assembly_order", [])
    if assembly:
        lines.append(f"$ASSEMBLY,{','.join(assembly)}")

    lines.append("$END")
    return "\r\n".join(lines) + "\r\n"


# ================================================================
# Send + wait-for-DONE  (called in a thread when freeze triggers)
# ================================================================

def send_and_wait_done(freeze_data: dict) -> bool:
    """Send protocol data over serial, then block until arm replies $DONE.

    Returns True on success, False on serial error.
    Reports all TX and RX activity to the serial monitor web UI.
    Calls the unfreeze_callback when $DONE is received.
    """
    global _listener_running, _listener_thread
    formatted = format_protocol(freeze_data)

    # --- console preview ---
    print("[SERIAL] ====== TX (to arm) ======")
    for line in formatted.strip().split("\r\n"):
        print(f"[SERIAL]  {line}")
    print("[SERIAL] ==========================")

    try:
        import serial
    except ImportError:
        print("[SERIAL] pyserial not installed — SKIP")
        print("[SERIAL] Install: pip install pyserial")
        return False

    was_listening = _listener_running

    try:
        # Pause background listener so we can take over the port
        if _listener_running:
            _listener_running = False
            time.sleep(0.6)  # give listener time to release port

        with _serial_lock:
            ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT)
            ser.write(formatted.encode("ascii"))
            ser.flush()
            print(f"[SERIAL] TX done ({len(formatted)} bytes) — waiting for $DONE...")

            # Report each TX line to the serial monitor
            for line in formatted.strip().split("\r\n"):
                _report_to_monitor("TX", line)

        # ---- Wait for $DONE response (2-minute timeout) ----
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                line = ser.readline()
                if line:
                    text = line.decode("ascii", errors="replace").strip()
                    if text:
                        _report_to_monitor("RX", text)
                        print(f"[SERIAL] RX ← arm: {text}")
                        if "$DONE" in text or "$UNFREEZE" in text:
                            print("[SERIAL] ✓ DONE received — unfreezing")
                            if _unfreeze_callback is not None:
                                _unfreeze_callback()
                            break
            except Exception:
                time.sleep(0.05)
        else:
            print("[SERIAL] TIMEOUT waiting for $DONE (120s)")

        ser.close()
        return True

    except Exception as exc:
        print(f"[SERIAL] ERROR: {exc}")
        return False

    finally:
        # Restart background listener if it was running before
        if was_listening and _unfreeze_callback is not None:
            _listener_running = True
            _listener_thread = threading.Thread(
                target=_listener_loop, daemon=True, name="serial-listener"
            )
            _listener_thread.start()


# ================================================================
# Background listener  (optional — for async commands like $STOP)
# ================================================================

def _listener_loop():
    """Background thread: continuously reads serial for incoming commands."""
    global _listener_running
    try:
        import serial
    except ImportError:
        return

    print("[SERIAL] listener thread started")
    while _listener_running:
        try:
            with _serial_lock:
                ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT)
            while _listener_running:
                try:
                    line = ser.readline().decode("ascii", errors="replace").strip()
                except Exception:
                    time.sleep(0.1)
                    continue

                if not line:
                    continue

                print(f"[SERIAL] RX ← arm: {line}")

                _report_to_monitor("RX", line)

                if line.startswith("$DONE") or line.startswith("$UNFREEZE"):
                    print("[SERIAL] ✓ unfreeze command received")
                    if _unfreeze_callback is not None:
                        _unfreeze_callback()

            ser.close()
        except Exception as exc:
            print(f"[SERIAL] listener error: {exc}")
            if _listener_running:
                time.sleep(RECONNECT_DELAY)


def start_listener(unfreeze_cb) -> threading.Thread | None:
    """Start background serial listener thread.

    Args:
        unfreeze_cb: callable, invoked when arm sends $DONE or $UNFREEZE.
    Returns the thread, or None if pyserial is not installed.
    """
    global _listener_thread, _listener_running, _unfreeze_callback

    try:
        import serial  # noqa: F401
    except ImportError:
        print("[SERIAL] pyserial not installed — listener not started")
        return None

    _unfreeze_callback = unfreeze_cb
    _listener_running = True
    _listener_thread = threading.Thread(
        target=_listener_loop, daemon=True, name="serial-listener"
    )
    _listener_thread.start()
    return _listener_thread


def stop_listener():
    """Stop the background listener thread."""
    global _listener_running
    _listener_running = False


# ================================================================
# Standalone test
# ================================================================

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            data = json.load(f)
    else:
        data = {
            "frozen": True,
            "pieces": [
                {
                    "id": "piece_3",
                    "pick_mm": [-21.25, 59.75],
                    "place_mm": [175.93, 103.71],
                    "rotate_deg": 35.98,
                    "area_mm2": 1466.8,
                    "vertices": 3,
                },
                {
                    "id": "piece_1",
                    "pick_mm": [38.25, 80.75],
                    "place_mm": [115.64, 123.17],
                    "rotate_deg": 62.58,
                    "area_mm2": 581.4,
                    "vertices": 3,
                },
                {
                    "id": "piece_2",
                    "pick_mm": [23.00, 152.50],
                    "place_mm": [144.19, 125.60],
                    "rotate_deg": -78.28,
                    "area_mm2": 864.8,
                    "vertices": 5,
                },
                {
                    "id": "piece_4",
                    "pick_mm": [-28.75, 156.50],
                    "place_mm": [126.79, 95.21],
                    "rotate_deg": -57.20,
                    "area_mm2": 2086.4,
                    "vertices": 4,
                },
            ],
            "pickup_order": ["piece_3", "piece_1", "piece_2", "piece_4"],
            "assembly_order": ["piece_1", "piece_2", "piece_3", "piece_4"],
            "solve_info": {"mode": "unknown-pattern", "fill_ratio": 0.9679},
        }

    formatted = format_protocol(data)
    print("=== Protocol Preview ===")
    print(formatted, end="")
    print("========================")