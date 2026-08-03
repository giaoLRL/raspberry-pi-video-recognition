#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serial Protocol Module — Bidirectional (TX + RX)
================================================
Physical:  Pi 5  → GPIO 14 TX (Pin 8) + GPIO 15 RX (Pin 10) + GND (Pin 6)
Port:     /dev/ttyAMA0, 115200 8N1, 3.3V TTL

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
SERIAL_PORT = "/dev/ttyAMA0"       # Pi 5 UART0 (GPIO14/15)
SERIAL_BAUDRATE = 115200
SERIAL_TIMEOUT = 1.0               # readline timeout (seconds)
RECONNECT_DELAY = 2.0              # serial reconnect interval

# ---- shared state (injected by caller) ----
_listener_thread = None
_listener_running = False
_listener_stopped = threading.Event()  # signals listener has released port
_serial_lock = threading.Lock()        # guards port open/close
_unfreeze_callback = None              # callable, invoked on $DONE / $UNFREEZE
MONITOR_URL = "http://127.0.0.1:8081/log"  # serial_monitor web endpoint


# ================================================================
# Protocol formatter
# ================================================================


def _report_to_monitor(direction: str, text: str) -> None:
    """Post a log entry to the serial monitor web UI (port 8081).

    Silently ignores errors when the monitor is not running.
    """
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


def _match_done(text):
    """Check if text is a $DONE or $UNFREEZE command (exact prefix match)."""
    return text.startswith("$DONE") or text.startswith("$UNFREEZE")


def format_protocol(freeze_data: dict) -> str:
    """Convert freeze_data dict → serial protocol string (CRLF terminated).

    Validates piece data defensively: missing keys and malformed coordinates
    are skipped rather than crashing.
    """
    lines = []
    pieces = freeze_data.get("pieces", [])
    solve_info = freeze_data.get("solve_info", {})

    mode = solve_info.get("mode", "unknown")
    lines.append(f"$BEGIN,N={len(pieces)},MODE={mode}")

    # Build piece map — skip entries without an id
    piece_map = {}
    for p in pieces:
        pid = p.get("id") if isinstance(p, dict) else None
        if pid:
            piece_map[pid] = p

    pickup_order = freeze_data.get("pickup_order", [])

    for pid in pickup_order:
        p = piece_map.get(pid)
        if p is None:
            continue

        pick = p.get("pick_mm")
        if not pick or len(pick) < 2:
            continue  # skip pieces without valid pick coordinates
        px, py = float(pick[0]), float(pick[1])

        place = p.get("place_mm", [0.0, 0.0])
        if not place or len(place) < 2:
            place = [0.0, 0.0]
        tx, ty = float(place[0]), float(place[1])

        rot = float(p.get("rotate_deg", 0.0))
        lines.append(
            f"$PIECE,{pid},{px:.2f},{py:.2f},{tx:.2f},{ty:.2f},{rot:.2f}"
        )

    assembly = freeze_data.get("assembly_order", [])
    if assembly:
        lines.append(f"$ASSEMBLY,{','.join(str(x) for x in assembly)}")

    lines.append("$END")
    return "\r\n".join(lines) + "\r\n"


# ================================================================
# Send + wait-for-DONE  (called in a thread when freeze triggers)
# ================================================================


_ARM_DONE_TIMEOUT = 120  # seconds to wait for arm $DONE response


def send_and_wait_done(freeze_data: dict) -> bool:
    """Send protocol data over serial, then block until arm replies $DONE.

    Returns True if $DONE was received, False on timeout or error.
    Always calls the unfreeze callback so the system never stays frozen.
    Reports all TX and RX activity to the serial monitor web UI.
    """
    global _listener_running, _listener_thread

    try:
        import serial
    except ImportError:
        print("[SERIAL] pyserial not installed — SKIP")
        print("[SERIAL] Install: pip install pyserial")
        # Still unfreeze so the system can recover
        if _unfreeze_callback is not None:
            _unfreeze_callback()
        return False

    formatted = format_protocol(freeze_data)

    # --- console preview ---
    print("[SERIAL] ====== TX (to arm) ======")
    for line in formatted.strip().split("\r\n"):
        print(f"[SERIAL]  {line}")
    print("[SERIAL] ==========================")

    was_listening = _listener_running
    ser = None
    done_received = False

    try:
        # ── Stop background listener with proper handshake ──
        if was_listening:
            _listener_stopped.clear()
            _listener_running = False
            # Wait for listener to actually close its port (Event-based, not sleep)
            if not _listener_stopped.wait(timeout=2.0):
                print("[SERIAL] WARNING: listener did not stop in time")
            # Join the old thread so we don't accumulate zombie threads
            if _listener_thread is not None and _listener_thread.is_alive():
                _listener_thread.join(timeout=1.0)

        # ── Send protocol data ──
        with _serial_lock:
            ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT)
            try:
                ser.write(formatted.encode("ascii"))
                ser.flush()
            except Exception:
                # Ensure port is closed on partial-write failure
                if ser and ser.is_open:
                    ser.close()
                ser = None
                raise

        print(f"[SERIAL] TX done ({len(formatted)} bytes) — waiting for $DONE...")

        # Report each TX line to the serial monitor (outside lock)
        for line in formatted.strip().split("\r\n"):
            _report_to_monitor("TX", line)

        # ── Wait for $DONE response ──
        deadline = time.time() + _ARM_DONE_TIMEOUT
        consecutive_errors = 0
        rx_buf = b""
        while ser and time.time() < deadline:
            try:
                # Read available data in bulk, then split by \r\n or \r.
                # readline() fails when the arm sends bare \r line endings.
                waiting = ser.in_waiting
                if waiting:
                    rx_buf += ser.read(waiting)
                elif rx_buf:
                    pass  # data already buffered, keep processing
                else:
                    time.sleep(0.01)
                    continue

                # Split buffer into complete lines
                while True:
                    # Try \r\n first, then bare \r, then bare \n
                    for sep in (b"\r\n", b"\r", b"\n"):
                        idx = rx_buf.find(sep)
                        if idx >= 0:
                            line = rx_buf[:idx]
                            rx_buf = rx_buf[idx + len(sep):]
                            text = line.decode("ascii", errors="replace").strip()
                            if text:
                                _report_to_monitor("RX", text)
                                print(f"[SERIAL] RX ← arm: {text}")
                                if _match_done(text):
                                    print("[SERIAL] ✓ DONE received — unfreezing")
                                    done_received = True
                            break
                    else:
                        break  # no complete line found

                if done_received:
                    break
                consecutive_errors = 0

            except Exception:
                consecutive_errors += 1
                if consecutive_errors > 10:
                    print("[SERIAL] Too many read errors — aborting wait")
                    break
                time.sleep(0.05)

        if not done_received:
            print(f"[SERIAL] TIMEOUT waiting for $DONE ({_ARM_DONE_TIMEOUT}s)")

    except Exception as exc:
        print(f"[SERIAL] ERROR: {exc}")

    finally:
        # ── Always close the send port ──
        if ser is not None:
            try:
                if ser.is_open:
                    ser.close()
            except Exception:
                pass

        # ── Always unfreeze — the system MUST recover ──
        if _unfreeze_callback is not None:
            _unfreeze_callback()

        # ── Restart background listener if it was running before ──
        if was_listening and _unfreeze_callback is not None:
            _listener_running = True
            _listener_thread = threading.Thread(
                target=_listener_loop, daemon=True, name="serial-listener"
            )
            _listener_thread.start()

    return done_received


# ================================================================
# Background listener  (optional — for async commands like $STOP)
# ================================================================


def _listener_loop():
    """Background thread: continuously reads serial for incoming commands."""
    global _listener_running

    try:
        import serial
    except ImportError:
        _listener_stopped.set()
        return

    print("[SERIAL] listener thread started")
    while _listener_running:
        ser = None
        try:
            with _serial_lock:
                ser = serial.Serial(
                    SERIAL_PORT, SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT
                )

            while _listener_running:
                try:
                    # Bulk read + split by \r\n or \r (handles arm's bare CR).
                    waiting = ser.in_waiting
                    if waiting:
                        data = ser.read(waiting)
                    else:
                        time.sleep(0.02)
                        continue

                    for sep in (b"\r\n", b"\r", b"\n"):
                        parts = data.split(sep)
                        if len(parts) > 1:
                            data = parts.pop(-1)  # last fragment (incomplete line)
                            for part in parts:
                                line = part.decode("ascii", errors="replace").strip()
                                if not line:
                                    continue
                                print(f"[SERIAL] RX ← arm: {line}")
                                _report_to_monitor("RX", line)
                                if _match_done(line):
                                    print("[SERIAL] ✓ unfreeze command received")
                                    if _unfreeze_callback is not None:
                                        _unfreeze_callback()
                            break
                except Exception:
                    time.sleep(0.1)

        except Exception as exc:
            print(f"[SERIAL] listener error: {exc}")
        finally:
            if ser is not None:
                try:
                    if ser.is_open:
                        ser.close()
                except Exception:
                    pass

        if _listener_running:
            time.sleep(RECONNECT_DELAY)

    # Signal that we've released the port
    _listener_stopped.set()
    print("[SERIAL] listener thread stopped")


def start_listener(unfreeze_cb):
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
    """Stop the background listener thread and wait for it to exit."""
    global _listener_running, _listener_thread
    _listener_stopped.clear()
    _listener_running = False
    if _listener_thread is not None and _listener_thread.is_alive():
        _listener_thread.join(timeout=2.0)


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
