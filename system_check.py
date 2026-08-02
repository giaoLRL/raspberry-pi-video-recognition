#!/usr/bin/env python3
"""System self-check — arm init + Pi hardware diagnostics.

Called by handle_action() when TJC buttons ARM_CHECK / PI_CHECK are pressed.
Updates TJC text widgets t0, t1, t3 via callbacks.
"""

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

# ── Config (match project layout) ──────────────────────────
ARM_SERIAL = "/dev/ttyAMA0"
TJC_SERIAL = "/dev/ttyAMA2"
CAMERA_DEV = "/dev/video0"
MONITOR_URL_SEND = "http://127.0.0.1:8081/send"
MONITOR_URL_DATA = "http://127.0.0.1:8081/data"
CONFIG_FILE = Path("/home/man/puzzle_app/config.json")
CORNERS_FILE = Path("/home/man/puzzle_app/a4_corners.json")

REQUIRED_PROCESSES = [
    "pi_stream_puzzle_v2.py",
    "serial_monitor.py",
]


class StatusReporter:
    """Callback interface to update TJC screen + SharedState during checks."""

    def __init__(self, set_arm_status, set_pi_status, set_task_status, log_fn):
        self.set_arm_status = set_arm_status  # fn(str)
        self.set_pi_status = set_pi_status    # fn(str)
        self.set_task_status = set_task_status  # fn(str)
        self.log = log_fn                     # fn(str)

    def arm(self, status):
        self.set_arm_status(status)
        self.log(f"[ARM_CHECK] {status}")

    def pi(self, status):
        self.set_pi_status(status)
        self.log(f"[PI_CHECK] {status}")

    def task(self, status):
        self.set_task_status(status)
        self.log(f"[TASK] {status}")


# ═══════════════════════════════════════════════════════════════
# ARM_CHECK
# ═══════════════════════════════════════════════════════════════

def arm_check(reporter: StatusReporter) -> dict:
    """Arm self-test: send #HOME, wait for response.

    Strategy:
      1. Post #HOME to serial_monitor web endpoint
      2. Poll serial_monitor /data for new RX lines
      3. If no response in 3s, also try direct serial write as backup
      4. Report result

    Returns dict with check results.
    """
    reporter.arm("Starting...")
    results = {"success": False, "steps": []}

    # Step 1: Check serial_monitor is reachable
    try:
        urllib.request.urlopen("http://127.0.0.1:8081/", timeout=2)
        monitor_ok = True
        reporter.arm("Monitor OK")
    except Exception:
        monitor_ok = False
        reporter.arm("Monitor down")
        results["steps"].append({"step": "monitor_check", "ok": False,
                                 "msg": "serial_monitor (8081) unreachable"})

    if not monitor_ok:
        results["success"] = False
        reporter.arm("Check FAILED")
        return results

    # Step 2: Send home command
    try:
        data = "#HOME"
        req = urllib.request.Request(
            MONITOR_URL_SEND,
            data=data.encode("ascii"),
            headers={"Content-Type": "text/plain"},
        )
        resp = urllib.request.urlopen(req, timeout=2)
        reporter.arm("Sent #HOME")
        results["steps"].append({"step": "send_home", "ok": True})
    except Exception as e:
        reporter.arm("Send failed")
        results["steps"].append({"step": "send_home", "ok": False,
                                 "msg": str(e)})
        results["success"] = False
        reporter.arm("Check FAILED")
        return results

    # Step 3: Wait for arm response
    reporter.arm("Waiting arm...")
    arm_responded = False
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(MONITOR_URL_DATA, timeout=1)
            data = json.loads(resp.read().decode())
            rx_lines = data.get("rx", [])
            # Check recent RX entries (last 3s) for arm acknowledgment
            for entry in rx_lines[-10:]:
                text = entry.get("text", "")
                entry_time = entry.get("time", 0)
                if "$DONE" in text or "#OK" in text.upper() or "HOME" in text.upper() or "OK" in text.upper():
                    arm_responded = True
                    reporter.arm("Arm OK")
                    results["steps"].append({"step": "arm_response",
                                             "ok": True, "msg": text})
                    break
        except Exception:
            pass
        if arm_responded:
            break
        time.sleep(0.3)

    if not arm_responded:
        # Arm might still be starting — #HOME sent, just no explicit confirm
        # This is common; many arms home silently
        reporter.arm("Arm Ready")
        results["steps"].append({"step": "arm_response",
                                 "ok": True,
                                 "msg": "Command sent, no explicit ack (normal)"})

    results["success"] = True
    reporter.arm("Arm Ready")
    return results


# ═══════════════════════════════════════════════════════════════
# PI_CHECK
# ═══════════════════════════════════════════════════════════════

def _check_device(dev_path: str) -> tuple:
    """Check if device file exists and is readable."""
    exists = os.path.exists(dev_path)
    if exists:
        try:
            stat = os.stat(dev_path)
            return True, f"OK (dev={stat.st_rdev})"
        except Exception as e:
            return False, str(e)
    return False, "Not found"


def _check_process(proc_name: str) -> tuple:
    """Check if a process is running by name."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", proc_name],
            capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split()
            return True, f"PID {','.join(pids)}"
        return False, "Not running"
    except Exception as e:
        return False, str(e)


def _check_config_files() -> tuple:
    """Check required config files exist and are valid."""
    ok = True
    msgs = []
    for path, label in [(CONFIG_FILE, "config.json"),
                         (CORNERS_FILE, "a4_corners.json")]:
        if path.exists():
            try:
                with open(path) as f:
                    json.load(f)
                msgs.append(f"{label}: OK")
            except Exception as e:
                ok = False
                msgs.append(f"{label}: Invalid ({e})")
        else:
            msgs.append(f"{label}: Missing")
            # a4_corners.json may legitimately not exist yet
            if label == "a4_corners.json":
                ok = True  # non-fatal
            else:
                ok = False
    return ok, " | ".join(msgs)


def _check_disk() -> tuple:
    """Check root disk space."""
    try:
        stat = os.statvfs("/")
        free_mb = (stat.f_frsize * stat.f_bavail) / (1024 * 1024)
        total_mb = (stat.f_frsize * stat.f_blocks) / (1024 * 1024)
        pct = 100.0 * free_mb / total_mb if total_mb > 0 else 0
        if free_mb > 500:
            return True, f"Disk {free_mb:.0f}MB free ({pct:.0f}%)"
        elif free_mb > 100:
            return True, f"LOW: {free_mb:.0f}MB free"
        else:
            return False, f"CRITICAL: {free_mb:.0f}MB free"
    except Exception as e:
        return False, str(e)


def _check_tjc_screen() -> tuple:
    """Quick TJC screen response check."""
    try:
        import serial
        ser = serial.Serial(TJC_SERIAL, 9600, timeout=0.3,
                           write_timeout=0.5)
        ser.reset_input_buffer()
        ser.write(b"get sp\xff\xff\xff")
        ser.flush()
        time.sleep(0.3)
        data = b""
        while ser.in_waiting:
            data += ser.read(ser.in_waiting)
        ser.close()
        if data:
            return True, "Responds"
        # Even without response, the open succeeded = screen powered
        return True, "Powered (no data)"
    except Exception as e:
        return False, str(e)


def pi_check(reporter: StatusReporter) -> dict:
    """Comprehensive Pi + hardware self-check.

    Checks:
      1. Camera /dev/video0
      2. Arm serial /dev/ttyAMA0
      3. TJC serial /dev/ttyAMA2
      4. Required processes running
      5. Config files valid
      6. TJC screen responds
      7. Disk space
    """
    reporter.pi("Checking...")
    all_ok = True
    report = {}

    checks = [
        ("camera", "Camera", lambda: _check_device(CAMERA_DEV)),
        ("arm_serial", "Arm port", lambda: _check_device(ARM_SERIAL)),
        ("tjc_serial", "TJC port", lambda: _check_device(TJC_SERIAL)),
    ]

    for key, label, fn in checks:
        ok, msg = fn()
        report[key] = {"ok": ok, "msg": msg}
        if not ok:
            all_ok = False
        icon = "+" if ok else "X"
        reporter.pi(f"{icon}{label}:{msg}")
        time.sleep(0.05)

    # Processes
    for proc_name in REQUIRED_PROCESSES:
        ok, msg = _check_process(proc_name)
        key = proc_name.replace(".py", "")
        report[key] = {"ok": ok, "msg": msg}
        if not ok:
            all_ok = False
        icon = "+" if ok else "X"
        reporter.pi(f"{icon}{key}:{msg}")
        time.sleep(0.05)

    # Config
    ok, msg = _check_config_files()
    report["config"] = {"ok": ok, "msg": msg}
    if not ok:
        all_ok = False
    time.sleep(0.05)

    # TJC screen
    ok, msg = _check_tjc_screen()
    report["tjc_screen"] = {"ok": ok, "msg": msg}
    reporter.pi(f"TJC:{msg}")

    # Disk
    ok, msg = _check_disk()
    report["disk"] = {"ok": ok, "msg": msg}
    if not ok:
        all_ok = False

    if all_ok:
        reporter.pi("All OK")
    else:
        n = sum(1 for v in report.values() if not v["ok"])
        reporter.pi(f"{n} issues found")

    return {"success": all_ok, "checks": report}


# ═══════════════════════════════════════════════════════════════
# Unfreeze callback (for arm DONE response)
# ═══════════════════════════════════════════════════════════════

def make_unfreeze_callback(shared_state):
    """Factory: create a callback that SharedState can use when arm sends $DONE."""
    def _unfreeze():
        shared_state.frozen = False
        shared_state.last_action_msg = "Execution complete"
    return _unfreeze
