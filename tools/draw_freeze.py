#!/usr/bin/env python3
"""Standalone TJC screen drawer — reads freeze.json, renders via tjc_display module."""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tjc_display import open_tjc, draw_state


WF = str(Path(__file__).resolve().parent.parent / "puzzle_app" / "freeze.json")


def draw_freeze(tjc):
    """Read freeze.json and draw on TJC screen.

    Returns a success boolean — True if freeze.json was loaded and drawn,
    False if the file was missing or malformed.
    """
    try:
        with open(WF, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Cannot read {WF}: {e}")
        return False

    try:
        pieces = data.get("pieces", [])
        # Build info without mutating the original dict
        info = dict(data.get("solve_info", {}))
        info["pieces_count"] = len(pieces)
        info["assembly_order"] = data.get("assembly_order", [])

        plan_items = []
        for p in pieces:
            if not isinstance(p, dict):
                continue
            pid = p.get("id", "")
            place = p.get("place_mm")
            rot = p.get("rotate_deg", 0)
            if place and not (abs(place[0]) < 0.01 and abs(place[1]) < 0.01):
                plan_items.append({
                    "piece_id": pid,
                    "place_mm": place,
                    "rotate_deg": rot,
                    "target_polygon_mm": p.get("polygon_arm_mm"),
                })

        frozen = data.get("frozen", False)
        draw_state(tjc, pieces, plan_items, info, frozen=frozen)
        print(f"[DRAW] {len(pieces)} pieces, frozen={frozen}")
        return True
    except Exception as e:
        print(f"[DRAW] Error rendering: {e}")
        return False


if __name__ == "__main__":
    once = "--once" in sys.argv

    tjc = open_tjc()
    if not tjc:
        print("Cannot open TJC serial port")
        sys.exit(1)

    print("TJC screen ready")

    if once:
        draw_freeze(tjc)
        print("Done!")
        tjc.close()
        sys.exit(0)

    # Watch mode
    print(f"Watching {WF}...")
    last_hash = ""
    last_mtime = 0

    # Initial draw
    try:
        if os.path.exists(WF):
            with open(WF, "r") as f:
                data = json.load(f)
            draw_freeze(tjc)
            last_hash = json.dumps(data, sort_keys=True)
    except (FileNotFoundError, json.JSONDecodeError):
        tjc.cls()

    try:
        while True:
            try:
                mtime = os.path.getmtime(WF)
            except OSError:
                time.sleep(0.5)
                continue

            if mtime != last_mtime:
                # Record mtime immediately so transient errors don't prevent retry
                last_mtime = mtime
                time.sleep(0.1)

                try:
                    with open(WF, "r") as f:
                        data = json.load(f)
                    h = json.dumps(data, sort_keys=True)
                    if h != last_hash:
                        last_hash = h
                        draw_freeze(tjc)
                except json.JSONDecodeError:
                    # File may be mid-write — keep old hash so we retry on next
                    # mtime change, but don't crash.
                    print(f"[DRAW] JSON decode error (file may be mid-write), will retry")
                except Exception as e:
                    print(f"[DRAW] Error: {e}")

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nExit")
        tjc.cls()
        tjc.close()
