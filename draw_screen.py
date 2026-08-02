#!/usr/bin/env python3
"""Standalone TJC screen drawer — reads freeze.json, renders via tjc_display module."""

import json, os, sys, time

sys.path.insert(0, "/home/man/puzzle_robot_project")
from tjc_display import open_tjc, draw_state, TJC

WF = "/home/man/puzzle_app/freeze.json"


def draw_freeze_json(tjc: TJC):
    """Read freeze.json and draw on TJC screen."""
    try:
        with open(WF, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Cannot read {WF}: {e}")
        tjc.cls()
        return

    pieces = data.get("pieces", [])
    info = data.get("solve_info", {})
    info["pieces_count"] = len(pieces)
    info["assembly_order"] = data.get("assembly_order", [])

    plan_items = []
    for p in pieces:
        pid = p.get("id", "")
        place = p.get("place_mm", [0, 0])
        rot = p.get("rotate_deg", 0)
        if place and place != [0.0, 0.0]:
            plan_items.append({
                "piece_id": pid,
                "place_mm": place,
                "rotate_deg": rot,
                "target_polygon_mm": p.get("polygon_arm_mm"),
            })

    frozen = data.get("frozen", False)
    draw_state(tjc, pieces, plan_items, info, frozen=frozen)
    print(f"[DRAW] {len(pieces)} pieces, frozen={frozen}")


if __name__ == "__main__":
    once = "--once" in sys.argv

    tjc = open_tjc()
    if not tjc:
        print("Cannot open TJC serial port")
        sys.exit(1)

    print(f"TJC screen ready")

    if once:
        draw_freeze_json(tjc)
        print("Done!")
        tjc.close()
        sys.exit(0)

    # Watch mode
    print(f"Watching {WF}...")
    last_hash = None
    last_mtime = 0

    try:
        with open(WF, "r") as f:
            data = json.load(f)
        draw_freeze_json(tjc)
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
                last_mtime = mtime
                time.sleep(0.1)
                try:
                    with open(WF, "r") as f:
                        data = json.load(f)
                    h = json.dumps(data, sort_keys=True)
                    if h != last_hash:
                        last_hash = h
                        draw_freeze_json(tjc)
                except Exception as e:
                    print(f"Error: {e}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nExit")
        tjc.cls()
        tjc.close()
