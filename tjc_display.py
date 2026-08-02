#!/usr/bin/env python3
"""TJC Serial Screen Display — library module for real-time puzzle visualization.

Used by both pi_stream_puzzle_v2.py (integrated, real-time) and
draw_screen.py (standalone, reads freeze.json).
"""

import math
import time
from typing import Optional

try:
    import serial
except ImportError:
    serial = None

END = b"\xff\xff\xff"

# ── Screen constants (TJC 480x272) ──────────────────────────
SW, SH = 480, 272
MG = 10
UW = SW - MG * 2  # 460
UH = SH - MG * 2  # 252

# ── arm-mm coordinate bounds ────────────────────────────────
# arm-mm: X[0,210] right→left, Y[-75,222] top→bottom
ARM_X0, ARM_X1 = 0.0, 210.0
ARM_Y0, ARM_Y1 = -75.0, 222.0
ARM_W = ARM_X1 - ARM_X0  # 210 mm
ARM_H = ARM_Y1 - ARM_Y0  # 297 mm

# Uniform scale: fit A4 (portrait) into landscape screen, no rotation
SCALE = min(UW / ARM_W, UH / ARM_H) * 0.80
A4_PX_W = ARM_W * SCALE
A4_PX_H = ARM_H * SCALE
A4_OX = MG + (UW - A4_PX_W) / 2  # center X
A4_OY = MG + (UH - A4_PX_H) / 2  # center Y

# ── Colors (RGB565) ──────────────────────────────────────────
W = 65535; B = 0; GY = 21140; LGY = 42280; DGY = 31680; R = 63488
G = 2016; BL = 31; Y = 65504; C = 2047; M = 63519

PC = [R, G, BL, C, M, Y, 42280, 21140]  # piece colors


def arm_to_screen(ax: float, ay: float) -> tuple:
    """arm-mm → screen pixel (uniform scale, no rotation)"""
    # arm X: right→left → screen left→right
    sx = int(A4_OX + (ARM_X1 - ax) * SCALE)
    # arm Y: top→bottom → screen top→bottom
    sy = int(A4_OY + (ay - ARM_Y0) * SCALE)
    return max(0, min(SW - 1, sx)), max(0, min(SH - 1, sy))


# ── TJC driver ───────────────────────────────────────────────

class TJC:
    """TJC USART HMI serial display driver."""

    def __init__(self, ser):
        self.s = ser
        self._batch = []
        self._batch_mode = False

    def _send(self, cmd: str):
        if self._batch_mode:
            self._batch.append(cmd)
        else:
            self.s.reset_input_buffer()
            self.s.write(cmd.encode("gbk") + END)
            self.s.flush()
            time.sleep(0.005)

    def begin_batch(self):
        self._batch_mode = True
        self._batch = []

    def flush_batch(self):
        self._batch_mode = False
        for cmd in self._batch:
            self.s.reset_input_buffer()
            self.s.write(cmd.encode("gbk") + END)
            self.s.flush()
            time.sleep(0.004)
        self._batch = []

    def cls(self, c=W): self._send(f"cls {c}")

    def fill(self, x, y, w, h, c):
        if w > 0 and h > 0:
            self._send(f"fill {x},{y},{w},{h},{c}")

    def line(self, x1, y1, x2, y2, c):
        self._send(f"line {x1},{y1},{x2},{y2},{c}")

    def cir(self, x, y, r, c):
        self._send(f"cir {x},{y},{r},{c}")

    def txt(self, x, y, t, c=B, bg=W, w=60, h=16):
        t = str(t).replace('"', "'")
        self._send(f'xstr {x},{y},{w},{h},0,{bg},{c},0,0,"{t}"')

    def polyline(self, pts, color):
        """Draw polygon outline from screen (x,y) tuples."""
        if len(pts) < 2:
            return
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            self.line(x1, y1, x2, y2, color)

    def filled_poly(self, pts, color_fill, color_edge):
        """Fill polygon by scanning horizontal lines (simple scanline fill).

        Only appropriate for small polygons on sparse displays (few pixels).
        For larger fills, use TJC's built-in fill rect as approximation.
        """
        if len(pts) < 3:
            return
        # Outline
        self.polyline(pts, color_edge)
        # Approximate fill with bounding-box fill + outline redraw trick:
        # just draw the outline clearly

    def bkcmd(self, lv=0):
        self._send(f"bkcmd={lv}")

    def page(self, n=0):
        self._send(f"page {n}")

    def close(self):
        if self.s and self.s.is_open:
            self.s.close()


# ── Drawing functions ────────────────────────────────────────

def _pts_arm_to_screen(poly_arm):
    """Convert polygon from arm-mm coords to screen pixel list."""
    return [arm_to_screen(v[0], v[1]) for v in poly_arm]


def _centroid(poly):
    if not poly:
        return 0, 0
    return sum(v[0] for v in poly) / len(poly), sum(v[1] for v in poly) / len(poly)


def _rotate_poly(poly, angle_deg):
    """Rotate polygon around centroid by angle_deg CW."""
    if not poly or len(poly) < 2 or abs(angle_deg) < 0.01:
        return [list(v) for v in poly]
    cx, cy = _centroid(poly)
    rad = math.radians(-angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    return [[cx + (v[0] - cx) * cos_a - (v[1] - cy) * sin_a,
             cy + (v[0] - cx) * sin_a + (v[1] - cy) * cos_a] for v in poly]


def _translate_poly(poly, dx, dy):
    return [[v[0] + dx, v[1] + dy] for v in poly]


def draw_state(tjc: TJC,
               pieces_arm: list = None,
               plan_items: list = None,
               info: dict = None,
               frozen: bool = False,
               recognition: bool = True):
    """Draw puzzle state on TJC screen — minimal commands for 9600 baud."""

    pieces_arm = pieces_arm or []
    plan_items = plan_items or []
    info = info or {}

    tjc.begin_batch()

    # ── Background: just clear, no heavy fill ──
    tjc.cls(W)

    # ── A4 outline only (no fill) ──
    a4 = [arm_to_screen(210, -75), arm_to_screen(0, -75),
          arm_to_screen(0, 222), arm_to_screen(210, 222)]
    tjc.polyline(a4, GY)

    # ── Origin dot ──
    o = arm_to_screen(0, 0)
    tjc.cir(o[0], o[1], 3, B)

    # ── Build pick_map ──
    pick_map = {}
    for p in pieces_arm:
        if p.get("pick_mm"):
            pick_map[p.get("id", "")] = p["pick_mm"]

    # ── Piece outlines (pick position) ──
    for i, p in enumerate(pieces_arm):
        color = PC[i % len(PC)]
        poly = p.get("polygon_arm_mm")
        pick = p.get("pick_mm")
        if poly:
            tjc.polyline(_pts_arm_to_screen(poly), color)
        if pick:
            pt = arm_to_screen(pick[0], pick[1])
            tjc.cir(pt[0], pt[1], 3, color)

    # ── Plan: target polygons (already at final position) + place dots + connecting lines ──
    for i, item in enumerate(plan_items):
        color = PC[i % len(PC)]
        pid = item.get("piece_id", "")
        place = item.get("place_mm")
        target = item.get("target_polygon_mm")
        pick = pick_map.get(pid)

        # Target polygon: already rotated+placed by solver, draw as-is
        if target:
            tjc.polyline(_pts_arm_to_screen(target), color)

        # Place dot
        if place and place != [0.0, 0.0]:
            pt = arm_to_screen(place[0], place[1])
            tjc.cir(pt[0], pt[1], 3, color)

        # Pick → Place connecting line
        if pick and place and place != [0.0, 0.0]:
            p1 = arm_to_screen(pick[0], pick[1])
            p2 = arm_to_screen(place[0], place[1])
            tjc.line(p1[0], p1[1], p2[0], p2[1], color)

    # ── Header (minimal) ──
    n = info.get("pieces_count", len(pieces_arm))
    mode = info.get("mode", "--")
    tjc.txt(2, 2, f"{mode} Pcs:{n}", B, W, 140, 14)
    if frozen:
        tjc.txt(SW - 52, 2, "FROZEN", R, W, 50, 14)

    tjc.flush_batch()


# ── Recompute layout on resolution change ──

def recalc_layout():
    global SCALE, A4_PX_W, A4_PX_H, A4_OX, A4_OY
    SCALE = min(UW / ARM_W, UH / ARM_H) * 0.80
    A4_PX_W = ARM_W * SCALE
    A4_PX_H = ARM_H * SCALE
    A4_OX = MG + (UW - A4_PX_W) / 2
    A4_OY = MG + (UH - A4_PX_H) / 2


# ── Open connection ──

def open_tjc(port: str = "/dev/ttyAMA2", baud: int = 9600) -> Optional[TJC]:
    """Open TJC serial connection. Returns None if unavailable."""
    if serial is None:
        return None
    try:
        ser = serial.Serial(port, baud, timeout=0.5, write_timeout=0.5)
        tjc = TJC(ser)
        tjc.bkcmd(0)
        tjc.page(0)
        time.sleep(0.1)
        return tjc
    except Exception:
        return None
