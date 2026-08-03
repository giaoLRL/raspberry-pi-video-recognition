#!/usr/bin/env python3
"""TJC USART HMI Display — MCU-control pattern.

Architecture (standard USART HMI):
  Screen: HMI project with components (buttons b0-b5, text widgets t0,t1,t3)
  MCU:    Python handles ALL logic — reads button prints, updates widgets, draws A4

Commands end with 0xFF 0xFF 0xFF (END marker).
"""

import math
import time
import numpy as np
from typing import Optional, List

try:
    import serial
except ImportError:
    serial = None

END = b"\xff\xff\xff"

# Screen
SW, SH = 480, 272

# Layout (matches TJC HMI project)
STATUS_H = 18       # top: t0, t1, t3 text widgets
BTN_ROW_H = 19      # each button row height
BTN_ROW1_Y = 234    # b0, b1, b2 (mode buttons)
BTN_ROW2_Y = 253    # b3, b4, b5 (system buttons)
UI_BOT = BTN_ROW_H * 2   # 38
UI_TOP = STATUS_H         # 18

MG = 5
UW = SW - MG * 2                     # 470
UH = SH - MG * 2 - UI_TOP - UI_BOT   # remaining for A4

# arm-mm bounds
ARM_X0, ARM_X1 = 0.0, 210.0           # X: right to left
ARM_Y0, ARM_Y1 = -75.0, 222.0         # Y: top to bottom
ARM_W = ARM_X1 - ARM_X0               # 210 mm
ARM_H = ARM_Y1 - ARM_Y0               # 297 mm

# Axes swapped: arm Y (297 mm, long side) -> screen X (horizontal),
# arm X (210 mm, short side) -> screen Y (vertical).
# SCALE uses the effective dimensions after swap.
SCALE = min(UW / ARM_H, UH / ARM_W) * 0.70
A4_PX_W = ARM_H * SCALE                  # 297 mm -> screen X
A4_PX_H = ARM_W * SCALE                  # 210 mm -> screen Y
A4_OX = MG + (UW - A4_PX_W) / 2
A4_OY = MG + UI_TOP + (UH - A4_PX_H) / 2

# Colors (RGB565)
W = 65535; B = 0; GY = 21140; DGY = 31680; R = 63488
G = 2016; BL = 31; Y_C = 65504; C = 2047; M = 63519
PC = [R, G, BL, C, M, Y_C, 42280, 21140]


def arm_to_screen(ax, ay):
    """arm-mm to screen pixel (axes swapped, uniform scale).

    Arm Y (long side, 297 mm) -> screen X (horizontal).
    Arm X (short side, 210 mm) -> screen Y (vertical)."""
    sx = int(A4_OX + (ay - ARM_Y0) * SCALE)
    sy = int(A4_OY + (ARM_X1 - ax) * SCALE)
    return max(0, min(SW - 1, sx)), max(0, min(SH - 1, sy))


class TJC:
    """TJC USART HMI serial driver.

    Pattern:
      - Screen components (buttons, text widgets) defined in HMI project
      - Python sends txt= commands to update widgets
      - Python reads prints messages from button presses
      - All logic stays in Python (MCU)
    """

    def __init__(self, ser, bkcmd=0):
        self.s = ser
        self.bkcmd = bkcmd
        self._msg_buf = b""
        self._batch = []
        self._batch_mode = False

    def _send(self, cmd):
        """Send one command + END marker. No per-cmd sleep (was 4ms bottleneck)."""
        self.s.write(cmd.encode("gbk") + END)
        self.s.flush()

    def _cmd(self, cmd):
        """Send, respecting batch mode."""
        if self._batch_mode:
            self._batch.append(cmd)
        else:
            self._send(cmd)

    # Widget control (standard USART HMI)

    def set_text(self, widget, text):
        """Update text widget: t0.txt="hello" + END"""
        text = str(text).replace('"', "'").replace("\n", " ")
        self._cmd(widget + '.txt="' + text + '"')

    def set_visible(self, widget, visible):
        """Show/hide widget."""
        v = "1" if visible else "0"
        self._cmd(widget + ".visible=" + v)

    # Message reading (button presses)

    def poll_messages(self):
        """Read prints output from TJC button presses.

        Uses bulk read(self.s.in_waiting) instead of byte-by-byte read(1)
        for much faster polling (was ~4ms/byte, now ~0.5ms total).

        Returns list of decoded message strings.
        """
        messages = []
        try:
            available = self.s.in_waiting
            if available > 0:
                self._msg_buf += self.s.read(available)

                while True:
                    idx = self._msg_buf.find(END)
                    if idx < 0:
                        if len(self._msg_buf) > 128:
                            self._msg_buf = b""
                        break

                    frame = self._msg_buf[:idx]
                    self._msg_buf = self._msg_buf[idx + 3:]

                    if frame:
                        cleaned = bytes(c for c in frame if c != 0x01)
                        try:
                            msg = cleaned.decode("gbk").strip()
                        except Exception:
                            msg = cleaned.decode("ascii", errors="replace").strip()
                        if msg:
                            messages.append(msg)
        except Exception:
            self._msg_buf = b""
        return messages

    # Drawing commands (for A4 + pieces area)

    def begin_batch(self):
        self._batch_mode = True
        self._batch = []

    def flush_batch(self):
        self._batch_mode = False
        for i, cmd in enumerate(self._batch):
            self.s.write(cmd.encode("gbk") + END)
            if i % 8 == 7:
                self.s.flush()
        if self._batch:
            self.s.flush()
        self._batch = []

    def cls(self, c=W):
        self._cmd("cls " + str(c))

    def fill(self, x, y, w, h, c):
        if w > 0 and h > 0:
            self._cmd("fill {},{},{},{},".format(x, y, w, h) + str(c))

    def line(self, x1, y1, x2, y2, c):
        self._cmd("line {},{},{},{},".format(x1, y1, x2, y2) + str(c))

    def cir(self, x, y, r, c):
        self._cmd("cir {},{},{},".format(x, y, r) + str(c))

    def polyline(self, pts, color):
        if len(pts) < 2:
            return
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            self.line(x1, y1, x2, y2, color)

    def xstr(self, x, y, text, fg, bg, w=60, h=16):
        """Draw text string."""
        text = str(text).replace('"', "'")
        self._cmd('xstr {},{},{},{},0,{},{},0,0,"{}"'.format(x, y, w, h, bg, fg, text))

    def page(self, n=0):
        self._cmd("page " + str(n))

    def close(self):
        if self.s and self.s.is_open:
            self.s.close()


def _pts_arm_to_screen(poly_arm):
    return [arm_to_screen(v[0], v[1]) for v in poly_arm]


def draw_state(tjc,
               pieces_arm=None,
               plan_items=None,
               info=None,
               frozen=False,
               recognition=True,
               arm_status="",
               pi_status="",
               task_status="",
               project_loaded=True,
               a4_rotation_deg=0.0,
               a4_corners_arm=None,
               cam_frame_arm=None):
    """Draw puzzle state on TJC screen.

    project_loaded=True (HMI project on screen):
      Updates text widgets t0,t1,t3 + draws A4 outline + pieces only.
      Drawing area cleared via fill() to prevent frame overlap.

    project_loaded=False (bare screen):
      Falls back to drawing status bar + visual buttons manually.

    a4_rotation_deg: A4 rotation in degrees (positive = clockwise in arm space).
    cam_frame_arm: optional list of [x,y] arm-mm points for camera FOV (blue rect).
    """
    pieces_arm = pieces_arm or []
    plan_items = plan_items or []
    info = info or {}

    # 1. Update text widgets (HMI project components)
    if project_loaded:
        if arm_status:
            tjc.set_text("t0", arm_status)
        if pi_status:
            tjc.set_text("t1", pi_status)
        if task_status:
            tjc.set_text("t3", task_status)

    # 2. Draw A4 + pieces
    tjc.begin_batch()

    # ============================================================
    # Step 1: COMPUTE all screen coordinates first
    # ============================================================
    a4_raw = [[210.0, -75.0], [0.0, -75.0], [0.0, 222.0], [210.0, 222.0]]
    if abs(a4_rotation_deg) > 0.05:
        cx, cy = 105.0, 73.5
        rad = math.radians(a4_rotation_deg)
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        a4_rot = []
        for ax, ay in a4_raw:
            rx = cx + (ax - cx) * cos_r - (ay - cy) * sin_r
            ry = cy + (ax - cx) * sin_r + (ay - cy) * cos_r
            a4_rot.append([rx, ry])
        a4 = [arm_to_screen(ax, ay) for ax, ay in a4_rot]
    else:
        a4 = [arm_to_screen(ax, ay) for ax, ay in a4_raw]

    cam_pts = None
    if cam_frame_arm and len(cam_frame_arm) >= 4:
        cam_pts = [arm_to_screen(ax, ay) for ax, ay in cam_frame_arm[:4]]

    # ============================================================
    # Step 2: FILL — clear content bounding box with generous padding
    # ============================================================
    if project_loaded:
        all_pts = list(a4)
        if cam_pts:
            all_pts = all_pts + cam_pts
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        pad = 20
        fx0 = max(0, int(min(xs)) - pad)
        fy0 = max(UI_TOP, int(min(ys)) - pad)
        fx1 = min(SW, int(max(xs)) + pad)
        fy1 = min(BTN_ROW1_Y - 4, int(max(ys)) + pad)
        if fx1 > fx0 and fy1 > fy0:
            tjc.fill(fx0, fy0, fx1 - fx0, fy1 - fy0, W)

    if not project_loaded:
        hdr_bg = 1057
        hdr_fg = 2016
        tjc.cls(W)
        tjc.fill(0, 0, SW, STATUS_H, hdr_bg)
        n = info.get("pieces_count", 0)
        mode = info.get("mode", "--")
        status = "Mode:{} Pcs:{}".format(mode, n)
        if frozen:
            status += " [FROZEN]"
        rec = "R:ON" if recognition else "R:OFF"
        tjc.xstr(4, 1, status + " | " + rec, hdr_fg, hdr_bg, SW - 8, 16)

        tjc.fill(0, BTN_ROW1_Y, SW, UI_BOT, hdr_bg)
        btn_defs = [
            ("1", 5, 75), ("2", 85, 75), ("3", 165, 75),
            ("Arm", 250, 73), ("Pi", 328, 73), ("Pos", 406, 69),
        ]
        for label, bx, bw in btn_defs:
            tjc.fill(bx, BTN_ROW1_Y, bw - 4, BTN_ROW_H, 25585)
            tjc.xstr(bx + 4, BTN_ROW1_Y + 2, label, W, 25585, bw - 8, BTN_ROW_H - 2)

    # ============================================================
    # Step 3: DRAW
    # ============================================================
    # A4 outline (grey)
    tjc.polyline(a4, GY)

    # Camera frame (blue, axis-aligned)
    if cam_pts:
        tjc.polyline(cam_pts, BL)

    # Origin
    o = arm_to_screen(0, 0)
    tjc.cir(o[0], o[1], 3, B)

    # Pick map
    pick_map = {}
    for p in pieces_arm:
        if p.get("pick_mm"):
            pick_map[p.get("id", "")] = p["pick_mm"]

    # Piece outlines at pick position
    for i, p in enumerate(pieces_arm):
        color = PC[i % len(PC)]
        poly = p.get("polygon_arm_mm")
        pick = p.get("pick_mm")
        if poly:
            tjc.polyline(_pts_arm_to_screen(poly), color)
        if pick:
            pt = arm_to_screen(pick[0], pick[1])
            tjc.cir(pt[0], pt[1], 3, color)

    # Plan: targets + place dots + lines
    for i, item in enumerate(plan_items):
        color = PC[i % len(PC)]
        place = item.get("place_mm")
        target = item.get("target_polygon_mm")
        pick = pick_map.get(item.get("piece_id", ""))

        if target:
            tjc.polyline(_pts_arm_to_screen(target), color)
        if place and place != [0.0, 0.0]:
            pt = arm_to_screen(place[0], place[1])
            tjc.cir(pt[0], pt[1], 3, color)
        if pick and place and place != [0.0, 0.0]:
            p1 = arm_to_screen(pick[0], pick[1])
            p2 = arm_to_screen(place[0], place[1])
            tjc.line(p1[0], p1[1], p2[0], p2[1], color)

    tjc.flush_batch()


def recalc_layout():
    global SCALE, A4_PX_W, A4_PX_H, A4_OX, A4_OY, UH
    UH = SH - MG * 2 - UI_TOP - UI_BOT
    SCALE = min(UW / ARM_H, UH / ARM_W) * 0.70
    A4_PX_W = ARM_H * SCALE
    A4_PX_H = ARM_W * SCALE
    A4_OX = MG + (UW - A4_PX_W) / 2
    A4_OY = MG + UI_TOP + (UH - A4_PX_H) / 2


def open_tjc(port="/dev/ttyAMA2", baud=9600, bkcmd=0):
    """Open TJC serial connection.

    bkcmd=0: no command acks (cleanest for MCU-control pattern).
    """
    if serial is None:
        return None
    try:
        ser = serial.Serial(port, baud, timeout=0.1, write_timeout=0.5)
        tjc = TJC(ser, bkcmd=bkcmd)
        tjc._send("bkcmd=" + str(bkcmd))
        tjc.page(0)
        time.sleep(0.1)
        return tjc
    except Exception:
        return None


_cached_mm_px = [None, None]  # [mm_px_X, mm_px_Y]

# ?? Injected from main.py ??
_ctx = {}
def setup_tjc_state(*, shared_state, mode_labels, image_to_arm):
    _ctx['SharedState'] = shared_state
    _ctx['MODE_LABELS'] = mode_labels
    _ctx['image_to_arm'] = image_to_arm

def _build_tjc_state(pieces, reconst, fps=0, last_action="", selected_mode="AUTO",
                     a4_corners_camera=None, cam_w=1920, cam_h=1080):
    """Convert vision data to arm-mm format for TJC display.

    a4_corners_camera: optional [4x2] A4 corners in camera image coords.
    cam_w, cam_h: camera image dimensions.
    """
    # Compute A4 rotation + camera view frame
    global _cached_mm_px
    a4_rot = 0.0
    cam_frame_arm = None
    if a4_corners_camera is not None and len(a4_corners_camera) == 4:
        tl = np.array(a4_corners_camera[0], dtype=np.float64)
        tr = np.array(a4_corners_camera[1], dtype=np.float64)
        bl = np.array(a4_corners_camera[3], dtype=np.float64)
        br = np.array(a4_corners_camera[2], dtype=np.float64)

        # Rotation from top edge
        cam_deg = math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0]))
        a4_rot = -cam_deg

        # Camera frame: find A4 position in camera image, extend to full FOV
        left_px   = float(min(tl[0], bl[0]))
        right_px  = float(max(tr[0], br[0]))
        top_px    = float(min(tl[1], tr[1]))
        bottom_px = float(max(bl[1], br[1]))
        a4_bbox_w = right_px - left_px
        a4_bbox_h = bottom_px - top_px

        if a4_bbox_w > 10 and a4_bbox_h > 10:
            # One mm/px from A4 (separate axes = actual physical proportions)
            mm_px_X = 210.0 / a4_bbox_w
            mm_px_Y = 297.0 / a4_bbox_h

            # Cache mm_px when A4 is fully visible, use cached value when
            # A4 is near image edges (partial/false detection) to avoid
            # distorted camera frame.
            _edge = 20
            a4_fully_visible = (
                left_px > _edge and right_px < cam_w - _edge and
                top_px > _edge and bottom_px < cam_h - _edge
            )
            if a4_fully_visible:
                _cached_mm_px[0] = mm_px_X
                _cached_mm_px[1] = mm_px_Y
            elif _cached_mm_px[0] is not None:
                mm_px_X = _cached_mm_px[0]
                mm_px_Y = _cached_mm_px[1]

            # A4 arm-mm bounds
            A4_L, A4_R = 0.0, 210.0
            A4_T, A4_B = -75.0, 222.0

            # Extend A4 bounds by camera margin to get FOV
            fov_left   = A4_L - left_px * mm_px_X
            fov_right  = A4_R + (cam_w - right_px) * mm_px_X
            fov_top    = A4_T - top_px * mm_px_Y
            fov_bottom = A4_B + (cam_h - bottom_px) * mm_px_Y

            # Axis-aligned rectangle
            cam_frame_arm = [
                [round(fov_right, 1), round(fov_top, 1)],
                [round(fov_left, 1),  round(fov_top, 1)],
                [round(fov_left, 1),  round(fov_bottom, 1)],
                [round(fov_right, 1), round(fov_bottom, 1)],
            ]
    pieces_arm = []
    for pp in pieces:
        pick_arm = list(_ctx['image_to_arm'](pp.pickup_x_image, pp.pickup_y_image))
        poly_arm = []
        poly = pp.polygon
        if poly is not None and len(poly) > 0:
            arr = np.asarray(poly, dtype=np.float64)
            if arr.ndim == 3:
                arr = arr.reshape(-1, 2)
            for vx, vy in arr:
                ax, ay = _ctx['image_to_arm'](float(vx), float(vy))
                poly_arm.append([round(ax, 2), round(ay, 2)])
        pieces_arm.append({
            "id": f"piece_{pp.piece_id}",
            "pick_mm": [round(pick_arm[0], 2), round(pick_arm[1], 2)],
            "polygon_arm_mm": poly_arm,
        })

    plan_items = []
    if reconst is not None and reconst.plan:
        for item in reconst.plan:
            plan_items.append({
                "piece_id": str(item.get("piece_id", "")),
                "place_mm": item.get("place_mm", [0, 0]),
                "rotate_deg": item.get("rotate_deg", 0),
                "target_polygon_mm": item.get("target_polygon_mm"),
            })

    info = {
        "mode": reconst.selected_mode if reconst else "--",
        "fill_ratio": reconst.solver_info.get("fill_ratio", 0) if reconst else 0,
        "pieces_count": len(pieces),
        "fps": round(fps, 1),
        "last_action": last_action,
        "selected_mode": selected_mode,
    }
    if reconst and reconst.plan:
        info["assembly_order"] = [str(it["piece_id"]) for it in reconst.plan]

    return pieces_arm, plan_items, info, a4_rot, cam_frame_arm


