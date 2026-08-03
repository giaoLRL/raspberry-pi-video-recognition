#!/usr/bin/env python3
"""Rendering functions for puzzle piece overlay visualization.

Extracted from main.py. Dependencies injected via setup().
"""

import cv2
import numpy as np

# Injected by setup()
_ctx = {}

# ?? Constants ??
WARP_WIDTH = 840
WARP_HEIGHT = 1188
A4_HEIGHT_MM = 297.0


def setup(*, image_to_arm, arm_to_warp, pixels_per_mm, min_piece_area_px, max_piece_area_px,
          shared_state=None,
          log_print):
    """Inject dependencies before using rendering functions."""
    _ctx["image_to_arm"] = image_to_arm
    _ctx["arm_to_warp"] = arm_to_warp
    _ctx["PIXELS_PER_MM"] = pixels_per_mm
    _ctx["MIN_PIECE_AREA_PX"] = min_piece_area_px
    _ctx["MAX_PIECE_AREA_PX"] = max_piece_area_px
    _ctx["log_print"] = log_print
    _ctx["SharedState"] = shared_state


# ?? Original functions from main.py (deps replaced with _ctx) ??

def warp_to_camera(pts, mat):
    pts_arr = np.atleast_3d(np.asarray(pts, dtype=np.float64)).astype(np.float32)
    if pts_arr.ndim == 2:
        pts_arr = pts_arr.reshape(-1, 1, 2)
    m = np.asarray(mat, dtype=np.float64).astype(np.float32)
    try:
        out = cv2.perspectiveTransform(pts_arr, m)
    except Exception:
        pts2d = pts_arr.reshape(-1, 2).astype(np.float64)
        ones = np.ones((pts2d.shape[0], 1), dtype=np.float64)
        homo = np.hstack([pts2d, ones])
        t = homo @ m.astype(np.float64).T
        w = t[:, 2:3].copy()
        w[w == 0] = 1e-10
        out = (t[:, :2] / w).reshape(-1, 1, 2).astype(np.float32)
    if out.size == 1:
        out = np.array([[[float(out), float(out)]]], dtype=np.float32)
    return out.reshape(-1, 2)


def draw_overlay(frame, corners, pieces, reconst, w2c, area_mode=0):

    out = frame.copy()

    if corners is not None:

        cv2.polylines(out, [np.round(corners).astype(np.int32)], True, (255, 255, 0), 2, cv2.LINE_AA)

    # Draw detected pieces
    piece_pickup_camera = {}  # piece_id -> (cx, cy) in camera pixels
    if w2c is not None:

        for p in pieces:

            # Convert pickup point (warp px) to arm-mm coords
            px_mm, py_mm = _ctx['image_to_arm'](p.pickup_x_image, p.pickup_y_image)


            # Original contour (thin grey)

            ccam = warp_to_camera(p.contour, w2c)

            cv2.drawContours(out, [np.round(ccam).astype(np.int32)], -1, (128, 128, 128), 1, cv2.LINE_AA)

            # Simplified polygon (yellow)

            pcam = warp_to_camera(p.polygon, w2c)

            cv2.polylines(out, [np.round(pcam).astype(np.int32)], True, (0, 255, 255), 3, cv2.LINE_AA)

            # Pickup point = safe_interior_point (red filled circle)

            ccam = warp_to_camera(np.array([[[p.pickup_x_image, p.pickup_y_image]]], dtype=np.float32), w2c)[0]

            cxi, cyi = int(ccam[0]), int(ccam[1])

            cv2.circle(out, (cxi, cyi), 7, (0, 0, 255), -1)

            cv2.circle(out, (cxi, cyi), 10, (0, 0, 255), 2, cv2.LINE_AA)

            # Save pickup camera coords for connecting-line reference
            piece_pickup_camera[p.piece_id] = (cxi, cyi)

            # ID label

            cv2.putText(out, f"ID:{p.piece_id}",

                        (cxi + 14, cyi - 18),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            # Pickup coordinate in mm (white text with dark background for readability)

            coord_str = f"({px_mm:.1f}, {py_mm:.1f})mm"

            (tw, th), _ = cv2.getTextSize(coord_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            cv2.rectangle(out,

                          (cxi - tw // 2 - 4, cyi + 16 - th - 2),

                          (cxi + tw // 2 + 4, cyi + 16 + 2),

                          (0, 0, 0), -1, cv2.LINE_AA)

            cv2.putText(out, coord_str,

                        (cxi - tw // 2, cyi + 16),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Log to console

            area_mm2 = p.area_px / (_ctx['PIXELS_PER_MM'] ** 2)

            nv = len(p.polygon)

            print(f"[PICKUP] ID:{p.piece_id} pos=({px_mm:.1f},{py_mm:.1f})mm area={area_mm2:.0f}mm2 sides={nv}", flush=True)

    cv2.putText(out, f"Pieces: {len(pieces)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Area reference boxes

    if area_mode in (1, 2) and w2c is not None:

        area_px = _ctx['MIN_PIECE_AREA_PX'] if area_mode == 1 else _ctx['MAX_PIECE_AREA_PX']

        label = "MIN AREA" if area_mode == 1 else "MAX AREA"

        color = (0, 255, 0) if area_mode == 1 else (255, 0, 255)

        side = int(np.sqrt(area_px))

        cx_w, cy_w = WARP_WIDTH // 2, WARP_HEIGHT // 2

        square_warp = np.array([[cx_w - side // 2, cy_w - side // 2],

                                [cx_w + side // 2, cy_w - side // 2],

                                [cx_w + side // 2, cy_w + side // 2],

                                [cx_w - side // 2, cy_w + side // 2]], dtype=np.float32)

        square_cam = np.round(warp_to_camera(square_warp, w2c)).astype(np.int32)

        cv2.polylines(out, [square_cam], True, color, 3, cv2.LINE_AA)

        cv2.putText(out, f"{label}: {area_px}px", (square_cam[0][0], square_cam[0][1] - 8),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # Reconstruction plan

    # reconst can be a list (from main.py plan_to_draw) or RecoveryResult
    plan_items = reconst if isinstance(reconst, list) else (reconst.plan if hasattr(reconst, 'plan') else None)
    if plan_items and w2c is not None:

        tpts = []

        for idx, item in enumerate(plan_items, start=1):

            pid = item.get("piece_id", idx)

            # 绿色：目标放置位置（arm mm → warp → camera）
            poly_arm = np.asarray(item["target_polygon_mm"], dtype=np.float32).reshape(-1, 2)
            poly_warp = np.array([[_ctx['arm_to_warp'](float(x), float(y)) for x, y in poly_arm]], dtype=np.float32).reshape(-1, 2)
            pcam = np.round(warp_to_camera(poly_warp, w2c)).astype(np.int32)

            tpts.append(pcam)

            cv2.polylines(out, [pcam], True, (0, 255, 0), 3, cv2.LINE_AA)

            # 碎片ID标签（绿色）

            label_pos = tuple(pcam[0])

            cv2.putText(out, f"#{pid}", (label_pos[0] - 5, label_pos[1] - 8),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            # 红色1px：同源数据验证

            cv2.polylines(out, [pcam], True, (0, 0, 255), 1, cv2.LINE_AA)

            # ── Place point (safe_interior at target) + connecting line ──
            # Distinct BGR colors for up to 6 pieces
            LINE_PALETTE = [
                (255, 0, 255),   # Magenta
                (255, 255, 0),   # Cyan
                (0, 140, 255),   # Orange
                (0, 255, 128),   # Lime green
                (0, 255, 255),   # Yellow
                (128, 0, 255),   # Pink
            ]
            line_color = LINE_PALETTE[(idx - 1) % len(LINE_PALETTE)]

            # place_mm is the safe_interior_point at target, already in arm-mm
            place_arm = np.array(item["place_mm"], dtype=np.float64)  # [x_mm, y_mm]
            tc_wx, tc_wy = _ctx['arm_to_warp'](float(place_arm[0]), float(place_arm[1]))
            tc_cam = warp_to_camera(np.array([[[tc_wx, tc_wy]]], dtype=np.float32), w2c)[0]
            tc_cx, tc_cy = int(round(tc_cam[0])), int(round(tc_cam[1]))

            # Find matching original pickup point and draw connecting line
            if isinstance(pid, str) and pid.startswith("piece_"):
                piece_num = int(pid.split("_")[1])
            else:
                piece_num = int(pid) if pid is not None else idx
            if piece_num in piece_pickup_camera:
                orig_cx, orig_cy = piece_pickup_camera[piece_num]
                # Color-coded line: original pickup → target place
                cv2.line(out, (orig_cx, orig_cy), (tc_cx, tc_cy), line_color, 2, cv2.LINE_AA)

            # Target place marker (color matches the connecting line)
            cv2.circle(out, (tc_cx, tc_cy), 7, line_color, -1)
            cv2.circle(out, (tc_cx, tc_cy), 11, line_color, 2, cv2.LINE_AA)

            # Target place coordinate label (color matches the marker)
            tcoord_str = f"({place_arm[0]:.1f},{place_arm[1]:.1f})mm"
            (tw, th), _ = cv2.getTextSize(tcoord_str, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 2)
            cv2.rectangle(out,
                          (tc_cx - tw // 2 - 4, tc_cy + 14 - th - 2),
                          (tc_cx + tw // 2 + 4, tc_cy + 14 + 2),
                          (0, 0, 0), -1, cv2.LINE_AA)
            cv2.putText(out, tcoord_str,
                        (tc_cx - tw // 2, tc_cy + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, line_color, 2, cv2.LINE_AA)

            # Log target place point
            print(f"[TARGET] {pid}: place=({place_arm[0]:.1f},{place_arm[1]:.1f})mm", flush=True)

        if tpts:

            # 蓝框：目标矩形（arm mm → warp → camera）
            info = reconst.solver_info if hasattr(reconst, "solver_info") else {}

            target_origin = info.get("target_origin_mm")

            target_size = info.get("target_size_mm")

            if target_origin is not None and target_size is not None:

                arm_ox = float(target_origin[0])
                arm_oy = float(target_origin[1])
                w = float(target_size[0])
                h = float(target_size[1])
                # target_origin in arm = top-right corner (solver TL flipped to arm)
                rect_arm = np.array([
                    [arm_ox, arm_oy],           # top-right
                    [arm_ox - w, arm_oy],       # top-left
                    [arm_ox - w, arm_oy + h],   # bottom-left
                    [arm_ox, arm_oy + h],       # bottom-right
                ], dtype=np.float64)

                box_pts = []
                for ax, ay in rect_arm:
                    wx, wy = _ctx['arm_to_warp'](float(ax), float(ay))
                    pc = warp_to_camera(np.array([[[wx, wy]]], dtype=np.float32), w2c)[0]
                    box_pts.append([int(round(pc[0])), int(round(pc[1]))])
                box = np.array(box_pts, dtype=np.int32)

            else:

                merged = np.vstack(tpts)

                box = np.round(cv2.boxPoints(cv2.minAreaRect(merged.astype(np.float32)))).astype(np.int32)

            cv2.polylines(out, [box], True, (255, 0, 0), 2, cv2.LINE_AA)

            if not hasattr(draw_overlay, "_last_box_hash"):

                draw_overlay._last_box_hash = None

            box_hash = hash(box.tobytes())

            if box_hash != draw_overlay._last_box_hash:

                draw_overlay._last_box_hash = box_hash

                corners_str = ", ".join(f"({x},{y})" for x, y in box)

                if target_size is not None and target_origin is not None:

                    sz = np.asarray(target_size, dtype=np.float64)
                    _ctx['log_print'](f"BlueBox: [{corners_str}] size={sz[0]:.1f}x{sz[1]:.1f}mm arm=({float(target_origin[0]):.1f},{float(target_origin[1]):.1f})")

                else:

                    _ctx['log_print'](f"BlueBox: [{corners_str}] (minAreaRect fallback)")

        mode_label = reconst.selected_mode if hasattr(reconst, "selected_mode") else "SOLVED"
        cv2.putText(out, f"Restored: {mode_label}", (20, 70),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2, cv2.LINE_AA)

    # ============================================================
    # ARM COORDINATE SYSTEM (authoritative for all display rendering)
    #
    #   Origin: A4 right edge + 75mm down => warp pixel (839, 300)
    #   X axis: leftward  (0 at BR edge, ~210 at TL edge)
    #   Y axis: downward   (0 at 75mm from top, ~222 at bottom edge)
    #
    #   TL = arm(209.8, -75.0)    TR = arm(0.0, -75.0)
    #   BL = arm(209.8, 221.8)    BR = arm(0.0, 221.8)
    #
    #   Conversions (coords.py):
    #     _ctx['image_to_arm'](wx, wy)   : warp px -> arm mm
    #     _ctx['arm_to_warp'](ax, ay)    : arm mm  -> warp px
    #
    #   Rendering: arm mm -> arm_to_warp -> warp_to_camera -> screen
    # ============================================================
    if corners is not None and w2c is not None:

        # --- Origin in warp ---

        origin_wx = WARP_WIDTH - 1  # BR right edge (pixel 839)

        origin_warp = np.array([origin_wx, 300.0])  # TR + 75mm down

        origin_cam = warp_to_camera(np.array([[origin_warp]], dtype=np.float32), w2c)[0]

        ox, oy = int(round(origin_cam[0])), int(round(origin_cam[1]))


        def phys_to_cam(x_mm, y_mm):

            wx, wy = _ctx['arm_to_warp'](x_mm, y_mm)


            pcam = warp_to_camera(np.array([[[wx, wy]]], dtype=np.float32), w2c)[0]

            return int(round(pcam[0])), int(round(pcam[1]))


        X_MIN_MM = 0.0       # at BR (origin)
        X_MAX_MM = 210.0     # A4 width (X leftward)
        Y_MAX_MM = 297.0     # A4 height (Y downward from origin)

        # --- Draw X axis (red) ---

        x0_cam = phys_to_cam(X_MIN_MM, 0.0)

        x1_cam = phys_to_cam(X_MAX_MM, 0.0)

        cv2.line(out, x0_cam, x1_cam, (0, 0, 255), 2, cv2.LINE_AA)

        # Arrow head at top

        temp1 = phys_to_cam(X_MAX_MM - 10.0, -6.0)

        temp2 = phys_to_cam(X_MAX_MM - 10.0, 6.0)

        cv2.line(out, x1_cam, temp1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.line(out, x1_cam, temp2, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(out, "X", (x1_cam[0] + 8, x1_cam[1] - 8),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


        # --- Draw Y axis (green) ---

        y0_cam = phys_to_cam(0.0, 0.0)

        y1_cam = phys_to_cam(0.0, Y_MAX_MM)

        cv2.line(out, y0_cam, y1_cam, (0, 255, 0), 2, cv2.LINE_AA)

        # Arrow head at right

        temp1 = phys_to_cam(-6.0, Y_MAX_MM - 10.0)

        temp2 = phys_to_cam(6.0, Y_MAX_MM - 10.0)

        cv2.line(out, y1_cam, temp1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.line(out, y1_cam, temp2, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(out, "Y", (y1_cam[0] + 8, y1_cam[1] + 4),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


        # --- X-axis tick marks and labels ---

        # X-axis ticks: 0 to 210mm, step=20mm

        x_step = 20.0

        for x_mm in np.arange(X_MIN_MM, X_MAX_MM + 0.1, x_step):

            t0 = phys_to_cam(x_mm, -5.0)

            t1 = phys_to_cam(x_mm, 5.0)

            cv2.line(out, t0, t1, (0, 0, 255), 1, cv2.LINE_AA)

            # Label: place to left of axis

            tlbl = phys_to_cam(x_mm, -10.0)

            cv2.putText(out, f"{x_mm:.0f}",

                        (tlbl[0] - 30, tlbl[1] + 4),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)


        # --- Y-axis tick marks and labels ---

        y_step = 20.0

        for y_mm in np.arange(0.0, Y_MAX_MM + 0.1, y_step):

            t0 = phys_to_cam(-5.0, y_mm)

            t1 = phys_to_cam(5.0, y_mm)

            cv2.line(out, t0, t1, (0, 255, 0), 1, cv2.LINE_AA)

            # Label below axis

            tlbl = phys_to_cam(-10.0, y_mm)

            cv2.putText(out, f"{y_mm:.0f}",

                        (tlbl[0] - 28, tlbl[1] + 8),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)


        # --- Origin marker ---

        cv2.circle(out, (ox, oy), 8, (0, 255, 255), -1, cv2.LINE_AA)

        cv2.circle(out, (ox, oy), 14, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(out, "O(0,0)@75mm",

                    (ox + 18, oy - 14),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)


        # --- Corners with labels ---

        warp_corners_all = np.array([

            [0, 0], [WARP_WIDTH - 1, 0],

            [WARP_WIDTH - 1, WARP_HEIGHT - 1], [0, WARP_HEIGHT - 1],

        ], dtype=np.float32)

        cc_all = warp_to_camera(warp_corners_all, w2c)

        tl_c, tr_c, br_c, bl_c = cc_all[:4]


        # Arm coords of A4 corners
        _arm_tl = _ctx['image_to_arm'](0.0, 0.0)
        _arm_tr = _ctx['image_to_arm'](float(WARP_WIDTH - 1), 0.0)
        _arm_bl = _ctx['image_to_arm'](0.0, float(WARP_HEIGHT - 1))
        _arm_br = _ctx['image_to_arm'](float(WARP_WIDTH - 1), float(WARP_HEIGHT - 1))

        corner_defs = [
            ("TL", tl_c, (0, 255, 0),    24, -10, f"({_arm_tl[0]:.1f},{_arm_tl[1]:.1f})"),
            ("TR", tr_c, (255, 160, 0),  -52, -10, f"({_arm_tr[0]:.1f},{_arm_tr[1]:.1f})"),
            ("BL", bl_c, (0, 255, 200),   24, 22,  f"({_arm_bl[0]:.1f},{_arm_bl[1]:.1f})"),
            ("BR", br_c, (255, 100, 255), -52, 22,  f"({_arm_br[0]:.1f},{_arm_br[1]:.1f})"),
        ]

        for label, pt, color, dx, dy, coord_str in corner_defs:

            px, py = int(round(pt[0])), int(round(pt[1]))

            cv2.circle(out, (px, py), 8, color, 2, cv2.LINE_AA)

            cv2.circle(out, (px, py), 2, color, -1, cv2.LINE_AA)

            cv2.putText(out, f"{label}{coord_str}",

                        (px + dx, py + dy),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


        # --- Grid lines (subtle) ---

        for x_mm in np.arange(X_MIN_MM, X_MAX_MM + 0.1, x_step):

            if abs(x_mm) < 0.01:

                continue  # skip origin, Y axis already drawn

            g0 = phys_to_cam(x_mm, 0.0)

            g1 = phys_to_cam(x_mm, Y_MAX_MM)

            cv2.line(out, g0, g1, (60, 60, 60), 1, cv2.LINE_AA)


        for y_mm in np.arange(y_step, Y_MAX_MM + 0.1, y_step):

            g0 = phys_to_cam(X_MIN_MM, y_mm)

            g1 = phys_to_cam(X_MAX_MM, y_mm)

            cv2.line(out, g0, g1, (60, 60, 60), 1, cv2.LINE_AA)


        # --- Assembly target zone: lower half of A4 (mid-line to bottom) ---
        # A4 lower half in solver mm: [0, 148.5, 210, 297]
        # In arm mm: X=0(BR edge) to X=210(TL edge), Y=73.5(mid) to Y=221.8(bottom)
        _zone_x1 = 0.0
        _zone_x2 = (WARP_WIDTH - 1) / _ctx['PIXELS_PER_MM']                  # ~209.75
        _zone_y1 = A4_HEIGHT_MM * 0.5 - 300.0 / _ctx['PIXELS_PER_MM']       # 148.5 - 75 = 73.5
        _zone_y2 = (WARP_HEIGHT - 1 - 300.0) / _ctx['PIXELS_PER_MM']         # (1187-300)/4 = 221.75

        zone_corners_phys = [(_zone_x1, _zone_y1), (_zone_x2, _zone_y1),
                             (_zone_x2, _zone_y2), (_zone_x1, _zone_y2)]

        zone_pts_cam = [phys_to_cam(x, y) for x, y in zone_corners_phys]

        # Draw filled translucent zone

        zone_overlay = out.copy()

        cv2.fillPoly(zone_overlay, [np.array(zone_pts_cam, dtype=np.int32)], (200, 150, 50))

        out = cv2.addWeighted(out, 0.75, zone_overlay, 0.25, 0)

        # Cyan border

        cv2.polylines(out, [np.array(zone_pts_cam, dtype=np.int32)], True, (255, 200, 0), 3, cv2.LINE_AA)

        # Corner labels

        zone_labels = [f"Z1(0,{_zone_y1:.0f})", f"Z2({_zone_x2:.0f},{_zone_y1:.0f})",
                       f"Z3({_zone_x2:.0f},{_zone_y2:.0f})", f"Z4(0,{_zone_y2:.0f})"]

        for zlbl, (zx, zy) in zip(zone_labels, zone_pts_cam):

            cv2.circle(out, (zx, zy), 6, (255, 200, 0), -1, cv2.LINE_AA)

            cv2.putText(out, zlbl, (zx + 8, zy - 8),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 200, 0), 1, cv2.LINE_AA)

        # Center label

        zcx, zcy = phys_to_cam((_zone_x1 + _zone_x2) * 0.5, (_zone_y1 + _zone_y2) * 0.5)

        cv2.putText(out, "ASSEMBLY ZONE",

                    (zcx - 60, zcy),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)


    # A4 label

    if corners is not None:

        pts = np.round(corners).astype(np.int32)

        cv2.putText(out, "A4", (pts[0][0] + 5, pts[0][1] - 8),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

    else:

        cv2.putText(out, "Searching A4...", (20, 100),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # Render log messages

    with _ctx['SharedState'].lock:

        logs = list(_ctx['SharedState'].log_lines)

    if logs:

        fh = out.shape[0]

        line_h = 16

        margin = 8

        log_h = len(logs) * line_h + margin * 2

        overlay = out.copy()

        cv2.rectangle(overlay, (0, fh - log_h), (520, fh), (0, 0, 0), -1)

        out = cv2.addWeighted(out, 0.6, overlay, 0.4, 0)

        for i, line in enumerate(logs):

            text = line if len(line) <= 90 else line[:87] + "..."

            cv2.putText(out, text, (margin + 4, fh - margin - (len(logs) - 1 - i) * line_h),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    return out


# ============================================================

# Web UI and shared state

# ============================================================

# ============================================================


