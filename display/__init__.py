#!/usr/bin/env python3
"""TJC display module — screen drivers, rendering, and coordinate conversion.

Re-exports from tjc_display.py (canonical location at project root).
"""

from tjc_display import (
    TJC, TjcManager, draw_state, arm_to_screen, open_tjc,
    SW, SH, W, B, GY, R, G, BL, PC,
    END,
    STATUS_H, BTN_ROW_H, BTN_ROW1_Y, BTN_ROW2_Y,
    UI_TOP, UI_BOT, MG, CONTENT_Y0, CONTENT_Y1,
)
