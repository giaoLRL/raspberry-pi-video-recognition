#!/usr/bin/env python3
"""System health and diagnostics module.

Provides arm self-check, Pi hardware diagnostics.
Re-exports from system_check.py (canonical location at project root).
"""

from system_check import arm_check, pi_check, StatusReporter, make_unfreeze_callback
