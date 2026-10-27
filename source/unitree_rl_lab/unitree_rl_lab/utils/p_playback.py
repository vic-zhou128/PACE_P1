"""Keyboard speed caps inside a P1 checkpoint's sampled command envelope."""

from copy import deepcopy
import math


def keyboard_speed_ranges(hard_limits, vx: float, vy: float, trained_ranges=None):
    result = deepcopy(hard_limits)
    for axis, speed in (("lin_vel_x", vx), ("lin_vel_y", vy)):
        if not math.isfinite(speed) or speed < 0:
            raise ValueError("Keyboard speeds must be finite and nonnegative")
        low, high = getattr(hard_limits, axis)
        if trained_ranges is not None:
            trained_low, trained_high = map(float, trained_ranges[axis])
            if not all(map(math.isfinite, (trained_low, trained_high))) or trained_low > trained_high:
                raise ValueError(f"Invalid trained command range: {axis}")
            low, high = max(low, trained_low), min(high, trained_high)
        # Standing is always allowed. A forward-only stage disables S rather
        # than mapping S to a small positive velocity; zero lateral ranges disable A/D.
        setattr(result, axis, (min(0.0, max(low, -speed)), max(0.0, min(high, speed))))
    return result
