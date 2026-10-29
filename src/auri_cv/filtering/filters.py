"""Deterministic temporal filters. Filters never overwrite the raw signal; callers keep both.

A missing sample (None / non-finite) clears filter state, so smoothing never bridges a
tracking gap, and the filter is reseeded on the next valid sample.
"""

from collections import deque
from math import isfinite
from typing import Final

DEFAULT_SMA_WINDOW: Final[int] = 5
DEFAULT_EMA_ALPHA: Final[float] = 0.4


class MovingAverageFilter:
    """Simple moving average over the last `window` consecutive valid samples."""

    def __init__(self, window: int = DEFAULT_SMA_WINDOW) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._buf: deque[float] = deque(maxlen=window)

    def reset(self) -> None:
        self._buf.clear()

    def update(self, value: float | None) -> float | None:
        if value is None or not isfinite(value):
            self._buf.clear()
            return None
        self._buf.append(float(value))
        return sum(self._buf) / len(self._buf)


class ExponentialFilter:
    """EMA: y_t = a * x_t + (1 - a) * y_{t-1}; reseeded to x_t after a gap."""

    def __init__(self, alpha: float = DEFAULT_EMA_ALPHA) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = alpha
        self._state: float | None = None

    def reset(self) -> None:
        self._state = None

    def update(self, value: float | None) -> float | None:
        if value is None or not isfinite(value):
            self._state = None
            return None
        x = float(value)
        self._state = (
            x if self._state is None else self._alpha * x + (1 - self._alpha) * self._state
        )
        return self._state
