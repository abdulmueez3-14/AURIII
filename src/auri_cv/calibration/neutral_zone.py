"""Display-layer neutral zone (deadband) with dual-threshold hysteresis.

The deadband acts on the DISPLAY value only. Callers keep the raw and relative values.
"""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Final

DEFAULT_DELTA_ENTER_DEG: Final[float] = 3.0
DEFAULT_DELTA_EXIT_DEG: Final[float] = 5.0


class NeutralZoneState(StrEnum):
    IN_NEUTRAL = "IN_NEUTRAL"
    ACTIVE_EXCURSION = "ACTIVE_EXCURSION"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class DisplayReading:
    state: NeutralZoneState
    display_deg: float | None  # None when UNAVAILABLE -- never a fake 0.0


class NeutralZone:
    """Hysteresis: leave neutral at |d| >= delta_exit; re-enter at |d| <= delta_enter."""

    def __init__(
        self,
        delta_enter_deg: float = DEFAULT_DELTA_ENTER_DEG,
        delta_exit_deg: float = DEFAULT_DELTA_EXIT_DEG,
    ) -> None:
        if not 0.0 <= delta_enter_deg < delta_exit_deg:
            raise ValueError("require 0 <= delta_enter < delta_exit")
        self._enter = delta_enter_deg
        self._exit = delta_exit_deg
        self._in_neutral = True

    def reset(self) -> None:
        self._in_neutral = True

    def update(self, relative_deg: float | None) -> DisplayReading:
        if relative_deg is None or not isfinite(relative_deg):
            return DisplayReading(NeutralZoneState.UNAVAILABLE, None)
        mag = abs(relative_deg)
        if self._in_neutral and mag >= self._exit:
            self._in_neutral = False
        elif not self._in_neutral and mag <= self._enter:
            self._in_neutral = True
        if self._in_neutral:
            return DisplayReading(NeutralZoneState.IN_NEUTRAL, 0.0)
        return DisplayReading(NeutralZoneState.ACTIVE_EXCURSION, relative_deg)
