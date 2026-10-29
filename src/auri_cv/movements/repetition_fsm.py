"""Repetition detection finite-state machine (docs/MATHEMATICAL_MODEL.md section 9).

Input is the relative excursion (degrees, already calibrated and preferably filtered),
or None when tracking is unavailable. The FSM never fabricates progress across a gap.
"""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Final

THETA_START_DEG: Final[float] = 15.0
THETA_MIN_PEAK_DEG: Final[float] = 45.0
THETA_RETURN_DEG: Final[float] = 10.0
PEAK_HYSTERESIS_DEG: Final[float] = 5.0
MAX_GAP_S: Final[float] = 1.0


class RepState(StrEnum):
    IDLE = "IDLE"
    RAISING = "RAISING"
    PEAK_HOLD = "PEAK_HOLD"
    LOWERING = "LOWERING"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class Repetition:
    index: int  # 1-based
    peak_deg: float
    time_to_peak_s: float  # from crossing THETA_START to the peak
    duration_s: float  # from crossing THETA_START to return below THETA_RETURN


class RepetitionFSM:
    """IDLE -> RAISING -> PEAK_HOLD -> LOWERING -> COMPLETE -> IDLE."""

    def __init__(
        self,
        start_deg: float = THETA_START_DEG,
        min_peak_deg: float = THETA_MIN_PEAK_DEG,
        return_deg: float = THETA_RETURN_DEG,
        peak_hysteresis_deg: float = PEAK_HYSTERESIS_DEG,
        max_gap_s: float = MAX_GAP_S,
    ) -> None:
        if not return_deg < start_deg < min_peak_deg:
            raise ValueError("require return < start < min_peak")
        self._start = start_deg
        self._min_peak = min_peak_deg
        self._return = return_deg
        self._hyst = peak_hysteresis_deg
        self._max_gap = max_gap_s
        self.repetitions: list[Repetition] = []
        self.invalidated_count = 0
        self.aborted_count = 0  # started but never reached the minimum peak
        self.reset()

    @property
    def rep_count(self) -> int:
        return len(self.repetitions)

    @property
    def state(self) -> RepState:
        return self._state

    def reset(self) -> None:
        """Returns to IDLE and drops in-progress state (completed reps are kept)."""
        self._state = RepState.IDLE
        self._t_start = 0.0
        self._t_peak = 0.0
        self._peak = 0.0
        self._prev: float | None = None
        self._gap_start: float | None = None
        # A repetition may only begin after the signal has been seen in the resting zone,
        # so resuming mid-movement (session start, after a tracking gap) never counts.
        self._armed = False

    def _invalidate(self) -> None:
        self.invalidated_count += 1
        self.reset()

    def update(self, excursion_deg: float | None, t_s: float) -> RepState:
        """Advances the FSM by one sample. Returns the state AFTER this sample."""
        if self._state == RepState.COMPLETE:
            self._state = RepState.IDLE

        if excursion_deg is None or not isfinite(excursion_deg):
            if self._state != RepState.IDLE:
                if self._gap_start is None:
                    self._gap_start = t_s
                if t_s - self._gap_start > self._max_gap:
                    self._invalidate()
            self._prev = None
            return self._state
        self._gap_start = None
        x = float(excursion_deg)
        prev, self._prev = self._prev, x

        if self._state == RepState.IDLE:
            if x <= self._return:
                self._armed = True
            if self._armed and x > self._start:
                self._state = RepState.RAISING
                self._t_start, self._t_peak, self._peak = t_s, t_s, x
        elif self._state == RepState.RAISING:
            if x > self._peak:
                self._peak, self._t_peak = x, t_s
            if x <= self._return:
                self.aborted_count += 1
                self.reset()
                self._armed = True
            elif prev is not None and x <= prev and self._peak >= self._min_peak:
                self._state = RepState.PEAK_HOLD
        elif self._state == RepState.PEAK_HOLD:
            if x > self._peak:
                self._peak, self._t_peak = x, t_s
            elif x < self._peak - self._hyst:
                self._state = RepState.LOWERING
        elif self._state == RepState.LOWERING:
            if x > self._peak:  # re-raised: peak was not final
                self._peak, self._t_peak = x, t_s
                self._state = RepState.PEAK_HOLD
            elif x <= self._return:
                self.repetitions.append(
                    Repetition(
                        index=self.rep_count + 1,
                        peak_deg=self._peak,
                        time_to_peak_s=self._t_peak - self._t_start,
                        duration_s=t_s - self._t_start,
                    )
                )
                self.reset()
                self._armed = True
                self._state = RepState.COMPLETE
                self._prev = x
        return self._state
