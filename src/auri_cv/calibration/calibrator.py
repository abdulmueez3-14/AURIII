"""Multi-frame baseline calibration (theta_0) for a single angle channel."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Final

import numpy as np

DEFAULT_CALIBRATION_FRAMES: Final[int] = 30
DEFAULT_MAX_IQR_DEG: Final[float] = 3.0


class CalibrationState(StrEnum):
    COLLECTING = "COLLECTING"
    COMPLETE = "COMPLETE"
    REJECTED = "REJECTED"  # subject moved during calibration (IQR too large)


@dataclass(frozen=True)
class Baseline:
    """Result of a successful calibration. All values in degrees."""

    theta0_deg: float  # median of the N raw samples
    iqr_deg: float  # Q3 - Q1 of the N raw samples
    n_frames: int


class BaselineCalibrator:
    """Collects N CONSECUTIVE valid raw samples and sets the median as theta_0.

    A missing/non-finite sample (tracking loss) restarts collection, so a baseline is
    never built from a gapped sequence. If IQR > `max_iqr_deg` the result is REJECTED
    and `reset()` must be called to retry; no baseline is produced.
    """

    def __init__(
        self,
        n_frames: int = DEFAULT_CALIBRATION_FRAMES,
        max_iqr_deg: float = DEFAULT_MAX_IQR_DEG,
    ) -> None:
        if n_frames < 2:
            raise ValueError("n_frames must be >= 2")
        if max_iqr_deg <= 0.0:
            raise ValueError("max_iqr_deg must be positive")
        self._n = n_frames
        self._max_iqr = max_iqr_deg
        self.reset()

    def reset(self) -> None:
        self._samples: list[float] = []
        self._state = CalibrationState.COLLECTING
        self._baseline: Baseline | None = None

    @property
    def state(self) -> CalibrationState:
        return self._state

    @property
    def baseline(self) -> Baseline | None:
        return self._baseline

    @property
    def progress(self) -> float:
        return min(1.0, len(self._samples) / self._n)

    def add_sample(self, raw_deg: float | None) -> CalibrationState:
        """Feeds one raw angle (None = unavailable). No-op once COMPLETE/REJECTED."""
        if self._state != CalibrationState.COLLECTING:
            return self._state
        if raw_deg is None or not isfinite(raw_deg):
            self._samples.clear()
            return self._state
        self._samples.append(float(raw_deg))
        if len(self._samples) < self._n:
            return self._state

        q1, med, q3 = np.percentile(self._samples, [25.0, 50.0, 75.0])
        iqr = float(q3 - q1)
        if iqr > self._max_iqr:
            self._state = CalibrationState.REJECTED
        else:
            self._baseline = Baseline(float(med), iqr, self._n)
            self._state = CalibrationState.COMPLETE
        return self._state
