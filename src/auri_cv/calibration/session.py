"""Calibrated angle channel: raw -> relative excursion -> display, with quality gating."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Final

from auri_cv.calibration.calibrator import Baseline, BaselineCalibrator, CalibrationState
from auri_cv.calibration.neutral_zone import NeutralZone, NeutralZoneState
from auri_cv.quality.landmark_quality import QualityStatus

DEFAULT_MAX_GAP_FRAMES: Final[int] = 15


class ChannelStatus(StrEnum):
    CALIBRATING = "CALIBRATING"
    CALIBRATION_REJECTED = "CALIBRATION_REJECTED"
    OK = "OK"
    UNAVAILABLE = "UNAVAILABLE"
    RECALIBRATION_REQUIRED = "RECALIBRATION_REQUIRED"


@dataclass(frozen=True)
class ChannelReading:
    """Raw, relative and display values are kept side by side; none overwrites another."""

    status: ChannelStatus
    raw_deg: float | None
    relative_deg: float | None  # raw - theta_0, never clipped by the deadband
    display_deg: float | None  # deadbanded; None whenever status is not OK
    neutral_state: NeutralZoneState | None


class CalibratedChannel:
    """One angle channel (e.g. elevation) through calibration and the neutral zone.

    Becomes RECALIBRATION_REQUIRED (latched until `recalibrate()`) when the quality gate
    says so, or when tracking is lost for more than `max_gap_frames` consecutive frames.
    """

    def __init__(
        self,
        calibrator: BaselineCalibrator | None = None,
        neutral_zone: NeutralZone | None = None,
        max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES,
    ) -> None:
        self._cal = calibrator or BaselineCalibrator()
        self._nz = neutral_zone or NeutralZone()
        self._max_gap = max_gap_frames
        self._gap = 0
        self._recal_required = False

    @property
    def baseline(self) -> Baseline | None:
        return self._cal.baseline

    def recalibrate(self) -> None:
        self._cal.reset()
        self._nz.reset()
        self._gap = 0
        self._recal_required = False

    def update(
        self, raw_deg: float | None, quality: QualityStatus, pose_gated: bool = False
    ) -> ChannelReading:
        """`pose_gated`: raw is None because the pose is outside the movement's definition while
        tracking is fine; this is not a tracking gap and never latches recalibration."""
        usable = (
            raw_deg is not None
            and isfinite(raw_deg)
            and quality in (QualityStatus.VALID, QualityStatus.VALID_WITH_WARNING)
        )
        raw = raw_deg if usable else None
        calibrated = self._cal.state == CalibrationState.COMPLETE

        if calibrated:
            if quality == QualityStatus.RECALIBRATION_REQUIRED:
                self._recal_required = True
            tracking_ok = usable or (pose_gated and quality != QualityStatus.UNAVAILABLE)
            self._gap = 0 if tracking_ok else self._gap + 1
            if self._gap > self._max_gap:
                self._recal_required = True
        else:
            state = self._cal.add_sample(raw)
            if state == CalibrationState.REJECTED:
                return ChannelReading(ChannelStatus.CALIBRATION_REJECTED, raw, None, None, None)
            if state == CalibrationState.COLLECTING:
                return ChannelReading(ChannelStatus.CALIBRATING, raw, None, None, None)
            # This frame completed calibration; fall through (relative excursion ~0).

        if self._recal_required:
            return ChannelReading(ChannelStatus.RECALIBRATION_REQUIRED, raw, None, None, None)
        baseline = self._cal.baseline
        if raw is None or baseline is None:
            return ChannelReading(ChannelStatus.UNAVAILABLE, None, None, None, None)

        rel = raw - baseline.theta0_deg
        disp = self._nz.update(rel)
        return ChannelReading(ChannelStatus.OK, raw, rel, disp.display_deg, disp.state)
