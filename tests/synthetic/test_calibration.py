"""Synthetic tests for baseline calibration, neutral zone and the calibrated channel."""

import numpy as np
import pytest

from auri_cv.calibration import (
    BaselineCalibrator,
    CalibratedChannel,
    CalibrationState,
    ChannelStatus,
    NeutralZone,
    NeutralZoneState,
)
from auri_cv.quality.landmark_quality import QualityStatus

VALID = QualityStatus.VALID


def test_baseline_is_median_with_known_iqr():
    cal = BaselineCalibrator(n_frames=5)
    for v in (10.0, 11.0, 12.0, 13.0, 14.0):
        cal.add_sample(v)
    assert cal.state == CalibrationState.COMPLETE
    assert cal.baseline is not None
    assert cal.baseline.theta0_deg == 12.0
    assert cal.baseline.iqr_deg == 2.0


def test_median_rejects_outlier():
    cal = BaselineCalibrator(n_frames=5, max_iqr_deg=3.0)
    for v in (10.0, 10.0, 10.0, 10.0, 40.0):
        cal.add_sample(v)
    assert cal.baseline is not None
    assert cal.baseline.theta0_deg == 10.0


def test_calibration_rejected_when_subject_moves():
    cal = BaselineCalibrator(n_frames=10, max_iqr_deg=3.0)
    for v in np.linspace(0.0, 20.0, 10):
        cal.add_sample(float(v))
    assert cal.state == CalibrationState.REJECTED
    assert cal.baseline is None
    cal.reset()
    assert cal.state == CalibrationState.COLLECTING


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_gap_restarts_collection(bad):
    cal = BaselineCalibrator(n_frames=4)
    for v in (5.0, 5.0, 5.0):
        cal.add_sample(v)
    cal.add_sample(bad)
    assert cal.progress == 0.0
    for v in (6.0, 6.0, 6.0, 6.0):
        cal.add_sample(v)
    assert cal.baseline is not None and cal.baseline.theta0_deg == 6.0


def test_neutral_zone_hysteresis_no_flicker():
    nz = NeutralZone(3.0, 5.0)
    assert nz.update(4.9).display_deg == 0.0  # below exit: still neutral
    r = nz.update(5.0)
    assert r.state == NeutralZoneState.ACTIVE_EXCURSION and r.display_deg == 5.0
    # between enter and exit: stays active, shows the real value
    assert nz.update(4.0).display_deg == 4.0
    assert nz.update(3.1).display_deg == 3.1
    assert nz.update(3.0).display_deg == 0.0  # re-enter at <= enter
    assert nz.update(-4.9).display_deg == 0.0  # symmetric, negative side
    assert nz.update(-6.0).display_deg == -6.0


def test_neutral_zone_jitter_around_exit_does_not_flicker():
    nz = NeutralZone(3.0, 5.0)
    out = [nz.update(v).display_deg for v in (5.5, 4.5, 5.2, 4.1, 5.4)]
    assert 0.0 not in out


def test_neutral_zone_unavailable_is_never_zero():
    r = NeutralZone().update(None)
    assert r.state == NeutralZoneState.UNAVAILABLE and r.display_deg is None
    assert NeutralZone().update(float("nan")).display_deg is None


def test_neutral_zone_invalid_thresholds():
    with pytest.raises(ValueError):
        NeutralZone(5.0, 3.0)


def _calibrated(theta0: float = 20.0, n: int = 5) -> CalibratedChannel:
    ch = CalibratedChannel(BaselineCalibrator(n_frames=n))
    for _ in range(n):
        ch.update(theta0, VALID)
    return ch


def test_channel_relative_and_display_and_raw_preserved():
    ch = _calibrated(20.0)
    r = ch.update(21.0, VALID)  # +1 deg: inside the deadband
    assert r.status == ChannelStatus.OK
    assert r.raw_deg == 21.0 and r.relative_deg == pytest.approx(1.0) and r.display_deg == 0.0
    r = ch.update(50.0, VALID)  # +30 deg
    assert r.relative_deg == pytest.approx(30.0) and r.display_deg == pytest.approx(30.0)


def test_channel_calibrating_shows_no_angle():
    ch = CalibratedChannel(BaselineCalibrator(n_frames=5))
    r = ch.update(20.0, VALID)
    assert r.status == ChannelStatus.CALIBRATING and r.display_deg is None


def test_channel_calibration_rejected():
    ch = CalibratedChannel(BaselineCalibrator(n_frames=5, max_iqr_deg=1.0))
    reading = None
    for v in (0.0, 10.0, 20.0, 30.0, 40.0):
        reading = ch.update(v, VALID)
    assert reading is not None and reading.status == ChannelStatus.CALIBRATION_REJECTED


@pytest.mark.parametrize("q", [QualityStatus.UNAVAILABLE, QualityStatus.RECALIBRATION_REQUIRED])
def test_channel_bad_quality_never_displays_zero(q):
    ch = _calibrated()
    r = ch.update(20.0, q)  # raw looks like baseline, but quality says no
    assert r.display_deg is None and r.relative_deg is None and r.raw_deg is None
    assert r.status != ChannelStatus.OK


def test_channel_missing_raw_is_unavailable_then_latches_recalibration():
    ch = CalibratedChannel(BaselineCalibrator(n_frames=5), max_gap_frames=3)
    for _ in range(5):
        ch.update(20.0, VALID)
    for _ in range(3):
        assert ch.update(None, VALID).status == ChannelStatus.UNAVAILABLE
    assert ch.update(None, VALID).status == ChannelStatus.RECALIBRATION_REQUIRED
    # Latched: valid data no longer reports OK until recalibrated
    assert ch.update(20.0, VALID).status == ChannelStatus.RECALIBRATION_REQUIRED
    ch.recalibrate()
    assert ch.update(20.0, VALID).status == ChannelStatus.CALIBRATING


def test_channel_short_gap_recovers():
    ch = _calibrated()
    assert ch.update(None, VALID).status == ChannelStatus.UNAVAILABLE
    assert ch.update(20.5, VALID).status == ChannelStatus.OK


def test_channel_quality_recalibration_latches():
    ch = _calibrated()
    assert (
        ch.update(25.0, QualityStatus.RECALIBRATION_REQUIRED).status
        == ChannelStatus.RECALIBRATION_REQUIRED
    )
    assert ch.update(25.0, VALID).status == ChannelStatus.RECALIBRATION_REQUIRED
