"""Synthetic time-series tests for filters and the repetition FSM."""

import pytest

from auri_cv.filtering import ExponentialFilter, MovingAverageFilter
from auri_cv.movements import RepetitionFSM, RepState

DT = 1.0 / 30.0


def _rep_profile(peak: float, rate_dps: float = 60.0, hold_frames: int = 3) -> list[float]:
    """Triangle excursion 0 -> peak -> 0 at rate_dps with a short hold at the top."""
    step = rate_dps * DT
    up = [min(peak, i * step) for i in range(int(peak / step) + 2)]
    return [0.0] * 5 + up + [peak] * hold_frames + up[::-1] + [0.0] * 5


def _run(fsm: RepetitionFSM, series, t0: float = 0.0):
    states = []
    for i, x in enumerate(series):
        states.append(fsm.update(x, t0 + i * DT))
    return states


def test_sma_known_values_and_gap_reset():
    f = MovingAverageFilter(window=3)
    assert [f.update(v) for v in (3.0, 6.0, 9.0, 12.0)] == [3.0, 4.5, 6.0, 9.0]
    assert f.update(None) is None
    assert f.update(100.0) == 100.0  # no bridging across the gap


def test_ema_known_values_and_reseed():
    f = ExponentialFilter(alpha=0.5)
    assert f.update(10.0) == 10.0
    assert f.update(20.0) == 15.0
    assert f.update(float("nan")) is None
    assert f.update(40.0) == 40.0


def test_filter_parameter_validation():
    with pytest.raises(ValueError):
        MovingAverageFilter(0)
    with pytest.raises(ValueError):
        ExponentialFilter(0.0)


def test_single_valid_repetition_metrics():
    fsm = RepetitionFSM()
    states = _run(fsm, _rep_profile(90.0))
    assert fsm.rep_count == 1
    assert RepState.PEAK_HOLD in states and RepState.LOWERING in states
    assert RepState.COMPLETE in states
    assert states[-1] == RepState.IDLE
    rep = fsm.repetitions[0]
    assert rep.index == 1 and rep.peak_deg == pytest.approx(90.0)
    # ramp 15 -> 90 at 60 deg/s takes ~1.25 s; whole rep is roughly twice that plus hold
    assert rep.time_to_peak_s == pytest.approx(1.25, abs=0.1)
    assert rep.duration_s > 2 * rep.time_to_peak_s - 0.2


def test_multiple_repetitions_counted():
    fsm = RepetitionFSM()
    _run(fsm, _rep_profile(80.0) + _rep_profile(70.0) + _rep_profile(100.0))
    assert fsm.rep_count == 3
    assert [r.peak_deg for r in fsm.repetitions] == pytest.approx([80.0, 70.0, 100.0])


def test_shallow_movement_below_min_peak_not_counted():
    fsm = RepetitionFSM()
    _run(fsm, _rep_profile(35.0))
    assert fsm.rep_count == 0 and fsm.aborted_count == 1 and fsm.state == RepState.IDLE


def test_noise_below_start_threshold_ignored():
    fsm = RepetitionFSM()
    _run(fsm, [0.0, 4.0, -3.0, 10.0, 14.9, 2.0] * 20)
    assert fsm.rep_count == 0 and fsm.state == RepState.IDLE


def test_long_tracking_loss_mid_rep_invalidates():
    fsm = RepetitionFSM()
    prof = _rep_profile(90.0)
    peak_idx = prof.index(90.0)
    series = prof[:peak_idx] + [None] * 40 + prof[peak_idx:]  # 1.33 s gap
    _run(fsm, series)
    assert fsm.invalidated_count == 1
    assert fsm.rep_count == 0  # the tail of the profile alone is not a full repetition


def test_short_tracking_loss_does_not_invalidate():
    fsm = RepetitionFSM()
    prof = _rep_profile(90.0)
    peak_idx = prof.index(90.0)
    series = prof[:peak_idx] + [None] * 10 + prof[peak_idx:]  # 0.33 s gap
    _run(fsm, series)
    assert fsm.invalidated_count == 0 and fsm.rep_count == 1


def test_unavailable_while_idle_is_harmless():
    fsm = RepetitionFSM()
    _run(fsm, [None] * 100)
    assert fsm.state == RepState.IDLE and fsm.invalidated_count == 0


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        RepetitionFSM(start_deg=50.0, min_peak_deg=45.0)


def test_signal_already_elevated_at_start_is_not_a_repetition():
    fsm = RepetitionFSM()
    _run(fsm, [80.0] * 10 + [60.0, 40.0, 20.0, 5.0, 0.0])
    assert fsm.rep_count == 0
