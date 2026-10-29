"""End-to-end synthetic tests of ShoulderAnalysisPipeline (no camera)."""

import numpy as np
import pytest

from auri_cv.calibration import ChannelStatus
from auri_cv.movements import CompensationWarning
from auri_cv.pipeline import FrameResult, ShoulderAnalysisPipeline
from tests.synthetic.pose_builder import elevation_profile, make_observation

FPS = 30.0
DT_MS = 1000.0 / FPS
REST = 40  # > 30 calibration frames


def _run(elevations, **kw) -> tuple[ShoulderAnalysisPipeline, list[FrameResult]]:
    pipe = ShoulderAnalysisPipeline()
    out = []
    for i, e in enumerate(elevations):
        kw_i = {k: (v[i] if isinstance(v, list) else v) for k, v in kw.items()}
        if e is None:
            obs = make_observation(0.0, i * DT_MS, i, drop=True)
        else:
            obs = make_observation(e, i * DT_MS, i, **kw_i)
        out.append(pipe.process(obs))
    return pipe, out


def test_calibration_then_resting_display_is_exactly_zero():
    _, res = _run([0.0] * 60)
    assert res[0].channel.status == ChannelStatus.CALIBRATING
    assert res[0].channel.display_deg is None  # never a fake 0 before a baseline exists
    ok = [r for r in res if r.channel.status == ChannelStatus.OK]
    assert ok and all(r.channel.display_deg == 0.0 for r in ok)
    assert all(abs(r.channel.relative_deg) < 1e-4 for r in ok)


def test_two_clean_repetitions_counted_with_correct_peaks():
    seq = (
        [0.0] * REST + elevation_profile(90.0) + [0.0] * 10 + elevation_profile(120.0) + [0.0] * 10
    )
    pipe, res = _run(seq)
    assert pipe.fsm.rep_count == 2
    assert [r.peak_deg for r in pipe.fsm.repetitions] == pytest.approx([90.0, 120.0], abs=1.0)
    assert res[-1].rep_count == 2
    # raw solver output is untouched by smoothing/deadband and matches ground truth
    peak_frame = max(res, key=lambda r: r.raw.elevation_deg if r.raw else -1.0)
    assert peak_frame.raw is not None
    assert peak_frame.raw.elevation_deg == pytest.approx(120.0, abs=1e-4)
    assert peak_frame.raw.abduction_deg == pytest.approx(120.0, abs=1e-4)


def test_forward_flexion_plane_reported_in_raw():
    _, res = _run([0.0] * REST + [90.0] * 5, plane_deg=[0.0] * REST + [90.0] * 5)
    last = res[-1].raw
    assert last is not None
    assert last.flexion_deg == pytest.approx(90.0, abs=1e-4)
    assert last.abduction_deg is None
    assert last.plane_of_elevation_deg == pytest.approx(90.0, abs=1e-4)


def test_noisy_landmarks_still_stable_and_counted():
    rng = np.random.default_rng(7)
    seq = [0.0] * REST + elevation_profile(90.0) + [0.0] * 15
    pipe, res = _run(seq, noise_m=0.003, rng=rng)
    assert pipe.fsm.rep_count == 1
    resting = [r for r in res[REST - 5 : REST + 0] if r.channel.status == ChannelStatus.OK]
    assert all(r.channel.display_deg == 0.0 for r in resting)


def test_dropout_never_displays_zero_and_latches_recalibration():
    seq = [0.0] * REST + elevation_profile(90.0)[:30] + [None] * 60 + [30.0] * 20
    pipe, res = _run(seq)
    drop = res[REST + 30 : REST + 90]
    assert all(r.channel.display_deg is None for r in drop)
    assert all(r.channel.status != ChannelStatus.OK for r in drop)
    assert res[-1].channel.status == ChannelStatus.RECALIBRATION_REQUIRED
    assert pipe.fsm.rep_count == 0

    pipe.recalibrate()
    again = pipe.process(make_observation(0.0, 1e6, 999))
    assert again.channel.status == ChannelStatus.CALIBRATING


def test_trunk_lean_during_rep_flagged_and_hiking_flagged():
    n = REST + 20
    seq = [0.0] * REST + [60.0] * 20
    lean = [0.0] * REST + [18.0] * 20
    _, res = _run(seq, trunk_roll_deg=lean)
    comp = res[-1].compensation
    assert comp is not None
    assert comp.lateral_lean_deg == pytest.approx(18.0, abs=1e-3)
    assert CompensationWarning.LEAN_DETECTED in comp.warnings

    lift = [0.0] * REST + [0.05] * 20
    _, res = _run(seq, right_shoulder_lift_m=lift)
    comp = res[n - 1].compensation
    assert comp is not None and CompensationWarning.HIKING_DETECTED in comp.warnings
    assert comp.shoulder_hiking_m == pytest.approx(0.05, abs=3e-3)


def test_compensation_absent_before_calibration_and_when_unavailable():
    _, res = _run([0.0] * 10)
    assert all(r.compensation is None for r in res)


def test_low_visibility_is_unavailable_not_zero():
    pipe = ShoulderAnalysisPipeline()
    for i in range(REST):
        pipe.process(make_observation(0.0, i * DT_MS, i))
    r = pipe.process(make_observation(0.0, REST * DT_MS, REST, visibility=0.2))
    assert r.raw is None and r.channel.display_deg is None
    assert r.channel.status == ChannelStatus.UNAVAILABLE


# ---- shoulders-only (assumed vertical) mode -------------------------------------------
from auri_cv.pipeline import TorsoMode  # noqa: E402

UP_Z = (0.0, 0.0, 1.0)  # fixture frame is Z-up (MediaPipe world is Y-down)


def _shoulder_pipe() -> ShoulderAnalysisPipeline:
    return ShoulderAnalysisPipeline(
        torso_mode=TorsoMode.SHOULDERS_ONLY_ASSUMED_VERTICAL, up_world=UP_Z
    )


def test_hips_mode_is_unavailable_without_hips_but_shoulder_mode_works():
    full = ShoulderAnalysisPipeline()
    sho = _shoulder_pipe()
    for i in range(REST + 10):
        obs = make_observation(0.0 if i < REST else 60.0, i * DT_MS, i, drop_hips=True)
        r_full, r_sho = full.process(obs), sho.process(obs)
    assert r_full.raw is None and r_full.channel.display_deg is None
    assert r_sho.channel.status == ChannelStatus.OK
    assert r_sho.torso_mode == TorsoMode.SHOULDERS_ONLY_ASSUMED_VERTICAL
    assert r_sho.channel.relative_deg == pytest.approx(60.0, abs=0.5)  # type: ignore[arg-type]
    assert r_sho.compensation is None  # lean/hiking are not measurable without hips


def test_shoulder_mode_matches_hips_mode_when_trunk_upright():
    a, b = ShoulderAnalysisPipeline(), _shoulder_pipe()
    for e in (0.0, 30.0, 90.0, 140.0):
        obs = make_observation(e, 0.0, 0)
        ra, rb = a.process(obs), b.process(obs)
        assert ra.raw is not None and rb.raw is not None
        assert rb.raw.elevation_deg == pytest.approx(ra.raw.elevation_deg, abs=1e-6)
        assert rb.raw.abduction_deg == pytest.approx(ra.raw.abduction_deg, abs=1e-6)


def test_shoulder_mode_confounds_trunk_lean_with_arm_elevation():
    """Documented limitation: a leaning trunk changes the apparent elevation (arm fixed to it)."""
    pipe = _shoulder_pipe()
    r = pipe.process(make_observation(0.0, 0.0, 0, trunk_roll_deg=15.0))
    assert r.raw is not None
    # Arm hangs along the (leaning) trunk. Hips-mode would read 0 deg; assumed vertical reads ~15.
    assert r.raw.elevation_deg == pytest.approx(15.0, abs=0.5)
