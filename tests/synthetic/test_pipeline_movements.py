"""End-to-end: explicit movement selection drives movement-specific metrics through the pipeline."""

import numpy as np
import pytest

from auri_cv.calibration import ChannelStatus
from auri_cv.movements import (
    CameraAxes,
    CompensationWarning,
    MotionStatus,
    ShoulderMovement,
)
from auri_cv.pipeline import FrameResult, ShoulderAnalysisPipeline, TorsoMode
from tests.synthetic.pose_builder import make_observation

M = ShoulderMovement
FIXTURE_CAMERA = CameraAxes(lateral=(1.0, 0.0, 0.0), depth=(0.0, 1.0, 0.0))
DT = 1000.0 / 30.0
REST, HOLD = 40, 40


def _pipe(movement, arm="right", **kw) -> ShoulderAnalysisPipeline:
    return ShoulderAnalysisPipeline(
        laterality=arm, movement=movement, camera_axes=FIXTURE_CAMERA, **kw
    )


def _drive(pipe, start, goal, arm="right", yaw=0.0, **kw) -> list[FrameResult]:
    """REST frames at `start`, then HOLD frames at `goal`. Each is (elev, plane, forearm|None)."""
    out = []
    for i in range(REST + HOLD):
        e, p, f = start if i < REST else goal
        kw_i = {k: (v[i] if isinstance(v, list) else v) for k, v in kw.items()}
        obs = make_observation(
            e, i * DT, i, plane_deg=p, arm=arm, forearm_dir_torso=f, body_yaw_deg=yaw, **kw_i
        )
        out.append(pipe.process(obs))
    return out


def _rot(er_deg: float, arm: str = "right") -> tuple[float, float, float]:
    t, s = np.radians(er_deg), (1.0 if arm == "right" else -1.0)
    return (s * float(np.sin(t)), float(np.cos(t)), 0.0)


NEUTRAL = (0.0, 0.0, None)

# movement, start, goal, yaw (body turn: 90 = side-on), expected angle at goal, expected status
CASES = [
    (M.FLEXION, NEUTRAL, (90.0, 90.0, None), 90.0, 90.0, MotionStatus.OK),
    (M.EXTENSION, NEUTRAL, (35.0, -90.0, None), 90.0, 35.0, MotionStatus.OK),
    (M.ABDUCTION, NEUTRAL, (90.0, 0.0, None), 0.0, 90.0, MotionStatus.OK),
    (M.ADDUCTION, NEUTRAL, (20.0, 180.0, None), 0.0, 20.0, MotionStatus.OK),
    (M.SCAPTION, NEUTRAL, (110.0, 30.0, None), 45.0, 110.0, MotionStatus.OK),
    (M.HORIZONTAL_ADDUCTION, (90.0, 0.0, None), (90.0, 60.0, None), 0.0, 60.0, MotionStatus.OK),
    (M.HORIZONTAL_ABDUCTION, (90.0, 0.0, None), (90.0, -30.0, None), 0.0, 30.0, MotionStatus.OK),
]


@pytest.mark.parametrize("arm", ["right", "left"])
@pytest.mark.parametrize("movement,start,goal,yaw,expected,status", CASES)
def test_each_movement_end_to_end_both_arms(movement, start, goal, yaw, expected, status, arm):
    res = _drive(_pipe(movement, arm), start, goal, arm=arm, yaw=yaw)
    last = res[-1]
    assert last.motion is not None and last.motion.movement == movement
    assert last.motion.angle_deg == pytest.approx(expected, abs=1e-3)
    assert last.motion.status == status
    # The calibrated channel tracks THIS movement's metric, relative to its own baseline.
    assert last.channel.status == ChannelStatus.OK
    assert last.channel.relative_deg == pytest.approx(expected, abs=1.0)
    assert last.channel.display_deg == pytest.approx(expected, abs=1.0)
    # resting frames display exactly 0 in the deadband
    rest = [r for r in res[REST - 5 : REST] if r.channel.status == ChannelStatus.OK]
    assert rest and all(r.channel.display_deg == 0.0 for r in rest)


@pytest.mark.parametrize("arm", ["right", "left"])
@pytest.mark.parametrize("movement,sign", [(M.EXTERNAL_ROTATION, 1.0), (M.INTERNAL_ROTATION, -1.0)])
def test_rotation_end_to_end_is_flagged_low_confidence(movement, sign, arm):
    start = (0.0, 0.0, _rot(0.0, arm))
    goal = (0.0, 0.0, _rot(40.0 * sign, arm))
    res = _drive(_pipe(movement, arm), start, goal, arm=arm)
    last = res[-1]
    assert last.motion is not None and last.motion.angle_deg == pytest.approx(40.0, abs=1e-3)
    assert last.motion.status == MotionStatus.LOW_CONFIDENCE and last.motion.confidence <= 0.5
    assert last.channel.relative_deg == pytest.approx(40.0, abs=1.0)
    # rotation did not change any elevation metric
    assert last.raw is not None and last.raw.elevation_deg == pytest.approx(0.0, abs=1e-3)


def test_rotation_without_wrist_is_unavailable_not_guessed():
    pipe = _pipe(M.EXTERNAL_ROTATION)
    res = _drive(pipe, (0.0, 0.0, _rot(0.0)), (0.0, 0.0, _rot(30.0)))
    assert res[-1].motion.angle_deg is not None  # sanity with wrist
    # straight elbow (no 90 deg protocol): same arm, forearm along the upper arm
    res = _drive(_pipe(M.EXTERNAL_ROTATION), NEUTRAL, NEUTRAL)
    assert all(r.motion is not None and r.motion.angle_deg is None for r in res)
    assert all(r.channel.display_deg is None for r in res)


def test_wrong_exercise_selected_withholds_number_and_suggests_the_real_one():
    pipe = _pipe(M.FLEXION)
    res = _drive(pipe, NEUTRAL, (90.0, 0.0, None), yaw=90.0)  # patient actually abducts
    last = res[-1]
    assert last.motion is not None and last.motion.status == MotionStatus.NOT_MEASURABLE
    assert last.motion.angle_deg is None and last.channel.display_deg is None
    assert last.suggestion is not None and last.suggestion.movement == M.ABDUCTION
    # Pose-gating is not a tracking loss: it must not latch recalibration (HOLD > gap limit).
    assert HOLD > 15 and last.channel.status != ChannelStatus.RECALIBRATION_REQUIRED


def test_incorrect_camera_placement_lowers_confidence_but_keeps_geometry():
    good = _drive(_pipe(M.ABDUCTION), NEUTRAL, (90.0, 0.0, None), yaw=0.0)[-1].motion
    bad = _drive(_pipe(M.ABDUCTION), NEUTRAL, (90.0, 0.0, None), yaw=90.0)[-1].motion
    assert good is not None and bad is not None
    assert good.status == MotionStatus.OK and bad.status != MotionStatus.OK
    assert bad.confidence < good.confidence
    assert any("view" in r for r in bad.reasons) and bad.view_yaw_deg == pytest.approx(90.0)


def test_missing_landmarks_never_report_zero():
    pipe = _pipe(M.ABDUCTION)
    _drive(pipe, NEUTRAL, NEUTRAL)
    r = pipe.process(make_observation(0.0, 9e6, 999, drop=True))
    assert r.motion is None and r.channel.display_deg is None
    assert r.channel.status != ChannelStatus.OK


def test_noisy_landmarks_keep_angle_close_to_truth():
    rng = np.random.default_rng(11)
    res = _drive(_pipe(M.ABDUCTION), NEUTRAL, (90.0, 0.0, None), noise_m=0.003, rng=rng)
    vals = [r.motion.angle_deg for r in res[-20:] if r.motion and r.motion.angle_deg is not None]
    assert len(vals) >= 15 and abs(float(np.mean(vals)) - 90.0) < 4.0


def test_trunk_lean_is_separated_from_shoulder_movement_with_hips():
    """Trunk leans 15 deg while the arm stays 60 deg abducted RELATIVE TO THE TRUNK."""
    pipe = _pipe(M.ABDUCTION)
    res = _drive(
        pipe,
        NEUTRAL,
        (60.0, 0.0, None),
        trunk_roll_deg=[0.0] * REST + [15.0] * HOLD,
    )
    last = res[-1]
    assert last.motion is not None and last.motion.angle_deg == pytest.approx(60.0, abs=1e-3)
    assert last.compensation is not None
    assert last.compensation.lateral_lean_deg == pytest.approx(15.0, abs=1e-3)
    assert CompensationWarning.LEAN_DETECTED in last.compensation.warnings


def test_shoulders_only_mode_cannot_separate_trunk_lean():
    pipe = ShoulderAnalysisPipeline(
        movement=M.ABDUCTION,
        camera_axes=FIXTURE_CAMERA,
        torso_mode=TorsoMode.SHOULDERS_ONLY_ASSUMED_VERTICAL,
        up_world=(0.0, 0.0, 1.0),
    )
    res = _drive(pipe, NEUTRAL, (60.0, 0.0, None), trunk_roll_deg=[0.0] * REST + [15.0] * HOLD)
    m = res[-1].motion
    assert m is not None and m.angle_deg == pytest.approx(
        45.0, abs=0.5
    )  # 60 - 15: leaning toward the arm lowers it
    assert res[-1].compensation is None  # and no lean metric is offered


def test_legacy_behaviour_without_selected_movement_is_unchanged():
    res = _drive(ShoulderAnalysisPipeline(camera_axes=FIXTURE_CAMERA), NEUTRAL, (90.0, 90.0, None))
    last = res[-1]
    assert last.motion is None and last.suggestion is None
    assert last.channel.relative_deg == pytest.approx(90.0, abs=1.0)  # elevation tracking


def test_all_movements_mode_reports_each_definition_independently():
    pipe = ShoulderAnalysisPipeline(camera_axes=FIXTURE_CAMERA, compute_all_movements=True)
    res = _drive(pipe, NEUTRAL, (90.0, 0.0, None))  # arm abducted to 90 deg
    am = res[-1].all_motions
    assert am is not None and set(am) == set(M)
    assert am[M.ABDUCTION].angle_deg == pytest.approx(90.0, abs=1e-3)
    for m in (M.FLEXION, M.EXTENSION, M.SCAPTION, M.EXTERNAL_ROTATION, M.INTERNAL_ROTATION):
        assert am[m].angle_deg is None, m  # not confused with abduction
    assert am[M.HORIZONTAL_ADDUCTION].angle_deg == pytest.approx(0.0, abs=1e-3)
    assert res[-1].motion is None  # no channel movement was selected
    assert res[-1].channel.relative_deg == pytest.approx(90.0, abs=1.0)  # legacy elevation channel


def test_all_movements_off_by_default_and_none_without_segment():
    res = _drive(ShoulderAnalysisPipeline(camera_axes=FIXTURE_CAMERA), NEUTRAL, NEUTRAL)
    assert all(r.all_motions is None for r in res)
    pipe = ShoulderAnalysisPipeline(camera_axes=FIXTURE_CAMERA, compute_all_movements=True)
    assert pipe.process(make_observation(0.0, 0.0, 0, drop=True)).all_motions is None
