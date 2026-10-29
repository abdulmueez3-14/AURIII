"""Synthetic analytic tests for compensation metrics (rigid rotations of the fixture torso)."""

import numpy as np
import pytest

from auri_cv.frames.torso_frame import estimate_torso_frame
from auri_cv.movements import (
    CompensationWarning,
    compute_compensation,
    shoulder_height_above_hips,
)
from tests.synthetic.fixtures import make_neutral_standing_pose


def _rot(axis: str, deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


def _frame(pose, rot=None, right_shoulder_lift=0.0):
    pts = [pose.left_shoulder, pose.right_shoulder, pose.left_hip, pose.right_hip]
    pts = [np.array(p, dtype=float) for p in pts]
    pts[1] = pts[1] + np.array([0.0, 0.0, right_shoulder_lift])
    if rot is not None:
        pivot = 0.5 * (pts[2] + pts[3])
        pts = [pivot + rot @ (p - pivot) for p in pts]
    return estimate_torso_frame(*pts), pts[1]


@pytest.fixture
def baseline():
    pose = make_neutral_standing_pose()
    frame, rs = _frame(pose)
    return pose, frame, shoulder_height_above_hips(rs, frame)


def test_identity_gives_zero_metrics(baseline):
    pose, frame, h0 = baseline
    m = compute_compensation(frame, frame, pose.right_shoulder, h0)
    assert (
        m.lateral_lean_deg,
        m.forward_lean_deg,
        m.trunk_rotation_deg,
        m.shoulder_hiking_m,
    ) == pytest.approx((0.0, 0.0, 0.0, 0.0), abs=1e-6)
    assert m.warnings == ()


@pytest.mark.parametrize("deg", [5.0, 15.0, 30.0])
def test_lateral_lean_toward_subject_right(baseline, deg):
    pose, base, h0 = baseline
    frame, rs = _frame(pose, _rot("y", -deg))  # top of trunk toward world -X = subject right
    m = compute_compensation(frame, base, rs, h0)
    assert m.lateral_lean_deg == pytest.approx(deg, abs=1e-5)
    assert m.forward_lean_deg == pytest.approx(0.0, abs=1e-5)
    assert (CompensationWarning.LEAN_DETECTED in m.warnings) == (deg > 10.0)


@pytest.mark.parametrize("deg", [8.0, 20.0])
def test_forward_lean(baseline, deg):
    pose, base, h0 = baseline
    frame, rs = _frame(pose, _rot("x", deg))
    m = compute_compensation(frame, base, rs, h0)
    assert m.forward_lean_deg == pytest.approx(deg, abs=1e-5)
    assert m.lateral_lean_deg == pytest.approx(0.0, abs=1e-5)
    assert (CompensationWarning.LEAN_DETECTED in m.warnings) == (deg > 10.0)


@pytest.mark.parametrize("deg", [-25.0, 10.0, 40.0])
def test_trunk_rotation_about_vertical(baseline, deg):
    pose, base, h0 = baseline
    frame, rs = _frame(pose, _rot("z", deg))
    m = compute_compensation(frame, base, rs, h0)
    assert m.trunk_rotation_deg == pytest.approx(deg, abs=1e-5)
    assert m.lateral_lean_deg == pytest.approx(0.0, abs=1e-5)


def test_lean_is_invariant_to_rigid_camera_rotation(baseline):
    """Rotating the entire scene (baseline and current) must not change the metrics."""
    pose, _, _ = baseline
    cam = _rot("x", 33.0) @ _rot("y", -20.0)
    base_cam, _ = _frame(pose, cam)
    cur_cam, _ = _frame(pose, cam @ _rot("y", -12.0))
    base, rs = _frame(pose)
    cur, _ = _frame(pose, _rot("y", -12.0))
    a = compute_compensation(cur, base, rs, 0.0)
    b = compute_compensation(cur_cam, base_cam, rs, 0.0)
    assert a.lateral_lean_deg == pytest.approx(b.lateral_lean_deg, abs=1e-5)
    assert a.lateral_lean_deg == pytest.approx(12.0, abs=1e-5)


@pytest.mark.parametrize("lift", [0.01, 0.025, 0.04, 0.08])
def test_shoulder_hiking(baseline, lift):
    pose, base, h0 = baseline
    frame, rs = _frame(pose, right_shoulder_lift=lift)
    m = compute_compensation(frame, base, rs, h0)
    assert m.shoulder_hiking_m == pytest.approx(lift, abs=2e-3)
    assert (CompensationWarning.HIKING_DETECTED in m.warnings) == (lift > 0.03)


def test_shoulder_drop_is_not_hiking(baseline):
    pose, base, h0 = baseline
    frame, rs = _frame(pose, right_shoulder_lift=-0.05)
    m = compute_compensation(frame, base, rs, h0)
    assert m.shoulder_hiking_m < 0.0 and CompensationWarning.HIKING_DETECTED not in m.warnings
