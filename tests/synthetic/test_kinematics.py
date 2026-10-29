"""Synthetic analytical tests for the Phase 5 joint-angle solver."""

import numpy as np
import pytest

from auri_cv.anatomy.segments import ArmLaterality, estimate_arm_segments
from auri_cv.frames.torso_frame import estimate_torso_frame
from auri_cv.geometry.exceptions import DegenerateVectorError, NonFiniteVectorError
from auri_cv.kinematics import (
    calculate_coronal_abduction,
    calculate_humerothoracic_elevation,
    calculate_plane_of_elevation,
    calculate_sagittal_flexion,
    solve_arm_kinematics,
)
from tests.synthetic.fixtures import (
    make_neutral_standing_pose,
    make_right_arm_90deg_abduction_pose,
    make_right_arm_90deg_forward_flexion_pose,
    make_right_arm_180deg_overhead_pose,
)


def _dir(elev_deg: float, plane_deg: float, lateral_sign: float = 1.0) -> np.ndarray:
    """Unit vector in Torso Frame: elevation from -Z_T, plane from lateral toward anterior."""
    e, p = np.radians(elev_deg), np.radians(plane_deg)
    return np.array([lateral_sign * np.sin(e) * np.cos(p), np.sin(e) * np.sin(p), -np.cos(e)])


@pytest.mark.parametrize("elev", [0.0, 30.0, 45.0, 90.0, 135.0, 180.0])
@pytest.mark.parametrize("plane", [0.0, 45.0, 90.0])
def test_elevation_is_plane_independent(elev, plane):
    assert calculate_humerothoracic_elevation(_dir(elev, plane)) == pytest.approx(elev, abs=1e-6)


@pytest.mark.parametrize("elev", [30.0, 60.0])
def test_pure_abduction_and_flexion(elev):
    assert calculate_coronal_abduction(_dir(elev, 0.0)) == pytest.approx(elev, abs=1e-6)
    assert calculate_sagittal_flexion(_dir(elev, 0.0)) == pytest.approx(0.0, abs=1e-6)
    assert calculate_sagittal_flexion(_dir(elev, 90.0)) == pytest.approx(elev, abs=1e-6)
    assert calculate_coronal_abduction(_dir(elev, 90.0)) == pytest.approx(0.0, abs=1e-6)


def test_projection_undefined_when_arm_leaves_plane():
    lateral = _dir(90.0, 0.0)  # no sagittal component: flexion must not be a noisy 0/180
    with pytest.raises(DegenerateVectorError):
        calculate_sagittal_flexion(lateral)
    forward = _dir(90.0, 90.0)  # no coronal component
    with pytest.raises(DegenerateVectorError):
        calculate_coronal_abduction(forward)


def test_extension_and_adduction_are_negative():
    assert calculate_sagittal_flexion(_dir(30.0, -90.0)) == pytest.approx(-30.0, abs=1e-6)
    assert calculate_coronal_abduction(_dir(20.0, 0.0, lateral_sign=-1.0)) == pytest.approx(
        -20.0, abs=1e-6
    )


def test_scaption_plane_and_elevation():
    v = _dir(90.0, 45.0)  # VALIDATION_PLAN case 4
    assert calculate_humerothoracic_elevation(v) == pytest.approx(90.0, abs=1e-6)
    assert calculate_plane_of_elevation(v) == pytest.approx(45.0, abs=1e-6)


def test_laterality_symmetry():
    right = _dir(60.0, 30.0, lateral_sign=1.0)
    left = _dir(60.0, 30.0, lateral_sign=-1.0)  # mirror image across the sagittal plane
    for fn in (calculate_coronal_abduction, calculate_plane_of_elevation):
        assert fn(right, ArmLaterality.RIGHT) == pytest.approx(
            fn(left, ArmLaterality.LEFT), abs=1e-9
        )
    assert calculate_sagittal_flexion(right) == pytest.approx(calculate_sagittal_flexion(left))


def test_plane_of_elevation_undefined_near_vertical():
    with pytest.raises(DegenerateVectorError):
        calculate_plane_of_elevation(np.array([0.0, 0.0, -1.0]))
    with pytest.raises(DegenerateVectorError):
        calculate_plane_of_elevation(np.array([0.0, 0.0, 1.0]))


def test_degenerate_and_nonfinite_inputs_rejected():
    for fn in (
        calculate_humerothoracic_elevation,
        calculate_sagittal_flexion,
        calculate_coronal_abduction,
    ):
        with pytest.raises(DegenerateVectorError):
            fn(np.zeros(3))
        with pytest.raises(NonFiniteVectorError):
            fn(np.array([np.nan, 0.0, -1.0]))


def test_unnormalized_input_and_radians():
    v = 3.0 * _dir(90.0, 0.0)
    assert calculate_humerothoracic_elevation(v, in_degrees=False) == pytest.approx(np.pi / 2)


def _solve(pose, timestamp_ms=0.0):
    frame = estimate_torso_frame(
        pose.left_shoulder, pose.right_shoulder, pose.left_hip, pose.right_hip
    )
    seg = estimate_arm_segments(
        pose.right_shoulder, pose.right_elbow, pose.right_wrist, frame, ArmLaterality.RIGHT
    )
    return solve_arm_kinematics(seg, timestamp_ms)


def test_solver_on_fixture_poses():
    k = _solve(make_neutral_standing_pose(), 10.0)
    assert k.elevation_deg == pytest.approx(0.0, abs=1e-5)
    assert k.plane_of_elevation_deg is None
    assert (k.flexion_deg, k.abduction_deg) == pytest.approx((0.0, 0.0), abs=1e-5)
    assert k.timestamp_ms == 10.0

    k = _solve(make_right_arm_90deg_abduction_pose())
    assert (k.elevation_deg, k.abduction_deg) == pytest.approx((90.0, 90.0), abs=1e-5)
    assert k.flexion_deg is None
    assert k.plane_of_elevation_deg == pytest.approx(0.0, abs=1e-5)

    k = _solve(make_right_arm_90deg_forward_flexion_pose())
    assert (k.elevation_deg, k.flexion_deg) == pytest.approx((90.0, 90.0), abs=1e-5)
    assert k.abduction_deg is None
    assert k.plane_of_elevation_deg == pytest.approx(90.0, abs=1e-5)

    k = _solve(make_right_arm_180deg_overhead_pose())
    assert k.elevation_deg == pytest.approx(180.0, abs=1e-5)
    assert k.plane_of_elevation_deg is None
