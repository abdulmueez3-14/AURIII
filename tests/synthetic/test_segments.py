"""Synthetic ground-truth tests for anatomical segment estimation."""

import numpy as np

from auri_cv.anatomy.segments import (
    ArmLaterality,
    estimate_arm_segments,
)
from auri_cv.frames.torso_frame import estimate_torso_frame
from tests.synthetic.fixtures import (
    make_neutral_standing_pose,
    make_right_arm_90deg_abduction_pose,
    make_right_arm_90deg_forward_flexion_pose,
    make_right_arm_180deg_overhead_pose,
)


def _get_torso_frame_from_fixture(pose):
    return estimate_torso_frame(
        pose.left_shoulder,
        pose.right_shoulder,
        pose.left_hip,
        pose.right_hip,
    )


def test_synthetic_neutral_pose_segment_directions():
    """In neutral pose, arms hang vertically down along the torso (-Z_T)."""
    pose = make_neutral_standing_pose()
    frame = _get_torso_frame_from_fixture(pose)

    # Right arm
    r_seg = estimate_arm_segments(
        shoulder=pose.right_shoulder,
        elbow=pose.right_elbow,
        wrist=pose.right_wrist,
        torso_frame=frame,
        laterality=ArmLaterality.RIGHT,
    )
    np.testing.assert_allclose(r_seg.upper_arm_dir_torso, [0.0, 0.0, -1.0], atol=1e-6)
    assert r_seg.forearm_dir_torso is not None
    np.testing.assert_allclose(r_seg.forearm_dir_torso, [0.0, 0.0, -1.0], atol=1e-6)
    assert np.isclose(r_seg.upper_arm_length_m, 0.30, atol=1e-6)
    assert np.isclose(r_seg.forearm_length_m, 0.25, atol=1e-6)
    assert np.isclose(r_seg.elbow_flexion_deg, 0.0, atol=1e-6)

    # Left arm
    l_seg = estimate_arm_segments(
        shoulder=pose.left_shoulder,
        elbow=pose.left_elbow,
        wrist=pose.left_wrist,
        torso_frame=frame,
        laterality=ArmLaterality.LEFT,
    )
    np.testing.assert_allclose(l_seg.upper_arm_dir_torso, [0.0, 0.0, -1.0], atol=1e-6)


def test_synthetic_90deg_abduction_segment_direction():
    """At 90 deg abduction, right upper arm points laterally to the right (+X_T)."""
    pose = make_right_arm_90deg_abduction_pose()
    frame = _get_torso_frame_from_fixture(pose)

    r_seg = estimate_arm_segments(
        shoulder=pose.right_shoulder,
        elbow=pose.right_elbow,
        wrist=pose.right_wrist,
        torso_frame=frame,
        laterality=ArmLaterality.RIGHT,
    )

    # In Torso Frame, +X_T points to subject's right
    np.testing.assert_allclose(r_seg.upper_arm_dir_torso, [1.0, 0.0, 0.0], atol=1e-6)
    assert r_seg.forearm_dir_torso is not None
    np.testing.assert_allclose(r_seg.forearm_dir_torso, [1.0, 0.0, 0.0], atol=1e-6)


def test_synthetic_180deg_overhead_segment_direction():
    """At 180 deg overhead, right upper arm points superiorly (+Z_T)."""
    pose = make_right_arm_180deg_overhead_pose()
    frame = _get_torso_frame_from_fixture(pose)

    r_seg = estimate_arm_segments(
        shoulder=pose.right_shoulder,
        elbow=pose.right_elbow,
        wrist=pose.right_wrist,
        torso_frame=frame,
        laterality=ArmLaterality.RIGHT,
    )

    # In Torso Frame, +Z_T points headward
    np.testing.assert_allclose(r_seg.upper_arm_dir_torso, [0.0, 0.0, 1.0], atol=1e-6)
    assert r_seg.forearm_dir_torso is not None
    np.testing.assert_allclose(r_seg.forearm_dir_torso, [0.0, 0.0, 1.0], atol=1e-6)


def test_synthetic_90deg_forward_flexion_segment_direction():
    """At 90 deg forward flexion, right upper arm points anteriorly (+Y_T)."""
    pose = make_right_arm_90deg_forward_flexion_pose()
    frame = _get_torso_frame_from_fixture(pose)

    r_seg = estimate_arm_segments(
        shoulder=pose.right_shoulder,
        elbow=pose.right_elbow,
        wrist=pose.right_wrist,
        torso_frame=frame,
        laterality=ArmLaterality.RIGHT,
    )

    # In Torso Frame, +Y_T points forward from chest
    np.testing.assert_allclose(r_seg.upper_arm_dir_torso, [0.0, 1.0, 0.0], atol=1e-6)
    assert r_seg.forearm_dir_torso is not None
    np.testing.assert_allclose(r_seg.forearm_dir_torso, [0.0, 1.0, 0.0], atol=1e-6)
