"""Verification tests for synthetic pose fixtures.

Validates that synthetic fixtures satisfy basic anatomical constraints,
positive segment lengths, non-degeneracy, and expected analytical angles.
"""

import numpy as np
import pytest

from tests.synthetic.fixtures import (
    make_neutral_standing_pose,
    make_right_arm_90deg_abduction_pose,
    make_right_arm_90deg_forward_flexion_pose,
    make_right_arm_180deg_overhead_pose,
)


@pytest.mark.parametrize(
    "pose_factory",
    [
        make_neutral_standing_pose,
        make_right_arm_90deg_abduction_pose,
        make_right_arm_90deg_forward_flexion_pose,
        make_right_arm_180deg_overhead_pose,
    ],
)
def test_synthetic_pose_anatomical_integrity(pose_factory):
    """Verify that all synthetic poses maintain valid segment lengths and symmetry."""
    pose = pose_factory()

    # Shoulder width must be 0.40 m
    shoulder_width = np.linalg.norm(pose.left_shoulder - pose.right_shoulder)
    assert np.isclose(shoulder_width, 0.40, atol=1e-5)

    # Hip width must be 0.30 m
    hip_width = np.linalg.norm(pose.left_hip - pose.right_hip)
    assert np.isclose(hip_width, 0.30, atol=1e-5)

    # Right upper arm length must be 0.30 m
    r_arm_len = np.linalg.norm(pose.right_elbow - pose.right_shoulder)
    assert np.isclose(r_arm_len, 0.30, atol=1e-5)

    # Right forearm length must be 0.25 m
    r_forearm_len = np.linalg.norm(pose.right_wrist - pose.right_elbow)
    assert np.isclose(r_forearm_len, 0.25, atol=1e-5)


def test_analytical_elevation_angle_verification():
    """Verify analytical elevation calculation directly on synthetic vectors."""
    downward_ref = np.array([0.0, 0.0, -1.0])

    # 1. Neutral pose: arm vector should be [0, 0, -1] -> 0 degrees
    neutral = make_neutral_standing_pose()
    v_arm_neutral = (neutral.right_elbow - neutral.right_shoulder) / np.linalg.norm(
        neutral.right_elbow - neutral.right_shoulder
    )
    cos_theta = np.clip(np.dot(downward_ref, v_arm_neutral), -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(cos_theta))
    assert np.isclose(theta_deg, neutral.expected_right_elevation_deg, atol=1e-5)

    # 2. 90 deg Abduction: arm vector along -X -> 90 degrees
    abd = make_right_arm_90deg_abduction_pose()
    v_arm_abd = (abd.right_elbow - abd.right_shoulder) / np.linalg.norm(
        abd.right_elbow - abd.right_shoulder
    )
    cos_theta = np.clip(np.dot(downward_ref, v_arm_abd), -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(cos_theta))
    assert np.isclose(theta_deg, abd.expected_right_elevation_deg, atol=1e-5)

    # 3. 180 deg Overhead: arm vector along +Z -> 180 degrees
    overhead = make_right_arm_180deg_overhead_pose()
    v_arm_overhead = (overhead.right_elbow - overhead.right_shoulder) / np.linalg.norm(
        overhead.right_elbow - overhead.right_shoulder
    )
    cos_theta = np.clip(np.dot(downward_ref, v_arm_overhead), -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(cos_theta))
    assert np.isclose(theta_deg, overhead.expected_right_elevation_deg, atol=1e-5)
