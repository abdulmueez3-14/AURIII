"""Synthetic perturbation and invariance tests for the Torso Coordinate Frame."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from auri_cv.frames.torso_frame import estimate_torso_frame
from tests.synthetic.fixtures import (
    make_neutral_standing_pose,
    make_right_arm_90deg_abduction_pose,
    make_right_arm_90deg_forward_flexion_pose,
    make_right_arm_180deg_overhead_pose,
)


def _apply_rigid_transform(
    points: dict[str, np.ndarray],
    R: np.ndarray,
    t: np.ndarray,
) -> dict[str, np.ndarray]:
    """Applies global rotation R and translation t to a dictionary of 3D points."""
    transformed = {}
    for name, p in points.items():
        transformed[name] = np.dot(R, p) + t
    return transformed


@pytest.mark.parametrize(
    "pose_factory",
    [
        make_neutral_standing_pose,
        make_right_arm_90deg_abduction_pose,
        make_right_arm_90deg_forward_flexion_pose,
        make_right_arm_180deg_overhead_pose,
    ],
)
@pytest.mark.parametrize(
    ("yaw_deg", "pitch_deg", "roll_deg", "t_world"),
    [
        (0.0, 0.0, 0.0, np.array([1.5, -2.0, 0.5])),  # Pure translation
        (30.0, 0.0, 0.0, np.array([0.0, 0.0, 0.0])),  # Pure yaw (trunk rotation)
        (0.0, 20.0, 0.0, np.array([0.0, 0.0, 0.0])),  # Pure pitch (trunk flexion)
        (0.0, 0.0, 15.0, np.array([0.0, 0.0, 0.0])),  # Pure roll (lateral trunk lean)
        (25.0, -12.0, 18.0, np.array([2.3, -1.1, 0.7])),  # Combined 6-DoF perturbation
    ],
)
def test_torso_relative_arm_invariance_under_rigid_perturbation(
    pose_factory,
    yaw_deg,
    pitch_deg,
    roll_deg,
    t_world,
):
    """Verifies that arm landmarks expressed in Torso Frame are invariant to global trunk motion.

    Whether the subject leans, twists, or steps around the room, the torso-relative
    positions of the shoulder, elbow, and wrist must remain constant.
    """
    pose = pose_factory()
    pts_unperturbed = {
        "s_l": pose.left_shoulder,
        "s_r": pose.right_shoulder,
        "h_l": pose.left_hip,
        "h_r": pose.right_hip,
        "e_r": pose.right_elbow,
        "w_r": pose.right_wrist,
        "e_l": pose.left_elbow,
        "w_l": pose.left_wrist,
    }

    # 1. Unperturbed Torso Frame & relative arm coordinates
    frame_orig = estimate_torso_frame(
        pts_unperturbed["s_l"],
        pts_unperturbed["s_r"],
        pts_unperturbed["h_l"],
        pts_unperturbed["h_r"],
    )
    e_r_orig_T = frame_orig.transform_point_to_torso(pts_unperturbed["e_r"])
    w_r_orig_T = frame_orig.transform_point_to_torso(pts_unperturbed["w_r"])

    # 2. Apply global 6-DoF rigid perturbation
    R_global = Rotation.from_euler("zyx", [yaw_deg, pitch_deg, roll_deg], degrees=True).as_matrix()
    pts_perturbed = _apply_rigid_transform(pts_unperturbed, R_global, t_world)

    # 3. Re-estimate Torso Frame in the perturbed coordinate system
    frame_perturbed = estimate_torso_frame(
        pts_perturbed["s_l"],
        pts_perturbed["s_r"],
        pts_perturbed["h_l"],
        pts_perturbed["h_r"],
    )

    # 4. Transform perturbed arm points into the new Torso Frame
    e_r_perturbed_T = frame_perturbed.transform_point_to_torso(pts_perturbed["e_r"])
    w_r_perturbed_T = frame_perturbed.transform_point_to_torso(pts_perturbed["w_r"])

    # 5. Assert invariance: coordinates in Torso Frame must match within 1e-5 meters
    np.testing.assert_allclose(e_r_perturbed_T, e_r_orig_T, atol=1e-5)
    np.testing.assert_allclose(w_r_perturbed_T, w_r_orig_T, atol=1e-5)
