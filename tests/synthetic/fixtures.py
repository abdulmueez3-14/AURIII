"""Synthetic ground-truth body landmark generators for kinematic verification.

These fixtures provide mathematically exact 3D landmark coordinates with known
analytical joint angles, used to verify the geometry and kinematics engine
without camera or model uncertainty.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticBodyLandmarks:
    """Exact 3D coordinates (in meters) for torso and arm landmarks."""

    left_shoulder: np.ndarray
    right_shoulder: np.ndarray
    left_hip: np.ndarray
    right_hip: np.ndarray
    right_elbow: np.ndarray
    right_wrist: np.ndarray
    left_elbow: np.ndarray
    left_wrist: np.ndarray

    # Analytical ground-truth properties
    expected_right_elevation_deg: float
    expected_right_abduction_deg: float
    expected_right_flexion_deg: float
    expected_left_elevation_deg: float
    expected_left_abduction_deg: float
    expected_left_flexion_deg: float


def make_neutral_standing_pose() -> SyntheticBodyLandmarks:
    """Constructs a subject in ideal resting posture.

    Torso frame is aligned with the world frame:
    - Shoulders at z = 1.40 m, width = 0.40 m
    - Hips at z = 0.90 m, width = 0.30 m
    - Arms hanging vertically down along the sides (length = 0.30 m upper arm, 0.25 m forearm)
    - Expected elevation: 0.0 deg for both arms
    """
    s_l = np.array([0.20, 0.0, 1.40])
    s_r = np.array([-0.20, 0.0, 1.40])
    h_l = np.array([0.15, 0.0, 0.90])
    h_r = np.array([-0.15, 0.0, 0.90])

    # Hanging vertically down: -Z direction
    e_r = np.array([-0.20, 0.0, 1.10])
    w_r = np.array([-0.20, 0.0, 0.85])

    e_l = np.array([0.20, 0.0, 1.10])
    w_l = np.array([0.20, 0.0, 0.85])

    return SyntheticBodyLandmarks(
        left_shoulder=s_l,
        right_shoulder=s_r,
        left_hip=h_l,
        right_hip=h_r,
        right_elbow=e_r,
        right_wrist=w_r,
        left_elbow=e_l,
        left_wrist=w_l,
        expected_right_elevation_deg=0.0,
        expected_right_abduction_deg=0.0,
        expected_right_flexion_deg=0.0,
        expected_left_elevation_deg=0.0,
        expected_left_abduction_deg=0.0,
        expected_left_flexion_deg=0.0,
    )


def make_right_arm_90deg_abduction_pose() -> SyntheticBodyLandmarks:
    """Right arm abducted 90 deg horizontally in the frontal plane (-X direction)."""
    pose = make_neutral_standing_pose()
    # Right shoulder is at [-0.20, 0.0, 1.40]. Moving laterally along -X:
    e_r = np.array([-0.50, 0.0, 1.40])
    w_r = np.array([-0.75, 0.0, 1.40])

    return SyntheticBodyLandmarks(
        left_shoulder=pose.left_shoulder,
        right_shoulder=pose.right_shoulder,
        left_hip=pose.left_hip,
        right_hip=pose.right_hip,
        right_elbow=e_r,
        right_wrist=w_r,
        left_elbow=pose.left_elbow,
        left_wrist=pose.left_wrist,
        expected_right_elevation_deg=90.0,
        expected_right_abduction_deg=90.0,
        expected_right_flexion_deg=0.0,
        expected_left_elevation_deg=0.0,
        expected_left_abduction_deg=0.0,
        expected_left_flexion_deg=0.0,
    )


def make_right_arm_180deg_overhead_pose() -> SyntheticBodyLandmarks:
    """Right arm elevated 180 deg straight overhead (+Z direction)."""
    pose = make_neutral_standing_pose()
    e_r = np.array([-0.20, 0.0, 1.70])
    w_r = np.array([-0.20, 0.0, 1.95])

    return SyntheticBodyLandmarks(
        left_shoulder=pose.left_shoulder,
        right_shoulder=pose.right_shoulder,
        left_hip=pose.left_hip,
        right_hip=pose.right_hip,
        right_elbow=e_r,
        right_wrist=w_r,
        left_elbow=pose.left_elbow,
        left_wrist=pose.left_wrist,
        expected_right_elevation_deg=180.0,
        expected_right_abduction_deg=0.0,
        expected_right_flexion_deg=0.0,
        expected_left_elevation_deg=0.0,
        expected_left_abduction_deg=0.0,
        expected_left_flexion_deg=0.0,
    )


def make_right_arm_90deg_forward_flexion_pose() -> SyntheticBodyLandmarks:
    """Right arm flexed 90 deg forward in the sagittal plane.

    Subject faces anteriorly (ventral direction is -Y in world frame).
    """
    pose = make_neutral_standing_pose()
    e_r = np.array([-0.20, -0.30, 1.40])
    w_r = np.array([-0.20, -0.55, 1.40])

    return SyntheticBodyLandmarks(
        left_shoulder=pose.left_shoulder,
        right_shoulder=pose.right_shoulder,
        left_hip=pose.left_hip,
        right_hip=pose.right_hip,
        right_elbow=e_r,
        right_wrist=w_r,
        left_elbow=pose.left_elbow,
        left_wrist=pose.left_wrist,
        expected_right_elevation_deg=90.0,
        expected_right_abduction_deg=0.0,
        expected_right_flexion_deg=90.0,
        expected_left_elevation_deg=0.0,
        expected_left_abduction_deg=0.0,
        expected_left_flexion_deg=0.0,
    )
