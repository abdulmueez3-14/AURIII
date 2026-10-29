"""Unit tests for anatomical arm segment estimation."""

import numpy as np
import pytest

from auri_cv.anatomy.segments import (
    ArmLaterality,
    ArmSegmentState,
    estimate_arm_segments,
    estimate_arm_segments_from_pose,
)
from auri_cv.frames.torso_frame import estimate_torso_frame
from auri_cv.geometry.exceptions import (
    DegenerateGeometryError,
    NonFiniteVectorError,
)
from auri_cv.pose.data_models import Landmark3D, PoseObservation


@pytest.fixture
def standard_torso():
    """Standard torso frame in upright posture."""
    s_l = np.array([0.20, 0.0, 1.40])
    s_r = np.array([-0.20, 0.0, 1.40])
    h_l = np.array([0.15, 0.0, 0.90])
    h_r = np.array([-0.15, 0.0, 0.90])
    return estimate_torso_frame(s_l, s_r, h_l, h_r)


class TestArmSegmentEstimation:
    def test_straight_hanging_right_arm(self, standard_torso):
        # Shoulder at [-0.20, 0, 1.40]
        # Elbow at [-0.20, 0, 1.10] -> arm vector down [0, 0, -1]
        # Wrist at [-0.20, 0, 0.85] -> forearm vector down [0, 0, -1]
        s_r = np.array([-0.20, 0.0, 1.40])
        e_r = np.array([-0.20, 0.0, 1.10])
        w_r = np.array([-0.20, 0.0, 0.85])

        seg = estimate_arm_segments(
            shoulder=s_r,
            elbow=e_r,
            wrist=w_r,
            torso_frame=standard_torso,
            laterality=ArmLaterality.RIGHT,
        )

        assert isinstance(seg, ArmSegmentState)
        assert seg.laterality == ArmLaterality.RIGHT

        # Upper arm vector in World and Torso
        np.testing.assert_allclose(seg.upper_arm_dir_world, [0.0, 0.0, -1.0], atol=1e-6)
        np.testing.assert_allclose(seg.upper_arm_dir_torso, [0.0, 0.0, -1.0], atol=1e-6)
        assert np.isclose(seg.upper_arm_length_m, 0.30, atol=1e-6)

        # Forearm vector in World and Torso
        assert seg.forearm_dir_world is not None
        assert seg.forearm_dir_torso is not None
        np.testing.assert_allclose(seg.forearm_dir_world, [0.0, 0.0, -1.0], atol=1e-6)
        np.testing.assert_allclose(seg.forearm_dir_torso, [0.0, 0.0, -1.0], atol=1e-6)
        assert np.isclose(seg.forearm_length_m, 0.25, atol=1e-6)

        # Straight arm has 0 deg elbow flexion
        assert seg.elbow_flexion_deg is not None
        assert np.isclose(seg.elbow_flexion_deg, 0.0, atol=1e-6)

    def test_bent_elbow_90deg_flexion(self, standard_torso):
        # Arm hanging down [0, 0, -1]
        # Forearm bent forward along -Y in world (chest forward) -> [0, -1, 0]
        s_r = np.array([-0.20, 0.0, 1.40])
        e_r = np.array([-0.20, 0.0, 1.10])
        w_r = np.array([-0.20, -0.25, 1.10])

        seg = estimate_arm_segments(
            shoulder=s_r,
            elbow=e_r,
            wrist=w_r,
            torso_frame=standard_torso,
            laterality=ArmLaterality.RIGHT,
        )

        # Forearm points along -Y in world -> anterior in Torso (+Y_T)
        np.testing.assert_allclose(seg.forearm_dir_world, [0.0, -1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(seg.forearm_dir_torso, [0.0, 1.0, 0.0], atol=1e-6)

        # Elbow flexion should be exactly 90 deg
        assert seg.elbow_flexion_deg is not None
        assert np.isclose(seg.elbow_flexion_deg, 90.0, atol=1e-6)

    def test_wrist_missing_still_estimates_upper_arm(self, standard_torso):
        s_r = np.array([-0.20, 0.0, 1.40])
        e_r = np.array([-0.20, 0.0, 1.10])

        seg = estimate_arm_segments(
            shoulder=s_r,
            elbow=e_r,
            wrist=None,
            torso_frame=standard_torso,
            laterality=ArmLaterality.RIGHT,
        )

        assert np.isclose(seg.upper_arm_length_m, 0.30, atol=1e-6)
        assert seg.forearm_dir_world is None
        assert seg.forearm_dir_torso is None
        assert seg.forearm_length_m is None
        assert seg.elbow_flexion_deg is None

    def test_left_arm_laterality(self, standard_torso):
        s_l = np.array([0.20, 0.0, 1.40])
        e_l = np.array([0.50, 0.0, 1.40])  # Abducted laterally to the left (+X in world)

        seg = estimate_arm_segments(
            shoulder=s_l,
            elbow=e_l,
            wrist=None,
            torso_frame=standard_torso,
            laterality=ArmLaterality.LEFT,
        )

        assert seg.laterality == ArmLaterality.LEFT
        np.testing.assert_allclose(seg.upper_arm_dir_world, [1.0, 0.0, 0.0], atol=1e-6)
        # In torso coordinates, +X is subject's right, so left is -X_T: [-1, 0, 0]
        np.testing.assert_allclose(seg.upper_arm_dir_torso, [-1.0, 0.0, 0.0], atol=1e-6)


class TestDegenerateSegments:
    def test_collapsed_arm_raises_degenerate_error(self, standard_torso):
        s_r = np.array([-0.20, 0.0, 1.40])
        e_r = np.array([-0.20, 0.0, 1.45])  # Length = 0.05 m (< 0.10 m)

        with pytest.raises(DegenerateGeometryError, match="Upper arm length too small"):
            estimate_arm_segments(
                shoulder=s_r,
                elbow=e_r,
                wrist=None,
                torso_frame=standard_torso,
                min_length=0.10,
            )

    def test_nan_coordinates_raise_non_finite_error(self, standard_torso):
        s_r = np.array([np.nan, 0.0, 1.40])
        e_r = np.array([-0.20, 0.0, 1.10])

        with pytest.raises(NonFiniteVectorError):
            estimate_arm_segments(
                shoulder=s_r,
                elbow=e_r,
                wrist=None,
                torso_frame=standard_torso,
            )


class TestEstimateArmSegmentsFromPose:
    def test_valid_pose_extraction(self, standard_torso):
        lm_dict = {
            "RIGHT_SHOULDER": Landmark3D(
                name="RIGHT_SHOULDER",
                index=12,
                x=-0.20,
                y=0.0,
                z=1.40,
                visibility=0.9,
                presence=0.9,
            ),
            "RIGHT_ELBOW": Landmark3D(
                name="RIGHT_ELBOW", index=14, x=-0.20, y=0.0, z=1.10, visibility=0.9, presence=0.9
            ),
            "RIGHT_WRIST": Landmark3D(
                name="RIGHT_WRIST", index=16, x=-0.20, y=0.0, z=0.85, visibility=0.9, presence=0.9
            ),
        }
        pose = PoseObservation(
            timestamp_ms=0.0,
            frame_index=0,
            is_detected=True,
            landmarks_2d={},
            landmarks_3d=lm_dict,
        )

        seg = estimate_arm_segments_from_pose(
            pose=pose,
            torso_frame=standard_torso,
            laterality="right",
        )
        assert isinstance(seg, ArmSegmentState)
        assert np.isclose(seg.upper_arm_length_m, 0.30, atol=1e-6)

    def test_missing_elbow_returns_none(self, standard_torso):
        lm_dict = {
            "RIGHT_SHOULDER": Landmark3D(
                name="RIGHT_SHOULDER",
                index=12,
                x=-0.20,
                y=0.0,
                z=1.40,
                visibility=0.9,
                presence=0.9,
            ),
            # Missing RIGHT_ELBOW
        }
        pose = PoseObservation(
            timestamp_ms=0.0,
            frame_index=0,
            is_detected=True,
            landmarks_2d={},
            landmarks_3d=lm_dict,
        )

        seg = estimate_arm_segments_from_pose(
            pose=pose,
            torso_frame=standard_torso,
            laterality="right",
        )
        assert seg is None
