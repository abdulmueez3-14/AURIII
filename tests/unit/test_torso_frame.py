"""Unit tests for torso coordinate frame estimation and transformations."""

import numpy as np
import pytest

from auri_cv.frames.torso_frame import (
    TorsoFrame,
    estimate_torso_frame,
    estimate_torso_frame_from_pose,
)
from auri_cv.geometry.exceptions import (
    DegenerateGeometryError,
    NonFiniteVectorError,
)
from auri_cv.geometry.rotations import is_valid_rotation_matrix
from auri_cv.pose.data_models import Landmark3D, PoseObservation


class TestTorsoFrameEstimation:
    @pytest.fixture
    def standard_torso_landmarks(self):
        """Standard upright standing posture facing negative Y (camera at origin)."""
        s_l = np.array([0.20, 0.0, 1.40])
        s_r = np.array([-0.20, 0.0, 1.40])
        h_l = np.array([0.15, 0.0, 0.90])
        h_r = np.array([-0.15, 0.0, 0.90])
        return s_l, s_r, h_l, h_r

    def test_standard_torso_frame_axes(self, standard_torso_landmarks):
        s_l, s_r, h_l, h_r = standard_torso_landmarks
        frame = estimate_torso_frame(s_l, s_r, h_l, h_r)

        # Origin should be shoulder midpoint: [0.0, 0.0, 1.40]
        np.testing.assert_allclose(frame.origin, [0.0, 0.0, 1.40], atol=1e-6)

        # Superior axis +Z_T should be directly upward [0, 0, 1]
        np.testing.assert_allclose(frame.z_axis, [0.0, 0.0, 1.0], atol=1e-6)

        # Lateral axis +X_T (subject's right) should point along -X in world [-1, 0, 0]
        np.testing.assert_allclose(frame.x_axis, [-1.0, 0.0, 0.0], atol=1e-6)

        # Anterior axis +Y_T (forward) should point along -Y in world [0, -1, 0]
        np.testing.assert_allclose(frame.y_axis, [0.0, -1.0, 0.0], atol=1e-6)

        # Validation of SO(3) rotation matrices
        assert is_valid_rotation_matrix(frame.R_world_from_torso)
        assert is_valid_rotation_matrix(frame.R_torso_from_world)
        np.testing.assert_allclose(frame.R_torso_from_world, frame.R_world_from_torso.T, atol=1e-6)

        assert np.isclose(frame.shoulder_width, 0.40, atol=1e-6)
        assert np.isclose(frame.torso_length, 0.50, atol=1e-6)

    def test_point_transformations_in_torso_frame(self, standard_torso_landmarks):
        s_l, s_r, h_l, h_r = standard_torso_landmarks
        frame = estimate_torso_frame(s_l, s_r, h_l, h_r)

        # 1. Shoulder center in world should be [0, 0, 0] in torso coordinates
        p_origin_T = frame.transform_point_to_torso(frame.origin)
        np.testing.assert_allclose(p_origin_T, [0.0, 0.0, 0.0], atol=1e-6)

        # 2. Right shoulder in world should have positive X_T (+half_width = +0.20 m)
        s_r_T = frame.transform_point_to_torso(s_r)
        np.testing.assert_allclose(s_r_T, [0.20, 0.0, 0.0], atol=1e-6)

        # 3. Left shoulder in world should have negative X_T (-half_width = -0.20 m)
        s_l_T = frame.transform_point_to_torso(s_l)
        np.testing.assert_allclose(s_l_T, [-0.20, 0.0, 0.0], atol=1e-6)

        # 4. Hip midpoint in world should have negative Z_T (-torso_length = -0.50 m)
        h_c = (h_l + h_r) / 2.0
        h_c_T = frame.transform_point_to_torso(h_c)
        np.testing.assert_allclose(h_c_T, [0.0, 0.0, -0.50], atol=1e-6)

        # 5. Round trip transformation
        s_r_recovered = frame.transform_point_to_world(s_r_T)
        np.testing.assert_allclose(s_r_recovered, s_r, atol=1e-6)

    def test_vector_transformations_ignore_translation(self, standard_torso_landmarks):
        s_l, s_r, h_l, h_r = standard_torso_landmarks
        frame = estimate_torso_frame(s_l, s_r, h_l, h_r)

        # Direction vector pointing straight down in world: [0, 0, -1]
        down_W = np.array([0.0, 0.0, -1.0])
        down_T = frame.transform_vector_to_torso(down_W)
        np.testing.assert_allclose(down_T, [0.0, 0.0, -1.0], atol=1e-6)

        # Direction vector pointing toward subject's right in world: [-1, 0, 0]
        right_W = np.array([-1.0, 0.0, 0.0])
        right_T = frame.transform_vector_to_torso(right_W)
        np.testing.assert_allclose(right_T, [1.0, 0.0, 0.0], atol=1e-6)

        # Round trip
        right_W_recovered = frame.transform_vector_to_world(right_T)
        np.testing.assert_allclose(right_W_recovered, right_W, atol=1e-6)


class TestDegenerateTorsoHandling:
    def test_shoulders_too_close_raises_degenerate_error(self):
        s_l = np.array([0.05, 0.0, 1.40])
        s_r = np.array([-0.05, 0.0, 1.40])  # Width = 0.10 m (< 0.15 m)
        h_l = np.array([0.15, 0.0, 0.90])
        h_r = np.array([-0.15, 0.0, 0.90])

        with pytest.raises(DegenerateGeometryError, match="Inter-shoulder distance too small"):
            estimate_torso_frame(s_l, s_r, h_l, h_r, min_span=0.15)

    def test_hips_too_close_raises_degenerate_error(self):
        s_l = np.array([0.20, 0.0, 1.40])
        s_r = np.array([-0.20, 0.0, 1.40])
        h_l = np.array([0.02, 0.0, 0.90])
        h_r = np.array([-0.02, 0.0, 0.90])  # Hip width = 0.04 m (< 0.15 m)

        with pytest.raises(DegenerateGeometryError, match="Inter-hip distance too small"):
            estimate_torso_frame(s_l, s_r, h_l, h_r, min_span=0.15)

    def test_torso_length_too_small_raises_degenerate_error(self):
        s_l = np.array([0.20, 0.0, 1.00])
        s_r = np.array([-0.20, 0.0, 1.00])
        h_l = np.array([0.15, 0.0, 0.95])
        h_r = np.array([-0.15, 0.0, 0.95])  # Spine length = 0.05 m (< 0.15 m)

        with pytest.raises(DegenerateGeometryError, match="Torso length"):
            estimate_torso_frame(s_l, s_r, h_l, h_r, min_span=0.15)

    def test_collinear_shoulders_and_spine_raises_degenerate_error(self):
        # Shoulders positioned vertically along Z axis (collinear with spine)
        s_l = np.array([0.0, 0.0, 1.60])
        s_r = np.array([0.0, 0.0, 1.20])
        h_l = np.array([0.15, 0.0, 0.90])
        h_r = np.array([-0.15, 0.0, 0.90])

        with pytest.raises(DegenerateGeometryError, match="collinear"):
            estimate_torso_frame(s_l, s_r, h_l, h_r, min_span=0.15)

    def test_nan_coordinates_raise_non_finite_error(self):
        s_l = np.array([np.nan, 0.0, 1.40])
        s_r = np.array([-0.20, 0.0, 1.40])
        h_l = np.array([0.15, 0.0, 0.90])
        h_r = np.array([-0.15, 0.0, 0.90])

        with pytest.raises(NonFiniteVectorError):
            estimate_torso_frame(s_l, s_r, h_l, h_r)


class TestEstimateTorsoFrameFromPose:
    def test_valid_pose_returns_torso_frame(self):
        lm_dict = {
            "LEFT_SHOULDER": Landmark3D(
                name="LEFT_SHOULDER", index=11, x=0.20, y=0.0, z=1.40, visibility=0.9, presence=0.9
            ),
            "RIGHT_SHOULDER": Landmark3D(
                name="RIGHT_SHOULDER",
                index=12,
                x=-0.20,
                y=0.0,
                z=1.40,
                visibility=0.9,
                presence=0.9,
            ),
            "LEFT_HIP": Landmark3D(
                name="LEFT_HIP", index=23, x=0.15, y=0.0, z=0.90, visibility=0.9, presence=0.9
            ),
            "RIGHT_HIP": Landmark3D(
                name="RIGHT_HIP", index=24, x=-0.15, y=0.0, z=0.90, visibility=0.9, presence=0.9
            ),
        }
        pose = PoseObservation(
            timestamp_ms=10.0,
            frame_index=0,
            is_detected=True,
            landmarks_2d={},
            landmarks_3d=lm_dict,
        )

        frame = estimate_torso_frame_from_pose(pose)
        assert isinstance(frame, TorsoFrame)
        np.testing.assert_allclose(frame.origin, [0.0, 0.0, 1.40])

    def test_missing_landmark_returns_none(self):
        lm_dict = {
            "LEFT_SHOULDER": Landmark3D(
                name="LEFT_SHOULDER", index=11, x=0.20, y=0.0, z=1.40, visibility=0.9, presence=0.9
            ),
            # Missing RIGHT_SHOULDER
            "LEFT_HIP": Landmark3D(
                name="LEFT_HIP", index=23, x=0.15, y=0.0, z=0.90, visibility=0.9, presence=0.9
            ),
            "RIGHT_HIP": Landmark3D(
                name="RIGHT_HIP", index=24, x=-0.15, y=0.0, z=0.90, visibility=0.9, presence=0.9
            ),
        }
        pose = PoseObservation(
            timestamp_ms=10.0,
            frame_index=0,
            is_detected=True,
            landmarks_2d={},
            landmarks_3d=lm_dict,
        )

        frame = estimate_torso_frame_from_pose(pose)
        assert frame is None
