"""Unit tests for rotation matrices and coordinate transformations."""

import numpy as np
import pytest

from auri_cv.geometry.exceptions import (
    InvalidRotationMatrixError,
    NonFiniteVectorError,
)
from auri_cv.geometry.rotations import (
    assert_valid_rotation_matrix,
    is_valid_rotation_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
    rotation_matrix_from_axes,
    rotation_matrix_from_axis_angle,
    transform_point,
    transform_vector,
)


class TestRotationMatrixValidation:
    def test_identity_is_valid(self):
        I3 = np.eye(3)
        assert is_valid_rotation_matrix(I3)
        assert_valid_rotation_matrix(I3)

    def test_pure_z_rotation_is_valid(self):
        theta = np.pi / 4.0  # 45 deg
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        assert is_valid_rotation_matrix(R)
        assert_valid_rotation_matrix(R)

    def test_reflection_matrix_is_rejected(self):
        # Reflection across xy plane: det(R) = -1.0
        R_reflect = np.diag([1.0, 1.0, -1.0])
        assert not is_valid_rotation_matrix(R_reflect)
        with pytest.raises(InvalidRotationMatrixError, match="proper rotation"):
            assert_valid_rotation_matrix(R_reflect)

    def test_non_orthogonal_matrix_is_rejected(self):
        R_skew = np.array(
            [
                [1.0, 0.5, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        assert not is_valid_rotation_matrix(R_skew)
        with pytest.raises(InvalidRotationMatrixError, match="not orthogonal"):
            assert_valid_rotation_matrix(R_skew)

    def test_wrong_shape_is_rejected(self):
        R_4x4 = np.eye(4)
        assert not is_valid_rotation_matrix(R_4x4)
        with pytest.raises(InvalidRotationMatrixError, match="must be a 3x3"):
            assert_valid_rotation_matrix(R_4x4)

    def test_nan_matrix_is_rejected(self):
        R_nan = np.eye(3)
        R_nan[0, 0] = np.nan
        assert not is_valid_rotation_matrix(R_nan)
        with pytest.raises(InvalidRotationMatrixError, match="non-finite"):
            assert_valid_rotation_matrix(R_nan)


class TestRotationMatrixFromAxes:
    def test_standard_basis_axes(self):
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        z = np.array([0.0, 0.0, 1.0])
        R = rotation_matrix_from_axes(x, y, z)
        np.testing.assert_allclose(R, np.eye(3))

    def test_permuted_right_handed_axes(self):
        # Cyclic permutation: x->y, y->z, z->x
        x = np.array([0.0, 1.0, 0.0])
        y = np.array([0.0, 0.0, 1.0])
        z = np.array([1.0, 0.0, 0.0])
        R = rotation_matrix_from_axes(x, y, z)
        assert is_valid_rotation_matrix(R)

    def test_left_handed_axes_raises_error(self):
        # Left-handed system: z = - (x x y)
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        z = np.array([0.0, 0.0, -1.0])
        with pytest.raises(InvalidRotationMatrixError):
            rotation_matrix_from_axes(x, y, z)


class TestTransformPointAndVector:
    def test_point_round_trip_transformation(self):
        # Point in frame A
        p_A = np.array([1.5, -2.0, 0.75])
        origin_B = np.array([10.0, 20.0, 30.0])
        R_B_from_A = rotation_matrix_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.radians(60.0))

        # Forward transform to Frame B
        p_B = transform_point(p_A, R_B_from_A, origin=origin_B, inverse=False)

        # Inverse transform back to Frame A
        p_A_recovered = transform_point(p_B, R_B_from_A, origin=origin_B, inverse=True)
        np.testing.assert_allclose(p_A_recovered, p_A, atol=1e-6)

    def test_vector_transformation_ignores_translation(self):
        # A direction vector must only be rotated, never translated
        v_A = np.array([0.0, 1.0, 0.0])
        R_90z = rotation_matrix_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2.0)

        # 90 deg rotation of [0, 1, 0] around +Z gives [-1, 0, 0]
        v_B = transform_vector(v_A, R_90z, inverse=False)
        np.testing.assert_allclose(v_B, [-1.0, 0.0, 0.0], atol=1e-6)

        # Norm must be strictly preserved
        assert np.isclose(np.linalg.norm(v_B), np.linalg.norm(v_A))

        # Round trip
        v_A_recovered = transform_vector(v_B, R_90z, inverse=True)
        np.testing.assert_allclose(v_A_recovered, v_A, atol=1e-6)

    def test_transform_non_finite_raises(self):
        with pytest.raises(NonFiniteVectorError):
            transform_point(np.array([np.nan, 0.0, 0.0]), np.eye(3))


class TestQuaternionConversions:
    def test_identity_quaternion(self):
        I3 = np.eye(3)
        q = matrix_to_quaternion(I3)
        # Hamilton convention (w, x, y, z): [1.0, 0.0, 0.0, 0.0]
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-6)

        R_recovered = quaternion_to_matrix(q)
        np.testing.assert_allclose(R_recovered, I3, atol=1e-6)

    def test_rotation_quaternion_round_trip(self):
        axis = np.array([1.0, 1.0, 0.0])
        angle = np.radians(72.0)
        R = rotation_matrix_from_axis_angle(axis, angle)

        q = matrix_to_quaternion(R)
        assert np.isclose(np.linalg.norm(q), 1.0)

        R_recovered = quaternion_to_matrix(q)
        np.testing.assert_allclose(R_recovered, R, atol=1e-6)
