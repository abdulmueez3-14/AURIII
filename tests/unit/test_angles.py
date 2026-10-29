"""Unit tests for angular kinematics and signed/unsigned angle calculations."""

import numpy as np
import pytest

from auri_cv.geometry.angles import (
    angle_between_vectors,
    degrees_to_radians,
    radians_to_degrees,
    signed_angle_around_axis,
    wrap_to_180,
    wrap_to_pi,
)
from auri_cv.geometry.exceptions import (
    DegenerateVectorError,
    NonFiniteVectorError,
    SingularProjectionError,
)


class TestAngleConversionsAndWrapping:
    def test_radians_degrees_round_trip(self):
        deg = 45.0
        rad = degrees_to_radians(deg)
        assert np.isclose(rad, np.pi / 4.0)
        assert np.isclose(radians_to_degrees(rad), deg)

    def test_wrap_to_pi(self):
        assert np.isclose(wrap_to_pi(3.0 * np.pi), -np.pi)
        assert np.isclose(wrap_to_pi(-3.0 * np.pi), -np.pi)
        assert np.isclose(wrap_to_pi(np.pi / 2.0), np.pi / 2.0)

    def test_wrap_to_180(self):
        assert np.isclose(wrap_to_180(200.0), -160.0)
        assert np.isclose(wrap_to_180(-190.0), 170.0)
        assert np.isclose(wrap_to_180(90.0), 90.0)


class TestAngleBetweenVectors:
    def test_zero_angle_identical_vectors(self):
        v = np.array([0.0, 0.0, 5.0])
        theta_rad = angle_between_vectors(v, v, in_degrees=False)
        assert np.isclose(theta_rad, 0.0)

        theta_deg = angle_between_vectors(v, v, in_degrees=True)
        assert np.isclose(theta_deg, 0.0)

    def test_180_deg_opposing_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-2.0, 0.0, 0.0])
        theta_deg = angle_between_vectors(a, b, in_degrees=True)
        assert np.isclose(theta_deg, 180.0)

    def test_90_deg_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 3.0, 0.0])
        theta_deg = angle_between_vectors(a, b, in_degrees=True)
        assert np.isclose(theta_deg, 90.0)

    def test_45_deg_known_angle(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 1.0, 0.0])
        theta_deg = angle_between_vectors(a, b, in_degrees=True)
        assert np.isclose(theta_deg, 45.0)

    def test_degenerate_vector_raises(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        with pytest.raises(DegenerateVectorError):
            angle_between_vectors(a, b)

    def test_nan_vector_raises(self):
        a = np.array([np.nan, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        with pytest.raises(NonFiniteVectorError):
            angle_between_vectors(a, b)


class TestSignedAngleAroundAxis:
    def test_counter_clockwise_positive_90_deg(self):
        # Around +Z axis: rotating from +X [1, 0, 0] to +Y [0, 1, 0] is +90 deg
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        z_axis = np.array([0.0, 0.0, 1.0])

        theta_deg = signed_angle_around_axis(a, b, axis=z_axis, in_degrees=True)
        assert np.isclose(theta_deg, 90.0)

    def test_clockwise_negative_90_deg(self):
        # Around +Z axis: rotating from +Y [0, 1, 0] to +X [1, 0, 0] is -90 deg
        a = np.array([0.0, 1.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        z_axis = np.array([0.0, 0.0, 1.0])

        theta_deg = signed_angle_around_axis(a, b, axis=z_axis, in_degrees=True)
        assert np.isclose(theta_deg, -90.0)

    def test_reversed_axis_flips_sign(self):
        # If rotation axis is reversed (-Z), positive rotation convention inverts
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        neg_z_axis = np.array([0.0, 0.0, -1.0])

        theta_deg = signed_angle_around_axis(a, b, axis=neg_z_axis, in_degrees=True)
        assert np.isclose(theta_deg, -90.0)

    def test_non_planar_vectors_projected_correctly(self):
        # Vectors having non-zero z-components; projected onto xy plane
        a = np.array([1.0, 0.0, 50.0])
        b = np.array([0.0, 1.0, -100.0])
        z_axis = np.array([0.0, 0.0, 1.0])

        theta_deg = signed_angle_around_axis(a, b, axis=z_axis, in_degrees=True)
        assert np.isclose(theta_deg, 90.0)

    def test_collinear_with_axis_raises_singular_projection(self):
        # Vector a is along z-axis, cannot be projected onto xy plane
        a = np.array([0.0, 0.0, 5.0])
        b = np.array([1.0, 0.0, 0.0])
        z_axis = np.array([0.0, 0.0, 1.0])

        with pytest.raises(SingularProjectionError):
            signed_angle_around_axis(a, b, axis=z_axis)
