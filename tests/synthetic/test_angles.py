"""Synthetic ground-truth tests for known analytical angles across 3D planes."""

import numpy as np
import pytest

from auri_cv.geometry.angles import angle_between_vectors, signed_angle_around_axis


@pytest.mark.parametrize(
    "angle_deg",
    [0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 150.0, 180.0],
)
def test_synthetic_known_unsigned_angles_xy_plane(angle_deg):
    """Verifies that vector angle computation accurately recovers exact analytical angles."""
    angle_rad = np.radians(angle_deg)
    # Vector a along +X
    a = np.array([1.0, 0.0, 0.0])
    # Vector b rotated by angle_rad in XY plane
    b = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])

    recovered_deg = angle_between_vectors(a, b, in_degrees=True)
    assert np.isclose(recovered_deg, angle_deg, atol=1e-6)


@pytest.mark.parametrize(
    "angle_deg",
    [0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 180.0],
)
def test_synthetic_known_unsigned_angles_arbitrary_3d_plane(angle_deg):
    """Verifies angle accuracy in an arbitrary 3D inclined plane."""
    angle_rad = np.radians(angle_deg)

    # Define two orthonormal vectors spanning an inclined plane
    u = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    # v perpendicular to u: [-1, 1, 2] / sqrt(6)
    v = np.array([-1.0, 1.0, 2.0]) / np.sqrt(6.0)

    # Vector a along u
    a = u * 3.5  # Arbitrary magnitude
    # Vector b at angle_rad in plane
    b = (np.cos(angle_rad) * u + np.sin(angle_rad) * v) * 2.1

    recovered_deg = angle_between_vectors(a, b, in_degrees=True)
    assert np.isclose(recovered_deg, angle_deg, atol=1e-6)


@pytest.mark.parametrize(
    "angle_deg",
    [-179.0, -135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0],
)
def test_synthetic_known_signed_angles_around_normal(angle_deg):
    """Verifies signed rotation around an arbitrary 3D normal axis."""
    angle_rad = np.radians(angle_deg)

    # Arbitrary unit normal axis
    normal = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)

    # Vector a perpendicular to normal
    # Let's find a vector perp to normal: [-2, 1, 0] . [1, 2, 3] = -2 + 2 + 0 = 0
    perp1 = np.array([-2.0, 1.0, 0.0]) / np.sqrt(5.0)
    perp2 = np.cross(normal, perp1)

    a = perp1
    # Rotate perp1 around normal by angle_rad using Rodrigues formula
    b = np.cos(angle_rad) * perp1 + np.sin(angle_rad) * perp2

    recovered_deg = signed_angle_around_axis(a, b, axis=normal, in_degrees=True)

    # Handle wrapping at +/- 180
    if np.isclose(abs(angle_deg), 180.0):
        assert np.isclose(abs(recovered_deg), 180.0, atol=1e-6)
    else:
        assert np.isclose(recovered_deg, angle_deg, atol=1e-6)
