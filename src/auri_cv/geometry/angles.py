"""Angular kinematics and vector angle functions with numerical safeguards for AURI."""

import numpy as np

from auri_cv.geometry.vectors import (
    DEFAULT_EPSILON,
    normalize_vector,
    project_vector_onto_plane,
    safe_cosine,
)


def radians_to_degrees(rad: float) -> float:
    """Converts radians to degrees."""
    return float(np.degrees(rad))


def degrees_to_radians(deg: float) -> float:
    """Converts degrees to radians."""
    return float(np.radians(deg))


def wrap_to_pi(rad: float) -> float:
    """Wraps an angle in radians to the interval [-pi, pi]."""
    return float((rad + np.pi) % (2.0 * np.pi) - np.pi)


def wrap_to_180(deg: float) -> float:
    """Wraps an angle in degrees to the interval [-180, 180]."""
    return float((deg + 180.0) % 360.0 - 180.0)


def angle_between_vectors(
    a: np.ndarray,
    b: np.ndarray,
    in_degrees: bool = False,
    eps: float = DEFAULT_EPSILON,
) -> float:
    """Computes the unsigned angle between two 3D vectors.

    theta = arccos(clip((a . b) / (||a|| * ||b||), -1.0, 1.0))

    Args:
        a: First 3D vector.
        b: Second 3D vector.
        in_degrees: If True, returns angle in degrees; otherwise radians.
        eps: Minimum magnitude threshold.

    Returns:
        Unsigned angle in [0, pi] radians (or [0, 180] degrees).

    Raises:
        NonFiniteVectorError: If inputs contain NaN or Inf.
        DegenerateVectorError: If either vector has near-zero magnitude.
    """
    cos_val = safe_cosine(a, b, eps=eps)
    theta_rad = float(np.arccos(cos_val))

    if in_degrees:
        return radians_to_degrees(theta_rad)
    return theta_rad


def signed_angle_around_axis(
    a: np.ndarray,
    b: np.ndarray,
    axis: np.ndarray,
    in_degrees: bool = False,
    eps: float = DEFAULT_EPSILON,
) -> float:
    """Computes the signed angle from vector a to vector b around a rotation axis.

    The vectors are projected onto the plane perpendicular to the axis, and the
    signed angle is evaluated using the right-hand rule:
    theta = atan2(axis . (a_perp x b_perp), a_perp . b_perp)

    Args:
        a: Initial vector.
        b: Target vector.
        axis: Unit normal vector defining the rotation axis and sign convention.
        in_degrees: If True, returns angle in degrees; otherwise radians.
        eps: Minimum magnitude threshold for projections.

    Returns:
        Signed angle in [-pi, pi] radians (or [-180, +180] degrees).
        Positive indicates counter-clockwise rotation about the axis (right-hand rule).

    Raises:
        NonFiniteVectorError: If inputs contain NaN or Inf.
        DegenerateVectorError: If axis or vectors have near-zero magnitude.
        SingularProjectionError: If either vector is collinear with the axis.
    """
    unit_axis = normalize_vector(axis, eps=eps)

    # Project vectors onto the plane perpendicular to the unit axis
    a_perp = project_vector_onto_plane(a, unit_axis, eps=eps)
    b_perp = project_vector_onto_plane(b, unit_axis, eps=eps)

    # Normalize projected vectors
    u_a = normalize_vector(a_perp, eps=eps)
    u_b = normalize_vector(b_perp, eps=eps)

    cross_prod = np.cross(u_a, u_b)
    sin_term = float(np.dot(unit_axis, cross_prod))
    cos_term = float(np.dot(u_a, u_b))

    theta_rad = float(np.atan2(sin_term, cos_term))

    if in_degrees:
        return radians_to_degrees(theta_rad)
    return theta_rad
