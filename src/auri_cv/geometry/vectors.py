"""Vector math primitives with strict numerical safeguards for AURI."""

from typing import Final

import numpy as np

from auri_cv.geometry.exceptions import (
    DegenerateVectorError,
    NonFiniteVectorError,
    SingularProjectionError,
)

DEFAULT_EPSILON: Final[float] = 1e-7


def assert_finite_vector(v: np.ndarray, name: str = "vector") -> None:
    """Verifies that all components of a vector are finite (not NaN or Inf).

    Args:
        v: Numpy array of any shape.
        name: Parameter name for descriptive error reporting.

    Raises:
        NonFiniteVectorError: If vector contains NaN or infinite values.
    """
    if not np.all(np.isfinite(v)):
        raise NonFiniteVectorError(f"{name} contains non-finite values (NaN or Inf): {v}")


def vector_norm(v: np.ndarray) -> float:
    """Calculates the Euclidean L2 norm of a vector.

    Args:
        v: 1D vector.

    Returns:
        Euclidean magnitude as a float.

    Raises:
        NonFiniteVectorError: If coordinates are non-finite.
    """
    assert_finite_vector(v)
    return float(np.linalg.norm(v))


def normalize_vector(v: np.ndarray, eps: float = DEFAULT_EPSILON) -> np.ndarray:
    """Normalizes a 3D vector to a unit vector.

    Args:
        v: Input 3D vector.
        eps: Minimum magnitude threshold to prevent division by zero.

    Returns:
        Unit vector with Euclidean norm of 1.0.

    Raises:
        NonFiniteVectorError: If coordinates are non-finite.
        DegenerateVectorError: If magnitude is less than eps.
    """
    assert_finite_vector(v)
    mag = float(np.linalg.norm(v))
    if mag < eps:
        raise DegenerateVectorError(
            f"Cannot normalize degenerate vector with near-zero magnitude ({mag:.2e} < {eps:.2e}): {v}"
        )
    return (v / mag).astype(np.float64)


def safe_dot(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the scalar dot product between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Dot product as float.

    Raises:
        NonFiniteVectorError: If either vector contains non-finite values.
    """
    assert_finite_vector(a, "vector a")
    assert_finite_vector(b, "vector b")
    return float(np.dot(a, b))


def safe_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes the cross product between two 3D vectors.

    Args:
        a: First 3D vector.
        b: Second 3D vector.

    Returns:
        Cross product 3D vector (a x b).

    Raises:
        NonFiniteVectorError: If either vector contains non-finite values.
    """
    assert_finite_vector(a, "vector a")
    assert_finite_vector(b, "vector b")
    return np.cross(a, b).astype(np.float64)


def safe_cosine(a: np.ndarray, b: np.ndarray, eps: float = DEFAULT_EPSILON) -> float:
    """Calculates the cosine of the angle between two vectors, clamped to [-1.0, 1.0].

    Prevents domain errors in arccos caused by floating-point inaccuracy.

    Args:
        a: First vector.
        b: Second vector.
        eps: Minimum magnitude threshold.

    Returns:
        Cosine value guaranteed to be within [-1.0, 1.0].

    Raises:
        NonFiniteVectorError: If coordinates are non-finite.
        DegenerateVectorError: If either vector is near-zero length.
    """
    assert_finite_vector(a, "vector a")
    assert_finite_vector(b, "vector b")

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    if norm_a < eps or norm_b < eps:
        raise DegenerateVectorError(
            f"Cannot compute cosine for near-zero vector: norm(a)={norm_a:.2e}, norm(b)={norm_b:.2e}"
        )

    cos_val = float(np.dot(a, b) / (norm_a * norm_b))
    return float(np.clip(cos_val, -1.0, 1.0))


def project_vector_onto_plane(
    v: np.ndarray,
    normal: np.ndarray,
    eps: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """Projects vector v onto the plane perpendicular to the given unit normal.

    v_perp = v - (v . normal) * normal

    Args:
        v: 3D vector to project.
        normal: Plane unit normal vector (must be normalized).
        eps: Minimum magnitude threshold for projected vector.

    Returns:
        Projected vector in the plane.

    Raises:
        SingularProjectionError: If the projected vector is collinear with normal
            (magnitude < eps).
        NonFiniteVectorError: If coordinates are non-finite.
    """
    assert_finite_vector(v, "vector v")
    assert_finite_vector(normal, "plane normal")

    # Ensure normal is unit
    norm_unit = normalize_vector(normal, eps=eps)
    v_perp = v - np.dot(v, norm_unit) * norm_unit

    perp_mag = float(np.linalg.norm(v_perp))
    if perp_mag < eps:
        raise SingularProjectionError(
            f"Vector {v} is collinear with plane normal {normal}; projection magnitude is {perp_mag:.2e} < {eps:.2e}"
        )

    return v_perp.astype(np.float64)
