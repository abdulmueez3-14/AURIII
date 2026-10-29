"""Rotation matrix and spatial transformation operations with SO(3) validation."""

import numpy as np
from scipy.spatial.transform import Rotation

from auri_cv.geometry.exceptions import (
    InvalidRotationMatrixError,
)
from auri_cv.geometry.vectors import (
    DEFAULT_EPSILON,
    assert_finite_vector,
    normalize_vector,
)

DEFAULT_ROTATION_TOLERANCE: float = 1e-5


def is_valid_rotation_matrix(
    R: np.ndarray,
    tol: float = DEFAULT_ROTATION_TOLERANCE,
) -> bool:
    """Verifies that a 3x3 matrix is an orthonormal proper rotation in SO(3).

    Checks:
    1. Shape is exactly (3, 3).
    2. All elements are finite.
    3. R^T * R = I (orthonormality, Frobenius norm difference < tol).
    4. det(R) = +1.0 (proper rotation, no reflection; |det(R) - 1.0| < tol).

    Args:
        R: Candidate matrix.
        tol: Numerical tolerance.

    Returns:
        True if R is a valid rotation matrix, False otherwise.
    """
    if not isinstance(R, np.ndarray) or R.shape != (3, 3):
        return False

    if not np.all(np.isfinite(R)):
        return False

    # Check R^T * R = I
    identity = np.eye(3)
    orthogonality_error = float(np.linalg.norm(np.dot(R.T, R) - identity))
    if orthogonality_error > tol:
        return False

    # Check det(R) = +1.0
    det = float(np.linalg.det(R))
    return abs(det - 1.0) <= tol


def assert_valid_rotation_matrix(
    R: np.ndarray,
    tol: float = DEFAULT_ROTATION_TOLERANCE,
    name: str = "matrix",
) -> None:
    """Asserts that a 3x3 matrix is an orthonormal proper rotation in SO(3).

    Args:
        R: Candidate matrix.
        tol: Numerical tolerance.
        name: Name of matrix for error message.

    Raises:
        InvalidRotationMatrixError: If matrix violates shape, orthonormality,
            or has determinant != +1.0.
    """
    if not isinstance(R, np.ndarray) or R.shape != (3, 3):
        raise InvalidRotationMatrixError(
            f"{name} must be a 3x3 numpy array, got shape {getattr(R, 'shape', type(R))}"
        )

    if not np.all(np.isfinite(R)):
        raise InvalidRotationMatrixError(f"{name} contains non-finite values (NaN or Inf): {R}")

    identity = np.eye(3)
    orthogonality_error = float(np.linalg.norm(np.dot(R.T, R) - identity))
    if orthogonality_error > tol:
        raise InvalidRotationMatrixError(
            f"{name} is not orthogonal: ||R^T R - I||_F = {orthogonality_error:.2e} > {tol:.2e}"
        )

    det = float(np.linalg.det(R))
    if abs(det - 1.0) > tol:
        raise InvalidRotationMatrixError(
            f"{name} is not a proper rotation: det(R) = {det:.6f} != +1.0 (difference > {tol:.2e})"
        )


def rotation_matrix_from_axes(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    tol: float = DEFAULT_ROTATION_TOLERANCE,
) -> np.ndarray:
    """Forms a 3x3 rotation matrix from three orthonormal basis column vectors.

    R = [x_axis | y_axis | z_axis]

    Args:
        x_axis: Unit vector for x basis.
        y_axis: Unit vector for y basis.
        z_axis: Unit vector for z basis.
        tol: Numerical tolerance for validation.

    Returns:
        3x3 rotation matrix with column vectors [x, y, z].

    Raises:
        InvalidRotationMatrixError: If the constructed matrix is not in SO(3).
    """
    R = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float64)
    assert_valid_rotation_matrix(R, tol=tol, name="constructed axis matrix")
    return R


def transform_point(
    point: np.ndarray,
    R: np.ndarray,
    origin: np.ndarray | None = None,
    inverse: bool = False,
    tol: float = DEFAULT_ROTATION_TOLERANCE,
) -> np.ndarray:
    """Transforms a 3D point using rotation matrix R and optional translation origin.

    Forward (inverse=False):
        p' = R * point + origin
    Inverse (inverse=True):
        p' = R^T * (point - origin)

    Args:
        point: 3D point coordinates.
        R: 3x3 orthonormal rotation matrix in SO(3).
        origin: 3D origin vector (defaults to zeros).
        inverse: If True, applies inverse transform (R^T).
        tol: Validation tolerance for R.

    Returns:
        Transformed 3D point.
    """
    assert_finite_vector(point, "point")
    assert_valid_rotation_matrix(R, tol=tol)

    origin_vec = np.zeros(3, dtype=np.float64) if origin is None else origin
    assert_finite_vector(origin_vec, "origin")

    if not inverse:
        return (np.dot(R, point) + origin_vec).astype(np.float64)
    return np.dot(R.T, point - origin_vec).astype(np.float64)


def transform_vector(
    vector: np.ndarray,
    R: np.ndarray,
    inverse: bool = False,
    tol: float = DEFAULT_ROTATION_TOLERANCE,
) -> np.ndarray:
    """Transforms a 3D direction vector using rotation matrix R (translation is NOT applied).

    Forward (inverse=False):
        v' = R * vector
    Inverse (inverse=True):
        v' = R^T * vector

    Args:
        vector: 3D direction vector.
        R: 3x3 orthonormal rotation matrix in SO(3).
        inverse: If True, applies inverse transform (R^T).
        tol: Validation tolerance for R.

    Returns:
        Transformed 3D direction vector.
    """
    assert_finite_vector(vector, "vector")
    assert_valid_rotation_matrix(R, tol=tol)

    if not inverse:
        return np.dot(R, vector).astype(np.float64)
    return np.dot(R.T, vector).astype(np.float64)


def rotation_matrix_from_axis_angle(
    axis: np.ndarray,
    angle_rad: float,
    eps: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """Constructs a 3x3 rotation matrix from an axis and rotation angle.

    Args:
        axis: 3D rotation axis vector.
        angle_rad: Angle of rotation in radians.
        eps: Threshold to check non-zero axis length.

    Returns:
        3x3 rotation matrix in SO(3).
    """
    unit_axis = normalize_vector(axis, eps=eps)
    rotvec = unit_axis * angle_rad
    R = Rotation.from_rotvec(rotvec).as_matrix()
    return R.astype(np.float64)


def matrix_to_quaternion(R: np.ndarray, tol: float = DEFAULT_ROTATION_TOLERANCE) -> np.ndarray:
    """Converts a 3x3 rotation matrix to a scalar-first Hamilton quaternion (w, x, y, z).

    Args:
        R: 3x3 rotation matrix in SO(3).
        tol: Validation tolerance.

    Returns:
        Unit quaternion as [w, x, y, z] array of shape (4,).
    """
    assert_valid_rotation_matrix(R, tol=tol)
    # SciPy's as_quat() returns (x, y, z, w) - scalar-last
    scipy_quat = Rotation.from_matrix(R).as_quat()
    # Rearrange to scalar-first Hamilton convention (w, x, y, z)
    return np.array([scipy_quat[3], scipy_quat[0], scipy_quat[1], scipy_quat[2]], dtype=np.float64)


def quaternion_to_matrix(q: np.ndarray, eps: float = DEFAULT_EPSILON) -> np.ndarray:
    """Converts a scalar-first Hamilton quaternion (w, x, y, z) to a 3x3 rotation matrix.

    Args:
        q: Quaternion in (w, x, y, z) convention.
        eps: Tolerance for non-zero quaternion norm.

    Returns:
        3x3 rotation matrix in SO(3).
    """
    assert_finite_vector(q, "quaternion")
    if q.shape != (4,):
        raise ValueError(f"Quaternion must have shape (4,), got {q.shape}")

    q_norm = float(np.linalg.norm(q))
    if q_norm < eps:
        raise ValueError(f"Cannot convert near-zero quaternion (norm={q_norm:.2e})")

    q_unit = q / q_norm
    # Rearrange from Hamilton (w, x, y, z) to SciPy (x, y, z, w)
    scipy_quat = np.array([q_unit[1], q_unit[2], q_unit[3], q_unit[0]], dtype=np.float64)
    R = Rotation.from_quat(scipy_quat).as_matrix()
    return R.astype(np.float64)
