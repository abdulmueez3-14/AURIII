"""Geometry package for AURI Computer Vision & Biomedical Engine."""

from auri_cv.geometry.angles import (
    angle_between_vectors,
    degrees_to_radians,
    radians_to_degrees,
    signed_angle_around_axis,
    wrap_to_180,
    wrap_to_pi,
)
from auri_cv.geometry.exceptions import (
    DegenerateGeometryError,
    DegenerateVectorError,
    GeometryError,
    InvalidRotationMatrixError,
    NonFiniteVectorError,
    SingularProjectionError,
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
from auri_cv.geometry.vectors import (
    assert_finite_vector,
    normalize_vector,
    project_vector_onto_plane,
    safe_cosine,
    safe_cross,
    safe_dot,
    vector_norm,
)

__all__ = [
    "GeometryError",
    "DegenerateGeometryError",
    "DegenerateVectorError",
    "NonFiniteVectorError",
    "InvalidRotationMatrixError",
    "SingularProjectionError",
    "assert_finite_vector",
    "vector_norm",
    "normalize_vector",
    "safe_dot",
    "safe_cross",
    "safe_cosine",
    "project_vector_onto_plane",
    "is_valid_rotation_matrix",
    "assert_valid_rotation_matrix",
    "rotation_matrix_from_axes",
    "transform_point",
    "transform_vector",
    "rotation_matrix_from_axis_angle",
    "matrix_to_quaternion",
    "quaternion_to_matrix",
    "radians_to_degrees",
    "degrees_to_radians",
    "wrap_to_pi",
    "wrap_to_180",
    "angle_between_vectors",
    "signed_angle_around_axis",
]
