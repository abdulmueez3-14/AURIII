"""Geometric and linear algebra exceptions for AURI."""


class GeometryError(ValueError):
    """Base exception for geometric errors."""

    pass


class DegenerateGeometryError(GeometryError):
    """Raised when an anatomical geometric configuration is degenerate or ill-conditioned."""

    pass


class DegenerateVectorError(GeometryError):
    """Raised when a vector has zero or near-zero magnitude."""

    pass


class NonFiniteVectorError(GeometryError):
    """Raised when a vector contains NaN, positive infinity, or negative infinity."""

    pass


class InvalidRotationMatrixError(GeometryError):
    """Raised when a matrix fails SO(3) proper rotation constraints."""

    pass


class SingularProjectionError(GeometryError):
    """Raised when a vector is collinear with the projection plane normal."""

    pass
