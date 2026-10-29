"""Unit tests for vector math operations and numerical safeguards."""

import numpy as np
import pytest

from auri_cv.geometry.exceptions import (
    DegenerateVectorError,
    NonFiniteVectorError,
    SingularProjectionError,
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


class TestFiniteVectorAssertions:
    def test_valid_vector_passes(self):
        v = np.array([1.0, -2.5, 3.14])
        assert_finite_vector(v)

    def test_nan_vector_raises(self):
        v = np.array([1.0, np.nan, 3.0])
        with pytest.raises(NonFiniteVectorError):
            assert_finite_vector(v)

    def test_inf_vector_raises(self):
        v = np.array([np.inf, 2.0, 3.0])
        with pytest.raises(NonFiniteVectorError):
            assert_finite_vector(v)

    def test_neg_inf_vector_raises(self):
        v = np.array([1.0, 2.0, -np.inf])
        with pytest.raises(NonFiniteVectorError):
            assert_finite_vector(v)


class TestVectorNormAndNormalize:
    def test_vector_norm_calculation(self):
        v = np.array([3.0, 4.0, 0.0])
        assert np.isclose(vector_norm(v), 5.0)

    def test_normalize_standard_vector(self):
        v = np.array([0.0, 0.0, 5.0])
        u = normalize_vector(v)
        np.testing.assert_allclose(u, [0.0, 0.0, 1.0])
        assert np.isclose(vector_norm(u), 1.0)

    def test_normalize_already_unit_vector(self):
        v = np.array([1.0, 0.0, 0.0])
        u = normalize_vector(v)
        np.testing.assert_allclose(u, v)

    def test_normalize_zero_vector_raises_degenerate_error(self):
        v = np.array([0.0, 0.0, 0.0])
        with pytest.raises(DegenerateVectorError):
            normalize_vector(v)

    def test_normalize_near_zero_vector_raises_degenerate_error(self):
        v = np.array([1e-9, 0.0, -1e-9])
        with pytest.raises(DegenerateVectorError):
            normalize_vector(v, eps=1e-7)

    def test_normalize_nan_vector_raises_non_finite_error(self):
        v = np.array([np.nan, 1.0, 0.0])
        with pytest.raises(NonFiniteVectorError):
            normalize_vector(v)


class TestDotAndCrossProducts:
    def test_safe_dot_parallel(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])
        assert np.isclose(safe_dot(a, b), 28.0)

    def test_safe_dot_orthogonal(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert np.isclose(safe_dot(a, b), 0.0)

    def test_safe_dot_nan_raises(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, np.nan, 0.0])
        with pytest.raises(NonFiniteVectorError):
            safe_dot(a, b)

    def test_safe_cross_basis_vectors(self):
        # i x j = k
        i = np.array([1.0, 0.0, 0.0])
        j = np.array([0.0, 1.0, 0.0])
        k = safe_cross(i, j)
        np.testing.assert_allclose(k, [0.0, 0.0, 1.0])

        # j x k = i
        np.testing.assert_allclose(safe_cross(j, k), i)

        # k x i = j
        np.testing.assert_allclose(safe_cross(k, i), j)

    def test_safe_cross_parallel_vectors_is_zero(self):
        a = np.array([2.0, 0.0, 0.0])
        b = np.array([5.0, 0.0, 0.0])
        np.testing.assert_allclose(safe_cross(a, b), [0.0, 0.0, 0.0])


class TestSafeCosine:
    def test_identical_vectors_is_one(self):
        v = np.array([1.5, -2.0, 3.0])
        assert np.isclose(safe_cosine(v, v), 1.0)

    def test_opposing_vectors_is_minus_one(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-3.0, 0.0, 0.0])
        assert np.isclose(safe_cosine(a, b), -1.0)

    def test_orthogonal_vectors_is_zero(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert np.isclose(safe_cosine(a, b), 0.0)

    def test_cosine_clamping_avoids_domain_error(self):
        # Vectors scaled such that floating point imprecision could yield 1.0000000000000002
        a = np.array([1.0, 1.0, 1.0])
        cos_val = safe_cosine(a, a)
        assert -1.0 <= cos_val <= 1.0
        # arccos must not raise domain error
        np.arccos(cos_val)

    def test_safe_cosine_degenerate_raises(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        with pytest.raises(DegenerateVectorError):
            safe_cosine(a, b)


class TestProjectVectorOntoPlane:
    def test_standard_projection(self):
        # Project vector v = [1, 2, 3] onto xy plane (normal = [0, 0, 1])
        v = np.array([1.0, 2.0, 3.0])
        normal = np.array([0.0, 0.0, 1.0])
        proj = project_vector_onto_plane(v, normal)
        np.testing.assert_allclose(proj, [1.0, 2.0, 0.0])

    def test_vector_already_in_plane(self):
        v = np.array([4.0, -1.0, 0.0])
        normal = np.array([0.0, 0.0, 1.0])
        proj = project_vector_onto_plane(v, normal)
        np.testing.assert_allclose(proj, v)

    def test_collinear_vector_raises_singular_projection(self):
        # Vector collinear with normal has zero projection onto plane
        v = np.array([0.0, 0.0, 5.0])
        normal = np.array([0.0, 0.0, 1.0])
        with pytest.raises(SingularProjectionError):
            project_vector_onto_plane(v, normal)
