"""Subject-centered orthonormal Torso Coordinate Frame (T) estimation."""

from dataclasses import dataclass
from typing import Final

import numpy as np

from auri_cv.geometry.exceptions import (
    DegenerateGeometryError,
    NonFiniteVectorError,
)
from auri_cv.geometry.rotations import (
    assert_valid_rotation_matrix,
    rotation_matrix_from_axes,
    transform_point,
    transform_vector,
)
from auri_cv.geometry.vectors import (
    assert_finite_vector,
    normalize_vector,
    safe_cross,
    safe_dot,
    vector_norm,
)
from auri_cv.pose.data_models import PoseObservation

DEFAULT_MIN_TORSO_SPAN: Final[float] = 0.15  # Minimum 15 cm span


@dataclass(frozen=True)
class TorsoFrame:
    """Rigid orthonormal Torso Coordinate Frame (T).

    Axes orientation:
    - +X_T: Subject's anatomical right (lateral)
    - +Y_T: Anterior (forward from chest)
    - +Z_T: Superior (headward along spine)
    Right-handed: x_hat x y_hat = z_hat
    """

    origin: np.ndarray  # (3,) origin in world coordinates (shoulder center)
    x_axis: np.ndarray  # (3,) lateral unit vector in world coordinates
    y_axis: np.ndarray  # (3,) anterior unit vector in world coordinates
    z_axis: np.ndarray  # (3,) superior unit vector in world coordinates
    R_world_from_torso: np.ndarray  # (3, 3) matrix mapping Torso coords to World coords
    R_torso_from_world: np.ndarray  # (3, 3) matrix mapping World coords to Torso coords
    shoulder_width: float  # Inter-shoulder distance in meters
    torso_length: float  # Shoulder-center to hip-center distance in meters
    # False for shoulders-only frames: the superior axis then comes from an ASSUMED vertical
    # and torso_length is nominal, so trunk lean / hiking are not measurable from this frame.
    hips_observed: bool = True

    def transform_point_to_torso(self, point_world: np.ndarray) -> np.ndarray:
        """Transforms a 3D point from World coordinates to Torso coordinates.

        p_T = R_T<-W * (p_W - origin)

        Args:
            point_world: Point coordinates in World Frame.

        Returns:
            Point coordinates in Torso Frame.
        """
        return transform_point(
            point_world,
            self.R_world_from_torso,
            origin=self.origin,
            inverse=True,
        )

    def transform_point_to_world(self, point_torso: np.ndarray) -> np.ndarray:
        """Transforms a 3D point from Torso coordinates to World coordinates.

        p_W = R_W<-T * p_T + origin

        Args:
            point_torso: Point coordinates in Torso Frame.

        Returns:
            Point coordinates in World Frame.
        """
        return transform_point(
            point_torso,
            self.R_world_from_torso,
            origin=self.origin,
            inverse=False,
        )

    def transform_vector_to_torso(self, vector_world: np.ndarray) -> np.ndarray:
        """Transforms a 3D direction vector from World to Torso (no translation applied).

        v_T = R_T<-W * v_W

        Args:
            vector_world: 3D direction vector in World Frame.

        Returns:
            Direction vector expressed in Torso Frame.
        """
        return transform_vector(
            vector_world,
            self.R_world_from_torso,
            inverse=True,
        )

    def transform_vector_to_world(self, vector_torso: np.ndarray) -> np.ndarray:
        """Transforms a 3D direction vector from Torso to World (no translation applied).

        v_W = R_W<-T * v_T

        Args:
            vector_torso: 3D direction vector in Torso Frame.

        Returns:
            Direction vector expressed in World Frame.
        """
        return transform_vector(
            vector_torso,
            self.R_world_from_torso,
            inverse=False,
        )


def estimate_torso_frame(
    left_shoulder: np.ndarray,
    right_shoulder: np.ndarray,
    left_hip: np.ndarray,
    right_hip: np.ndarray,
    min_span: float = DEFAULT_MIN_TORSO_SPAN,
) -> TorsoFrame:
    """Estimates an orthonormal Torso Frame from anatomical landmarks.

    Derivation:
    1. Shoulder midpoint: S_c = (S_L + S_R) / 2
    2. Hip midpoint: H_c = (H_L + H_R) / 2
    3. Superior axis: z_T = (S_c - H_c) / ||S_c - H_c||
    4. Lateral candidate: x_0 = S_R - S_L (subject right)
    5. Orthogonalize: x' = x_0 - (x_0 . z_T) * z_T
    6. Normalize: x_T = x' / ||x'||
    7. Anterior axis: y_T = z_T x x_T
    8. Validate SO(3) matrix: R = [x_T | y_T | z_T]

    Args:
        left_shoulder: Left shoulder 3D landmark in meters.
        right_shoulder: Right shoulder 3D landmark in meters.
        left_hip: Left hip 3D landmark in meters.
        right_hip: Right hip 3D landmark in meters.
        min_span: Minimum allowable distance threshold in meters.

    Returns:
        TorsoFrame containing basis axes, transformation matrices, and origin.

    Raises:
        NonFiniteVectorError: If any coordinate is NaN or Inf.
        DegenerateGeometryError: If landmarks are collinear or below minimum span.
    """
    assert_finite_vector(left_shoulder, "left_shoulder")
    assert_finite_vector(right_shoulder, "right_shoulder")
    assert_finite_vector(left_hip, "left_hip")
    assert_finite_vector(right_hip, "right_hip")

    # 1. Check anatomical spans
    shoulder_vec = right_shoulder - left_shoulder
    shoulder_width = vector_norm(shoulder_vec)
    if shoulder_width < min_span:
        raise DegenerateGeometryError(
            f"Inter-shoulder distance too small ({shoulder_width:.3f} m < {min_span:.3f} m); degenerate torso."
        )

    hip_vec = right_hip - left_hip
    hip_width = vector_norm(hip_vec)
    if hip_width < min_span:
        raise DegenerateGeometryError(
            f"Inter-hip distance too small ({hip_width:.3f} m < {min_span:.3f} m); degenerate torso."
        )

    # 2. Midpoints and spine axis
    s_c = (left_shoulder + right_shoulder) / 2.0
    h_c = (left_hip + right_hip) / 2.0

    spine_vec = s_c - h_c
    torso_length = vector_norm(spine_vec)
    if torso_length < min_span:
        raise DegenerateGeometryError(
            f"Torso length (spine distance) too small ({torso_length:.3f} m < {min_span:.3f} m); degenerate torso."
        )

    # Superior axis +Z_T (headward)
    z_hat = normalize_vector(spine_vec)

    # 3. Lateral axis candidate +X_T (subject right)
    # Project out any component along the superior axis (Gram-Schmidt)
    x_candidate = shoulder_vec
    proj_on_z = safe_dot(x_candidate, z_hat) * z_hat
    x_ortho = x_candidate - proj_on_z

    x_ortho_norm = vector_norm(x_ortho)
    if x_ortho_norm < 1e-4:
        raise DegenerateGeometryError(
            "Shoulder vector is collinear with spine axis; cannot determine lateral torso direction."
        )

    x_hat = normalize_vector(x_ortho)

    # 4. Anterior axis +Y_T (forward from chest)
    # Right-handed: z_hat x x_hat = y_hat, so x_hat x y_hat = z_hat
    y_hat = safe_cross(z_hat, x_hat)
    y_hat = normalize_vector(y_hat)

    # 5. Form rotation matrix R_W_from_T = [x_hat | y_hat | z_hat]
    R_w_from_t = rotation_matrix_from_axes(x_hat, y_hat, z_hat)
    assert_valid_rotation_matrix(R_w_from_t, name="Torso Frame R_W_from_T")
    R_t_from_w = R_w_from_t.T

    return TorsoFrame(
        origin=s_c.astype(np.float64),
        x_axis=x_hat,
        y_axis=y_hat,
        z_axis=z_hat,
        R_world_from_torso=R_w_from_t,
        R_torso_from_world=R_t_from_w,
        shoulder_width=shoulder_width,
        torso_length=torso_length,
    )


NOMINAL_TORSO_LENGTH_M: Final[float] = 0.50
MEDIAPIPE_WORLD_UP: Final[tuple[float, float, float]] = (0.0, -1.0, 0.0)  # Y-down world


def estimate_torso_frame_shoulders_only(
    left_shoulder: np.ndarray,
    right_shoulder: np.ndarray,
    up_world: np.ndarray,
    min_span: float = DEFAULT_MIN_TORSO_SPAN,
) -> TorsoFrame:
    """Fallback torso frame when hips are not observable.

    The superior axis is the ASSUMED vertical `up_world` (upright camera, upright trunk),
    NOT a measured spine direction. Elevation is then relative to the world vertical, so
    trunk lean is confounded with arm motion. The result has hips_observed=False.
    """
    assert_finite_vector(left_shoulder, "left_shoulder")
    assert_finite_vector(right_shoulder, "right_shoulder")
    assert_finite_vector(up_world, "up_world")
    up = normalize_vector(up_world)
    shoulder_vec = right_shoulder - left_shoulder
    x_perp = shoulder_vec - safe_dot(shoulder_vec, up) * up
    if vector_norm(x_perp) < min_span:
        raise DegenerateGeometryError("Shoulder line is (near) parallel to the assumed vertical.")
    s_c = (left_shoulder + right_shoulder) / 2.0
    h_c = s_c - NOMINAL_TORSO_LENGTH_M * up
    half = 0.375 * x_perp  # virtual hip width 0.75 * shoulder width; only the axes are used
    frame = estimate_torso_frame(
        left_shoulder, right_shoulder, h_c - half, h_c + half, min_span=min_span
    )
    return TorsoFrame(**{**frame.__dict__, "hips_observed": False})


def estimate_torso_frame_from_pose(
    pose: PoseObservation,
    min_span: float = DEFAULT_MIN_TORSO_SPAN,
) -> TorsoFrame | None:
    """Convenience helper extracting torso frame from a PoseObservation.

    Args:
        pose: PoseObservation containing 3D world landmarks.
        min_span: Minimum allowable landmark distance.

    Returns:
        TorsoFrame if all required landmarks are present and valid; None otherwise.
    """
    if not pose.is_detected:
        return None

    s_l = pose.get_world_point("LEFT_SHOULDER")
    s_r = pose.get_world_point("RIGHT_SHOULDER")
    h_l = pose.get_world_point("LEFT_HIP")
    h_r = pose.get_world_point("RIGHT_HIP")

    if s_l is None or s_r is None or h_l is None or h_r is None:
        return None

    try:
        return estimate_torso_frame(
            left_shoulder=s_l,
            right_shoulder=s_r,
            left_hip=h_l,
            right_hip=h_r,
            min_span=min_span,
        )
    except (DegenerateGeometryError, NonFiniteVectorError):
        return None
