"""Total humerothoracic elevation angle calculation."""

from typing import Final

import numpy as np

from auri_cv.geometry.angles import radians_to_degrees
from auri_cv.geometry.vectors import (
    assert_finite_vector,
    normalize_vector,
    safe_cosine,
)

# Downward reference vector in Torso Frame: -Z_T (headward is +Z_T)
DOWNWARD_TORSO_AXIS: Final[np.ndarray] = np.array([0.0, 0.0, -1.0], dtype=np.float64)


def calculate_humerothoracic_elevation(
    upper_arm_dir_torso: np.ndarray,
    in_degrees: bool = True,
) -> float:
    """Calculates total humerothoracic elevation angle relative to the torso vertical.

    CLINICAL BOUNDARY NOTE:
    This angle represents total humerothoracic elevation — the combined motion of
    the glenohumeral (GH) joint and the scapulothoracic (ST) articulation.
    It must NOT be presented or documented as isolated glenohumeral range of motion.

    Mathematical Formulation:
        d_T = [0, 0, -1]^T (downward torso vertical)
        theta = arccos(clip(d_T . v_arm_T, -1.0, 1.0))
        theta = arccos(clip(-v_z, -1.0, 1.0))

    Angles:
        0.0 deg: Arm hanging straight down along torso (-Z_T).
        90.0 deg: Arm horizontal in any plane (abduction, flexion, scaption).
        180.0 deg: Arm fully elevated overhead (+Z_T).

    Args:
        upper_arm_dir_torso: 3D unit direction vector of the upper arm in Torso Frame.
        in_degrees: If True, returns angle in degrees [0, 180]; otherwise radians [0, pi].

    Returns:
        Elevation angle in [0, 180] degrees (or [0, pi] radians).

    Raises:
        NonFiniteVectorError: If input contains NaN or Inf.
        DegenerateVectorError: If input has near-zero magnitude.
    """
    assert_finite_vector(upper_arm_dir_torso, "upper_arm_dir_torso")
    u_arm = normalize_vector(upper_arm_dir_torso)

    cos_theta = safe_cosine(DOWNWARD_TORSO_AXIS, u_arm)
    theta_rad = float(np.arccos(cos_theta))

    if in_degrees:
        return radians_to_degrees(theta_rad)
    return theta_rad
