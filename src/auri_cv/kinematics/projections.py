"""Planar projections and angular kinematic solvers in Torso Frame."""

from typing import Final

import numpy as np

from auri_cv.anatomy.segments import ArmLaterality
from auri_cv.geometry.angles import radians_to_degrees
from auri_cv.geometry.exceptions import DegenerateVectorError
from auri_cv.geometry.vectors import (
    assert_finite_vector,
    normalize_vector,
)

# sin(5 deg): below this in-plane component a planar projection angle is not observable
# (atan2 of ~(0, 0) is rounding noise, e.g. flipping between 0 and 180 deg).
MIN_HORIZONTAL_COMPONENT: Final[float] = 0.0871


def _require_in_plane(a: float, b: float, what: str) -> None:
    if float(np.hypot(a, b)) < MIN_HORIZONTAL_COMPONENT:
        raise DegenerateVectorError(
            f"{what} is undefined: arm has (near) no component in that plane "
            f"(< {MIN_HORIZONTAL_COMPONENT})."
        )


def calculate_sagittal_flexion(
    upper_arm_dir_torso: np.ndarray,
    in_degrees: bool = True,
) -> float:
    """Calculates forward shoulder flexion angle in the sagittal plane (Y_T - Z_T).

    Mathematical Formulation:
        theta_flexion = atan2(v_y, -v_z)

    Sign Convention:
        Positive (+): Forward flexion (anterior excursion, v_y > 0).
        Negative (-): Shoulder extension (posterior excursion, v_y < 0).
        0.0 deg: Arm hanging straight down (-Z_T).
        +90.0 deg: Arm horizontal pointing directly forward (+Y_T).
        180.0 deg: Full overhead elevation (+Z_T).

    Args:
        upper_arm_dir_torso: Upper arm unit direction vector in Torso Frame.
        in_degrees: If True, returns degrees [-180, 180]; otherwise radians.

    Returns:
        Signed flexion angle in [-180, 180] degrees (or [-pi, pi] radians).
    """
    assert_finite_vector(upper_arm_dir_torso, "upper_arm_dir_torso")
    u_arm = normalize_vector(upper_arm_dir_torso)

    vy = float(u_arm[1])
    vz = float(u_arm[2])
    _require_in_plane(vy, vz, "sagittal flexion")

    theta_rad = float(np.atan2(vy, -vz))
    if in_degrees:
        return radians_to_degrees(theta_rad)
    return theta_rad


def calculate_coronal_abduction(
    upper_arm_dir_torso: np.ndarray,
    laterality: ArmLaterality | str = ArmLaterality.RIGHT,
    in_degrees: bool = True,
) -> float:
    """Calculates shoulder abduction angle in the coronal plane (X_T - Z_T).

    Mathematical Formulation:
        For Right Arm: lateral is +X_T -> x_lat = v_x
        For Left Arm:  lateral is -X_T -> x_lat = -v_x
        theta_abduction = atan2(x_lat, -v_z)

    Sign Convention:
        Positive (+): Outward abduction (moving arm away from trunk).
        Negative (-): Cross-body adduction (crossing midline).
        0.0 deg: Arm hanging straight down (-Z_T).
        +90.0 deg: Arm horizontal in pure lateral abduction.
        180.0 deg: Full overhead elevation (+Z_T).

    Args:
        upper_arm_dir_torso: Upper arm unit direction vector in Torso Frame.
        laterality: ArmLaterality.RIGHT or ArmLaterality.LEFT.
        in_degrees: If True, returns degrees [-180, 180]; otherwise radians.

    Returns:
        Signed abduction angle in [-180, 180] degrees (or [-pi, pi] radians).
    """
    assert_finite_vector(upper_arm_dir_torso, "upper_arm_dir_torso")
    u_arm = normalize_vector(upper_arm_dir_torso)
    arm_lat = ArmLaterality(laterality)

    # In Torso Frame: +X_T is subject's right
    # For Right arm: lateral is +X_T
    # For Left arm: lateral is -X_T
    vx = float(u_arm[0])
    vz = float(u_arm[2])
    x_lat = vx if arm_lat == ArmLaterality.RIGHT else -vx
    _require_in_plane(x_lat, vz, "coronal abduction")

    theta_rad = float(np.atan2(x_lat, -vz))
    if in_degrees:
        return radians_to_degrees(theta_rad)
    return theta_rad


def calculate_plane_of_elevation(
    upper_arm_dir_torso: np.ndarray,
    laterality: ArmLaterality | str = ArmLaterality.RIGHT,
    in_degrees: bool = True,
) -> float:
    """Calculates the plane of elevation (scaption angle) relative to the coronal plane.

    Mathematical Formulation:
        x_lat = v_x (right) or -v_x (left)
        phi_plane = atan2(v_y, x_lat)

    Sign Convention:
        0.0 deg: Pure coronal abduction (lateral).
        30.0 - 45.0 deg: Scaption plane (scapular plane anterior to coronal).
        90.0 deg: Pure sagittal flexion (anterior).

    Args:
        upper_arm_dir_torso: Upper arm unit direction vector in Torso Frame.
        laterality: ArmLaterality.RIGHT or ArmLaterality.LEFT.
        in_degrees: If True, returns degrees [-180, 180]; otherwise radians.

    Returns:
        Plane of elevation angle in [-180, 180] degrees (or [-pi, pi] radians).
    """
    assert_finite_vector(upper_arm_dir_torso, "upper_arm_dir_torso")
    u_arm = normalize_vector(upper_arm_dir_torso)
    arm_lat = ArmLaterality(laterality)

    vx = float(u_arm[0])
    vy = float(u_arm[1])
    x_lat = vx if arm_lat == ArmLaterality.RIGHT else -vx

    # Near the vertical the horizontal component vanishes and the plane is unobservable.
    _require_in_plane(x_lat, vy, "plane of elevation")

    phi_rad = float(np.atan2(vy, x_lat))
    if in_degrees:
        return radians_to_degrees(phi_rad)
    return phi_rad
