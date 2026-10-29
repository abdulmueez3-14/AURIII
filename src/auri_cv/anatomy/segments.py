"""Upper-limb anatomical segment estimation and direction vectors."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np

from auri_cv.frames.torso_frame import TorsoFrame
from auri_cv.geometry.angles import angle_between_vectors
from auri_cv.geometry.exceptions import (
    DegenerateGeometryError,
    NonFiniteVectorError,
)
from auri_cv.geometry.vectors import (
    assert_finite_vector,
    normalize_vector,
    vector_norm,
)
from auri_cv.pose.data_models import PoseObservation

DEFAULT_MIN_SEGMENT_LENGTH: Final[float] = 0.10  # Minimum 10 cm segment length


class ArmLaterality(StrEnum):
    """Anatomical arm side designation."""

    RIGHT = "right"
    LEFT = "left"


@dataclass(frozen=True)
class ArmSegmentState:
    """Anatomical segment direction vectors and metric lengths.

    Boundary Guarantee:
    Segments are modeled strictly as unit direction vectors on S^2.
    They do NOT represent full 3D rotation matrices (SO(3)) and do NOT
    determine axial rotation about the longitudinal humeral axis.
    """

    laterality: ArmLaterality
    upper_arm_dir_world: np.ndarray  # (3,) unit vector (Shoulder -> Elbow in World)
    upper_arm_dir_torso: np.ndarray  # (3,) unit vector (Shoulder -> Elbow in Torso)
    upper_arm_length_m: float  # Metric distance between Shoulder and Elbow
    forearm_dir_world: np.ndarray | None  # (3,) unit vector (Elbow -> Wrist in World)
    forearm_dir_torso: np.ndarray | None  # (3,) unit vector (Elbow -> Wrist in Torso)
    forearm_length_m: float | None  # Metric distance between Elbow and Wrist
    elbow_flexion_deg: float | None  # Anatomical elbow flexion: 0 deg = straight, 90 deg = bent


def estimate_arm_segments(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray | None,
    torso_frame: TorsoFrame,
    laterality: ArmLaterality | str = ArmLaterality.RIGHT,
    min_length: float = DEFAULT_MIN_SEGMENT_LENGTH,
) -> ArmSegmentState:
    """Estimates upper-arm and optional forearm direction vectors in World and Torso frames.

    Args:
        shoulder: 3D shoulder joint center in meters.
        elbow: 3D elbow joint center in meters.
        wrist: Optional 3D wrist joint center in meters.
        torso_frame: Estimated TorsoFrame for the active frame.
        laterality: ArmLaterality.RIGHT or ArmLaterality.LEFT.
        min_length: Minimum allowable segment length in meters.

    Returns:
        ArmSegmentState containing S^2 unit vectors and lengths.

    Raises:
        NonFiniteVectorError: If any coordinate is NaN or Inf.
        DegenerateGeometryError: If segment length is below min_length.
    """
    assert_finite_vector(shoulder, "shoulder")
    assert_finite_vector(elbow, "elbow")

    arm_lat = ArmLaterality(laterality)

    # 1. Upper arm segment: Shoulder -> Elbow
    arm_vec_world = elbow - shoulder
    arm_len = vector_norm(arm_vec_world)
    if arm_len < min_length:
        raise DegenerateGeometryError(
            f"Upper arm length too small ({arm_len:.3f} m < {min_length:.3f} m); degenerate segment."
        )

    u_arm_world = normalize_vector(arm_vec_world)
    u_arm_torso = torso_frame.transform_vector_to_torso(u_arm_world)

    # 2. Forearm segment: Elbow -> Wrist (if available)
    u_forearm_world: np.ndarray | None = None
    u_forearm_torso: np.ndarray | None = None
    forearm_len: float | None = None
    elbow_flexion: float | None = None

    if wrist is not None:
        assert_finite_vector(wrist, "wrist")
        forearm_vec_world = wrist - elbow
        forearm_len = vector_norm(forearm_vec_world)

        if forearm_len >= min_length:
            u_forearm_world = normalize_vector(forearm_vec_world)
            u_forearm_torso = torso_frame.transform_vector_to_torso(u_forearm_world)
            # Anatomical elbow flexion: angle between upper arm and forearm direction vectors
            elbow_flexion = angle_between_vectors(
                u_arm_world,
                u_forearm_world,
                in_degrees=True,
            )

    return ArmSegmentState(
        laterality=arm_lat,
        upper_arm_dir_world=u_arm_world,
        upper_arm_dir_torso=u_arm_torso,
        upper_arm_length_m=arm_len,
        forearm_dir_world=u_forearm_world,
        forearm_dir_torso=u_forearm_torso,
        forearm_length_m=forearm_len,
        elbow_flexion_deg=elbow_flexion,
    )


def estimate_arm_segments_from_pose(
    pose: PoseObservation,
    torso_frame: TorsoFrame,
    laterality: ArmLaterality | str = ArmLaterality.RIGHT,
    min_length: float = DEFAULT_MIN_SEGMENT_LENGTH,
) -> ArmSegmentState | None:
    """Convenience helper extracting arm segment state from a PoseObservation.

    Args:
        pose: PoseObservation containing 3D world landmarks.
        torso_frame: Valid TorsoFrame for the observation.
        laterality: ArmLaterality.RIGHT or ArmLaterality.LEFT.
        min_length: Minimum segment length threshold.

    Returns:
        ArmSegmentState if shoulder and elbow are valid; None otherwise.
    """
    if not pose.is_detected:
        return None

    arm_lat = ArmLaterality(laterality)
    prefix = "RIGHT" if arm_lat == ArmLaterality.RIGHT else "LEFT"

    shoulder = pose.get_world_point(f"{prefix}_SHOULDER")
    elbow = pose.get_world_point(f"{prefix}_ELBOW")
    wrist = pose.get_world_point(f"{prefix}_WRIST")

    if shoulder is None or elbow is None:
        return None

    try:
        return estimate_arm_segments(
            shoulder=shoulder,
            elbow=elbow,
            wrist=wrist,
            torso_frame=torso_frame,
            laterality=arm_lat,
            min_length=min_length,
        )
    except (DegenerateGeometryError, NonFiniteVectorError):
        return None
