"""Joint-angle solver: ArmSegmentState -> RawKinematics (Torso Frame, degrees)."""

from collections.abc import Callable
from dataclasses import dataclass

from auri_cv.anatomy.segments import ArmSegmentState
from auri_cv.geometry.exceptions import DegenerateVectorError
from auri_cv.kinematics.elevation import calculate_humerothoracic_elevation
from auri_cv.kinematics.projections import (
    calculate_coronal_abduction,
    calculate_plane_of_elevation,
    calculate_sagittal_flexion,
)


@dataclass(frozen=True)
class RawKinematics:
    """Raw (uncalibrated, undeadbanded) shoulder angles at the external boundary (degrees).

    `elevation_deg` is TOTAL HUMEROTHORACIC elevation (glenohumeral + scapulothoracic),
    not isolated glenohumeral range of motion. All angles are measured relative to the
    downward Torso-Frame axis (-Z_T); calibration and display deadband are applied later.

    Planar projection angles are `None` (never a fabricated 0.0) when the arm has no
    measurable component in the relevant plane, e.g. flexion for a purely lateral arm.
    """

    elevation_deg: float  # [0, 180]
    flexion_deg: float | None  # sagittal projection, [-180, 180], + = forward
    abduction_deg: float | None  # coronal projection, [-180, 180], + = away from midline
    plane_of_elevation_deg: float | None  # 0 = coronal, 90 = sagittal
    timestamp_ms: float


def _or_none(fn: Callable[[], float]) -> float | None:
    try:
        return fn()
    except DegenerateVectorError:
        return None


def solve_arm_kinematics(segment: ArmSegmentState, timestamp_ms: float) -> RawKinematics:
    """Solves raw shoulder angles from the upper-arm direction in the Torso Frame."""
    u = segment.upper_arm_dir_torso
    lat = segment.laterality
    return RawKinematics(
        elevation_deg=calculate_humerothoracic_elevation(u),
        flexion_deg=_or_none(lambda: calculate_sagittal_flexion(u)),
        abduction_deg=_or_none(lambda: calculate_coronal_abduction(u, lat)),
        plane_of_elevation_deg=_or_none(lambda: calculate_plane_of_elevation(u, lat)),
        timestamp_ms=timestamp_ms,
    )
