"""Compensation metrics: trunk lean, trunk rotation, shoulder hiking.

Lean and rotation are measured relative to a BASELINE TorsoFrame captured at calibration,
so they are independent of the world-axis convention and of camera tilt. They describe
change in trunk posture since calibration, not absolute gravity-referenced lean.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np

from auri_cv.frames.torso_frame import TorsoFrame
from auri_cv.geometry.angles import radians_to_degrees
from auri_cv.geometry.vectors import assert_finite_vector

LEAN_THRESHOLD_DEG: Final[float] = 10.0
HIKING_THRESHOLD_M: Final[float] = 0.03


class CompensationWarning(StrEnum):
    LEAN_DETECTED = "LEAN_DETECTED"
    HIKING_DETECTED = "HIKING_DETECTED"


@dataclass(frozen=True)
class CompensationMetrics:
    lateral_lean_deg: float  # + = trunk tilted toward subject's right, vs baseline
    forward_lean_deg: float  # + = trunk tilted anteriorly, vs baseline
    trunk_rotation_deg: float  # + = right shoulder moved anteriorly (about baseline z_T)
    shoulder_hiking_m: float  # + = ipsilateral shoulder raised vs baseline
    warnings: tuple[CompensationWarning, ...]


def shoulder_height_above_hips(shoulder_world: np.ndarray, frame: TorsoFrame) -> float:
    """dh = (S - H_c) . z_T in meters, with H_c = origin - torso_length * z_T."""
    assert_finite_vector(shoulder_world, "shoulder_world")
    hip_center = frame.origin - frame.torso_length * frame.z_axis
    return float(np.dot(shoulder_world - hip_center, frame.z_axis))


def compute_compensation(
    frame: TorsoFrame,
    baseline_frame: TorsoFrame,
    shoulder_world: np.ndarray,
    baseline_shoulder_height_m: float,
    lean_threshold_deg: float = LEAN_THRESHOLD_DEG,
    hiking_threshold_m: float = HIKING_THRESHOLD_M,
) -> CompensationMetrics:
    """Computes compensation versus the baseline frame.

    Args:
        frame: Current TorsoFrame (world coordinates).
        baseline_frame: TorsoFrame at calibration (same world frame).
        shoulder_world: Current ipsilateral (exercising-side) shoulder, world, meters.
        baseline_shoulder_height_m: `shoulder_height_above_hips` at calibration.
    """
    r = baseline_frame.R_torso_from_world
    z_b = r @ frame.z_axis  # current trunk axis in baseline torso coordinates
    x_b = r @ frame.x_axis  # current lateral axis in baseline torso coordinates

    lateral = radians_to_degrees(float(np.arctan2(z_b[0], z_b[2])))
    forward = radians_to_degrees(float(np.arctan2(z_b[1], z_b[2])))
    rotation = radians_to_degrees(float(np.arctan2(x_b[1], x_b[0])))
    hiking = shoulder_height_above_hips(shoulder_world, frame) - baseline_shoulder_height_m

    warnings: list[CompensationWarning] = []
    if max(abs(lateral), abs(forward)) > lean_threshold_deg:
        warnings.append(CompensationWarning.LEAN_DETECTED)
    if hiking > hiking_threshold_m:
        warnings.append(CompensationWarning.HIKING_DETECTED)
    return CompensationMetrics(lateral, forward, rotation, hiking, tuple(warnings))
