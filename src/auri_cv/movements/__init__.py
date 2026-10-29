"""Repetition detection and compensation analysis for AURI."""

from auri_cv.movements.circumduction import CircumductionState, CircumductionTracker
from auri_cv.movements.compensation import (
    CompensationMetrics,
    CompensationWarning,
    compute_compensation,
    shoulder_height_above_hips,
)
from auri_cv.movements.repetition_fsm import Repetition, RepetitionFSM, RepState
from auri_cv.movements.shoulder_movements import (
    MEDIAPIPE_CAMERA,
    SPECS,
    CameraAxes,
    CameraView,
    MotionEstimate,
    MotionStatus,
    MovementSpec,
    MovementSuggestion,
    ShoulderMovement,
    classify_view,
    estimate_motion,
    estimate_subject_yaw_deg,
    suggest_movement,
)

__all__ = [
    "MEDIAPIPE_CAMERA",
    "SPECS",
    "CameraAxes",
    "CameraView",
    "CircumductionState",
    "CircumductionTracker",
    "MotionEstimate",
    "MotionStatus",
    "MovementSpec",
    "MovementSuggestion",
    "ShoulderMovement",
    "classify_view",
    "estimate_motion",
    "estimate_subject_yaw_deg",
    "suggest_movement",
    "CompensationMetrics",
    "CompensationWarning",
    "RepState",
    "Repetition",
    "RepetitionFSM",
    "compute_compensation",
    "shoulder_height_above_hips",
]
