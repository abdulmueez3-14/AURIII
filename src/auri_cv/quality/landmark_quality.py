"""Landmark quality assessment and validation gates for pose observations."""

from enum import StrEnum
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from auri_cv.pose.data_models import (
    LEFT_ARM_LANDMARK_NAMES,
    RIGHT_ARM_LANDMARK_NAMES,
    TORSO_LANDMARK_NAMES,
    PoseObservation,
)

DEFAULT_MIN_VISIBILITY: Final[float] = 0.5
DEFAULT_MIN_PRESENCE: Final[float] = 0.5
DEFAULT_WARNING_VISIBILITY: Final[float] = 0.7


class QualityStatus(StrEnum):
    """Discrete operational health and tracking quality states."""

    VALID = "VALID"
    VALID_WITH_WARNING = "VALID_WITH_WARNING"
    UNAVAILABLE = "UNAVAILABLE"
    RECALIBRATION_REQUIRED = "RECALIBRATION_REQUIRED"


class QualityAssessmentResult(BaseModel):
    """Synthesized quality metrics and operational status for an observation."""

    model_config = ConfigDict(frozen=True)

    status: QualityStatus = Field(description="Operational quality status")
    quality_score: float = Field(
        ge=0.0, le=1.0, description="Normalized continuous tracking quality score [0.0, 1.0]"
    )
    is_usable_for_kinematics: bool = Field(
        description="Whether observation satisfies minimum criteria for joint angle solving"
    )
    missing_landmarks: tuple[str, ...] = Field(
        default_factory=tuple, description="Landmarks not detected in the observation"
    )
    low_confidence_landmarks: tuple[str, ...] = Field(
        default_factory=tuple, description="Landmarks with visibility or presence below threshold"
    )
    warnings: tuple[str, ...] = Field(
        default_factory=tuple, description="Diagnostic warnings regarding observation quality"
    )


class LandmarkQualityGate:
    """Evaluates anatomical landmark presence, visibility, and numerical integrity."""

    def __init__(
        self,
        min_visibility: float = DEFAULT_MIN_VISIBILITY,
        min_presence: float = DEFAULT_MIN_PRESENCE,
        warning_visibility: float = DEFAULT_WARNING_VISIBILITY,
    ) -> None:
        """Initializes quality gate with configurable thresholds.

        Args:
            min_visibility: Strict minimum visibility threshold (below which
                landmarks are rejected).
            min_presence: Strict minimum presence threshold.
            warning_visibility: Threshold below which a warning is issued even if usable.
        """
        self.min_visibility = min_visibility
        self.min_presence = min_presence
        self.warning_visibility = warning_visibility

    def assess(
        self,
        pose: PoseObservation,
        target_arm: str | None = None,
        require_hips: bool = True,
    ) -> QualityAssessmentResult:
        """Assesses pose observation against anatomical requirements.

        Args:
            pose: PoseObservation to evaluate.
            target_arm: 'right', 'left', or None (both arms required).
            require_hips: False drops the hips from the required set (shoulders-only mode).

        Returns:
            QualityAssessmentResult with status and diagnostic lists.
        """
        if not pose.is_detected:
            return QualityAssessmentResult(
                status=QualityStatus.UNAVAILABLE,
                quality_score=0.0,
                is_usable_for_kinematics=False,
                missing_landmarks=(),
                low_confidence_landmarks=(),
                warnings=("No human pose detected in frame",),
            )

        # Assemble required landmarks
        required: list[str] = [n for n in TORSO_LANDMARK_NAMES if require_hips or "HIP" not in n]
        if target_arm == "right":
            required.extend(RIGHT_ARM_LANDMARK_NAMES)
        elif target_arm == "left":
            required.extend(LEFT_ARM_LANDMARK_NAMES)
        else:
            required.extend(RIGHT_ARM_LANDMARK_NAMES)
            required.extend(LEFT_ARM_LANDMARK_NAMES)

        # Remove duplicates while preserving order
        unique_required = tuple(dict.fromkeys(required))

        missing: list[str] = []
        low_confidence: list[str] = []
        warnings: list[str] = []
        visibilities: list[float] = []

        for name in unique_required:
            lm3d = pose.get_landmark_3d(name)
            if lm3d is None:
                missing.append(name)
                visibilities.append(0.0)
                continue

            # Check coordinate finiteness
            coords = np.array([lm3d.x, lm3d.y, lm3d.z])
            if not np.all(np.isfinite(coords)):
                missing.append(name)
                warnings.append(f"Landmark {name} contains non-finite coordinates: {coords}")
                visibilities.append(0.0)
                continue

            vis = lm3d.visibility
            presence = lm3d.presence
            visibilities.append(vis)

            if vis < self.min_visibility or presence < self.min_presence:
                low_confidence.append(name)
                warnings.append(
                    f"Landmark {name} below minimum confidence: vis={vis:.2f} < {self.min_visibility:.2f}, pres={presence:.2f}"
                )
            elif vis < self.warning_visibility:
                warnings.append(
                    f"Landmark {name} has marginal visibility: vis={vis:.2f} < {self.warning_visibility:.2f}"
                )

        mean_quality = float(np.mean(visibilities)) if visibilities else 0.0

        # Determine overall status
        if missing or low_confidence:
            status = QualityStatus.UNAVAILABLE
            usable = False
        elif warnings:
            status = QualityStatus.VALID_WITH_WARNING
            usable = True
        else:
            status = QualityStatus.VALID
            usable = True

        return QualityAssessmentResult(
            status=status,
            quality_score=float(np.clip(mean_quality, 0.0, 1.0)),
            is_usable_for_kinematics=usable,
            missing_landmarks=tuple(missing),
            low_confidence_landmarks=tuple(low_confidence),
            warnings=tuple(warnings),
        )
