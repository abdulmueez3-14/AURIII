"""Quality gates and observation assessment package for AURI."""

from auri_cv.quality.landmark_quality import (
    DEFAULT_MIN_PRESENCE,
    DEFAULT_MIN_VISIBILITY,
    DEFAULT_WARNING_VISIBILITY,
    LandmarkQualityGate,
    QualityAssessmentResult,
    QualityStatus,
)

__all__ = [
    "QualityStatus",
    "QualityAssessmentResult",
    "LandmarkQualityGate",
    "DEFAULT_MIN_VISIBILITY",
    "DEFAULT_MIN_PRESENCE",
    "DEFAULT_WARNING_VISIBILITY",
]
