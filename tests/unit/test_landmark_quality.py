"""Unit tests for landmark quality gates and assessment logic."""

import numpy as np

from auri_cv.pose.data_models import (
    LEFT_ARM_LANDMARK_NAMES,
    RIGHT_ARM_LANDMARK_NAMES,
    TORSO_LANDMARK_NAMES,
    Landmark3D,
    PoseObservation,
)
from auri_cv.quality.landmark_quality import (
    LandmarkQualityGate,
    QualityStatus,
)


def _build_synthetic_observation(
    landmarks_override: dict[str, tuple[float, float, float, float, float]] | None = None,
) -> PoseObservation:
    """Helper to build a complete test observation with optional landmark overrides.

    Format of tuple: (x, y, z, visibility, presence)
    """
    all_names = (
        set(TORSO_LANDMARK_NAMES) | set(RIGHT_ARM_LANDMARK_NAMES) | set(LEFT_ARM_LANDMARK_NAMES)
    )
    lm_dict: dict[str, Landmark3D] = {}

    for i, name in enumerate(all_names):
        # Default high-quality landmark
        x, y, z, vis, pres = 0.0, 0.0, 1.0, 0.95, 0.95
        if landmarks_override and name in landmarks_override:
            x, y, z, vis, pres = landmarks_override[name]

        lm_dict[name] = Landmark3D(
            name=name,
            index=i,
            x=x,
            y=y,
            z=z,
            visibility=vis,
            presence=pres,
        )

    return PoseObservation(
        timestamp_ms=100.0,
        frame_index=1,
        is_detected=True,
        landmarks_2d={},
        landmarks_3d=lm_dict,
    )


class TestLandmarkQualityGate:
    def test_undetected_pose_returns_unavailable(self):
        gate = LandmarkQualityGate()
        empty = PoseObservation.create_empty()
        result = gate.assess(empty)

        assert result.status == QualityStatus.UNAVAILABLE
        assert result.quality_score == 0.0
        assert not result.is_usable_for_kinematics
        assert "No human pose detected" in result.warnings[0]

    def test_perfect_pose_returns_valid(self):
        gate = LandmarkQualityGate()
        pose = _build_synthetic_observation()
        result = gate.assess(pose, target_arm="right")

        assert result.status == QualityStatus.VALID
        assert result.is_usable_for_kinematics
        assert result.quality_score >= 0.9
        assert len(result.missing_landmarks) == 0
        assert len(result.low_confidence_landmarks) == 0

    def test_missing_landmark_returns_unavailable(self):
        gate = LandmarkQualityGate()
        # Build pose missing RIGHT_ELBOW
        pose = _build_synthetic_observation()
        reduced_landmarks = {k: v for k, v in pose.landmarks_3d.items() if k != "RIGHT_ELBOW"}
        pose_missing = PoseObservation(
            timestamp_ms=100.0,
            frame_index=1,
            is_detected=True,
            landmarks_2d={},
            landmarks_3d=reduced_landmarks,
        )

        result = gate.assess(pose_missing, target_arm="right")
        assert result.status == QualityStatus.UNAVAILABLE
        assert not result.is_usable_for_kinematics
        assert "RIGHT_ELBOW" in result.missing_landmarks

    def test_low_visibility_returns_unavailable(self):
        gate = LandmarkQualityGate(min_visibility=0.5)
        # RIGHT_ELBOW with visibility 0.35 (below 0.5 threshold)
        pose = _build_synthetic_observation(
            landmarks_override={"RIGHT_ELBOW": (0.0, 0.0, 1.0, 0.35, 0.9)}
        )

        result = gate.assess(pose, target_arm="right")
        assert result.status == QualityStatus.UNAVAILABLE
        assert not result.is_usable_for_kinematics
        assert "RIGHT_ELBOW" in result.low_confidence_landmarks

    def test_marginal_visibility_returns_valid_with_warning(self):
        gate = LandmarkQualityGate(min_visibility=0.5, warning_visibility=0.7)
        # RIGHT_ELBOW with visibility 0.60 (between 0.5 and 0.7)
        pose = _build_synthetic_observation(
            landmarks_override={"RIGHT_ELBOW": (0.0, 0.0, 1.0, 0.60, 0.9)}
        )

        result = gate.assess(pose, target_arm="right")
        assert result.status == QualityStatus.VALID_WITH_WARNING
        assert result.is_usable_for_kinematics
        assert len(result.low_confidence_landmarks) == 0
        assert any("marginal visibility" in w for w in result.warnings)

    def test_nan_coordinates_returns_unavailable(self):
        gate = LandmarkQualityGate()
        # Torso shoulder with NaN coordinate
        pose = _build_synthetic_observation(
            landmarks_override={"RIGHT_SHOULDER": (np.nan, 0.0, 1.0, 0.9, 0.9)}
        )

        result = gate.assess(pose, target_arm="right")
        assert result.status == QualityStatus.UNAVAILABLE
        assert not result.is_usable_for_kinematics
        assert "RIGHT_SHOULDER" in result.missing_landmarks
        assert any("non-finite" in w for w in result.warnings)

    def test_target_arm_isolation(self):
        gate = LandmarkQualityGate()
        # LEFT_ELBOW is missing, but target_arm is 'right' -> should still be VALID
        pose = _build_synthetic_observation()
        reduced_landmarks = {k: v for k, v in pose.landmarks_3d.items() if k != "LEFT_ELBOW"}
        pose_no_left = PoseObservation(
            timestamp_ms=100.0,
            frame_index=1,
            is_detected=True,
            landmarks_2d={},
            landmarks_3d=reduced_landmarks,
        )

        # Evaluating right arm ignores left arm dropout
        result_right = gate.assess(pose_no_left, target_arm="right")
        assert result_right.status == QualityStatus.VALID

        # Evaluating left arm fails
        result_left = gate.assess(pose_no_left, target_arm="left")
        assert result_left.status == QualityStatus.UNAVAILABLE
