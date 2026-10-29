"""Synthetic quality gate tests using ground-truth body fixtures."""

from auri_cv.pose.data_models import Landmark3D, PoseObservation
from auri_cv.quality.landmark_quality import (
    LandmarkQualityGate,
    QualityStatus,
)
from tests.synthetic.fixtures import (
    SyntheticBodyLandmarks,
    make_neutral_standing_pose,
    make_right_arm_90deg_abduction_pose,
)


def _pose_observation_from_synthetic(
    synthetic: SyntheticBodyLandmarks,
    default_visibility: float = 0.95,
    default_presence: float = 0.95,
) -> PoseObservation:
    """Converts a synthetic ground truth fixture into a complete PoseObservation."""
    mapping = {
        "LEFT_SHOULDER": synthetic.left_shoulder,
        "RIGHT_SHOULDER": synthetic.right_shoulder,
        "LEFT_HIP": synthetic.left_hip,
        "RIGHT_HIP": synthetic.right_hip,
        "RIGHT_ELBOW": synthetic.right_elbow,
        "RIGHT_WRIST": synthetic.right_wrist,
        "LEFT_ELBOW": synthetic.left_elbow,
        "LEFT_WRIST": synthetic.left_wrist,
    }

    lm3d: dict[str, Landmark3D] = {}
    for i, (name, coords) in enumerate(mapping.items()):
        lm3d[name] = Landmark3D(
            name=name,
            index=i,
            x=float(coords[0]),
            y=float(coords[1]),
            z=float(coords[2]),
            visibility=default_visibility,
            presence=default_presence,
        )

    return PoseObservation(
        timestamp_ms=100.0,
        frame_index=1,
        is_detected=True,
        landmarks_2d={},
        landmarks_3d=lm3d,
    )


def test_synthetic_neutral_pose_quality():
    """Verify that ideal synthetic neutral pose yields 100% valid quality."""
    synthetic = make_neutral_standing_pose()
    obs = _pose_observation_from_synthetic(synthetic)

    gate = LandmarkQualityGate()
    result = gate.assess(obs, target_arm="right")

    assert result.status == QualityStatus.VALID
    assert result.is_usable_for_kinematics
    assert result.quality_score >= 0.90
    assert len(result.warnings) == 0


def test_synthetic_occluded_elbow_triggers_unavailable():
    """Verify that occluding the active elbow drops quality to UNAVAILABLE."""
    synthetic = make_right_arm_90deg_abduction_pose()
    obs = _pose_observation_from_synthetic(synthetic)

    # Artificially occlude the right elbow
    lm_dict = dict(obs.landmarks_3d)
    lm_dict["RIGHT_ELBOW"] = Landmark3D(
        name="RIGHT_ELBOW",
        index=14,
        x=synthetic.right_elbow[0],
        y=synthetic.right_elbow[1],
        z=synthetic.right_elbow[2],
        visibility=0.15,  # Heavily occluded
        presence=0.20,
    )
    occluded_obs = PoseObservation(
        timestamp_ms=obs.timestamp_ms,
        frame_index=obs.frame_index,
        is_detected=True,
        landmarks_2d={},
        landmarks_3d=lm_dict,
    )

    gate = LandmarkQualityGate()
    result = gate.assess(occluded_obs, target_arm="right")

    assert result.status == QualityStatus.UNAVAILABLE
    assert not result.is_usable_for_kinematics
    assert "RIGHT_ELBOW" in result.low_confidence_landmarks


def test_synthetic_marginal_visibility_triggers_warning():
    """Verify that marginal confidence (0.55) triggers VALID_WITH_WARNING."""
    synthetic = make_neutral_standing_pose()
    obs = _pose_observation_from_synthetic(synthetic)

    # Set right shoulder to marginal confidence
    lm_dict = dict(obs.landmarks_3d)
    lm_dict["RIGHT_SHOULDER"] = Landmark3D(
        name="RIGHT_SHOULDER",
        index=12,
        x=synthetic.right_shoulder[0],
        y=synthetic.right_shoulder[1],
        z=synthetic.right_shoulder[2],
        visibility=0.60,  # Marginal
        presence=0.85,
    )
    marginal_obs = PoseObservation(
        timestamp_ms=obs.timestamp_ms,
        frame_index=obs.frame_index,
        is_detected=True,
        landmarks_2d={},
        landmarks_3d=lm_dict,
    )

    gate = LandmarkQualityGate()
    result = gate.assess(marginal_obs, target_arm="right")

    assert result.status == QualityStatus.VALID_WITH_WARNING
    assert result.is_usable_for_kinematics
    assert any("marginal visibility" in w for w in result.warnings)
